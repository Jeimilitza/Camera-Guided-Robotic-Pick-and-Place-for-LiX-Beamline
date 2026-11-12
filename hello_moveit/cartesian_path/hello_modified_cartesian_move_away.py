#!/usr/bin/env python3
"""
Robust Cartesian move-away version for UR5e + MoveIt 2.

Sequence per cycle:
1. Move to TARGET joint configuration using joint-space planning.
2. Hold and optionally capture an image.
3. Compute the ACTUAL end-effector pose from the ACTUAL joint state.
4. Move away from that actual pose using Cartesian path planning.

This version avoids the most common Cartesian failures from the previous script:
- It does not build the Cartesian path from the ideal FK pose before the robot moves.
- It uses the actual reached joint state as the start of the Cartesian move.
- It uses a small Cartesian interpolation max_step.
- It uses a smaller default move-away distance.
- It tries fallback directions if the preferred direction cannot be planned.
- It does not return to the target after moving away.
"""

import copy
import math
import os
import sys
import time
import threading
from threading import Thread

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from vimba_capture import capture_once
except Exception:
    capture_once = None

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from pymoveit2 import MoveIt2

from geometry_msgs.msg import Pose
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionFK
from sensor_msgs.msg import JointState

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

JOINT_GOAL = [
    1.1046188835872111,
    -0.6567673975254662,
    -0.0429350995990605,
    -2.5001841534818769,
    5.1227158875285568,
    3.1796408312832698,
]

PLANNING_GROUP = "ur_manipulator"
BASE_LINK_NAME = "base_link"
END_EFFECTOR_NAME = "tool0"

NUM_CYCLES = 5

# Keep the joint target accurate, but do not reuse this extremely tight value
# for Cartesian pose planning.
JOINT_GOAL_TOLERANCE = 0.000001
CARTESIAN_POSITION_TOLERANCE = 0.002
CARTESIAN_ORIENTATION_TOLERANCE = 0.01

MAX_VELOCITY_SCALING = 0.10
MAX_ACCELERATION_SCALING = 0.10

# Cartesian planning settings. 1 mm interpolation is much more reliable than
# the 0.008 or 0.08 values used in earlier experiments.
CARTESIAN_MAX_STEP = 0.001
CARTESIAN_FRACTION_THRESHOLD = 0.80
CARTESIAN_AVOID_COLLISIONS = True
FK_TIMEOUT_SEC = 10.0
JOINT_STATE_TIMEOUT_SEC = 10.0

# Requested move-away distance. Start with 5 cm. Increase only after this works.
AWAY_OFFSET_METERS = 0.05

# Execute the away motion in small chunks. If a later chunk fails, the robot has
# still moved away from the target instead of failing the whole 5 cm move.
AWAY_STEP_METERS = 0.005

# Preferred direction for moving away. base_z usually means straight upward in
# the robot base frame, which is often safer than tool_x/tool_y near awkward poses.
PREFERRED_AWAY_AXIS = "base_z"

# Fallbacks are tried only if the preferred axis cannot plan the first step.
# You can reorder this list if your fixture defines "away" differently.
AWAY_AXIS_CANDIDATES = [
    PREFERRED_AWAY_AXIS,
    "base_neg_z",
    "tool_z",
    "tool_neg_z",
    "tool_neg_x",
    "tool_x",
    "tool_y",
    "tool_neg_y",
    "base_y",
    "base_neg_y",
    "base_x",
    "base_neg_x",
]

ENABLE_IMAGE_CAPTURE = True
HOLD_BEFORE_CAPTURE_SEC = 7.0
HOLD_AFTER_CAPTURE_SEC = 3.0


def _set_motion_scaling(moveit2):
    try:
        moveit2.max_velocity_scaling_factor = MAX_VELOCITY_SCALING
    except Exception:
        try:
            moveit2.max_velocity = MAX_VELOCITY_SCALING
        except Exception:
            pass

    try:
        moveit2.max_acceleration_scaling_factor = MAX_ACCELERATION_SCALING
    except Exception:
        try:
            moveit2.max_acceleration = MAX_ACCELERATION_SCALING
        except Exception:
            pass


def _configure_cartesian_options(moveit2):
    try:
        moveit2.cartesian_avoid_collisions = CARTESIAN_AVOID_COLLISIONS
    except Exception:
        pass
    try:
        moveit2.cartesian_jump_threshold = 0.0
    except Exception:
        pass


def _joint_state_from_positions(node, joint_positions):
    joint_state = JointState()
    joint_state.header.stamp = node.get_clock().now().to_msg()
    joint_state.name = list(JOINT_NAMES)
    joint_state.position = [float(value) for value in joint_positions]
    return joint_state


def compute_fk_pose(
    node,
    fk_client,
    joint_positions,
    link_name=END_EFFECTOR_NAME,
    frame_id=BASE_LINK_NAME,
    timeout_sec=FK_TIMEOUT_SEC,
):
    if not fk_client.wait_for_service(timeout_sec=timeout_sec):
        raise RuntimeError("MoveIt compute_fk service is not available")

    request = GetPositionFK.Request()
    request.header.stamp = node.get_clock().now().to_msg()
    request.header.frame_id = frame_id
    request.fk_link_names = [link_name]
    request.robot_state.joint_state = _joint_state_from_positions(node, joint_positions)
    request.robot_state.is_diff = False

    future = fk_client.call_async(request)
    deadline = time.monotonic() + timeout_sec
    while rclpy.ok() and not future.done():
        if time.monotonic() > deadline:
            raise TimeoutError("Timed out waiting for compute_fk response")
        time.sleep(0.01)

    response = future.result()
    if response is None:
        raise RuntimeError("compute_fk returned no response")
    if response.error_code.val != MoveItErrorCodes.SUCCESS:
        raise RuntimeError(f"compute_fk failed with error code {response.error_code.val}")
    if not response.pose_stamped:
        raise RuntimeError("compute_fk returned no poses")

    return response.pose_stamped[0].pose


def pose_position_list(pose: Pose):
    return [pose.position.x, pose.position.y, pose.position.z]


def pose_quat_xyzw_list(pose: Pose):
    return [
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ]


def normalize_quaternion_xyzw(q):
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("Quaternion has zero length")
    return [x / norm, y / norm, z / norm, w / norm]


def rotation_matrix_from_quat_xyzw(q):
    x, y, z, w = normalize_quaternion_xyzw(q)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def vector_neg(vec):
    return [-vec[0], -vec[1], -vec[2]]


def remove_duplicate_axes(axis_names):
    result = []
    seen = set()
    for axis in axis_names:
        axis = axis.lower().strip()
        if axis not in seen:
            result.append(axis)
            seen.add(axis)
    return result


def get_direction_in_base_frame(pose: Pose, axis_name: str):
    axis_name = axis_name.lower().strip()

    base_axes = {
        "base_x": [1.0, 0.0, 0.0],
        "base_neg_x": [-1.0, 0.0, 0.0],
        "base_y": [0.0, 1.0, 0.0],
        "base_neg_y": [0.0, -1.0, 0.0],
        "base_z": [0.0, 0.0, 1.0],
        "base_neg_z": [0.0, 0.0, -1.0],
    }
    if axis_name in base_axes:
        return base_axes[axis_name]

    rotation = rotation_matrix_from_quat_xyzw(pose_quat_xyzw_list(pose))
    tool_x = [rotation[0][0], rotation[1][0], rotation[2][0]]
    tool_y = [rotation[0][1], rotation[1][1], rotation[2][1]]
    tool_z = [rotation[0][2], rotation[1][2], rotation[2][2]]

    tool_axes = {
        "tool_x": tool_x,
        "tool_neg_x": vector_neg(tool_x),
        "tool_y": tool_y,
        "tool_neg_y": vector_neg(tool_y),
        "tool_z": tool_z,
        "tool_neg_z": vector_neg(tool_z),
    }
    if axis_name in tool_axes:
        return tool_axes[axis_name]

    raise ValueError(
        "Unsupported axis. Use base_x, base_neg_x, base_y, base_neg_y, "
        "base_z, base_neg_z, tool_x, tool_neg_x, tool_y, tool_neg_y, "
        "tool_z, or tool_neg_z."
    )


def shifted_pose_along_direction(pose: Pose, direction_in_base, distance_m: float):
    new_pose = copy.deepcopy(pose)
    new_pose.position.x += direction_in_base[0] * distance_m
    new_pose.position.y += direction_in_base[1] * distance_m
    new_pose.position.z += direction_in_base[2] * distance_m
    return new_pose


class JointStateTracker:
    def __init__(self, node):
        self._latest_joints = {}
        self._lock = threading.Lock()
        node.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)

    def _joint_state_cb(self, msg):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                self._latest_joints[name] = pos

    def get_current_joints(self):
        with self._lock:
            if not all(name in self._latest_joints for name in JOINT_NAMES):
                return None
            return [self._latest_joints[name] for name in JOINT_NAMES]

    def wait_for_current_joints(self, timeout_sec=JOINT_STATE_TIMEOUT_SEC):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            joints = self.get_current_joints()
            if joints is not None:
                return joints
            time.sleep(0.01)
        raise TimeoutError("Timed out waiting for /joint_states")


def get_actual_eef_pose(node, fk_client, joint_tracker):
    actual_joints = joint_tracker.wait_for_current_joints()
    return compute_fk_pose(node, fk_client, actual_joints)


def plan_cartesian_pose(logger, moveit2, joint_tracker, pose: Pose, label: str, threshold=None):
    if threshold is None:
        threshold = CARTESIAN_FRACTION_THRESHOLD

    start_joint_state = joint_tracker.wait_for_current_joints()
    logger.info(
        f"Planning Cartesian path to {label}: "
        f"x={pose.position.x:.4f}, y={pose.position.y:.4f}, z={pose.position.z:.4f}"
    )

    trajectory = moveit2.plan(
        position=pose_position_list(pose),
        quat_xyzw=pose_quat_xyzw_list(pose),
        frame_id=BASE_LINK_NAME,
        target_link=END_EFFECTOR_NAME,
        tolerance_position=CARTESIAN_POSITION_TOLERANCE,
        tolerance_orientation=CARTESIAN_ORIENTATION_TOLERANCE,
        start_joint_state=start_joint_state,
        cartesian=True,
        max_step=CARTESIAN_MAX_STEP,
        cartesian_fraction_threshold=threshold,
    )

    return trajectory


def execute_trajectory(logger, moveit2, trajectory, label: str):
    if trajectory is None:
        return False
    logger.info(f"Executing Cartesian path to {label}...")
    moveit2.execute(trajectory)
    if not moveit2.wait_until_executed():
        logger.warn(f"Cartesian execution to {label} did not complete")
        return False
    logger.info(f"Cartesian path to {label} done")
    return True


def plan_and_execute_cartesian_pose(logger, moveit2, joint_tracker, pose: Pose, label: str, threshold=None):
    trajectory = plan_cartesian_pose(logger, moveit2, joint_tracker, pose, label, threshold)
    if trajectory is None:
        logger.warn(f"Cartesian planning to {label} was not accepted")
        return False
    return execute_trajectory(logger, moveit2, trajectory, label)


def move_joint_space_to_target(logger, moveit2, joint_goal):
    logger.info("Planning and executing joint-space move to TARGET...")
    moveit2.move_to_configuration(
        joint_positions=joint_goal,
        tolerance=JOINT_GOAL_TOLERANCE,
    )
    if not moveit2.wait_until_executed():
        raise RuntimeError("Joint-space move to TARGET failed")
    logger.info("TARGET joint move done")


def log_pose(logger, label, pose: Pose):
    q = pose_quat_xyzw_list(pose)
    logger.info(
        f"{label}: position=({pose.position.x:.6f}, {pose.position.y:.6f}, {pose.position.z:.6f}), "
        f"quat_xyzw=({q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f})"
    )


def log_joint_error(logger, joint_tracker, joint_goal, label):
    actual = joint_tracker.wait_for_current_joints()
    logger.info(label)
    for j, name in enumerate(JOINT_NAMES):
        err = actual[j] - joint_goal[j]
        logger.info(
            f"  {name}: actual={actual[j]:.6f}  goal={joint_goal[j]:.6f}  err={err:.6f}"
        )


def try_one_cartesian_step(node, fk_client, logger, moveit2, joint_tracker, axis_name, step_distance):
    actual_pose = get_actual_eef_pose(node, fk_client, joint_tracker)
    direction = get_direction_in_base_frame(actual_pose, axis_name)
    target_pose = shifted_pose_along_direction(actual_pose, direction, step_distance)

    label = f"first away step {axis_name} {step_distance:+.4f} m"
    trajectory = plan_cartesian_pose(logger, moveit2, joint_tracker, target_pose, label)
    if trajectory is None:
        return False

    ok = execute_trajectory(logger, moveit2, trajectory, label)
    return ok


def choose_working_away_axis(node, fk_client, logger, moveit2, joint_tracker):
    axes = remove_duplicate_axes(AWAY_AXIS_CANDIDATES)
    logger.info(f"Finding a Cartesian move-away direction. Candidate axes: {axes}")

    # Try the normal step size first. If no direction works, try a 2 mm step,
    # then a 1 mm step. This avoids failing just because the first requested
    # chunk was too large for the current IK/collision state.
    test_steps = remove_duplicate_axes([])  # placeholder to keep linter quiet
    del test_steps
    for test_step in [AWAY_STEP_METERS, 0.002, 0.001]:
        for axis_name in axes:
            logger.info(f"Testing Cartesian away direction {axis_name} with {test_step:.4f} m step")
            try:
                if try_one_cartesian_step(node, fk_client, logger, moveit2, joint_tracker, axis_name, test_step):
                    logger.info(f"Selected away axis {axis_name}; first step already executed")
                    return axis_name, test_step
            except Exception as exc:
                logger.warn(f"Direction {axis_name} with {test_step:.4f} m step failed: {exc}")

    return None, 0.0


def continue_move_away(node, fk_client, logger, moveit2, joint_tracker, axis_name, already_moved):
    remaining = max(0.0, AWAY_OFFSET_METERS - already_moved)
    moved_total = already_moved
    step_index = 1

    while remaining > 1e-9:
        step_index += 1
        step_distance = min(AWAY_STEP_METERS, remaining)

        actual_pose = get_actual_eef_pose(node, fk_client, joint_tracker)
        direction = get_direction_in_base_frame(actual_pose, axis_name)
        next_pose = shifted_pose_along_direction(actual_pose, direction, step_distance)

        ok = plan_and_execute_cartesian_pose(
            logger,
            moveit2,
            joint_tracker,
            next_pose,
            f"away step {step_index} {axis_name} {step_distance:+.4f} m",
        )
        if not ok:
            # Try a smaller chunk before giving up on the remaining distance.
            reduced_step = min(0.001, remaining)
            logger.warn(
                f"Away step {step_index} failed at {step_distance:.4f} m; "
                f"trying reduced step {reduced_step:.4f} m"
            )
            actual_pose = get_actual_eef_pose(node, fk_client, joint_tracker)
            direction = get_direction_in_base_frame(actual_pose, axis_name)
            next_pose = shifted_pose_along_direction(actual_pose, direction, reduced_step)
            ok = plan_and_execute_cartesian_pose(
                logger,
                moveit2,
                joint_tracker,
                next_pose,
                f"reduced away step {step_index} {axis_name} {reduced_step:+.4f} m",
            )
            if not ok:
                logger.warn(
                    f"Stopped move-away after {moved_total * 100.0:.1f} cm. "
                    "The robot is already away from the target, but the rest of the "
                    "Cartesian path was not accepted."
                )
                return moved_total
            step_distance = reduced_step

        moved_total += step_distance
        remaining -= step_distance
        time.sleep(0.05)

    return moved_total


def capture_image_if_enabled(logger, cycle_index):
    if not ENABLE_IMAGE_CAPTURE:
        return
    if capture_once is None:
        logger.warn("Image capture is enabled, but vimba_capture could not be imported")
        return
    try:
        path = capture_once(cycle=cycle_index)
        logger.info(f"Image saved: {path}")
    except Exception as exc:
        logger.warn(f"Image capture failed, continuing motion: {exc}")


def cartesian_move_away_planner(node, moveit2, joint_goal, loop):
    logger = node.get_logger()
    joint_tracker = JointStateTracker(node)
    fk_client = node.create_client(GetPositionFK, "compute_fk")

    logger.info("Waiting for /joint_states...")
    joint_tracker.wait_for_current_joints()
    logger.info("Joint states are available")

    for i in range(loop):
        logger.info(f"===== CYCLE {i + 1} OF {loop} =====")
        start_time = time.monotonic()

        try:
            move_joint_space_to_target(logger, moveit2, joint_goal)
            time.sleep(0.2)
            log_joint_error(logger, joint_tracker, joint_goal, "TARGET actual joint values:")

            # Important: compute the Cartesian start pose from the actual reached
            # joint state, not from the ideal joint target.
            actual_target_pose = get_actual_eef_pose(node, fk_client, joint_tracker)
            log_pose(logger, "ACTUAL TARGET FK pose", actual_target_pose)

            time.sleep(HOLD_BEFORE_CAPTURE_SEC)
            capture_image_if_enabled(logger, i + 1)
            time.sleep(HOLD_AFTER_CAPTURE_SEC)

            logger.info(
                f"Moving away by up to {AWAY_OFFSET_METERS * 100.0:.1f} cm using Cartesian path "
                f"with max_step={CARTESIAN_MAX_STEP:.4f} m"
            )
            axis_name, first_step = choose_working_away_axis(node, fk_client, logger, moveit2, joint_tracker)
            if axis_name is None:
                logger.warn(
                    "No Cartesian move-away direction was accepted. Target move completed, "
                    "but the robot did not move away. Try reducing AWAY_OFFSET_METERS or "
                    "checking collisions/joint limits in RViz."
                )
                continue

            moved = continue_move_away(
                node,
                fk_client,
                logger,
                moveit2,
                joint_tracker,
                axis_name,
                first_step,
            )
            logger.info(
                f"Move-away complete: moved {moved * 100.0:.1f} cm along {axis_name}. "
                f"Cycle elapsed time: {time.monotonic() - start_time:.3f} seconds"
            )
        except Exception as exc:
            logger.error(f"Cycle {i + 1} stopped: {exc}")
            continue

        logger.info(f"==== END OF CYCLE {i + 1} ====")


def main():
    rclpy.init()

    node = Node("ur5e_move_py_cartesian_move_away_robust")
    callback_group = ReentrantCallbackGroup()

    moveit2 = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES,
        base_link_name=BASE_LINK_NAME,
        end_effector_name=END_EFFECTOR_NAME,
        group_name=PLANNING_GROUP,
        callback_group=callback_group,
    )
    _set_motion_scaling(moveit2)
    _configure_cartesian_options(moveit2)

    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    spinner = Thread(target=executor.spin, daemon=True)
    spinner.start()

    try:
        cartesian_move_away_planner(node, moveit2, JOINT_GOAL, NUM_CYCLES)
    finally:
        rclpy.shutdown()
        spinner.join()


if __name__ == "__main__":
    main()
