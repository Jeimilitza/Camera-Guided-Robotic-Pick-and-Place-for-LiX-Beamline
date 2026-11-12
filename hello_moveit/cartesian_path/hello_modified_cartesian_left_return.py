#!/usr/bin/env python3
"""
Robust Cartesian left/return version for UR5e + MoveIt 2.

Sequence per cycle:
1. Move to TARGET joint configuration.
2. Capture image.
3. Move 5 cm left using a Cartesian path in the TOOL frame.
4. Return back to the TARGET pose using a Cartesian path.

Note: I kept the move to TARGET as a joint-space move for reliability, then the left and return motions are Cartesian.

"""

import copy
import math
import os
import sys
import time
import threading
from threading import Thread

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vimba_capture import capture_once

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
GOAL_TOLERANCE = 0.00001

MAX_VELOCITY_SCALING = 0.10
MAX_ACCELERATION_SCALING = 0.10

# Cartesian planning settings.
# Smaller max_step gives MoveIt more interpolation samples and is usually more
# reliable near awkward configurations, but takes a little longer to plan.
CARTESIAN_MAX_STEP = 0.008
CARTESIAN_FRACTION_THRESHOLD = 0.90
CARTESIAN_AVOID_COLLISIONS = True
FK_TIMEOUT_SEC = 10.0
JOINT_STATE_TIMEOUT_SEC = 10.0

# Requested left motion
LEFT_OFFSET_METERS = 0.20

# Use small Cartesian chunks for the move.
LEFT_STEP_METERS = 0.005

# Choose the frame direction for "left".
# Options:
#   "tool_x", "tool_neg_x", "tool_y", "tool_neg_y",
#   "base_y", "base_neg_y", "base_x", "base_neg_x"
LEFT_AXIS = "tool_neg_x"

# The target move is kept as a joint move for reliability. The left and return
# motions are Cartesian paths. Set this to False if you specifically want the
# TARGET move itself to be attempted as a Cartesian path too.
USE_JOINT_MOVE_TO_TARGET = True


def _set_motion_scaling(moveit2):
    if hasattr(type(moveit2), "max_velocity"):
        moveit2.max_velocity = MAX_VELOCITY_SCALING
    else:
        moveit2.max_velocity_scaling_factor = MAX_VELOCITY_SCALING

    if hasattr(type(moveit2), "max_acceleration"):
        moveit2.max_acceleration = MAX_ACCELERATION_SCALING
    else:
        moveit2.max_acceleration_scaling_factor = MAX_ACCELERATION_SCALING


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
    norm = math.sqrt(x*x + y*y + z*z + w*w)
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


def vector_scale(vec, scalar):
    return [vec[0] * scalar, vec[1] * scalar, vec[2] * scalar]


def vector_neg(vec):
    return [-vec[0], -vec[1], -vec[2]]


def get_left_direction_in_base_frame(target_pose: Pose, axis_name: str):
    axis_name = axis_name.lower().strip()

    if axis_name == "base_x":
        return [1.0, 0.0, 0.0]
    if axis_name == "base_neg_x":
        return [-1.0, 0.0, 0.0]
    if axis_name == "base_y":
        return [0.0, 1.0, 0.0]
    if axis_name == "base_neg_y":
        return [0.0, -1.0, 0.0]

    rotation = rotation_matrix_from_quat_xyzw(pose_quat_xyzw_list(target_pose))
    tool_x = [rotation[0][0], rotation[1][0], rotation[2][0]]
    tool_y = [rotation[0][1], rotation[1][1], rotation[2][1]]

    if axis_name == "tool_x":
        return tool_x
    if axis_name == "tool_neg_x":
        return vector_neg(tool_x)
    if axis_name == "tool_y":
        return tool_y
    if axis_name == "tool_neg_y":
        return vector_neg(tool_y)

    raise ValueError(
        "Unsupported LEFT_AXIS. Use one of: tool_x, tool_neg_x, tool_y, "
        "tool_neg_y, base_y, base_neg_y, base_x, base_neg_x."
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
        self._node = node
        node.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)

    def _joint_state_cb(self, msg):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                self._latest_joints[name] = pos

    def have_all_joints(self):
        with self._lock:
            return all(name in self._latest_joints for name in JOINT_NAMES)

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
        tolerance_position=GOAL_TOLERANCE,
        tolerance_orientation=GOAL_TOLERANCE,
        start_joint_state=start_joint_state,
        cartesian=True,
        max_step=CARTESIAN_MAX_STEP,
        cartesian_fraction_threshold=threshold,
    )

    if trajectory is None:
        raise RuntimeError(f"Cartesian planning to {label} failed")
    return trajectory


def execute_trajectory(logger, moveit2, trajectory, label: str):
    logger.info(f"Executing Cartesian path to {label}...")
    moveit2.execute(trajectory)
    if not moveit2.wait_until_executed():
        raise RuntimeError(f"Cartesian execution to {label} failed")
    logger.info(f"Cartesian path to {label} done")


def move_to_cartesian_pose(logger, moveit2, joint_tracker, pose: Pose, label: str, threshold=None):
    trajectory = plan_cartesian_pose(logger, moveit2, joint_tracker, pose, label, threshold)
    execute_trajectory(logger, moveit2, trajectory, label)


def move_along_direction_stepwise(logger, moveit2, joint_tracker, start_pose: Pose, direction_in_base, total_distance_m: float):
    current_pose = copy.deepcopy(start_pose)
    direction_sign = 1.0 if total_distance_m >= 0.0 else -1.0
    remaining = abs(total_distance_m)
    step_index = 0

    while remaining > 1e-9:
        step_distance = min(LEFT_STEP_METERS, remaining) * direction_sign
        step_index += 1
        next_pose = shifted_pose_along_direction(current_pose, direction_in_base, step_distance)
        move_to_cartesian_pose(
            logger,
            moveit2,
            joint_tracker,
            next_pose,
            f"Cartesian step {step_index} distance {step_distance:+.4f} m",
        )
        current_pose = next_pose
        remaining -= abs(step_distance)
        time.sleep(0.05)

    return current_pose


def log_pose(logger, label, pose: Pose):
    q = pose_quat_xyzw_list(pose)
    logger.info(
        f"{label}: position=({pose.position.x:.6f}, {pose.position.y:.6f}, {pose.position.z:.6f}), "
        f"quat_xyzw=({q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f})"
    )


def cartesian_left_return_planner(node, moveit2, joint_goal, loop):
    logger = node.get_logger()
    joint_tracker = JointStateTracker(node)
    fk_client = node.create_client(GetPositionFK, "compute_fk")

    logger.info("Waiting for /joint_states...")
    joint_tracker.wait_for_current_joints()
    logger.info("Joint states are available")

    logger.info("Computing FK pose for TARGET joint configuration...")
    target_pose = compute_fk_pose(node, fk_client, joint_goal)
    log_pose(logger, "TARGET FK pose", target_pose)

    left_direction = get_left_direction_in_base_frame(target_pose, LEFT_AXIS)
    logger.info(
        f"LEFT_AXIS={LEFT_AXIS}, LEFT_OFFSET_METERS={LEFT_OFFSET_METERS:+.3f}, "
        f"direction in {BASE_LINK_NAME}=({left_direction[0]:+.4f}, "
        f"{left_direction[1]:+.4f}, {left_direction[2]:+.4f})"
    )

    left_pose = shifted_pose_along_direction(target_pose, left_direction, LEFT_OFFSET_METERS)
    log_pose(logger, "LEFT pose", left_pose)

    for i in range(loop):
        logger.info(f"===== CYCLE {i + 1} OF {loop} =====")
        start_time = time.monotonic()

        try:
            if USE_JOINT_MOVE_TO_TARGET:
                logger.info("Planning and executing joint-space move to TARGET first...")
                moveit2.move_to_configuration(joint_positions=joint_goal, tolerance=GOAL_TOLERANCE)
                if not moveit2.wait_until_executed():
                    raise RuntimeError("Joint-space move to TARGET failed")
                logger.info("TARGET joint move done")
            else:
                move_to_cartesian_pose(logger, moveit2, joint_tracker, target_pose, "TARGET pose")
        except Exception as e:
            logger.error(str(e))
            continue

        time.sleep(0.1)
        actual = joint_tracker.wait_for_current_joints()
        delta = [actual[j] - joint_goal[j] for j in range(len(joint_goal))]

        logger.info("TARGET actual joint values:")
        for j in range(len(joint_goal)):
            logger.info(
                f"  J{j}  actual={actual[j]:.6f}  goal={joint_goal[j]:.6f}  err={delta[j]:.6f}"
            )

        time.sleep(7)
        try:
            path = capture_once(cycle=i + 1)
            logger.info(f"Image saved: {path}")
        except Exception as e:
            logger.error(f"Image capture failed: {e}")

        time.sleep(3)
        logger.info(f"Time before left/return motion: {time.monotonic() - start_time:.3f} seconds")

        try:
            logger.info(
                f"Moving {abs(LEFT_OFFSET_METERS) * 100.0:.1f} cm along {LEFT_AXIS} "
                f"using {LEFT_STEP_METERS * 100.0:.1f} cm Cartesian steps..."
            )
            reached_left_pose = move_along_direction_stepwise(
                logger,
                moveit2,
                joint_tracker,
                target_pose,
                left_direction,
                LEFT_OFFSET_METERS,
            )

            logger.info("Returning back to TARGET pose using Cartesian steps...")
            move_along_direction_stepwise(
                logger,
                moveit2,
                joint_tracker,
                reached_left_pose,
                left_direction,
                -LEFT_OFFSET_METERS,
            )
        except Exception as e:
            logger.error(str(e))
            logger.error(
                "Cartesian left/return failed. Try LEFT_AXIS='tool_neg_x' first. "
                "If needed, try 'tool_y' or 'tool_neg_y'. Avoid base_y at this target unless "
                "you intentionally want base-frame motion near a singular direction."
            )
            continue

        actual_return = joint_tracker.wait_for_current_joints()
        logger.info("TARGET return actual joint values:")
        for j in range(len(joint_goal)):
            err = actual_return[j] - joint_goal[j]
            logger.info(
                f"  J{j}  actual={actual_return[j]:.6f}  goal={joint_goal[j]:.6f}  err={err:.6f}"
            )

        logger.info(f"==== END OF CYCLE {i + 1} ====")


def main():
    rclpy.init()

    node = Node("ur5e_move_py_cartesian_tool_left_return")
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
        cartesian_left_return_planner(node, moveit2, JOINT_GOAL, NUM_CYCLES)
    finally:
        rclpy.shutdown()
        spinner.join()


if __name__ == "__main__":
    main()
