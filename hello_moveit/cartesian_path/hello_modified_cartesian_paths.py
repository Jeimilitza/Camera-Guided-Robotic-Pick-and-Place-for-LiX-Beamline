#!/usr/bin/env python3
"""
Variant 2: use Cartesian paths for target, corrected, and home moves.

The original script defines HOME and TARGET as joint configurations. This file
keeps those same numbers, computes their end-effector poses through MoveIt's
compute_fk service, then executes those poses with pymoveit2.move_to_pose(...,
cartesian=True).

"""

import os
import sys
import time
import threading
from threading import Thread

# Allow importing vimba_capture installed alongside this script
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

HOME = [
    1.5683528658421044,
    -0.0041887902047864,
    -0.0397935069454707,
    -4.6891761013331656,
    4.7026151365735211,
    1.5910421461180309,
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
GOAL_TOLERANCE = 0.000001

MAX_VELOCITY_SCALING = 0.10
MAX_ACCELERATION_SCALING = 0.10

# Cartesian planning settings. Reduce max step for smoother paths; increase it
# if planning is too slow. Fraction threshold rejects partial Cartesian paths.
CARTESIAN_MAX_STEP = 0.0025
CARTESIAN_FRACTION_THRESHOLD = 0.95
CARTESIAN_AVOID_COLLISIONS = True
CARTESIAN_JUMP_THRESHOLD = 0.0
FK_TIMEOUT_SEC = 5.0


def _set_motion_scaling(moveit2):
    """Support both newer and older pymoveit2 property names."""
    if hasattr(type(moveit2), "max_velocity"):
        moveit2.max_velocity = MAX_VELOCITY_SCALING
    else:
        moveit2.max_velocity_scaling_factor = MAX_VELOCITY_SCALING

    if hasattr(type(moveit2), "max_acceleration"):
        moveit2.max_acceleration = MAX_ACCELERATION_SCALING
    else:
        moveit2.max_acceleration_scaling_factor = MAX_ACCELERATION_SCALING


def _configure_cartesian_options(moveit2):
    if hasattr(type(moveit2), "cartesian_avoid_collisions"):
        moveit2.cartesian_avoid_collisions = CARTESIAN_AVOID_COLLISIONS
    if hasattr(type(moveit2), "cartesian_jump_threshold"):
        moveit2.cartesian_jump_threshold = CARTESIAN_JUMP_THRESHOLD


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
    """Return the end-effector Pose for a joint configuration."""
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


def move_to_cartesian_pose(logger, moveit2, pose: Pose, label: str):
    logger.info(f"Planning and executing Cartesian path to {label}...")
    moveit2.move_to_pose(
        position=pose_position_list(pose),
        quat_xyzw=pose_quat_xyzw_list(pose),
        frame_id=BASE_LINK_NAME,
        target_link=END_EFFECTOR_NAME,
        tolerance_position=GOAL_TOLERANCE,
        tolerance_orientation=GOAL_TOLERANCE,
        cartesian=True,
        cartesian_max_step=CARTESIAN_MAX_STEP,
        cartesian_fraction_threshold=CARTESIAN_FRACTION_THRESHOLD,
    )
    if not moveit2.wait_until_executed():
        raise RuntimeError(f"Cartesian move to {label} failed")
    logger.info(f"Cartesian path to {label} done")


def cartesian_path_planner(node, moveit2, home, joint_goal, loop):
    logger = node.get_logger()

    latest_joints = {}
    lock = threading.Lock()

    def joint_state_cb(msg):
        with lock:
            for name, pos in zip(msg.name, msg.position):
                latest_joints[name] = pos

    node.create_subscription(JointState, "/joint_states", joint_state_cb, 10)

    def get_current_joints():
        with lock:
            return [latest_joints.get(name, 0.0) for name in JOINT_NAMES]

    fk_client = node.create_client(GetPositionFK, "compute_fk")

    logger.info("Computing FK poses for HOME and TARGET joint configurations...")
    home_pose = compute_fk_pose(node, fk_client, home)
    target_pose = compute_fk_pose(node, fk_client, joint_goal)
    logger.info("FK poses ready; Cartesian path mode is active")

    for i in range(loop):
        logger.info(f"===== CYCLE {i + 1} OF {loop} =====")

        start_time = time.monotonic()

        # MOVE TO TARGET POSE THROUGH A CARTESIAN PATH
        try:
            move_to_cartesian_pose(logger, moveit2, target_pose, "TARGET pose")
        except Exception as e:
            logger.error(str(e))
            continue

        # Compute delta at arrival, same correction logic as original script
        time.sleep(3.1)
        actual = get_current_joints()
        delta = [actual[j] - joint_goal[j] for j in range(len(joint_goal))]
        
        logger.info("TARGET actual joint values:")
        for j in range(len(joint_goal)):
            logger.info(
                f"  J{j}  actual={actual[j]:.6f}  goal={joint_goal[j]:.6f}  err={delta[j]:.6f}"
            )
        
        # 3s into the 10s hold - move to actual + delta as a Cartesian pose
        time.sleep(3)
        
        actual2 = get_current_joints()
        corrected = [actual2[j] + delta[j] for j in range(len(joint_goal))]

        logger.info("Computing FK pose for corrected position (actual + delta)...")
        try:
            corrected_pose = compute_fk_pose(node, fk_client, corrected)
            move_to_cartesian_pose(logger, moveit2, corrected_pose, "CORRECTED pose")
        except Exception as e:
            logger.error(str(e))
            continue
        
        # Give time for the oscillation to stop before capturing picture
        time.sleep(4)
        try:
            path = capture_once(cycle=i + 1)
            logger.info(f"Image saved: {path}")
        except Exception as e:
            logger.error(f"Image capture failed: {e}")

        # Remaining 2s of the hold
        time.sleep(3)

        elapsed = time.monotonic() - start_time
        logger.info(f"Time to reach target: {elapsed:.3f} seconds")

        # RETURN HOME POSE THROUGH A CARTESIAN PATH
        try:
            move_to_cartesian_pose(logger, moveit2, home_pose, "HOME pose")
        except Exception as e:
            logger.error(str(e))
            continue

        actual_home = get_current_joints()
        logger.info("HOME actual joint values:")
        for j in range(len(home)):
            err = actual_home[j] - home[j]
            logger.info(
                f"  J{j}  actual={actual_home[j]:.6f}  goal={home[j]:.6f}  err={err:.6f}"
            )

        logger.info(f"==== END OF CYCLE {i + 1} ====")


def main():
    rclpy.init()

    node = Node("ur5e_move_py_cartesian_paths")
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
        cartesian_path_planner(node, moveit2, HOME, JOINT_GOAL, NUM_CYCLES)
    finally:
        rclpy.shutdown()
        spinner.join()


if __name__ == "__main__":
    main()
