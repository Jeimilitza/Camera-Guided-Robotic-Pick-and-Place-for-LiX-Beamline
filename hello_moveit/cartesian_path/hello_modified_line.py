#!/usr/bin/env python3
"""
Attempt at making the repeatability test but only moving away from the target position in a straight line before returning.

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

from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionFK

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

HOME = [
    0.9672351786208672,
    -0.3086055347164280,
    -1.0451844767773690,
    -1.8504128812602052,
    5.2598497422905510,
    3.1889856586315040,
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
NUM_CYCLES = 200
GOAL_TOLERANCE = 0.000001


def _quat_inverse(q):
    x, y, z, w = q
    return [-x, -y, -z, w]


def _quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ]


def joint_trajectory_planner(node, moveit2, home, joint_goal, loop):
    logger = node.get_logger()

    latest_joints = {}
    lock = threading.Lock()

    def joint_state_cb(msg):
        with lock:
            for name, pos in zip(msg.name, msg.position):
                latest_joints[name] = pos

    node.create_subscription(JointState, '/joint_states', joint_state_cb, 10)

    def get_current_joints():
        with lock:
            return [latest_joints.get(name, 0.0) for name in JOINT_NAMES]

    for i in range(loop):
        logger.info(f"===== CYCLE {i + 1} OF {loop} =====")

        start_time = time.monotonic()

        # MOVE TO TARGET
        logger.info("Planning and executing move to TARGET...")
        moveit2.move_to_configuration(joint_positions=joint_goal, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")

        # Compute delta at arrival
        time.sleep(0.1)
        actual = get_current_joints()
        delta = [actual[j] - joint_goal[j] for j in range(len(joint_goal))]

        logger.info("TARGET actual joint values:")
        for j in range(len(joint_goal)):
            logger.info(
                f"  J{j}  actual={actual[j]:.6f}  goal={joint_goal[j]:.6f}  err={delta[j]:.6f}"
            )

        time.sleep(0.1)

        actual2 = get_current_joints()
        corrected = [actual2[j] + delta[j] for j in range(len(joint_goal))]

        logger.info("Moving to corrected position (actual + delta)...")
        moveit2.move_to_configuration(joint_positions=corrected, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("Corrected move done")
        
        #Give time for the oscillation to stop before capturing picture
        time.sleep(3)
        try:
            path = capture_once(cycle=i + 1)
            logger.info(f"Image saved: {path}")
        except Exception as e:
            logger.error(f"Image capture failed: {e}")

        time.sleep(0.1)

        elapsed = time.monotonic() - start_time
        logger.info(f"Time to reach target: {elapsed:.3f} seconds")

        # RETURN HOME
        logger.info("Planning and executing move to HOME...")
        moveit2.move_to_configuration(joint_positions=home, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("HOME execution done")

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

    node = Node("ur5e_move_py")

    callback_group = ReentrantCallbackGroup()

    moveit2 = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES,
        base_link_name="base_link",
        end_effector_name="tool0",
        group_name=PLANNING_GROUP,
        callback_group=callback_group,
    )
    moveit2.max_velocity_scaling_factor = 0.1
    moveit2.max_acceleration_scaling_factor = 0.1

    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    spinner = Thread(target=executor.spin, daemon=True)
    spinner.start()

    joint_trajectory_planner(node, moveit2, HOME, JOINT_GOAL, NUM_CYCLES)

    rclpy.shutdown()
    spinner.join()


if __name__ == "__main__":
    main()
