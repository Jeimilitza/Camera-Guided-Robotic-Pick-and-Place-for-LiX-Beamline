#!/usr/bin/env python3
"""
Script to make initial Pick and place experiment on UR5e.
"""
# ros2 launch ur_robot_driver ur_control.launch.py   ur_type:=ur5e   robot_ip:=192.167.2.100   use_mock_hardware:=false   headless_mode:=true   launch_rviz:=false   kinematics_param_file:="/home/kpatel2/ur5e_aug25.yaml" 
# ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e launch_rviz:=false
# ros2 launch robotiq_hande_driver gripper_controller_preview.launch.py  set_tool_voltage:="24" use_fake_hardware:=false   create_socat_tty:=true   tty_port:=/tmp/ttyUR   socat_ip_address:=192.167.2.100   socat_port:=54321   frequency_hz:=10   launch_rviz:=false 

# Standard Python imports.
import os
import sys
import time
import threading
from threading import Thread
from venv import logger

# ROS 2 imports.
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

# MoveIt interface used to send joint-space motion commands.
from pymoveit2 import MoveIt2

# Message used to read the current robot joint states.
from sensor_msgs.msg import JointState

# Service for forward kinematics.
from moveit_msgs.srv import GetPositionFK

# Action used to control the Robotiq parallel gripper.
from control_msgs.action import ParallelGripperCommand

from ament_index_python.packages import get_package_share_directory
from pathlib import Path
import yaml

# Dynamically find the share directory of the package
package_share_dir = get_package_share_directory("arm_gripper_description")

# Look inside the config/ folder
POSE_FILE = Path(package_share_dir) / "config" / "pick_poses.yaml"

# These are the six joints on the UR5e.
# The order here needs to match the order expected by MoveIt.
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

def load_poses(path=POSE_FILE):
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    poses = data["poses"]

    # Simply convert each list directly, already in the correct order
    return {
        pose_name: [float(value) for value in pose]
        for pose_name, pose in poses.items()
    }

POSES = load_poses(POSE_FILE)

HOME = POSES["home"]
FRONT_PICKUP = POSES["front_pickup"]
PICKUP = POSES["pickup"]
PLACE = POSES["place"]

# MoveIt planning group for the UR5e.
PLANNING_GROUP = "ur_manipulator"
NUM_CYCLES = 20 # Number of times the whole sequence will repeat.
GOAL_TOLERANCE = 0.000001 # How close the final joint positions need to be to the requested positions.

# Name of the ROS 2 action server used by the gripper.
GRIPPER_ACTION_NAME = "/gripper_action_controller/gripper_cmd"

# Joint name used by the Robotiq Hand-E gripper.
GRIPPER_JOINT_NAMES = ["hande_left_finger_joint"]
GRIPPER_CLOSED_POSITION = 0.00485     # width to grab the screwdriver in meters
GRIPPER_OPEN_POSITION = 0.025     # in meters -- fully open (per robotiq_hande_description: 0-25mm/finger)
GRIPPER_MAX_EFFORT = 20.0         # placeholder effort limit; tune to your gripper config
GRIPPER_ACTION_TIMEOUT = 10.0     # seconds to wait for the gripper action server
GRIPPER_SETTLE_TIME = 0.5         # seconds to let the gripper finish moving before continuing


def send_gripper_command(action_client, position, logger, max_effort=GRIPPER_MAX_EFFORT):
    """Send a ParallelGripperCommand goal and block (via an Event) until it completes.

    The goal's `command` field is a sensor_msgs/JointState: parallel arrays of joint
    name/position/effort, not a single float like the classic GripperCommand action.
    """
    if not action_client.wait_for_server(timeout_sec=GRIPPER_ACTION_TIMEOUT):
        logger.error(f"Gripper action server '{GRIPPER_ACTION_NAME}' not available")
        return False

    goal_msg = ParallelGripperCommand.Goal()
    goal_msg.command.name = GRIPPER_JOINT_NAMES
    goal_msg.command.position = [position] * len(GRIPPER_JOINT_NAMES)
    goal_msg.command.effort = [max_effort] * len(GRIPPER_JOINT_NAMES)

    done_event = threading.Event()
    outcome = {"accepted": False}

    # This callback gets called after the action server responds to our goal.
    def goal_response_callback(future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            logger.error("Gripper goal was rejected")
            done_event.set()
            return
        outcome["accepted"] = True
        result_future = goal_handle.get_result_async()

        # Callback used once the gripper action finishes.
        def result_callback(fut):
            done_event.set()

        result_future.add_done_callback(result_callback)

    # Send the gripper goal asynchronously.
    send_goal_future = action_client.send_goal_async(goal_msg)
    # Register the callback that handles the goal response.
    send_goal_future.add_done_callback(goal_response_callback)

    # Wait here until the action finishes or the timeout is reached.
    if not done_event.wait(timeout=GRIPPER_ACTION_TIMEOUT):
        logger.error("Timed out waiting for gripper action result")
        return False

    # Return whether the action goal was accepted.
    return outcome["accepted"]

# Simple helper function for closing the gripper.
def close_gripper(action_client, logger):
    logger.info("Closing gripper...")
    send_gripper_command(action_client, GRIPPER_CLOSED_POSITION, logger)

# Simple helper function for opening the gripper.
def open_gripper(action_client, logger):
    logger.info("Opening gripper...")
    send_gripper_command(action_client, GRIPPER_OPEN_POSITION, logger)


def joint_trajectory_planner(node, moveit2, gripper_client, home, front_pickup, pickup, place, loop):
    logger = node.get_logger()

    latest_joints = {}
    lock = threading.Lock()

    # Callback that receives new joint states from /joint_states.
    def joint_state_cb(msg):
        with lock:
            for name, pos in zip(msg.name, msg.position):
                latest_joints[name] = pos

    # Subscribe to the robot joint states.
    node.create_subscription(JointState, '/joint_states', joint_state_cb, 10)

    # Ensure a known gripper and arm state before the very first cycle
    # Move the robot to HOME
    logger.info("Planning and executing move to HOME...")
    moveit2.move_to_configuration(joint_positions=home, tolerance=GOAL_TOLERANCE)
    moveit2.wait_until_executed()
    logger.info("HOME execution done")
    # OPEN gripper at the START (home) position before departing
    open_gripper(gripper_client, logger)
    time.sleep(GRIPPER_SETTLE_TIME)

    for i in range(loop):
        logger.info(f"===== CYCLE {i + 1} OF {loop} =====")

        # MOVE TO above screwdriver.
        logger.info("Planning and executing move to above screwdriver...")
        moveit2.move_to_configuration(joint_positions=place, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(0.5)

        # CLOSE gripper at the END (target) position
        close_gripper(gripper_client, logger)
        time.sleep(GRIPPER_SETTLE_TIME)

        # RETURN HOME
        logger.info("Planning and executing move to HOME...")
        moveit2.move_to_configuration(joint_positions=home, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("HOME execution done")
        time.sleep(0.5)

        # MOVE to in front of pick-and-place location
        logger.info("Planning and executing move to front of pick-and-place location...")
        moveit2.move_to_configuration(joint_positions=front_pickup, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(0.5)

        # MOVE TO pick-and-place location
        logger.info("Planning and executing move to pick-and-place location...")
        moveit2.move_to_configuration(joint_positions=pickup, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(0.5)

        # OPEN gripper
        open_gripper(gripper_client, logger)
        time.sleep(GRIPPER_SETTLE_TIME)

        # CLOSE gripper at the END (target) position
        close_gripper(gripper_client, logger)
        time.sleep(GRIPPER_SETTLE_TIME)

        # Return to in front of pick-and-place location
        logger.info("Planning and executing move to in front of pick-and-place location...")
        moveit2.move_to_configuration(joint_positions=front_pickup, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(0.5)

        # RETURN HOME
        logger.info("Planning and executing move to HOME...")
        moveit2.move_to_configuration(joint_positions=home, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("HOME execution done")
        time.sleep(1)

        # MOVE TO above place location
        logger.info("Planning and executing move to above screwdriver...")
        moveit2.move_to_configuration(joint_positions=place, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(0.5)

        # OPEN gripper
        open_gripper(gripper_client, logger)
        time.sleep(GRIPPER_SETTLE_TIME)

        # RETURN HOME
        logger.info("Planning and executing move to HOME...")
        moveit2.move_to_configuration(joint_positions=home, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("HOME execution done")
        time.sleep(1)
        logger.info(f"==== END OF CYCLE {i + 1} ====")


def main():
    # Start ROS 2.
    rclpy.init()

    # Create the ROS 2 node.
    node = Node("ur5e_move_py")

    # Use a reentrant callback group so multiple callbacks
    # can be processed by the multithreaded executor.
    callback_group = ReentrantCallbackGroup()

    # Create the MoveIt2 interface for the UR5e.
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

    # Create an action client that communicates with the
    # Robotiq Hand-E gripper action server.
    gripper_client = ActionClient(
        node,
        ParallelGripperCommand,
        GRIPPER_ACTION_NAME,
        callback_group=callback_group,
    )

    # Use two executor threads so ROS callbacks can continue
    # running while the main thread waits for robot actions.
    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    spinner = Thread(target=executor.spin, daemon=True)
    spinner.start()

    # Run the full trajectory sequence for NUM_CYCLES cycles.
    joint_trajectory_planner(node, moveit2, gripper_client, HOME, FRONT_PICKUP, PICKUP, PLACE, NUM_CYCLES)


    # Shut down ROS 2 after the experiment is finished.
    rclpy.shutdown()

    # Wait for the executor thread to finish.
    spinner.join()


if __name__ == "__main__":
    main()

