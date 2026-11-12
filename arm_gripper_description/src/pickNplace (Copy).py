#!/usr/bin/env python3
"""
Script to make initial Pick and place experiment on UR5e.
"""
# ros2 launch ur_robot_driver ur_control.launch.py   ur_type:=ur5e   robot_ip:=192.167.2.100   use_mock_hardware:=false   headless_mode:=true   launch_rviz:=false   kinematics_param_file:="/home/kpatel2/ur5e_aug25.yaml" 
# ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e launch_rviz:=false
# ros2 launch robotiq_hande_driver gripper_controller_preview.launch.py  set_tool_voltage:="24" use_fake_hardware:=false   create_socat_tty:=true   tty_port:=/tmp/ttyUR   socat_ip_address:=192.167.2.100   socat_port:=54321   frequency_hz:=10   launch_rviz:=false 

##############################################################################################################################



# /opt/ros/jazzy/lib/ur_client_library/tool_communication.py 192.167.2.100 --tcp-port 54321 --device-name /tmp/ttyUR
# ros2 launch ur_robot_driver ur_control.launch.py   ur_type:=ur5e   robot_ip:=192.167.2.100   use_mock_hardware:=false   headless_mode:=true   launch_rviz:=false   description_launchfile:=/home/kpatel2/ws_moveit/src/arm_gripper_description/launch/ur_rsp.launch.py   controllers_file:=/home/kpatel2/ws_moveit/src/arm_gripper_description/controllers.yaml   use_tool_communication:=false
# ros2 launch arm_gripper_moveit_config move_group.launch.py

import os
import sys
import time
import threading
from threading import Thread
from venv import logger


import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from pymoveit2 import MoveIt2

from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionFK
from control_msgs.action import ParallelGripperCommand

from pathlib import Path
current_dir = Path(__file__).resolve().parent
apriltag_reader_dir = current_dir.parent / "apriltag-reader-nsls-ii"
sys.path.append(str(apriltag_reader_dir))
import apriltag_runner.py

from vimba_capture import capture_once

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

HOME = [
    1.2301520109176636,
    -1.2997780603221436,
    1.897787872944967,
    -3.732370515862936,
    4.974874019622803,
    5.086392402648926,
]

FRONT_PICKUP= [
    1.1080330610275269,
    -0.6375776094249268,
    0.48069698015321904,
    -1.430793122654297,
    4.6799750328063965,
    2.7098805904388428,
]

PICKUP= [
    1.0987077951431274,
    -0.5739746254733582,
    0.8095129171954554,
    -1.8395887813963832,
    4.682865619659424,
    2.641465902328491,
]

FRONT_PLACE= [
    1.2886873483657837,
    -0.5027454656413575,
    0.4940288702594202,
    -1.2308717531016846,
    -0.7296341101275843,
    2.700669527053833,
]

PLACE= [
    1.4092493057250977,
    -0.43961818635974126,
    0.6141875425921839,
    -1.5286696714213868,
    -0.715675179158346,
    2.872771739959717,
]

AWAY= [
    1.3688501119613647,
    -0.6605537694743653,
    0.8922932783709925,
    -1.8791781864561976,
    2.658952474594116,
    3.350132465362549,
]

AWAY2= [
    1.3610490560531616,
    -0.659830169086792,
    0.8925731817828577,
    -1.8923541508116664,
    0.9240360260009766,
    3.350092887878418,
]

PLANNING_GROUP = "ur_manipulator"
NUM_CYCLES = 200
GOAL_TOLERANCE = 0.000001

GRIPPER_ACTION_NAME = "/gripper_action_controller/gripper_cmd"

GRIPPER_JOINT_NAMES = ["hande_left_finger_joint"]
GRIPPER_CLOSED_POSITION = 0.00485     # meters
GRIPPER_OPEN_POSITION = 0.025     # meters -- fully open (per robotiq_hande_description: 0-25mm/finger)
GRIPPER_MAX_EFFORT = 20.0         # placeholder effort limit; tune to your gripper config
GRIPPER_ACTION_TIMEOUT = 10.0     # seconds to wait for the gripper action server
GRIPPER_SETTLE_TIME = 0.1         # seconds to let the gripper finish moving before continuing


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

    def goal_response_callback(future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            logger.error("Gripper goal was rejected")
            done_event.set()
            return
        outcome["accepted"] = True
        result_future = goal_handle.get_result_async()

        def result_callback(fut):
            done_event.set()

        result_future.add_done_callback(result_callback)

    send_goal_future = action_client.send_goal_async(goal_msg)
    send_goal_future.add_done_callback(goal_response_callback)

    if not done_event.wait(timeout=GRIPPER_ACTION_TIMEOUT):
        logger.error("Timed out waiting for gripper action result")
        return False

    return outcome["accepted"]


def close_gripper(action_client, logger):
    logger.info("Closing gripper...")
    send_gripper_command(action_client, GRIPPER_CLOSED_POSITION, logger)


def open_gripper(action_client, logger):
    logger.info("Opening gripper...")
    send_gripper_command(action_client, GRIPPER_OPEN_POSITION, logger)


def joint_trajectory_planner(node, moveit2, gripper_client, home, front_pickup, pickup, front_place, place, away, away2, loop):
    logger = node.get_logger()

    latest_joints = {}
    lock = threading.Lock()

    def joint_state_cb(msg):
        with lock:
            for name, pos in zip(msg.name, msg.position):
                latest_joints[name] = pos

    node.create_subscription(JointState, '/joint_states', joint_state_cb, 10)

    # Ensure a known gripper and arm state before the very first cycle
    logger.info("Planning and executing move to HOME...")
    moveit2.move_to_configuration(joint_positions=home, tolerance=GOAL_TOLERANCE)
    moveit2.wait_until_executed()
    logger.info("HOME execution done")
    # OPEN gripper at the START (home) position before departing
    open_gripper(gripper_client, logger)
    time.sleep(GRIPPER_SETTLE_TIME)

    for i in range(loop):
        logger.info(f"===== CYCLE {i + 1} OF {loop} =====")

        # MOVE TO in front of sample.
        logger.info("Planning and executing move to in front of sample...")
        moveit2.move_to_configuration(joint_positions=front_pickup, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(2)
        
        capture_once()
        sys.argv = ["apriltag_runner.py", "--image", "tag_images/capture.png", "--save-debug"]
        apriltag_runner.main()


        # MOVE TO sample
        logger.info("Planning and executing move to sample...")
        moveit2.move_to_configuration(joint_positions=pickup, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(0.1)

        # CLOSE gripper at the END (target) position
        close_gripper(gripper_client, logger)
        time.sleep(GRIPPER_SETTLE_TIME)
        
        # MOVE TO in front of sample.
        logger.info("Planning and executing move to in front of sample...")
        moveit2.move_to_configuration(joint_positions=front_pickup, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(0.1)
        
        # MOVE away from sample.
        logger.info("Planning and executing move away from sample...")
        moveit2.move_to_configuration(joint_positions=away, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(0.5)
        
        # MOVE away2 from sample.
        logger.info("Planning and executing move away2 from sample...")
        moveit2.move_to_configuration(joint_positions=away2, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(0.5)
        
        # MOVE TO in front of target
        logger.info("Planning and executing move to in front of target...")
        moveit2.move_to_configuration(joint_positions=front_place, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(0.1)

        # MOVE TO target
        logger.info("Planning and executing move to target...")
        moveit2.move_to_configuration(joint_positions=place, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(0.1)

        # OPEN gripper
        open_gripper(gripper_client, logger)
        time.sleep(GRIPPER_SETTLE_TIME)

        # MOVE TO in front of target
        logger.info("Planning and executing move to in front of target...")
        moveit2.move_to_configuration(joint_positions=front_place, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(0.1)

        # MOVE TO target
        logger.info("Planning and executing move to target...")
        moveit2.move_to_configuration(joint_positions=place, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(0.1)

        # CLOSE gripper at the END (target) position
        close_gripper(gripper_client, logger)
        time.sleep(GRIPPER_SETTLE_TIME)
        
        # MOVE TO in front of target
        logger.info("Planning and executing move to in front of target...")
        moveit2.move_to_configuration(joint_positions=front_place, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(0.1)
        
        # MOVE away2 from sample.
        logger.info("Planning and executing move away2 from sample...")
        moveit2.move_to_configuration(joint_positions=away2, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(0.1)
        
        # MOVE away from sample.
        logger.info("Planning and executing move away from sample...")
        moveit2.move_to_configuration(joint_positions=away, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(0.1)

        # MOVE TO in front of sample
        logger.info("Planning and executing move to in front of sample...")
        moveit2.move_to_configuration(joint_positions=front_pickup, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(0.1)

        # MOVE TO sample
        logger.info("Planning and executing move to sample...")
        moveit2.move_to_configuration(joint_positions=pickup, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(0.1)

        # OPEN gripper
        open_gripper(gripper_client, logger)
        time.sleep(GRIPPER_SETTLE_TIME)
        
        # MOVE TO in front of sample
        logger.info("Planning and executing move to in front of sample...")
        moveit2.move_to_configuration(joint_positions=front_pickup, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(0.1)

        # RETURN HOME
        logger.info("Planning and executing move to HOME...")
        moveit2.move_to_configuration(joint_positions=home, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("HOME execution done")
        time.sleep(2)
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

    gripper_client = ActionClient(
        node,
        ParallelGripperCommand,
        GRIPPER_ACTION_NAME,
        callback_group=callback_group,
    )

    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    spinner = Thread(target=executor.spin, daemon=True)
    spinner.start()

    joint_trajectory_planner(node, moveit2, gripper_client, HOME, FRONT_PICKUP, PICKUP, FRONT_PLACE, PLACE, AWAY, AWAY2, NUM_CYCLES)

    rclpy.shutdown()
    spinner.join()


if __name__ == "__main__":
    main()

"""

In front of pick up:
1.1022988557815552,
-0.5622729820064087,
0.6404545942889612,
4.641230745906494,
-1.5886171499835413,
-2.0360735098468226,


On sample pickup:
1.0995420217514038,
-0.5115587276271363,
0.640876595173971,
4.6412078577229,
-1.5887816588031214,
-2.0360263029681605,


Away:
1.1446586847305298,
-0.8566671770862122,
0.6437552610980433,
4.94017984250144,
-1.5735457579242151,
-2.111075226460592,


In front of place:
1.0616601705551147,
-0.5614001315883179,
0.6434920469867151,
4.625825567836426,
-1.564636532460348,
-2.109563652669088,


place:
1.07748281955719,
-0.5087118905833741,
0.646430794392721,
4.577051802272461,
-1.6292083899127405,
-2.109548870717184,


HOME:
1.328967809677124,
-0.9655317229083558,
0.5395215193377894,
4.770629408746519,
-1.6053670088397425,
-1.595947567616598,

















new pickup:


- elbow_joint
- shoulder_lift_joint
- shoulder_pan_joint
- wrist_1_joint
- wrist_2_joint
- wrist_3_joint
position:
- 0.34935218492616826
- -0.6009045404246827
- 1.8250353336334229
- 3.711465998286865
- -1.8315604368792933
- -1.6652067343341272






position 2
 elbow_joint
- shoulder_lift_joint
- shoulder_pan_joint
- wrist_1_joint
- wrist_2_joint
- wrist_3_joint
position:
- 0.34524518648256475
- -0.43508584917102056
- 1.1289507150650024
- 4.865842911392011
- -1.6191695372210901
- 4.352119445800781



position 3

- elbow_joint
- shoulder_lift_joint
- shoulder_pan_joint
- wrist_1_joint
- wrist_2_joint
- wrist_3_joint
position:
- 0.34819299379457647
- -0.3521687549403687
- 1.1121271848678589
- 4.908619972854414
- -1.6637714544879358
- 4.351080894470215

"""
