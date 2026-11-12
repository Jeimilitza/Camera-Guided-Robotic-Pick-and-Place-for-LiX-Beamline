#!/usr/bin/env python3
"""
Pick and place experiment on UR5e with AprilTag pose estimation.

The tag detection is done by calling apriltag_runner's preprocess() and detect_tags()
directly, so the pose is read as full-precision floats from the detection objects
(no argv faking, no stdout capture, no regex).
"""
# ros2 launch ur_robot_driver ur_control.launch.py   ur_type:=ur5e   robot_ip:=192.167.2.100   use_mock_hardware:=false   headless_mode:=true   launch_rviz:=false   kinematics_param_file:="/home/kpatel2/ur5e_aug25.yaml"
# ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e launch_rviz:=false
# ros2 launch robotiq_hande_driver gripper_controller_preview.launch.py  set_tool_voltage:="24" use_fake_hardware:=false   create_socat_tty:=true   tty_port:=/tmp/ttyUR   socat_ip_address:=192.167.2.100   socat_port:=54321   frequency_hz:=10   launch_rviz:=false

##############################################################################################################################

# /opt/ros/jazzy/lib/ur_client_library/tool_communication.py 192.167.2.100 --tcp-port 54321 --device-name /tmp/ttyUR
# ros2 launch ur_robot_driver ur_control.launch.py   ur_type:=ur5e   robot_ip:=192.167.2.100   use_mock_hardware:=false   headless_mode:=true   launch_rviz:=false   description_launchfile:=/home/kpatel2/ws_moveit/src/arm_gripper_description/launch/ur_rsp.launch.py   controllers_file:=/home/kpatel2/ws_moveit/src/arm_gripper_description/controllers.yaml   use_tool_communication:=false
# ros2 launch arm_gripper_moveit_config move_group.launch.py

import sys
import time
import threading
from pathlib import Path
from threading import Thread

import cv2
import numpy as np

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from pymoveit2 import MoveIt2

from sensor_msgs.msg import JointState
from control_msgs.action import ParallelGripperCommand

apriltag_reader_dir = Path("/home/kpatel2/ws_moveit/apriltag-reader-nsls-ii")
sys.path.append(str(apriltag_reader_dir))
from apriltag_runner import preprocess, detect_tags

this_dir = Path("/home/kpatel2/ws_moveit/src/arm_gripper_description/src")
sys.path.append(str(this_dir))
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
    1.0235360860824585,
    -1.3455487501672287,
    1.8680532614337366,
    -3.504127641717428,
    5.257328510284424,
    1.5359529256820679,
]

FRONT_PICKUP = [
    1.0642122030258179,
    -0.5147332710078736,
    -0.09414279460906982,
    -1.1568034452250977,
    4.806550979614258,
    -0.3926618734942835,
]

PLANNING_GROUP = "ur_manipulator"
NUM_CYCLES = 1
GOAL_TOLERANCE = 0.000001

GRIPPER_ACTION_NAME = "/gripper_action_controller/gripper_cmd"

GRIPPER_JOINT_NAMES = ["hande_left_finger_joint"]
GRIPPER_CLOSED_POSITION = 0.00485     # meters
GRIPPER_OPEN_POSITION = 0.025         # meters -- fully open (per robotiq_hande_description: 0-25mm/finger)
GRIPPER_MAX_EFFORT = 20.0             # placeholder effort limit; tune to your gripper config
GRIPPER_ACTION_TIMEOUT = 10.0         # seconds to wait for the gripper action server
GRIPPER_SETTLE_TIME = 0.1             # seconds to let the gripper finish moving before continuing

# ----------------------------
# AprilTag detection settings
# ----------------------------
IMAGE_PATH = "/home/kpatel2/ws_moveit/apriltag-reader-nsls-ii/tag_images/capture.png"

TAG_FAMILY = "tag36h11"        # same as the apriltag_runner CLI default (detect_tags itself defaults to tag25h9)
TAG_SIZE_M = 0.005             # black-square edge length in meters
FX = 11714.548810
FY = 11706.389936
CX = 639.771526
CY = 515.834163

TARGET_TAG_ID = None           # None = use the best detection; or set an int to require that ID
N_FRAMES = 5                   # frames captured per measurement (median / mean over valid ones)
MIN_VALID_FRAMES = 3           # fail the measurement if fewer frames give a good detection
MIN_DECISION_MARGIN = 20.0     # reject weak detections
MAX_HAMMING = 0                # reject detections decoded with bit errors
MAX_TRANSLATION_STD_M = 0.0005 # warn if frame-to-frame translation scatter exceeds this (0.5 mm)
FRAME_DELAY = 0.2              # seconds between captures


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


# ----------------------------
# AprilTag pose measurement
# ----------------------------


def detect_tag_pose_once(image_path=IMAGE_PATH):
    """Run the detector on a saved image and return the best tag detection as a dict.

    Returns None if no tag passes the quality gates. The pose is the tag in the camera
    frame, taken at full float precision from the detection object.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    gray = preprocess(img, use_clahe=False, use_sharpen=False, upscale=1.0)

    detections = detect_tags(
        gray,
        families=TAG_FAMILY,
        nthreads=4,
        decimate=1.0,
        sigma=0.0,
        refine_edges=False,
        decode_sharpening=0.25,
        do_pose=True,
        tag_size=TAG_SIZE_M,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
    )

    good = [
        d for d in detections
        if d.decision_margin >= MIN_DECISION_MARGIN
        and d.hamming <= MAX_HAMMING
        and (TARGET_TAG_ID is None or d.tag_id == TARGET_TAG_ID)
    ]
    if not good:
        return None

    best = max(good, key=lambda d: d.decision_margin)
    return {
        "tag_id": int(best.tag_id),
        "t": np.asarray(best.pose_t, dtype=float).reshape(3),
        "R": np.asarray(best.pose_R, dtype=float).reshape(3, 3),
        "margin": float(best.decision_margin),
    }


def nearest_rotation(R_mean):
    """Project an averaged 3x3 matrix back onto a proper rotation matrix (SVD)."""
    U, _, Vt = np.linalg.svd(R_mean)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def measure_tag_pose(logger):
    """Capture several frames and combine the valid detections into one pose.

    Returns (T_cam_tag, info) where T_cam_tag is a 4x4 homogeneous transform of the tag
    in the camera frame (meters), or (None, None) if the measurement failed.
    """
    samples = []
    for i in range(N_FRAMES):
        capture_once()
        det = detect_tag_pose_once()
        if det is None:
            logger.warn(f"Frame {i + 1}/{N_FRAMES}: no valid tag detection")
        else:
            samples.append(det)
        time.sleep(FRAME_DELAY)

    if len(samples) < MIN_VALID_FRAMES:
        logger.error(f"Only {len(samples)}/{N_FRAMES} valid detections (need {MIN_VALID_FRAMES})")
        return None, None

    ids = {s["tag_id"] for s in samples}
    if len(ids) > 1:
        logger.error(f"Inconsistent tag IDs across frames: {sorted(ids)}")
        return None, None

    ts = np.stack([s["t"] for s in samples])
    t_med = np.median(ts, axis=0)
    t_std = ts.std(axis=0)
    R_avg = nearest_rotation(np.mean([s["R"] for s in samples], axis=0))

    if np.any(t_std > MAX_TRANSLATION_STD_M):
        logger.warn(f"Translation scatter is high (std = {t_std} m); check lighting / settling")

    T_cam_tag = np.eye(4)
    T_cam_tag[:3, :3] = R_avg
    T_cam_tag[:3, 3] = t_med

    info = {
        "tag_id": samples[0]["tag_id"],
        "n_valid": len(samples),
        "t_std": t_std,
        "mean_margin": float(np.mean([s["margin"] for s in samples])),
    }
    return T_cam_tag, info


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
    # open_gripper(gripper_client, logger)
    time.sleep(GRIPPER_SETTLE_TIME)

    for i in range(loop):
        logger.info(f"===== CYCLE {i + 1} OF {loop} =====")

        # MOVE TO in front of sample.
        logger.info("Planning and executing move to in front of sample...")
        moveit2.move_to_configuration(joint_positions=front_pickup, tolerance=GOAL_TOLERANCE)
        moveit2.wait_until_executed()
        logger.info("TARGET execution done")
        time.sleep(2)  # let the arm settle before imaging

        # Measure the tag pose (camera frame)
        T_cam_tag, info = measure_tag_pose(logger)
        if T_cam_tag is None:
            logger.error("Tag pose measurement failed; aborting this cycle")
            continue

        logger.info(
            f"Tag {info['tag_id']}: {info['n_valid']}/{N_FRAMES} valid frames, "
            f"mean margin {info['mean_margin']:.1f}"
        )
        logger.info(f"t_cam_tag [m] = {np.array2string(T_cam_tag[:3, 3], precision=6)}")
        logger.info(f"t std [m]     = {np.array2string(info['t_std'], precision=6)}")
        logger.info(f"R_cam_tag =\n{np.array2string(T_cam_tag[:3, :3], precision=6)}")

        # NEXT STEP (needs hand-eye calibration, not done here):
        #   T_base_tag = T_base_tool0 @ T_tool0_camera @ T_cam_tag
        # T_base_tool0 : FK / TF of tool0 at FRONT_PICKUP
        # T_tool0_camera : hand-eye calibration result
        # then apply the fixed tag->object grasp offset and plan to that pose.

        time.sleep(5)


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
