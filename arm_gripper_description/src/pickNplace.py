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
import tf2_ros
from rclpy.duration import Duration
from rclpy.time import Time

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

HOME2 = [
    1.1409382820129395,
    -1.154371903543808,
    1.5909484068499964,
    -3.3006793461241664,
    -1.1069591681109827,
    2.423982858657837,
]

FRONT_PICKUP = [
    1.0883969068527222,
    -0.8867919606021424,
    0.6093910376178187,
    -1.3889123511365433,
    -1.6293609778033655,
    2.6778504848480225,
]

PICKUP = [
    1.0987077951431274,
    -0.5739746254733582,
    0.8095129171954554,
    -1.8395887813963832,
    4.682865619659424,
    2.641465902328491,
]

FRONT_PLACE = [
    1.108699917793274,
    -0.8345599931529541,
    0.783492390309469,
    -1.1829282206347962,
    -0.7980769316302698,
    2.5932281017303467,
]

PLACE = [
    1.4092493057250977,
    -0.43961818635974126,
    0.6141875425921839,
    -1.5286696714213868,
    -0.715675179158346,
    2.872771739959717,
]

AWAY = [
    1.3688501119613647,
    -0.6605537694743653,
    0.8922932783709925,
    -1.8791781864561976,
    2.658952474594116,
    3.350132465362549,
]

AWAY2 = [
    1.3610490560531616,
    -0.659830169086792,
    0.8925731817828577,
    -1.8923541508116664,
    0.9240360260009766,
    3.350092887878418,
]

PLANNING_GROUP = "ur_manipulator"
NUM_CYCLES = 5
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
N_FRAMES = 1                   # frames captured per measurement (median / mean over valid ones)
MIN_VALID_FRAMES = 1           # fail the measurement if fewer frames give a good detection
MIN_DECISION_MARGIN = 20.0     # reject weak detections
MAX_HAMMING = 0                # reject detections decoded with bit errors
MAX_TRANSLATION_STD_M = 0.0005 # warn if frame-to-frame translation scatter exceeds this (0.5 mm)
FRAME_DELAY = 0.2              # seconds between captures

# ----------------------------
# Move-to-tag settings
# ----------------------------
# 4x4 hand-eye result: pose of the camera frame expressed in tool0 (meters).
# MUST be set from your calibration; the script refuses to move while this is None.
T_TOOL0_CAMERA = np.array([[0.989920316, 0.128049373, 0.060507235, -0.007336026], [-0.136271600, 0.977551853, 0.160693574, 0.024411300], [-0.038572248, -0.167319251, 0.985147933, 0.162342708], [0, 0, 0, 1]])  # e.g. np.array([[...], [...], [...], [0, 0, 0, 1]])

# The gripper point that should land on the tag (e.g. fingertip center). Set ONE of:
GRIPPER_TCP_FRAME = None       # name of a TF frame in your URDF (looked up as tool0 -> frame)
GRIPPER_TCP_IN_TOOL0 = [0, 0, 0.160]    # or [x, y, z] in meters, expressed in the tool0 frame

TAG_TO_TARGET_OFFSET_BASE = [0.017, -0.006, 0.012]  # grasp offset from the tag, in base_link axes (m)
APPROACH_OFFSET_BASE = [0.0, 0.0, 0.06]      # hover this far from the target first, base_link axes (m); all zeros = skip
MAX_MOVE_DISTANCE_M = 0.50     # refuse any move longer than this from the capture pose (guards against bad calibration)
USE_CARTESIAN = True           # straight-line moves; set False to use normal joint-space planning
MOVE_TOL_POS = 0.0005          # meters
MOVE_TOL_ORI = 0.001           # radians
REQUIRE_CONFIRMATION = True    # wait for Enter in the terminal before any move toward the tag


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
        "corners": np.asarray(best.corners, dtype=float).reshape(4, 2),
        "center": np.asarray(best.center, dtype=float).reshape(2),
        "image": img,
    }

def show_detection(det, show_ms=3000, zoom_crop=True, crop_half=60):
    """Show the detected tag outlined on the image for show_ms milliseconds."""
    img = det["image"]
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()

    pts = det["corners"].astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts], True, (0, 255, 0), 1)
    cx, cy = det["center"].astype(int)
    cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)
    cv2.putText(vis, f"ID {det['tag_id']}  margin {det['margin']:.1f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if zoom_crop:
        # Zoomed view of the tag, since a 5 mm tag is tiny in the full frame
        y0, y1 = max(cy - crop_half, 0), min(cy + crop_half, vis.shape[0])
        x0, x1 = max(cx - crop_half, 0), min(cx + crop_half, vis.shape[1])
        crop = cv2.resize(vis[y0:y1, x0:x1], None, fx=6, fy=6,
                          interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Detected tag (zoom)", crop)

    cv2.imshow("Detected tag", vis)
    cv2.waitKey(show_ms)          # window stays up this long, then the script continues
    cv2.destroyAllWindows()
    cv2.waitKey(1)                # lets the windows actually close

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
            show_detection(det)
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


def quat_to_rot(x, y, z, w):
    """Quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def lookup_T(tf_buffer, target_frame, source_frame, timeout_s=2.0):
    """Return (4x4 transform, quaternion xyzw) of source_frame expressed in target_frame."""
    tf = tf_buffer.lookup_transform(
        target_frame, source_frame, Time(), timeout=Duration(seconds=timeout_s)
    )
    tr = tf.transform.translation
    q = tf.transform.rotation
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(q.x, q.y, q.z, q.w)
    T[:3, 3] = [tr.x, tr.y, tr.z]
    return T, [q.x, q.y, q.z, q.w]


def move_gripper_to_tag(moveit2, tf_buffer, T_cam_tag, logger):
    """Move the gripper point (not tool0) to the tag position, keeping the wrist orientation.

    Only x, y, z change: the tool0 orientation at the capture pose is reused for every goal.
    Returns True if the moves were sent, False if they were refused or aborted.
    """
    if T_TOOL0_CAMERA is None:
        logger.error("T_TOOL0_CAMERA (hand-eye calibration) is not set; refusing to move")
        return False

    try:
        T_base_tool0, quat_xyzw = lookup_T(tf_buffer, "base_link", "tool0")
        if GRIPPER_TCP_FRAME is not None:
            T_tool0_tcp, _ = lookup_T(tf_buffer, "tool0", GRIPPER_TCP_FRAME)
            p_tcp = T_tool0_tcp[:3, 3]
        elif GRIPPER_TCP_IN_TOOL0 is not None:
            p_tcp = np.asarray(GRIPPER_TCP_IN_TOOL0, dtype=float)
        else:
            logger.error("Set GRIPPER_TCP_FRAME or GRIPPER_TCP_IN_TOOL0; refusing to move")
            return False
    except tf2_ros.TransformException as e:
        logger.error(f"TF lookup failed: {e}")
        return False

    # Tag position in base_link: T_base_tag = T_base_tool0 @ T_tool0_camera @ T_cam_tag
    T_base_tag = T_base_tool0 @ np.asarray(T_TOOL0_CAMERA, dtype=float) @ T_cam_tag
    p_target = T_base_tag[:3, 3] + np.asarray(TAG_TO_TARGET_OFFSET_BASE, dtype=float)

    # tool0 position that puts the gripper point at a given base_link point,
    # with the tool0 orientation held fixed: p_tool0 = p_goal - R_base_tool0 @ p_tcp
    R_base_tool0 = T_base_tool0[:3, :3]
    p_tool0_now = T_base_tool0[:3, 3]

    goals = []
    approach = np.asarray(APPROACH_OFFSET_BASE, dtype=float)
    if np.any(approach != 0.0):
        goals.append(("approach", p_target + approach - R_base_tool0 @ p_tcp))
    goals.append(("target", p_target - R_base_tool0 @ p_tcp))

    logger.info(f"Tag in base_link [m]: {np.array2string(T_base_tag[:3, 3], precision=5)}")
    logger.info(f"Gripper point offset in tool0 [m]: {np.array2string(p_tcp, precision=5)}")
    for label, p in goals:
        dist = float(np.linalg.norm(p - p_tool0_now))
        logger.info(
            f"{label}: tool0 -> {np.array2string(p, precision=5)} m "
            f"({dist * 1000:.1f} mm from current tool0)"
        )
        if dist > MAX_MOVE_DISTANCE_M:
            logger.error(
                f"{label} is {dist:.3f} m away (limit {MAX_MOVE_DISTANCE_M} m); "
                "check T_TOOL0_CAMERA and the TCP offset. Refusing to move"
            )
            return False

    if REQUIRE_CONFIRMATION:
        answer = input("Move toward the tag? [y/N]: ").strip().lower()
        if answer != "y":
            logger.info("Move cancelled by user")
            return False

    for label, p in goals:
        logger.info(f"Moving to {label}...")
        moveit2.move_to_pose(
            position=[float(v) for v in p],
            quat_xyzw=quat_xyzw,
            target_link="tool0",
            frame_id="base_link",
            tolerance_position=MOVE_TOL_POS,
            tolerance_orientation=MOVE_TOL_ORI,
            cartesian=USE_CARTESIAN,
        )
        moveit2.wait_until_executed()
        logger.info(f"{label} execution done")
    return True


def joint_trajectory_planner(node, moveit2, gripper_client, home, home2, front_pickup, pickup, front_place, place, away, away2, loop, tf_buffer):
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
    moveit2.move_to_configuration(joint_positions=home2, tolerance=GOAL_TOLERANCE)
    moveit2.wait_until_executed()
    logger.info("HOME execution done")
    # OPEN gripper at the START (home) position before departing
    # open_gripper(gripper_client, logger)
    time.sleep(GRIPPER_SETTLE_TIME)

    for i in range(loop):
    
        logger.info(f"===== CYCLE {i + 1} OF {loop} =====")

        # MOVE TO in front of sample.
        logger.info("Planning and executing move to in front of sample...")
        moveit2.move_to_configuration(joint_positions=front_place, tolerance=GOAL_TOLERANCE)
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

        # Move the gripper point to the tag position (x, y, z only; wrist orientation unchanged)
        move_gripper_to_tag(moveit2, tf_buffer, T_cam_tag, logger)

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

    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer, node)

    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    spinner = Thread(target=executor.spin, daemon=True)
    spinner.start()

    joint_trajectory_planner(node, moveit2, gripper_client, HOME, HOME2, FRONT_PICKUP, PICKUP, FRONT_PLACE, PLACE, AWAY, AWAY2, NUM_CYCLES, tf_buffer)

    rclpy.shutdown()
    spinner.join()


if __name__ == "__main__":
    main()
