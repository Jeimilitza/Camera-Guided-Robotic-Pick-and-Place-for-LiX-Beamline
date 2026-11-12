#!/usr/bin/env python3
"""
Cartesian repeatability loop for a UR5e with MoveIt 2 and pymoveit2.

Normal sequence:
    1. Position the robot at HOME once with a joint-space move.
    2. Pre-plan a complete Cartesian HOME -> TARGET trajectory.
    3. Pre-plan the complete Cartesian TARGET -> HOME trajectory.
    4. Execute HOME -> TARGET.
    5. Wait for the robot to settle, activate Vimba, and capture one image.
    6. Execute TARGET -> HOME.
    7. Repeat.

The repeated HOME <-> TARGET motions are Cartesian. The one-time move to HOME
and emergency recovery use joint-space planning so the program always starts
from a known state and can recover safely.

"""

from __future__ import annotations

import copy
import math
import os
import sys
import threading
import time
from threading import Thread
from typing import Dict, List, Optional, Sequence, Tuple

# Allow importing vimba_capture.py when it is installed beside this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from vimba_capture import capture_once
except Exception as exc:  # Camera import failure must not prevent safe motion.
    capture_once = None
    VIMBA_IMPORT_ERROR = str(exc)
else:
    VIMBA_IMPORT_ERROR = ""

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath, GetPositionFK
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
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

TARGET = [
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

NUM_CYCLES = 200

# Used only for the one-time HOME move and emergency HOME recovery.
JOINT_MOVE_TOLERANCE = 0.00001
MAX_VELOCITY_SCALING = 0.30
MAX_ACCELERATION_SCALING = 0.30

# Cartesian planning parameters.
CARTESIAN_MAX_STEP = 0.005  # 1 mm interpolation inside each waypoint segment.
CARTESIAN_REQUIRED_FRACTION = 0.999
CARTESIAN_AVOID_COLLISIONS = True
CARTESIAN_JUMP_THRESHOLD = 0.0
CARTESIAN_PRISMATIC_JUMP_THRESHOLD = 0.0
CARTESIAN_REVOLUTE_JUMP_THRESHOLD = 0.0
MAX_CARTESIAN_SPEED_MPS = 0.05

# Start with 24 known-reachable FK waypoints. If a complete round trip cannot
# be planned, automatically retry with denser waypoint sets before moving.
CARTESIAN_WAYPOINT_COUNTS = (24, 48, 96)

# Repeatability timing.
TARGET_SETTLE_BEFORE_CAPTURE_SEC = 3.0
HOLD_AFTER_CAPTURE_SEC = 0.25
HOME_SETTLE_SEC = 1.0

ENABLE_IMAGE_CAPTURE = True

JOINT_STATE_TIMEOUT_SEC = 10.0
SERVICE_TIMEOUT_SEC = 20.0
MOTION_START_TIMEOUT_SEC = 5.0
MOTION_FINISH_TIMEOUT_SEC = 180.0

# If an endpoint differs from the requested Cartesian pose by more than these
# values, report it. These checks do not command a correction movement.
POSITION_REPORT_TOLERANCE_M = 0.002
ORIENTATION_REPORT_TOLERANCE_RAD = 0.02

# Used to decide whether a pre-planned return trajectory still starts close
# enough to the actual reached target joint state.
RETURN_START_JOINT_TOLERANCE_RAD = 0.02

# Only used after a failed execution or if the robot is unexpectedly not HOME.
HOME_RECOVERY_JOINT_TOLERANCE_RAD = 0.02


class JointStateTracker:
    """Thread-safe ordered view of /joint_states."""

    def __init__(self, node: Node, callback_group: ReentrantCallbackGroup):
        self._latest: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._subscription = node.create_subscription(
            JointState,
            "/joint_states",
            self._callback,
            10,
            callback_group=callback_group,
        )

    def _callback(self, msg: JointState) -> None:
        with self._lock:
            for name, position in zip(msg.name, msg.position):
                self._latest[name] = float(position)

    def get(self) -> Optional[List[float]]:
        with self._lock:
            if not all(name in self._latest for name in JOINT_NAMES):
                return None
            return [self._latest[name] for name in JOINT_NAMES]

    def wait(self, timeout_sec: float = JOINT_STATE_TIMEOUT_SEC) -> List[float]:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            current = self.get()
            if current is not None:
                return current
            time.sleep(0.01)
        raise TimeoutError("Timed out waiting for all required /joint_states")


def joint_state_message(node: Node, positions: Sequence[float]) -> JointState:
    msg = JointState()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.name = list(JOINT_NAMES)
    msg.position = [float(value) for value in positions]
    return msg


def wait_for_future(future, timeout_sec: float, operation: str):
    deadline = time.monotonic() + timeout_sec
    while rclpy.ok() and not future.done():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out while waiting for {operation}")
        time.sleep(0.01)

    if not future.done():
        raise RuntimeError(f"{operation} stopped because ROS was shut down")

    exception = future.exception()
    if exception is not None:
        raise RuntimeError(f"{operation} failed: {exception}")

    result = future.result()
    if result is None:
        raise RuntimeError(f"{operation} returned no response")
    return result


def compute_fk_pose(
    node: Node,
    fk_client,
    joint_positions: Sequence[float],
) -> Pose:
    request = GetPositionFK.Request()
    request.header.stamp = node.get_clock().now().to_msg()
    request.header.frame_id = BASE_LINK_NAME
    request.fk_link_names = [END_EFFECTOR_NAME]
    request.robot_state.joint_state = joint_state_message(node, joint_positions)
    request.robot_state.is_diff = False

    response = wait_for_future(
        fk_client.call_async(request),
        SERVICE_TIMEOUT_SEC,
        "compute_fk",
    )

    if response.error_code.val != MoveItErrorCodes.SUCCESS:
        raise RuntimeError(
            f"compute_fk returned MoveIt error code {response.error_code.val}"
        )
    if not response.pose_stamped:
        raise RuntimeError("compute_fk returned an empty pose list")

    return copy.deepcopy(response.pose_stamped[0].pose)


class FKCache:
    """Cache FK results because the same Cartesian routes repeat every cycle."""

    def __init__(self, node: Node, fk_client):
        self._node = node
        self._fk_client = fk_client
        self._cache: Dict[Tuple[float, ...], Pose] = {}

    @staticmethod
    def _key(joints: Sequence[float]) -> Tuple[float, ...]:
        return tuple(round(float(value), 12) for value in joints)

    def pose(self, joints: Sequence[float]) -> Pose:
        key = self._key(joints)
        if key not in self._cache:
            self._cache[key] = compute_fk_pose(
                self._node,
                self._fk_client,
                joints,
            )
        return copy.deepcopy(self._cache[key])


def interpolate_joints(
    start: Sequence[float],
    goal: Sequence[float],
    fraction: float,
) -> List[float]:
    return [
        float(start_value + fraction * (goal_value - start_value))
        for start_value, goal_value in zip(start, goal)
    ]


def quaternion_tuple(pose: Pose) -> Tuple[float, float, float, float]:
    return (
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    )


def make_quaternion_sign_continuous(previous: Pose, current: Pose) -> None:
    """q and -q are the same rotation; keep adjacent waypoint signs consistent."""

    previous_q = quaternion_tuple(previous)
    current_q = quaternion_tuple(current)
    dot = sum(a * b for a, b in zip(previous_q, current_q))
    if dot < 0.0:
        current.orientation.x *= -1.0
        current.orientation.y *= -1.0
        current.orientation.z *= -1.0
        current.orientation.w *= -1.0


def build_fk_waypoints(
    fk_cache: FKCache,
    start_joints: Sequence[float],
    goal_joints: Sequence[float],
    waypoint_count: int,
) -> List[Pose]:
    if waypoint_count < 1:
        raise ValueError("waypoint_count must be at least 1")

    waypoints: List[Pose] = []
    previous_pose = fk_cache.pose(start_joints)

    for index in range(1, waypoint_count + 1):
        fraction = index / waypoint_count
        reference_joints = interpolate_joints(start_joints, goal_joints, fraction)
        pose = fk_cache.pose(reference_joints)
        make_quaternion_sign_continuous(previous_pose, pose)
        waypoints.append(pose)
        previous_pose = pose

    return waypoints


def configure_cartesian_request_compatibility(request) -> None:
    """Set fields that differ slightly between MoveIt message releases."""

    if hasattr(request, "max_velocity_scaling_factor"):
        request.max_velocity_scaling_factor = MAX_VELOCITY_SCALING
    if hasattr(request, "max_acceleration_scaling_factor"):
        request.max_acceleration_scaling_factor = MAX_ACCELERATION_SCALING

    if hasattr(request, "prismatic_jump_threshold"):
        request.prismatic_jump_threshold = CARTESIAN_PRISMATIC_JUMP_THRESHOLD
    if hasattr(request, "revolute_jump_threshold"):
        request.revolute_jump_threshold = CARTESIAN_REVOLUTE_JUMP_THRESHOLD

    if hasattr(request, "cartesian_speed_limited_link"):
        request.cartesian_speed_limited_link = END_EFFECTOR_NAME
    elif hasattr(request, "cartesian_speed_end_effector_link"):
        request.cartesian_speed_end_effector_link = END_EFFECTOR_NAME

    if hasattr(request, "max_cartesian_speed"):
        request.max_cartesian_speed = MAX_CARTESIAN_SPEED_MPS


def plan_cartesian_waypoints(
    node: Node,
    cartesian_client,
    start_joints: Sequence[float],
    waypoints: Sequence[Pose],
):
    request = GetCartesianPath.Request()
    request.header.stamp = node.get_clock().now().to_msg()
    request.header.frame_id = BASE_LINK_NAME
    request.start_state.joint_state = joint_state_message(node, start_joints)
    request.start_state.is_diff = False
    request.group_name = PLANNING_GROUP
    request.link_name = END_EFFECTOR_NAME
    request.waypoints = [copy.deepcopy(pose) for pose in waypoints]
    request.max_step = CARTESIAN_MAX_STEP
    request.jump_threshold = CARTESIAN_JUMP_THRESHOLD
    request.avoid_collisions = CARTESIAN_AVOID_COLLISIONS
    configure_cartesian_request_compatibility(request)

    return wait_for_future(
        cartesian_client.call_async(request),
        SERVICE_TIMEOUT_SEC,
        "compute_cartesian_path",
    )


def trajectory_endpoint_in_required_order(joint_trajectory) -> List[float]:
    if not joint_trajectory.points:
        raise RuntimeError("Cartesian planner returned an empty trajectory")

    names = list(joint_trajectory.joint_names)
    positions = list(joint_trajectory.points[-1].positions)
    by_name = dict(zip(names, positions))

    missing = [name for name in JOINT_NAMES if name not in by_name]
    if missing:
        raise RuntimeError(
            "Cartesian trajectory is missing required joints: " + ", ".join(missing)
        )

    return [float(by_name[name]) for name in JOINT_NAMES]


def trajectory_has_valid_timing(joint_trajectory) -> bool:
    if not joint_trajectory.points:
        return False

    previous_time = -1.0
    has_positive_time = False
    for point in joint_trajectory.points:
        seconds = float(point.time_from_start.sec) + (
            float(point.time_from_start.nanosec) * 1e-9
        )
        if seconds < previous_time:
            return False
        if seconds > 0.0:
            has_positive_time = True
        previous_time = seconds

    return has_positive_time


def cartesian_response_is_complete(response) -> bool:
    return (
        response.error_code.val == MoveItErrorCodes.SUCCESS
        and response.fraction >= CARTESIAN_REQUIRED_FRACTION
        and bool(response.solution.joint_trajectory.points)
        and trajectory_has_valid_timing(response.solution.joint_trajectory)
    )


def angle_difference(actual: float, expected: float) -> float:
    return math.atan2(
        math.sin(actual - expected),
        math.cos(actual - expected),
    )


def maximum_joint_error(
    actual: Sequence[float],
    expected: Sequence[float],
) -> float:
    return max(
        abs(angle_difference(actual_value, expected_value))
        for actual_value, expected_value in zip(actual, expected)
    )


def pose_error(actual: Pose, expected: Pose) -> Tuple[float, float]:
    dx = actual.position.x - expected.position.x
    dy = actual.position.y - expected.position.y
    dz = actual.position.z - expected.position.z
    position_error = math.sqrt(dx * dx + dy * dy + dz * dz)

    aq = quaternion_tuple(actual)
    eq = quaternion_tuple(expected)
    an = math.sqrt(sum(value * value for value in aq))
    en = math.sqrt(sum(value * value for value in eq))
    if an <= 0.0 or en <= 0.0:
        orientation_error = math.inf
    else:
        dot = abs(sum(a * e for a, e in zip(aq, eq)) / (an * en))
        dot = max(-1.0, min(1.0, dot))
        orientation_error = 2.0 * math.acos(dot)

    return position_error, orientation_error


def motion_state_is_idle(moveit2: MoveIt2) -> bool:
    state = moveit2.query_state()
    return getattr(state, "name", str(state).split(".")[-1]) == "IDLE"


def wait_for_motion_without_spinning(
    moveit2: MoveIt2,
    description: str,
) -> bool:
    """Poll MoveIt2 state while the dedicated background executor handles callbacks."""

    start_deadline = time.monotonic() + MOTION_START_TIMEOUT_SEC
    saw_active_state = False

    while rclpy.ok() and time.monotonic() < start_deadline:
        if not motion_state_is_idle(moveit2):
            saw_active_state = True
            break
        time.sleep(0.01)

    if not saw_active_state:
        return False

    finish_deadline = time.monotonic() + MOTION_FINISH_TIMEOUT_SEC
    while rclpy.ok() and time.monotonic() < finish_deadline:
        if motion_state_is_idle(moveit2):
            error_code = moveit2.get_last_execution_error_code()
            return bool(
                moveit2.motion_suceeded
                and error_code is not None
                and error_code.val == MoveItErrorCodes.SUCCESS
            )
        time.sleep(0.02)

    try:
        moveit2.cancel_execution()
    except Exception:
        pass
    raise TimeoutError(f"Timed out while executing {description}")


def execute_joint_trajectory(
    logger,
    moveit2: MoveIt2,
    joint_trajectory,
    description: str,
) -> None:
    logger.info(f"Executing {description}...")
    moveit2.execute(joint_trajectory)
    if not wait_for_motion_without_spinning(moveit2, description):
        raise RuntimeError(f"Execution failed for {description}")
    logger.info(f"{description} complete")


def move_joint_space(
    logger,
    moveit2: MoveIt2,
    joint_positions: Sequence[float],
    description: str,
) -> None:
    logger.info(f"Planning and executing {description}...")
    moveit2.move_to_configuration(
        joint_positions=[float(value) for value in joint_positions],
        tolerance=JOINT_MOVE_TOLERANCE,
    )
    if not wait_for_motion_without_spinning(moveit2, description):
        raise RuntimeError(f"Joint-space motion failed for {description}")
    logger.info(f"{description} complete")


def log_joint_error(
    logger,
    actual: Sequence[float],
    expected: Sequence[float],
    label: str,
) -> None:
    logger.info(label)
    for name, actual_value, expected_value in zip(
        JOINT_NAMES,
        actual,
        expected,
    ):
        error = angle_difference(actual_value, expected_value)
        logger.info(
            f"  {name}: actual={actual_value:.6f} "
            f"goal={expected_value:.6f} err={error:.6f}"
        )


def log_pose_error(
    logger,
    node: Node,
    fk_client,
    actual_joints: Sequence[float],
    expected_pose: Pose,
    label: str,
) -> None:
    actual_pose = compute_fk_pose(node, fk_client, actual_joints)
    position_error, orientation_error = pose_error(actual_pose, expected_pose)
    logger.info(
        f"{label}: position error={position_error * 1000.0:.3f} mm, "
        f"orientation error={math.degrees(orientation_error):.4f} deg"
    )
    if (
        position_error > POSITION_REPORT_TOLERANCE_M
        or orientation_error > ORIENTATION_REPORT_TOLERANCE_RAD
    ):
        logger.warning(
            f"{label} exceeds the reporting tolerance; no correction move is applied"
        )


def capture_image(logger, cycle_number: int) -> None:
    if not ENABLE_IMAGE_CAPTURE:
        logger.info("Image capture disabled")
        return

    if capture_once is None:
        logger.warning(
            "Vimba capture is unavailable; returning HOME without an image. "
            f"Import error: {VIMBA_IMPORT_ERROR}"
        )
        return

    try:
        # capture_once opens/activates the Vimba camera, captures one frame,
        # saves it, and returns the saved path.
        image_path = capture_once(cycle=cycle_number)
        logger.info(f"Image saved: {image_path}")
    except Exception as exc:
        # Never leave the arm at TARGET because of a camera failure.
        logger.warning(f"Image capture failed; returning HOME: {exc}")


def build_route_candidates(
    fk_cache: FKCache,
    route_cache: Dict[int, Tuple[List[Pose], List[Pose]]],
    waypoint_count: int,
) -> Tuple[List[Pose], List[Pose]]:
    if waypoint_count not in route_cache:
        forward = build_fk_waypoints(fk_cache, HOME, TARGET, waypoint_count)
        reverse = build_fk_waypoints(fk_cache, TARGET, HOME, waypoint_count)
        route_cache[waypoint_count] = (forward, reverse)
    return route_cache[waypoint_count]


def plan_complete_round_trip(
    node: Node,
    logger,
    cartesian_client,
    fk_cache: FKCache,
    route_cache: Dict[int, Tuple[List[Pose], List[Pose]]],
    actual_home_joints: Sequence[float],
    preferred_waypoint_count: Optional[int],
):
    candidate_counts = list(CARTESIAN_WAYPOINT_COUNTS)
    if preferred_waypoint_count in candidate_counts:
        candidate_counts.remove(preferred_waypoint_count)
        candidate_counts.insert(0, preferred_waypoint_count)

    attempt_details = []

    for waypoint_count in candidate_counts:
        logger.info(
            f"Pre-planning complete Cartesian round trip with "
            f"{waypoint_count} FK waypoints per direction..."
        )
        forward_waypoints, reverse_waypoints = build_route_candidates(
            fk_cache,
            route_cache,
            waypoint_count,
        )

        forward_response = plan_cartesian_waypoints(
            node,
            cartesian_client,
            actual_home_joints,
            forward_waypoints,
        )
        forward_fraction = float(forward_response.fraction)

        if not cartesian_response_is_complete(forward_response):
            attempt_details.append(
                f"{waypoint_count}: outbound fraction={forward_fraction:.6f}, "
                f"error={forward_response.error_code.val}"
            )
            continue

        planned_target_joints = trajectory_endpoint_in_required_order(
            forward_response.solution.joint_trajectory
        )

        reverse_response = plan_cartesian_waypoints(
            node,
            cartesian_client,
            planned_target_joints,
            reverse_waypoints,
        )
        reverse_fraction = float(reverse_response.fraction)

        if not cartesian_response_is_complete(reverse_response):
            attempt_details.append(
                f"{waypoint_count}: outbound fraction={forward_fraction:.6f}, "
                f"return fraction={reverse_fraction:.6f}, "
                f"return error={reverse_response.error_code.val}"
            )
            continue

        logger.info(
            f"Complete round trip accepted with {waypoint_count} waypoints: "
            f"outbound fraction={forward_fraction:.6f}, "
            f"return fraction={reverse_fraction:.6f}"
        )
        return (
            waypoint_count,
            forward_response.solution.joint_trajectory,
            reverse_response.solution.joint_trajectory,
            planned_target_joints,
        )

    detail_text = "; ".join(attempt_details) if attempt_details else "no response details"
    raise RuntimeError(
        "MoveIt could not plan a complete Cartesian HOME -> TARGET -> HOME "
        f"round trip. Nothing was executed. Attempts: {detail_text}"
    )


def ensure_home(
    logger,
    moveit2: MoveIt2,
    joint_tracker: JointStateTracker,
) -> List[float]:
    actual = joint_tracker.wait()
    error = maximum_joint_error(actual, HOME)
    if error > HOME_RECOVERY_JOINT_TOLERANCE_RAD:
        move_joint_space(logger, moveit2, HOME, "one-time move/recovery to HOME")
        time.sleep(0.2)
        actual = joint_tracker.wait()
    return actual


def repeatability_loop(
    node: Node,
    moveit2: MoveIt2,
    joint_tracker: JointStateTracker,
    fk_client,
    cartesian_client,
) -> None:
    logger = node.get_logger()

    logger.info("Waiting for /joint_states...")
    joint_tracker.wait()
    logger.info("Joint states are available")

    if not fk_client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_SEC):
        raise RuntimeError("MoveIt compute_fk service is not available")
    if not cartesian_client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_SEC):
        raise RuntimeError("MoveIt compute_cartesian_path service is not available")

    fk_cache = FKCache(node, fk_client)
    route_cache: Dict[int, Tuple[List[Pose], List[Pose]]] = {}

    logger.info("Computing HOME and TARGET Cartesian poses from joint values...")
    home_pose = fk_cache.pose(HOME)
    target_pose = fk_cache.pose(TARGET)
    logger.info(
        f"HOME pose: x={home_pose.position.x:.6f}, "
        f"y={home_pose.position.y:.6f}, z={home_pose.position.z:.6f}"
    )
    logger.info(
        f"TARGET pose: x={target_pose.position.x:.6f}, "
        f"y={target_pose.position.y:.6f}, z={target_pose.position.z:.6f}"
    )

    # Establish the known starting state once. Normal loop motion after this is Cartesian.
    actual_home = ensure_home(logger, moveit2, joint_tracker)
    log_joint_error(logger, actual_home, HOME, "Initial HOME joint values:")
    time.sleep(HOME_SETTLE_SEC)

    preferred_waypoint_count: Optional[int] = None

    for cycle_index in range(NUM_CYCLES):
        cycle_number = cycle_index + 1
        logger.info(f"===== CYCLE {cycle_number} OF {NUM_CYCLES} =====")
        cycle_start = time.monotonic()

        # Verify/recover HOME only if a previous execution left the robot elsewhere.
        actual_home = ensure_home(logger, moveit2, joint_tracker)

        try:
            (
                preferred_waypoint_count,
                outbound_trajectory,
                return_trajectory,
                planned_target_joints,
            ) = plan_complete_round_trip(
                node,
                logger,
                cartesian_client,
                fk_cache,
                route_cache,
                actual_home,
                preferred_waypoint_count,
            )

            execute_joint_trajectory(
                logger,
                moveit2,
                outbound_trajectory,
                "Cartesian HOME -> TARGET trajectory",
            )

            # Endpoint diagnostics are useful, but a transient FK or joint-state
            # problem must never prevent the already-planned Cartesian return.
            actual_target = list(planned_target_joints)
            try:
                actual_target = joint_tracker.wait()
                log_joint_error(
                    logger,
                    actual_target,
                    TARGET,
                    "TARGET actual joint values (Cartesian pose may have an equivalent IK solution):",
                )
                log_pose_error(
                    logger,
                    node,
                    fk_client,
                    actual_target,
                    target_pose,
                    "TARGET Cartesian accuracy",
                )
            except Exception as diagnostic_exc:
                logger.warning(
                    "TARGET diagnostics failed; continuing with the pre-planned "
                    f"Cartesian return: {diagnostic_exc}"
                )

            logger.info(
                f"Holding TARGET for {TARGET_SETTLE_BEFORE_CAPTURE_SEC:.2f} seconds "
                "before activating Vimba..."
            )
            time.sleep(TARGET_SETTLE_BEFORE_CAPTURE_SEC)
            capture_image(logger, cycle_number)
            time.sleep(HOLD_AFTER_CAPTURE_SEC)

            # The return path was already proven feasible before the robot moved.
            # Verify its expected start still matches the reached target.
            return_start_error = maximum_joint_error(
                actual_target,
                planned_target_joints,
            )
            if return_start_error > RETURN_START_JOINT_TOLERANCE_RAD:
                logger.info(
                    "Actual TARGET differs from the preview return start; "
                    "re-planning the Cartesian return from the actual state"
                )
                _, reverse_waypoints = build_route_candidates(
                    fk_cache,
                    route_cache,
                    preferred_waypoint_count,
                )
                replanned_return = plan_cartesian_waypoints(
                    node,
                    cartesian_client,
                    actual_target,
                    reverse_waypoints,
                )
                if not cartesian_response_is_complete(replanned_return):
                    raise RuntimeError(
                        "Cartesian return re-plan was incomplete: "
                        f"fraction={replanned_return.fraction:.6f}, "
                        f"error={replanned_return.error_code.val}"
                    )
                return_trajectory = replanned_return.solution.joint_trajectory

            execute_joint_trajectory(
                logger,
                moveit2,
                return_trajectory,
                "Cartesian TARGET -> HOME trajectory",
            )

            try:
                actual_return = joint_tracker.wait()
                log_joint_error(logger, actual_return, HOME, "HOME actual joint values:")
                log_pose_error(
                    logger,
                    node,
                    fk_client,
                    actual_return,
                    home_pose,
                    "HOME Cartesian accuracy",
                )
            except Exception as diagnostic_exc:
                logger.warning(f"HOME diagnostics failed: {diagnostic_exc}")
            time.sleep(HOME_SETTLE_SEC)

            logger.info(
                f"==== END OF CYCLE {cycle_number}; "
                f"elapsed={time.monotonic() - cycle_start:.3f} seconds ===="
            )

        except Exception as exc:
            logger.error(f"Cycle {cycle_number} stopped safely: {exc}")
            logger.warning("Attempting joint-space recovery to HOME")
            try:
                move_joint_space(
                    logger,
                    moveit2,
                    HOME,
                    "emergency recovery to HOME",
                )
                time.sleep(HOME_SETTLE_SEC)
            except Exception as recovery_exc:
                logger.error(
                    "HOME recovery failed; stopping the repeatability loop: "
                    f"{recovery_exc}"
                )
                break


def main() -> None:
    rclpy.init()

    node = Node("ur5e_cartesian_repeatability")

    # Separate reentrant groups keep MoveIt actions and auxiliary services from
    # blocking one another. Every entity is created before executor.spin().
    moveit_callback_group = ReentrantCallbackGroup()
    service_callback_group = ReentrantCallbackGroup()

    moveit2 = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES,
        base_link_name=BASE_LINK_NAME,
        end_effector_name=END_EFFECTOR_NAME,
        group_name=PLANNING_GROUP,
        callback_group=moveit_callback_group,
        use_move_group_action=True,
    )

    # pymoveit2 used both property names across releases. Prefer the current
    # names, but keep compatibility with the version used by the earlier code.
    if hasattr(type(moveit2), "max_velocity"):
        moveit2.max_velocity = MAX_VELOCITY_SCALING
    else:
        moveit2.max_velocity_scaling_factor = MAX_VELOCITY_SCALING

    if hasattr(type(moveit2), "max_acceleration"):
        moveit2.max_acceleration = MAX_ACCELERATION_SCALING
    else:
        moveit2.max_acceleration_scaling_factor = MAX_ACCELERATION_SCALING

    joint_tracker = JointStateTracker(node, service_callback_group)
    fk_client = node.create_client(
        GetPositionFK,
        "compute_fk",
        callback_group=service_callback_group,
    )
    cartesian_client = node.create_client(
        GetCartesianPath,
        "compute_cartesian_path",
        callback_group=service_callback_group,
    )

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spinner = Thread(target=executor.spin, daemon=True)
    spinner.start()

    try:
        repeatability_loop(
            node,
            moveit2,
            joint_tracker,
            fk_client,
            cartesian_client,
        )
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
    except Exception as exc:
        node.get_logger().error(f"Repeatability program stopped: {exc}")
    finally:
        executor.shutdown()
        spinner.join(timeout=5.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
