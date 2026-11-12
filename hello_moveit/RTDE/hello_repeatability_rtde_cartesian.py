#!/usr/bin/env python3
"""
Fair-comparison RTDE repeatability loop for a UR5e.

Purpose
-------
This script is designed as a direct comparison with
``hello_repeatability_cartesian.py`` while replacing ROS 2 / MoveIt with
``ur_rtde``.

Normal sequence
---------------
1. Put the robot at HOME once with a joint-space ``moveJ`` if necessary.
2. Build a Cartesian HOME -> TARGET route from the same joint-space
   interpolation concept used by the MoveIt Cartesian script.
3. Convert every interpolated joint configuration to a calibrated Cartesian
   tool0 pose with RTDE forward kinematics.
4. Execute the route with blocking RTDE ``moveL(path)``.
5. Wait three seconds, activate Vimba, and capture one image.
6. Execute the Cartesian TARGET -> HOME route.
7. Wait one second at HOME and repeat.

Fair-comparison choices
-----------------------
* HOME, TARGET, cycle count, camera delays, and reporting tolerances match the
  MoveIt Cartesian script.
* There is no actual+delta correction movement.
* The repeated motions are Cartesian and use the same 0.05 m/s Cartesian speed
  cap as the MoveIt script.
* The active controller TCP is temporarily set to zero so RTDE Cartesian poses
  refer to the flange/tool0 frame, matching the MoveIt script's ``tool0`` link.
* Twenty-four FK-derived waypoints are used in each direction, matching the
  first waypoint density tried by the MoveIt script.
* The robot must be still for several consecutive RTDE samples before the
  endpoint is measured.
* Camera failure never prevents the return to HOME.

Important limitation
--------------------
MoveIt's acceleration setting is a dimensionless scaling factor applied to the
robot model's joint limits. RTDE ``moveL`` requires Cartesian acceleration in
m/s^2, so those two acceleration inputs cannot be made mathematically
identical. This file uses a conservative 0.10 m/s^2 while matching the explicit
0.05 m/s Cartesian speed cap exactly.

Safety
------
RTDE controller safety-limit checks are used when available, but RTDE does not
provide MoveIt's environment-aware planning scene or collision checking.
Validate the complete route in simulation and at low speed before using it on
real hardware. Do not run the ROS 2 UR driver, External Control, another RTDE
motion client, or another motion-owning PolyScope program at the same time.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# Allow importing vimba_capture.py when installed beside this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from vimba_capture import capture_once
except Exception as exc:  # Camera failure must not prevent safe robot motion.
    capture_once = None
    VIMBA_IMPORT_ERROR = str(exc)
else:
    VIMBA_IMPORT_ERROR = ""

try:
    from rtde_control import RTDEControlInterface as RTDEControl
    from rtde_receive import RTDEReceiveInterface as RTDEReceive
except ImportError as exc:
    raise SystemExit(
        "ur_rtde is not installed in this Python environment. Install it with:\n"
        "    python3 -m pip install --user ur_rtde"
    ) from exc


LOGGER = logging.getLogger("hello_repeatability_rtde_fair")

ROBOT_IP = os.environ.get("UR_ROBOT_IP", "192.168.2.100")

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

NUM_CYCLES = 100

# Match the MoveIt script's joint-space HOME/recovery goal-tolerance value.
# RTDE moveJ does not accept a goal-tolerance argument, so this value is used
# for endpoint reporting rather than controller-side planning.
JOINT_MOVE_TOLERANCE_RAD = 0.000001

# The repeated RTDE motion is Cartesian. Match the explicit Cartesian speed cap
# from hello_repeatability_cartesian.py exactly.
CARTESIAN_SPEED_MPS = 0.05

# MoveIt uses a dimensionless acceleration scaling factor. RTDE moveL instead
# requires m/s^2, so exact equivalence is impossible. This conservative value
# intentionally uses the same numeric value as the MoveIt scaling setting.
CARTESIAN_ACCELERATION_MPS2 = 0.10

# Initial/recovery moveJ settings are not part of the measured repeated route.
RECOVERY_MOVEJ_SPEED_RAD_S = 0.25
RECOVERY_MOVEJ_ACCELERATION_RAD_S2 = 0.25
STOPJ_ACCELERATION_RAD_S2 = 1.0

# Match the first waypoint density attempted by the MoveIt Cartesian script.
CARTESIAN_WAYPOINT_COUNT = 24

# Intermediate path points need blending so moveL(path) does not stop at every
# waypoint. The final waypoint always has blend=0, ensuring an exact stop at the
# requested endpoint. A conservative dynamic radius is capped at 2 mm.
MAX_BLEND_RADIUS_M = 0.002
BLEND_FRACTION_OF_ADJACENT_SEGMENT = 0.10

# Repeatability timing copied from the MoveIt Cartesian script.
TARGET_SETTLE_BEFORE_CAPTURE_SEC = 3.0
HOLD_AFTER_CAPTURE_SEC = 0.25
HOME_SETTLE_SEC = 1.0
ENABLE_IMAGE_CAPTURE = True

# Endpoint reporting tolerances copied from the MoveIt Cartesian script.
POSITION_REPORT_TOLERANCE_M = 0.002
ORIENTATION_REPORT_TOLERANCE_RAD = 0.02
RETURN_START_JOINT_TOLERANCE_RAD = 0.02
HOME_RECOVERY_JOINT_TOLERANCE_RAD = 0.02

# Independent settling check after each blocking RTDE motion.
SETTLED_JOINT_SPEED_TOLERANCE_RAD_S = 0.02
SETTLED_STATE_SAMPLES = 5
SETTLE_SAMPLE_PERIOD_SEC = 0.02
SETTLE_TIMEOUT_SEC = 15.0

# A zero TCP offset makes Cartesian poses describe the flange/tool0 frame.
TOOL0_TCP_OFFSET = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@dataclass(frozen=True)
class EndpointMeasurement:
    joints: Tuple[float, ...]
    pose: Tuple[float, ...]
    max_joint_speed: float


def _validate_six(values: Sequence[float], name: str) -> None:
    if len(values) != 6:
        raise ValueError(f"{name} must contain exactly 6 values")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{name} contains a non-finite value")


def _actual_joints(rtde_r: RTDEReceive) -> List[float]:
    values = rtde_r.getActualQ()
    if values is None or len(values) != 6:
        raise RuntimeError(f"ur_rtde returned invalid actual joints: {values!r}")
    return [float(value) for value in values]


def _actual_joint_speeds(rtde_r: RTDEReceive) -> List[float]:
    values = rtde_r.getActualQd()
    if values is None or len(values) != 6:
        raise RuntimeError(f"ur_rtde returned invalid joint speeds: {values!r}")
    return [float(value) for value in values]


def _actual_tool0_pose(rtde_r: RTDEReceive) -> List[float]:
    """Read actual [x, y, z, rx, ry, rz] after the active TCP is set to zero."""
    values = rtde_r.getActualTCPPose()
    if values is None or len(values) != 6:
        raise RuntimeError(f"ur_rtde returned invalid actual TCP pose: {values!r}")
    result = [float(value) for value in values]
    _validate_six(result, "actual tool0 pose")
    return result


def _raise_if_safety_stopped(rtde_r: RTDEReceive) -> None:
    if rtde_r.isEmergencyStopped():
        raise RuntimeError("Robot reports an emergency stop")
    if rtde_r.isProtectiveStopped():
        raise RuntimeError("Robot reports a protective stop")


def _angle_difference(actual: float, expected: float) -> float:
    return math.atan2(
        math.sin(actual - expected),
        math.cos(actual - expected),
    )


def _maximum_joint_error(
    actual: Sequence[float],
    expected: Sequence[float],
) -> float:
    return max(
        abs(_angle_difference(a, e))
        for a, e in zip(actual, expected)
    )


def _interpolate_joints(
    start: Sequence[float],
    goal: Sequence[float],
    fraction: float,
) -> List[float]:
    return [
        float(start_value + fraction * (goal_value - start_value))
        for start_value, goal_value in zip(start, goal)
    ]


def _forward_kinematics_tool0(
    rtde_c: RTDEControl,
    joints: Sequence[float],
) -> List[float]:
    """Return calibrated [x, y, z, rx, ry, rz] for the tool0/flange frame."""
    _validate_six(joints, "joints")
    pose = rtde_c.getForwardKinematics(
        [float(value) for value in joints],
        list(TOOL0_TCP_OFFSET),
    )
    if pose is None or len(pose) != 6:
        raise RuntimeError(f"getForwardKinematics returned invalid pose: {pose!r}")
    result = [float(value) for value in pose]
    _validate_six(result, "forward-kinematics pose")
    return result


def _rotation_vector_to_quaternion(
    rotation_vector: Sequence[float],
) -> Tuple[float, float, float, float]:
    rx, ry, rz = [float(value) for value in rotation_vector]
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)

    scale = math.sin(angle / 2.0) / angle
    return (
        rx * scale,
        ry * scale,
        rz * scale,
        math.cos(angle / 2.0),
    )


def _quaternion_to_rotation_vector(
    quaternion: Sequence[float],
) -> Tuple[float, float, float]:
    x, y, z, w = [float(value) for value in quaternion]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("Quaternion has zero length")

    x /= norm
    y /= norm
    z /= norm
    w /= norm
    w = max(-1.0, min(1.0, w))

    angle = 2.0 * math.acos(w)
    sin_half = math.sqrt(max(0.0, 1.0 - w * w))
    if sin_half < 1e-12 or angle < 1e-12:
        return (0.0, 0.0, 0.0)

    axis_x = x / sin_half
    axis_y = y / sin_half
    axis_z = z / sin_half
    return (
        axis_x * angle,
        axis_y * angle,
        axis_z * angle,
    )


def _make_rotation_vector_continuous(
    previous_pose: Sequence[float],
    current_pose: List[float],
) -> None:
    """Choose the equivalent orientation representation closest to the previous pose."""
    previous_q = _rotation_vector_to_quaternion(previous_pose[3:6])
    current_q = _rotation_vector_to_quaternion(current_pose[3:6])
    if sum(a * b for a, b in zip(previous_q, current_q)) < 0.0:
        current_q = tuple(-value for value in current_q)

    rx, ry, rz = _quaternion_to_rotation_vector(current_q)
    current_pose[3] = rx
    current_pose[4] = ry
    current_pose[5] = rz


def _pose_error(
    actual: Sequence[float],
    expected: Sequence[float],
) -> Tuple[float, float]:
    _validate_six(actual, "actual pose")
    _validate_six(expected, "expected pose")

    dx = float(actual[0]) - float(expected[0])
    dy = float(actual[1]) - float(expected[1])
    dz = float(actual[2]) - float(expected[2])
    position_error = math.sqrt(dx * dx + dy * dy + dz * dz)

    actual_q = _rotation_vector_to_quaternion(actual[3:6])
    expected_q = _rotation_vector_to_quaternion(expected[3:6])
    dot = abs(sum(a * e for a, e in zip(actual_q, expected_q)))
    dot = max(-1.0, min(1.0, dot))
    orientation_error = 2.0 * math.acos(dot)

    return position_error, orientation_error


def _position_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _build_fk_poses(
    rtde_c: RTDEControl,
    start_joints: Sequence[float],
    goal_joints: Sequence[float],
    waypoint_count: int = CARTESIAN_WAYPOINT_COUNT,
) -> List[List[float]]:
    if waypoint_count < 1:
        raise ValueError("waypoint_count must be at least 1")

    poses: List[List[float]] = []
    previous_pose = _forward_kinematics_tool0(rtde_c, start_joints)

    for index in range(1, waypoint_count + 1):
        fraction = index / waypoint_count
        reference_joints = _interpolate_joints(
            start_joints,
            goal_joints,
            fraction,
        )
        pose = _forward_kinematics_tool0(rtde_c, reference_joints)
        _make_rotation_vector_continuous(previous_pose, pose)
        poses.append(pose)
        previous_pose = pose

    return poses


def _dynamic_blend_radius(
    start_pose: Sequence[float],
    poses: Sequence[Sequence[float]],
    index: int,
) -> float:
    if index >= len(poses) - 1:
        return 0.0

    previous_pose = start_pose if index == 0 else poses[index - 1]
    current_pose = poses[index]
    next_pose = poses[index + 1]

    incoming = _position_distance(previous_pose, current_pose)
    outgoing = _position_distance(current_pose, next_pose)
    radius = min(
        MAX_BLEND_RADIUS_M,
        BLEND_FRACTION_OF_ADJACENT_SEGMENT * incoming,
        BLEND_FRACTION_OF_ADJACENT_SEGMENT * outgoing,
    )
    return max(0.0, float(radius))


def _build_movel_path(
    start_pose: Sequence[float],
    poses: Sequence[Sequence[float]],
) -> List[List[float]]:
    if not poses:
        raise ValueError("A Cartesian path must contain at least one pose")

    path: List[List[float]] = []
    for index, pose in enumerate(poses):
        _validate_six(pose, f"path pose {index}")
        blend = _dynamic_blend_radius(start_pose, poses, index)
        path.append(
            [float(value) for value in pose]
            + [
                CARTESIAN_SPEED_MPS,
                CARTESIAN_ACCELERATION_MPS2,
                blend,
            ]
        )

    # Explicitly guarantee a complete stop at the endpoint.
    path[-1][-1] = 0.0
    return path


def _validate_route_safety(
    rtde_c: RTDEControl,
    poses: Sequence[Sequence[float]],
    label: str,
) -> None:
    """Use controller safety-limit checks when the installed ur_rtde supports them."""
    checker = getattr(rtde_c, "isPoseWithinSafetyLimits", None)
    if checker is None:
        LOGGER.warning(
            "Installed ur_rtde has no isPoseWithinSafetyLimits(); "
            "%s cannot be prechecked against controller safety limits",
            label,
        )
        return

    for index, pose in enumerate(poses, start=1):
        if not bool(checker([float(value) for value in pose])):
            raise RuntimeError(
                f"{label} waypoint {index}/{len(poses)} is outside the "
                "controller safety limits"
            )


def _wait_until_still(
    rtde_c: RTDEControl,
    rtde_r: RTDEReceive,
    timeout_sec: float = SETTLE_TIMEOUT_SEC,
) -> EndpointMeasurement:
    """Require several consecutive low-velocity samples after a blocking move."""
    deadline = time.monotonic() + timeout_sec
    stable_samples = 0
    latest: Optional[EndpointMeasurement] = None

    while time.monotonic() < deadline:
        _raise_if_safety_stopped(rtde_r)
        joints = _actual_joints(rtde_r)
        speeds = _actual_joint_speeds(rtde_r)
        max_speed = max(abs(value) for value in speeds)
        pose = _actual_tool0_pose(rtde_r)
        latest = EndpointMeasurement(
            joints=tuple(joints),
            pose=tuple(pose),
            max_joint_speed=max_speed,
        )

        if max_speed <= SETTLED_JOINT_SPEED_TOLERANCE_RAD_S:
            stable_samples += 1
            if stable_samples >= SETTLED_STATE_SAMPLES:
                return latest
        else:
            stable_samples = 0

        time.sleep(SETTLE_SAMPLE_PERIOD_SEC)

    if latest is None:
        raise TimeoutError("No RTDE endpoint samples were received")
    raise TimeoutError(
        "Robot did not remain settled before timeout; "
        f"last max joint speed={latest.max_joint_speed:.6f} rad/s"
    )


def _log_joint_error(
    label: str,
    actual: Sequence[float],
    expected: Sequence[float],
) -> None:
    LOGGER.info("%s", label)
    for name, actual_value, expected_value in zip(
        JOINT_NAMES,
        actual,
        expected,
    ):
        error = _angle_difference(actual_value, expected_value)
        LOGGER.info(
            "  %s: actual=%.6f goal=%.6f err=%.6f",
            name,
            actual_value,
            expected_value,
            error,
        )

    max_error = _maximum_joint_error(actual, expected)
    if max_error > JOINT_MOVE_TOLERANCE_RAD:
        LOGGER.warning(
            "%s maximum joint error %.6f rad exceeds the matched reporting "
            "tolerance %.6f rad; no correction move is applied",
            label,
            max_error,
            JOINT_MOVE_TOLERANCE_RAD,
        )


def _log_pose_error(
    label: str,
    actual_pose: Sequence[float],
    expected_pose: Sequence[float],
) -> None:
    position_error, orientation_error = _pose_error(
        actual_pose,
        expected_pose,
    )
    LOGGER.info(
        "%s: position error=%.3f mm, orientation error=%.4f deg",
        label,
        position_error * 1000.0,
        math.degrees(orientation_error),
    )

    if (
        position_error > POSITION_REPORT_TOLERANCE_M
        or orientation_error > ORIENTATION_REPORT_TOLERANCE_RAD
    ):
        LOGGER.warning(
            "%s exceeds the matched Cartesian reporting tolerance; "
            "no correction move is applied",
            label,
        )


def _movej_recovery(
    rtde_c: RTDEControl,
    rtde_r: RTDEReceive,
    target: Sequence[float],
    description: str,
) -> EndpointMeasurement:
    _validate_six(target, description)
    _raise_if_safety_stopped(rtde_r)

    joint_checker = getattr(rtde_c, "isJointsWithinSafetyLimits", None)
    if joint_checker is not None and not bool(
        joint_checker([float(value) for value in target])
    ):
        raise RuntimeError(f"{description} is outside controller safety limits")

    LOGGER.info("Executing %s with RTDE moveJ...", description)
    succeeded = rtde_c.moveJ(
        [float(value) for value in target],
        RECOVERY_MOVEJ_SPEED_RAD_S,
        RECOVERY_MOVEJ_ACCELERATION_RAD_S2,
        False,
    )
    if not succeeded:
        raise RuntimeError(f"RTDE moveJ returned False for {description}")

    measurement = _wait_until_still(rtde_c, rtde_r)
    LOGGER.info("%s complete", description)
    return measurement


def _execute_cartesian_path(
    rtde_c: RTDEControl,
    rtde_r: RTDEReceive,
    start_pose: Sequence[float],
    poses: Sequence[Sequence[float]],
    description: str,
) -> EndpointMeasurement:
    _raise_if_safety_stopped(rtde_r)
    _validate_route_safety(rtde_c, poses, description)
    path = _build_movel_path(start_pose, poses)

    LOGGER.info(
        "Executing %s with %d Cartesian FK waypoints at %.3f m/s...",
        description,
        len(path),
        CARTESIAN_SPEED_MPS,
    )

    # The path overload is blocking by default. Each path entry is
    # [x, y, z, rx, ry, rz, velocity, acceleration, blend].
    succeeded = rtde_c.moveL(path)
    if not succeeded:
        raise RuntimeError(f"RTDE moveL(path) returned False for {description}")

    measurement = _wait_until_still(rtde_c, rtde_r)
    LOGGER.info("%s complete", description)
    return measurement


def _capture_image(cycle_number: int) -> None:
    if not ENABLE_IMAGE_CAPTURE:
        LOGGER.info("Image capture disabled")
        return

    if capture_once is None:
        LOGGER.warning(
            "Vimba capture is unavailable; returning HOME without an image. "
            "Import error: %s",
            VIMBA_IMPORT_ERROR,
        )
        return

    try:
        image_path = capture_once(cycle=cycle_number)
        LOGGER.info("Image saved: %s", image_path)
    except Exception:
        # Camera failure must never prevent the return to HOME.
        LOGGER.exception("Image capture failed; returning HOME")


def _ensure_home(
    rtde_c: RTDEControl,
    rtde_r: RTDEReceive,
    home_pose: Sequence[float],
) -> EndpointMeasurement:
    actual = _actual_joints(rtde_r)
    error = _maximum_joint_error(actual, HOME)

    if error > HOME_RECOVERY_JOINT_TOLERANCE_RAD:
        measurement = _movej_recovery(
            rtde_c,
            rtde_r,
            HOME,
            "one-time move/recovery to HOME",
        )
    else:
        measurement = _wait_until_still(rtde_c, rtde_r)

    _log_joint_error("HOME actual joint values:", measurement.joints, HOME)
    _log_pose_error("HOME Cartesian accuracy", measurement.pose, home_pose)
    return measurement


def run_cycles(rtde_c: RTDEControl, rtde_r: RTDEReceive) -> None:
    _validate_six(HOME, "HOME")
    _validate_six(TARGET, "TARGET")

    LOGGER.info("Computing calibrated HOME and TARGET tool0 poses...")
    home_pose = _forward_kinematics_tool0(rtde_c, HOME)
    target_pose = _forward_kinematics_tool0(rtde_c, TARGET)
    LOGGER.info(
        "HOME pose: x=%.6f, y=%.6f, z=%.6f",
        home_pose[0],
        home_pose[1],
        home_pose[2],
    )
    LOGGER.info(
        "TARGET pose: x=%.6f, y=%.6f, z=%.6f",
        target_pose[0],
        target_pose[1],
        target_pose[2],
    )

    # Cache the same FK-derived route concept used by the MoveIt version.
    forward_poses = _build_fk_poses(rtde_c, HOME, TARGET)
    reverse_poses = _build_fk_poses(rtde_c, TARGET, HOME)

    # Validate both directions before the robot leaves HOME.
    _validate_route_safety(rtde_c, forward_poses, "HOME -> TARGET route")
    _validate_route_safety(rtde_c, reverse_poses, "TARGET -> HOME route")

    _ensure_home(rtde_c, rtde_r, home_pose)
    time.sleep(HOME_SETTLE_SEC)

    for cycle_index in range(NUM_CYCLES):
        cycle_number = cycle_index + 1
        cycle_start = time.monotonic()
        LOGGER.info("===== CYCLE %d OF %d =====", cycle_number, NUM_CYCLES)

        try:
            # Begin every measured cycle from the same verified HOME state.
            home_measurement = _ensure_home(rtde_c, rtde_r, home_pose)

            # Recheck both precomputed routes before each outbound movement.
            _validate_route_safety(rtde_c, forward_poses, "HOME -> TARGET route")
            _validate_route_safety(rtde_c, reverse_poses, "TARGET -> HOME route")

            target_measurement = _execute_cartesian_path(
                rtde_c,
                rtde_r,
                home_measurement.pose,
                forward_poses,
                "RTDE Cartesian HOME -> TARGET path",
            )

            _log_joint_error(
                "TARGET actual joint values:",
                target_measurement.joints,
                TARGET,
            )
            _log_pose_error(
                "TARGET Cartesian accuracy",
                target_measurement.pose,
                target_pose,
            )

            LOGGER.info(
                "Holding TARGET for %.2f seconds before activating Vimba...",
                TARGET_SETTLE_BEFORE_CAPTURE_SEC,
            )
            time.sleep(TARGET_SETTLE_BEFORE_CAPTURE_SEC)
            _capture_image(cycle_number)
            time.sleep(HOLD_AFTER_CAPTURE_SEC)

            # Mirror the MoveIt return-start check. If the actual target joint
            # state differs materially, rebuild the return route from the real
            # measured state rather than assuming the nominal TARGET state.
            return_start_error = _maximum_joint_error(
                target_measurement.joints,
                TARGET,
            )
            if return_start_error > RETURN_START_JOINT_TOLERANCE_RAD:
                LOGGER.info(
                    "Actual TARGET differs from the nominal return start by "
                    "%.6f rad; rebuilding RTDE Cartesian return from the "
                    "measured state",
                    return_start_error,
                )
                cycle_reverse_poses = _build_fk_poses(
                    rtde_c,
                    target_measurement.joints,
                    HOME,
                )
            else:
                cycle_reverse_poses = reverse_poses

            _validate_route_safety(
                rtde_c,
                cycle_reverse_poses,
                "TARGET -> HOME route",
            )

            home_return = _execute_cartesian_path(
                rtde_c,
                rtde_r,
                target_measurement.pose,
                cycle_reverse_poses,
                "RTDE Cartesian TARGET -> HOME path",
            )

            _log_joint_error(
                "HOME return actual joint values:",
                home_return.joints,
                HOME,
            )
            _log_pose_error(
                "HOME return Cartesian accuracy",
                home_return.pose,
                home_pose,
            )

            time.sleep(HOME_SETTLE_SEC)
            LOGGER.info(
                "===== END OF CYCLE %d; elapsed=%.3f seconds =====",
                cycle_number,
                time.monotonic() - cycle_start,
            )

        except Exception as exc:
            LOGGER.exception("Cycle %d stopped: %s", cycle_number, exc)
            LOGGER.warning("Attempting RTDE joint-space recovery to HOME")
            try:
                recovered = _movej_recovery(
                    rtde_c,
                    rtde_r,
                    HOME,
                    "emergency recovery to HOME",
                )
                _log_joint_error(
                    "Recovered HOME joint values:",
                    recovered.joints,
                    HOME,
                )
                _log_pose_error(
                    "Recovered HOME Cartesian accuracy",
                    recovered.pose,
                    home_pose,
                )
                time.sleep(HOME_SETTLE_SEC)
            except Exception:
                LOGGER.exception(
                    "HOME recovery failed; stopping the repeatability loop"
                )
                break


def _attempt_stop(rtde_c: Optional[RTDEControl]) -> None:
    if rtde_c is None:
        return
    try:
        if rtde_c.isConnected():
            rtde_c.stopJ(STOPJ_ACCELERATION_RAD_S2, True)
            time.sleep(0.25)
    except Exception:
        LOGGER.exception("Could not send stopJ")


def _restore_tcp(
    rtde_c: Optional[RTDEControl],
    original_tcp: Optional[Sequence[float]],
) -> None:
    if rtde_c is None or original_tcp is None:
        return
    try:
        if rtde_c.isConnected():
            if not rtde_c.setTcp([float(value) for value in original_tcp]):
                LOGGER.error("RTDE setTcp returned False while restoring original TCP")
            else:
                LOGGER.info("Restored the controller's original TCP offset")
    except Exception:
        LOGGER.exception("Could not restore the controller's original TCP offset")


def _disconnect(
    rtde_c: Optional[RTDEControl],
    rtde_r: Optional[RTDEReceive],
) -> None:
    if rtde_c is not None:
        try:
            if rtde_c.isConnected():
                rtde_c.stopScript()
        except Exception:
            LOGGER.exception("Could not stop the ur_rtde control script")
        try:
            rtde_c.disconnect()
        except Exception:
            LOGGER.exception("Could not disconnect RTDE control interface")

    if rtde_r is not None:
        try:
            rtde_r.disconnect()
        except Exception:
            LOGGER.exception("Could not disconnect RTDE receive interface")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    rtde_c: Optional[RTDEControl] = None
    rtde_r: Optional[RTDEReceive] = None
    original_tcp: Optional[List[float]] = None

    try:
        LOGGER.info("Connecting ur_rtde to controller at %s...", ROBOT_IP)
        rtde_c = RTDEControl(ROBOT_IP)
        rtde_r = RTDEReceive(ROBOT_IP)

        if not rtde_c.isConnected() or not rtde_r.isConnected():
            raise ConnectionError("One or both ur_rtde interfaces are disconnected")

        _raise_if_safety_stopped(rtde_r)
        LOGGER.info(
            "Connected. Current joints: %s",
            ", ".join(f"{value:.6f}" for value in _actual_joints(rtde_r)),
        )

        # Save the active TCP, use zero TCP for a fair comparison with MoveIt's
        # tool0 link, and restore the original TCP during shutdown.
        tcp = rtde_c.getTCPOffset()
        if tcp is None or len(tcp) != 6:
            raise RuntimeError(f"getTCPOffset returned invalid data: {tcp!r}")
        original_tcp = [float(value) for value in tcp]

        if not rtde_c.setTcp(list(TOOL0_TCP_OFFSET)):
            raise RuntimeError("RTDE setTcp returned False for the zero/tool0 TCP")
        LOGGER.info(
            "Temporarily set active TCP to zero so Cartesian motion describes tool0"
        )

        run_cycles(rtde_c, rtde_r)
        return 0

    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; attempting stopJ before disconnecting")
        _attempt_stop(rtde_c)
        return 130

    except Exception:
        LOGGER.exception("RTDE repeatability sequence failed; attempting stopJ")
        _attempt_stop(rtde_c)
        return 1

    finally:
        _restore_tcp(rtde_c, original_tcp)
        _disconnect(rtde_c, rtde_r)


if __name__ == "__main__":
    raise SystemExit(main())
