#!/usr/bin/env python3
"""Repeat target -> camera capture -> Cartesian retract using ur_rtde.

Each cycle:
1. Move to JOINT_GOAL with moveJ.
2. Measure the joint error and perform the same correction used previously.
3. Allow the robot to settle and capture one image.
4. Move 10-20 cm away from the target along one Cartesian base-frame axis.

The former HOME joint waypoint and return-home motion have been removed.

No ROS 2, MoveIt, or explicit URScript strings are used in this file.

WARNING: Direct moveJ/moveL commands do not use a MoveIt planning scene or
perform MoveIt collision checking. Validate the complete path, retract
direction, workspace limits, and surroundings at reduced speed first.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional, Sequence

# Allow importing vimba_capture installed alongside this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vimba_capture import capture_once

try:
    from rtde_control import RTDEControlInterface as RTDEControl
    from rtde_receive import RTDEReceiveInterface as RTDEReceive
except ImportError as exc:
    raise SystemExit(
        "ur_rtde is not installed in this Python environment. Install it with:\n"
        "    python3 -m pip install --user ur_rtde"
    ) from exc


ROBOT_IP = os.environ.get("UR_ROBOT_IP", "192.168.2.100")

JOINT_GOAL = [
    1.1046188835872111,
    -0.6567673975254662,
    -0.0429350995990605,
    -2.5001841534818769,
    5.1227158875285568,
    3.1796408312832698,
]

NUM_CYCLES = 200
GOAL_TOLERANCE = 0.000001  # rad; checked after each blocking moveJ call

# Cartesian retract configuration.
# getActualTCPPose() is expressed in the robot BASE frame:
#     [x, y, z, rx, ry, rz]
# Therefore, this default moves the TCP 0.15 m along base-frame +Z while
# keeping its orientation unchanged. Use a negative distance for -X/-Y/-Z.
AWAY_AXIS = "y"             # "x", "y", or "z"
AWAY_DISTANCE_M = 0.15      # signed distance; magnitude must be 0.10-0.20 m
TCP_POSITION_TOLERANCE_M = 0.005

# Conservative direct-motion limits.
MOVEJ_ACCELERATION = 0.25  # rad/s^2
MOVEJ_VELOCITY = 0.25      # rad/s
MOVEL_ACCELERATION = 0.20  # m/s^2
MOVEL_VELOCITY = 0.05      # m/s
STOPJ_ACCELERATION = 1.0   # rad/s^2
STOPL_ACCELERATION = 0.5   # m/s^2

# +1.0 exactly preserves the previous algorithm:
#     delta = actual - goal
#     corrected = actual2 + delta
# Set this to -1.0 if the intention is to subtract/cancel the measured error.
CORRECTION_GAIN = +1.0

LOGGER = logging.getLogger("hello_polyscope_rtde")
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _validate_joint_vector(values: Sequence[float], name: str) -> None:
    if len(values) != 6:
        raise ValueError(f"{name} must contain exactly 6 joint values")


def _validate_away_configuration() -> None:
    axis = AWAY_AXIS.lower()
    if axis not in AXIS_INDEX:
        raise ValueError(
            f"AWAY_AXIS must be 'x', 'y', or 'z'; received {AWAY_AXIS!r}"
        )

    distance = abs(AWAY_DISTANCE_M)
    if not 0.10 <= distance <= 0.20:
        raise ValueError(
            "The magnitude of AWAY_DISTANCE_M must be between 0.10 and "
            f"0.20 m; received {AWAY_DISTANCE_M:.4f} m"
        )


def _get_actual_joints(rtde_r: RTDEReceive) -> list[float]:
    """Read the latest measured joint positions from the RTDE stream."""
    joints = rtde_r.getActualQ()
    if joints is None or len(joints) != 6:
        raise RuntimeError(f"ur_rtde returned invalid joint data: {joints!r}")
    return [float(value) for value in joints]


def _get_actual_tcp_pose(rtde_r: RTDEReceive) -> list[float]:
    """Return [x, y, z, rx, ry, rz] for the active TCP in the base frame."""
    pose = rtde_r.getActualTCPPose()
    if pose is None or len(pose) != 6:
        raise RuntimeError(f"ur_rtde returned invalid TCP pose data: {pose!r}")
    return [float(value) for value in pose]


def _raise_if_safety_stopped(rtde_r: RTDEReceive) -> None:
    """Fail before issuing a move if the controller reports a safety stop."""
    if rtde_r.isEmergencyStopped():
        raise RuntimeError("Robot reports an emergency stop")
    if rtde_r.isProtectiveStopped():
        raise RuntimeError("Robot reports a protective stop")


def _movej_and_wait(
    rtde_c: RTDEControl,
    rtde_r: RTDEReceive,
    target: Sequence[float],
) -> list[float]:
    """Execute a blocking RTDE moveJ and return the measured final joints."""
    _validate_joint_vector(target, "target")
    _raise_if_safety_stopped(rtde_r)

    succeeded = rtde_c.moveJ(
        [float(value) for value in target],
        MOVEJ_VELOCITY,
        MOVEJ_ACCELERATION,
        False,
    )
    if not succeeded:
        raise RuntimeError("RTDEControlInterface.moveJ() returned False")

    _raise_if_safety_stopped(rtde_r)
    actual = _get_actual_joints(rtde_r)

    max_error = max(abs(a - g) for a, g in zip(actual, target))
    if max_error > GOAL_TOLERANCE:
        LOGGER.warning(
            "Blocking moveJ completed, but maximum measured joint error is "
            "%.6f rad (configured tolerance %.6f rad)",
            max_error,
            GOAL_TOLERANCE,
        )

    return actual


def _movel_and_wait(
    rtde_c: RTDEControl,
    rtde_r: RTDEReceive,
    target_pose: Sequence[float],
) -> list[float]:
    """Execute a blocking linear TCP move and return the measured TCP pose."""
    if len(target_pose) != 6:
        raise ValueError("target_pose must contain [x, y, z, rx, ry, rz]")

    _raise_if_safety_stopped(rtde_r)

    pose = [float(value) for value in target_pose]

    # Check the pose against controller safety limits before commanding it.
    if hasattr(rtde_c, "isPoseWithinSafetyLimits"):
        if not rtde_c.isPoseWithinSafetyLimits(pose):
            raise RuntimeError(
                "Calculated away pose is outside the robot's configured "
                "safety limits or has no valid inverse-kinematics solution"
            )

    succeeded = rtde_c.moveL(
        pose,
        MOVEL_VELOCITY,
        MOVEL_ACCELERATION,
        False,
    )
    if not succeeded:
        raise RuntimeError("RTDEControlInterface.moveL() returned False")

    _raise_if_safety_stopped(rtde_r)
    actual_pose = _get_actual_tcp_pose(rtde_r)

    max_position_error = max(
        abs(actual_pose[index] - pose[index]) for index in range(3)
    )
    if max_position_error > TCP_POSITION_TOLERANCE_M:
        LOGGER.warning(
            "Blocking moveL completed, but maximum measured TCP position "
            "error is %.6f m (configured tolerance %.6f m)",
            max_position_error,
            TCP_POSITION_TOLERANCE_M,
        )

    return actual_pose


def _calculate_away_pose(current_pose: Sequence[float]) -> list[float]:
    """Offset the current TCP pose along one base-frame translation axis."""
    if len(current_pose) != 6:
        raise ValueError("current_pose must contain [x, y, z, rx, ry, rz]")

    away_pose = [float(value) for value in current_pose]
    away_pose[AXIS_INDEX[AWAY_AXIS.lower()]] += AWAY_DISTANCE_M
    return away_pose


def _log_joint_values(
    label: str,
    actual: Sequence[float],
    goal: Sequence[float],
) -> None:
    LOGGER.info("%s actual joint values:", label)
    for index, (actual_value, goal_value) in enumerate(zip(actual, goal)):
        LOGGER.info(
            "  J%d  actual=%.6f  goal=%.6f  err=%.6f",
            index,
            actual_value,
            goal_value,
            actual_value - goal_value,
        )


def _log_tcp_pose(label: str, pose: Sequence[float]) -> None:
    LOGGER.info(
        "%s TCP pose: x=%.6f y=%.6f z=%.6f "
        "rx=%.6f ry=%.6f rz=%.6f",
        label,
        *pose,
    )


def run_cycles(rtde_c: RTDEControl, rtde_r: RTDEReceive) -> None:
    """Run the target/correction/capture/Cartesian-away loop."""
    _validate_joint_vector(JOINT_GOAL, "JOINT_GOAL")
    _validate_away_configuration()

    LOGGER.info(
        "After each capture, retracting %.1f cm along base-frame %s%s",
        abs(AWAY_DISTANCE_M) * 100.0,
        "+" if AWAY_DISTANCE_M >= 0.0 else "-",
        AWAY_AXIS.upper(),
    )

    for cycle_index in range(NUM_CYCLES):
        cycle = cycle_index + 1
        LOGGER.info("===== CYCLE %d OF %d =====", cycle, NUM_CYCLES)
        target_phase_start = time.monotonic()

        # MOVE TO TARGET
        LOGGER.info("Moving to TARGET through ur_rtde...")
        _movej_and_wait(rtde_c, rtde_r, JOINT_GOAL)
        LOGGER.info("TARGET move done")

        # Compute delta at arrival.
        time.sleep(0.1)
        actual = _get_actual_joints(rtde_r)
        delta = [actual[j] - JOINT_GOAL[j] for j in range(6)]
        _log_joint_values("TARGET", actual, JOINT_GOAL)

        # Three seconds into the hold, command actual2 + delta.
        time.sleep(0.1)
        actual2 = _get_actual_joints(rtde_r)
        corrected = [
            actual2[j] + CORRECTION_GAIN * delta[j]
            for j in range(6)
        ]

        LOGGER.info(
            "Moving to corrected position: actual2 + (%.1f * delta)...",
            CORRECTION_GAIN,
        )
        _movej_and_wait(rtde_c, rtde_r, corrected)
        LOGGER.info("Corrected move done")

        # Give time for oscillation to stop before capturing the picture.
        time.sleep(3.0)
        try:
            path = capture_once(cycle=cycle)
            LOGGER.info("Image saved: %s", path)
        except Exception:
            # Camera failure is logged, but the loop continues and the robot
            # still moves away from the target before the next cycle.
            LOGGER.exception("Image capture failed")

       
        time.sleep(0.1)

        elapsed = time.monotonic() - target_phase_start
        LOGGER.info("Target/capture phase elapsed: %.3f seconds", elapsed)

        # MOVE AWAY FROM TARGET ALONG ONE CARTESIAN AXIS.
        target_tcp_pose = _get_actual_tcp_pose(rtde_r)
        away_pose = _calculate_away_pose(target_tcp_pose)
        _log_tcp_pose("Target", target_tcp_pose)
        _log_tcp_pose("Away target", away_pose)

        LOGGER.info(
            "Moving %.1f cm along base-frame %s%s with moveL...",
            abs(AWAY_DISTANCE_M) * 100.0,
            "+" if AWAY_DISTANCE_M >= 0.0 else "-",
            AWAY_AXIS.upper(),
        )
        actual_away_pose = _movel_and_wait(rtde_c, rtde_r, away_pose)
        LOGGER.info("Away move done")
        _log_tcp_pose("Away actual", actual_away_pose)

        LOGGER.info("===== END OF CYCLE %d =====", cycle)


def _attempt_stop(rtde_c: Optional[RTDEControl]) -> None:
    """Best-effort stop for interruption or failure handling."""
    if rtde_c is None:
        return

    try:
        if not rtde_c.isConnected():
            return

        # The sequence uses both moveJ and moveL, so try both stop types.
        try:
            rtde_c.stopL(STOPL_ACCELERATION, True)
        except Exception:
            LOGGER.debug("stopL was not accepted", exc_info=True)

        try:
            rtde_c.stopJ(STOPJ_ACCELERATION, True)
        except Exception:
            LOGGER.debug("stopJ was not accepted", exc_info=True)

        time.sleep(0.25)
    except Exception:
        LOGGER.exception("Could not stop robot motion")


def _disconnect(
    rtde_c: Optional[RTDEControl],
    rtde_r: Optional[RTDEReceive],
) -> None:
    """Stop the uploaded control script and close both RTDE connections."""
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

    try:
        LOGGER.info("Running file: %s", os.path.abspath(__file__))
        LOGGER.info("NUM_CYCLES = %d", NUM_CYCLES)
        LOGGER.info("Connecting ur_rtde to controller at %s...", ROBOT_IP)

        rtde_c = RTDEControl(ROBOT_IP)
        rtde_r = RTDEReceive(ROBOT_IP)

        if not rtde_c.isConnected() or not rtde_r.isConnected():
            raise ConnectionError("One or both ur_rtde interfaces are disconnected")

        LOGGER.info(
            "Connected. Current joints: %s",
            ", ".join(f"{value:.6f}" for value in _get_actual_joints(rtde_r)),
        )

        run_cycles(rtde_c, rtde_r)
        return 0

    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; attempting to stop motion before disconnecting")
        _attempt_stop(rtde_c)
        return 130

    except Exception:
        LOGGER.exception("Robot sequence failed; attempting to stop motion")
        _attempt_stop(rtde_c)
        return 1

    finally:
        _disconnect(rtde_c, rtde_r)


if __name__ == "__main__":
    raise SystemExit(main())
