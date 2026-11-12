#!/usr/bin/env python3
"""
UR5e micro-step precision test  (A -> B -> A -> B ...)

Sequence
  1. Warm-up        : HOME <-> A for --warmup-min minutes (same safe path as hello_moveit.py)
  2. Camera noise   : --noise-images images of the glass scale with the robot parked at A
  3. Step tests     : for every axis and every step size in the table below
                        move to A, settle, image A_0
                        repeat N times:  move to B, settle, image B_k ; move to A, settle, image A_k
  4. Everything is logged to <out-dir>/results.csv and the images are stored in <out-dir>/images/.
     Run analyze_precision.py on <out-dir> afterwards to get mean displacement, bias and
     repeatability of A.

Pose definition
  A = JOINT_GOAL (the pose where the camera sees the scale)
  B = A + step along a base_link axis, orientation unchanged.  B's joint angles are obtained
      by Newton iterations on MoveIt's own /compute_fk (nanometre-accurate in the MoveIt model),
      NOT with an IK plugin, whose tolerance (~10 um for KDL) would swamp a 5 um step.

Backends
  controller (default) : sends a 1-point FollowJointTrajectory goal directly to the UR controller.
                         Deterministic: the arm ends exactly at the commanded joint vector and the
                         move duration is set from --speed-mm-s.
  moveit               : uses pymoveit2 like hello_moveit.py.  Caveat: MoveIt's joint goal tolerance
                         lets the planner stop anywhere inside +/- tolerance, so keep --goal-tol
                         well below the smallest step (default 1e-8 rad).
"""

import argparse
import csv
import datetime as dt
import json
import math
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from threading import Thread

import numpy as np

# Allow importing vimba_capture / pose_math installed alongside this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_math import quat_to_matrix, solve_pose_offset
from vimba_capture import capture_once

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionFK
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


# --------------------------------------------------------------------------- #
# Robot / experiment constants
# --------------------------------------------------------------------------- #
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Same safe warm-up path as hello_moveit.py
HOME = [
    1.6205397844314575,
    -0.030809955005981493,
    0.22084552446474248,
    1.206824703807495,
    -1.0476668516742151,
    -0.09247094789613897,
]

# Pose A: the pose where the camera sees the glass scale
A_JOINTS = [
    1.1196330785751343,
    -0.46364779890094,
    -0.2839603126049042,
    3.646835966701172,
    -1.1288955847369593,
    0.11363356560468674,
]

PLANNING_GROUP = "ur_manipulator"
BASE_LINK = "base_link"
EE_LINK = "tool0"
CONTROLLER_ACTION = "/scaled_joint_trajectory_controller/follow_joint_trajectory"

# (commanded step in micrometres, repeats per axis)
CONDITIONS = [(1000, 30), (500, 30), (100, 50), (50, 50), (25, 100), (5, 100)]

AXIS_VECTORS = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}

QUINTIC_PEAK_FACTOR = 1.875   # peak velocity / mean velocity of a rest-to-rest quintic
UR_MAX_JOINT_SPEED = math.pi  # rad/s, used to turn a "velocity scale" into a joint speed

CSV_FIELDS = (
    ["seq", "t_epoch", "iso_time", "phase", "axis", "step_um", "repeat", "pose", "image", "move_s"]
    + [f"q_{j}" for j in JOINT_NAMES]
    + ["fk_x_m", "fk_y_m", "fk_z_m"]
)


# --------------------------------------------------------------------------- #
# ROS helpers
# --------------------------------------------------------------------------- #
def wait_future(future, timeout_sec, what):
    """Block the main thread while the background executor services the future."""
    deadline = time.monotonic() + timeout_sec
    while rclpy.ok() and not future.done():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for {what}")
        time.sleep(0.005)
    if not rclpy.ok():
        raise RuntimeError(f"ROS shut down while waiting for {what}")
    result = future.result()
    if result is None:
        raise RuntimeError(f"{what} returned no result: {future.exception()}")
    return result


class JointStateCache:
    def __init__(self, node, cb_group):
        self._lock = threading.Lock()
        self._pos = {}
        self._ready = threading.Event()
        node.create_subscription(JointState, "/joint_states", self._cb, 10, callback_group=cb_group)

    def _cb(self, msg):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                self._pos[name] = pos
            if all(j in self._pos for j in JOINT_NAMES):
                self._ready.set()

    def wait_ready(self, timeout=10.0):
        if not self._ready.wait(timeout):
            raise RuntimeError("Timed out waiting for all six joints on /joint_states")

    def get(self):
        with self._lock:
            return [self._pos[j] for j in JOINT_NAMES]


class Kinematics:
    """Forward kinematics through MoveIt's /compute_fk (same model MoveIt plans with)."""

    def __init__(self, node, cb_group):
        self.node = node
        self.client = node.create_client(GetPositionFK, "/compute_fk", callback_group=cb_group)

    def wait(self, timeout=10.0):
        if not self.client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("MoveIt /compute_fk service is unavailable")

    def fk(self, q):
        req = GetPositionFK.Request()
        req.header.frame_id = BASE_LINK
        req.header.stamp = self.node.get_clock().now().to_msg()
        req.fk_link_names = [EE_LINK]
        js = JointState()
        js.name = list(JOINT_NAMES)
        js.position = [float(v) for v in q]
        req.robot_state.joint_state = js
        resp = wait_future(self.client.call_async(req), 10.0, "compute_fk")
        if resp.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"compute_fk failed, error code {resp.error_code.val}")
        pose = resp.pose_stamped[0].pose
        p = np.array([pose.position.x, pose.position.y, pose.position.z])
        o = pose.orientation
        return p, quat_to_matrix([o.x, o.y, o.z, o.w])


# --------------------------------------------------------------------------- #
# Motion back-ends
# --------------------------------------------------------------------------- #
class ControllerMover:
    """Direct FollowJointTrajectory goals with a quintic (zero vel/acc at both ends) profile."""

    def __init__(self, node, cb_group, cache, args):
        self.cache = cache
        self.speed_m_s = args.speed_mm_s * 1e-3
        self.min_move_time = args.min_move_time
        self.warm_joint_speed = args.vel_scale_warmup * UR_MAX_JOINT_SPEED
        self.client = ActionClient(node, FollowJointTrajectory, args.controller_action,
                                   callback_group=cb_group)
        if not self.client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(
                f"Action server '{args.controller_action}' not available "
                f"(check `ros2 action list`, or use --backend moveit)")

    def move(self, q_target, step_m=None):
        if step_m is not None:
            duration = max(self.min_move_time, QUINTIC_PEAK_FACTOR * step_m / self.speed_m_s)
        else:
            cur = self.cache.get()
            dmax = max(abs(a - b) for a, b in zip(cur, q_target))
            duration = max(2.0, QUINTIC_PEAK_FACTOR * dmax / self.warm_joint_speed)

        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q_target]
        pt.velocities = [0.0] * len(JOINT_NAMES)
        pt.accelerations = [0.0] * len(JOINT_NAMES)
        sec = int(duration)
        pt.time_from_start = Duration(sec=sec, nanosec=int(round((duration - sec) * 1e9)))

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(JOINT_NAMES)
        goal.trajectory.points = [pt]

        handle = wait_future(self.client.send_goal_async(goal), 10.0, "trajectory goal response")
        if not handle.accepted:
            raise RuntimeError("Trajectory goal rejected by controller")
        res = wait_future(handle.get_result_async(), duration + 30.0, "trajectory result")
        if res.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(
                f"Trajectory failed: code {res.result.error_code} {res.result.error_string}")
        return duration


class MoveItMover:
    """pymoveit2 joint-space moves, as in hello_moveit.py."""

    def __init__(self, node, cb_group, args):
        from pymoveit2 import MoveIt2   # imported lazily: not needed by the controller backend
        self.m = MoveIt2(
            node=node,
            joint_names=JOINT_NAMES,
            base_link_name=BASE_LINK,
            end_effector_name=EE_LINK,
            group_name=PLANNING_GROUP,
            callback_group=cb_group,
        )
        self.vel_test = args.vel_scale_test
        self.vel_warm = args.vel_scale_warmup
        self.m.max_acceleration_scaling_factor = args.acc_scale
        self.tol = args.goal_tol

    def move(self, q_target, step_m=None):
        self.m.max_velocity_scaling_factor = self.vel_test if step_m is not None else self.vel_warm
        t0 = time.monotonic()
        self.m.move_to_configuration(joint_positions=[float(v) for v in q_target], tolerance=self.tol)
        ok = self.m.wait_until_executed()
        if ok is False:
            raise RuntimeError("MoveIt execution failed")
        return time.monotonic() - t0


# --------------------------------------------------------------------------- #
# The experiment
# --------------------------------------------------------------------------- #
class PrecisionTest:
    def __init__(self, args, node, cache, kin, mover):
        self.args = args
        self.node = node
        self.log = node.get_logger()
        self.cache = cache
        self.kin = kin
        self.mover = mover
        self.A = list(A_JOINTS)
        self.targets = {}      # (axis, step_um) -> joint vector of B
        self.seq = 0

        self.out_dir = Path(args.out_dir)
        self.img_dir = self.out_dir / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self.out_dir / "results.csv", "w", newline="")
        self._csv = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDS)
        self._csv.writeheader()

    # ---- setup ----------------------------------------------------------- #
    def build_targets(self, conditions):
        """Pose B for every (axis, step) via Newton on /compute_fk."""
        self.log.info("Computing B poses (A + step) with FK-based Newton solve ...")
        report = {}
        for axis in self.args.axes:
            for step_um, _ in conditions:
                offset = np.array(AXIS_VECTORS[axis]) * step_um * 1e-6
                qB, info = solve_pose_offset(self.kin.fk, self.A, offset)
                self.targets[(axis, step_um)] = [float(v) for v in qB]
                d_um = info["delta_m"] * 1e6
                self.log.info(
                    f"  {axis.upper()} +{step_um:>5} um : model delta = "
                    f"({d_um[0]:+.4f}, {d_um[1]:+.4f}, {d_um[2]:+.4f}) um, "
                    f"orientation err {info['rot_err_rad']:.1e} rad, "
                    f"max joint change {info['max_dq_rad']:.3e} rad, {info['iterations']} iterations")
                report[f"{axis}_{step_um}"] = {
                    "q_B": self.targets[(axis, step_um)],
                    "model_delta_um": d_um.tolist(),
                    "max_dq_rad": info["max_dq_rad"],
                }
        with open(self.out_dir / "run_config.json", "w") as f:
            json.dump({
                "args": vars(self.args),
                "q_A": self.A,
                "conditions": conditions,
                "targets": report,
                "started": dt.datetime.now().isoformat(),
            }, f, indent=2)

    # ---- low-level actions ---------------------------------------------- #
    def go_to_A(self):
        cur = self.cache.get()
        if max(abs(a - b) for a, b in zip(cur, self.A)) > 1e-4:
            self.log.info("Moving to A ...")
            self.mover.move(self.A)

    def acquire(self, phase, axis, step_um, repeat, pose, move_s=float("nan")):
        """Settle, read the robot state, grab an image, move it into out_dir, log a row."""
        time.sleep(self.args.settle)
        q = self.cache.get()
        try:
            fk_p, _ = self.kin.fk(q)
        except Exception as e:
            self.log.warning(f"FK for logging failed: {e}")
            fk_p = [float("nan")] * 3

        self.seq += 1
        now = time.time()
        rel_img = ""
        try:
            src = Path(str(capture_once(cycle=self.seq)))
            dest = self.img_dir / (
                f"{self.seq:05d}_{phase}_{axis}_{step_um:g}um_r{repeat:03d}_{pose}{src.suffix}")
            shutil.move(str(src), str(dest))
            rel_img = str(dest.relative_to(self.out_dir))
        except Exception as e:
            self.log.error(f"Image capture failed (seq {self.seq}): {e}")

        row = {
            "seq": self.seq, "t_epoch": f"{now:.3f}",
            "iso_time": dt.datetime.fromtimestamp(now).isoformat(timespec="milliseconds"),
            "phase": phase, "axis": axis, "step_um": step_um, "repeat": repeat, "pose": pose,
            "image": rel_img, "move_s": f"{move_s:.3f}",
            "fk_x_m": f"{fk_p[0]:.9f}", "fk_y_m": f"{fk_p[1]:.9f}", "fk_z_m": f"{fk_p[2]:.9f}",
        }
        for name, val in zip(JOINT_NAMES, q):
            row[f"q_{name}"] = f"{val:.12f}"
        self._csv.writerow(row)
        self._csv_file.flush()

    # ---- phases ---------------------------------------------------------- #
    def warm_up(self):
        minutes = self.args.warmup_min
        if minutes <= 0:
            self.log.info("Warm-up skipped")
            return
        self.log.info(f"Warm-up: HOME <-> A for {minutes:.1f} min ...")
        end = time.monotonic() + minutes * 60.0
        n = 0
        while time.monotonic() < end:
            self.mover.move(HOME)
            self.mover.move(self.A)
            n += 1
            self.log.info(f"  warm-up round trip {n} done, {max(0.0, end - time.monotonic()) / 60:.1f} min left")
        self.log.info("Warm-up finished")

    def noise_test(self):
        n = self.args.noise_images
        if n <= 0:
            return
        self.go_to_A()
        self.log.info(f"Camera noise/drift: {n} images at A, every {self.args.noise_interval:.1f} s ...")
        for i in range(n):
            if i:
                time.sleep(self.args.noise_interval)
            self.acquire("noise", "-", 0, i, "A")

    def run_condition(self, axis, step_um, repeats):
        qA, qB = self.A, self.targets[(axis, step_um)]
        step_m = step_um * 1e-6
        self.log.info(f"=== axis {axis.upper()}  step {step_um} um  x{repeats} ===")
        self.go_to_A()
        self.acquire("step", axis, step_um, 0, "A")
        for k in range(1, repeats + 1):
            d = self.mover.move(qB, step_m)
            self.acquire("step", axis, step_um, k, "B", d)
            d = self.mover.move(qA, step_m)
            self.acquire("step", axis, step_um, k, "A", d)
            if k % 10 == 0 or k == repeats:
                self.log.info(f"  {axis.upper()} {step_um} um: {k}/{repeats} repeats done")

    def close(self):
        self._csv_file.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=f"precision_{dt.datetime.now():%Y%m%d_%H%M%S}")
    ap.add_argument("--backend", choices=["controller", "moveit"], default="controller")
    ap.add_argument("--controller-action", default=CONTROLLER_ACTION)
    ap.add_argument("--axes", nargs="+", choices=list(AXIS_VECTORS), default=["x"],
                    help="base_link axes to test (default: x)")
    ap.add_argument("--steps", type=float, nargs="+", default=None,
                    help="subset of step sizes in um (default: the full table)")
    ap.add_argument("--quick", action="store_true",
                    help="dry-run: 3 repeats per condition, 5 noise images, no warm-up")
    ap.add_argument("--check-only", action="store_true",
                    help="only compute/print the B poses, do not move or capture")
    ap.add_argument("--speed-mm-s", type=float, default=50.0,
                    help="target peak tool speed for step moves (controller backend)")
    ap.add_argument("--min-move-time", type=float, default=0.25,
                    help="shortest allowed step-move duration in s (controller backend)")
    ap.add_argument("--vel-scale-test", type=float, default=0.05, help="moveit backend, step moves")
    ap.add_argument("--vel-scale-warmup", type=float, default=0.3,
                    help="warm-up / re-positioning speed as a fraction of max joint speed")
    ap.add_argument("--acc-scale", type=float, default=0.1, help="moveit backend")
    ap.add_argument("--goal-tol", type=float, default=1e-8, help="moveit backend joint goal tolerance (rad)")
    ap.add_argument("--settle", type=float, default=3.0, help="settle time before each image (s)")
    ap.add_argument("--warmup-min", type=float, default=20.0)
    ap.add_argument("--noise-images", type=int, default=100)
    ap.add_argument("--noise-interval", type=float, default=2.0)
    return ap.parse_args()


def main():
    args = parse_args()
    conditions = list(CONDITIONS)
    if args.steps:
        wanted = {float(s) for s in args.steps}
        conditions = [(s, n) for s, n in conditions if float(s) in wanted]
    if args.quick:
        conditions = [(s, min(n, 3)) for s, n in conditions]
        args.noise_images = min(args.noise_images, 5)
        args.warmup_min = 0.0
    if not conditions:
        sys.exit("No matching step sizes selected")

    rclpy.init()
    node = Node("ur5e_precision_test")
    log = node.get_logger()
    cb_group = ReentrantCallbackGroup()

    cache = JointStateCache(node, cb_group)
    kin = Kinematics(node, cb_group)
    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    spinner = Thread(target=executor.spin, daemon=True)
    spinner.start()

    test = None
    try:
        cache.wait_ready()
        kin.wait()

        if args.backend == "controller":
            mover = ControllerMover(node, cb_group, cache, args)
        else:
            mover = MoveItMover(node, cb_group, args)

        test = PrecisionTest(args, node, cache, kin, mover)
        test.build_targets(conditions)

        n_moves = sum(n for _, n in conditions) * 2 * len(args.axes)
        est_s = n_moves * (args.settle + 2.0) + args.noise_images * args.noise_interval + args.warmup_min * 60
        log.info(f"Output directory: {test.out_dir.resolve()}")
        log.info(f"Estimated run time: ~{est_s / 60:.0f} min")
        if args.check_only:
            log.info("--check-only: stopping before any motion")
            return

        test.warm_up()
        test.go_to_A()
        test.noise_test()
        for axis in args.axes:
            for step_um, repeats in conditions:
                test.run_condition(axis, step_um, repeats)
        test.go_to_A()
        log.info("All conditions finished. Now run: python3 analyze_precision.py "
                 f"{test.out_dir} --pitch-um <scale pitch>")
    except KeyboardInterrupt:
        log.warning("Interrupted by user - partial results are saved in results.csv")
    finally:
        if test is not None:
            test.close()
        rclpy.shutdown()
        spinner.join(timeout=2.0)


if __name__ == "__main__":
    main()
