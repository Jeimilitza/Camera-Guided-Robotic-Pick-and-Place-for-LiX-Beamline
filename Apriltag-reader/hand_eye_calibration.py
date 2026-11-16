#!/usr/bin/env python3
"""
Hand-eye calibration (eye-in-hand): finds T_tool0_camera, the pose of the camera
expressed in the tool0 frame, from 15-20 images of a fixed target.

INPUTS
  1. A directory of images, one per robot pose, all showing the same fixed target
     (an AprilTag or a ChArUco board). Take them after the arm has settled.
  2. A CSV with the tool0 pose in base_link for each image (same robot state as the
     image). Header and columns:

         image,x,y,z,qx,qy,qz,qw
         pose_01.png,0.412,-0.133,0.350,0.0,0.7071,0.0,0.7071
         ...

     x,y,z in meters, quaternion as [x, y, z, w]. Both are exactly what
         ros2 run tf2_ros tf2_echo base_link tool0
     prints (Translation, and Rotation "in Quaternion [x, y, z, w]").
     The "image" column is matched to the file name (or the name without extension).

OUTPUT
  A line you can paste into the pick-and-place script:
      T_TOOL0_CAMERA = np.array([[...], [...], [...], [0, 0, 0, 1]])

METHOD
  For each image the detector gives T_cam_target. With T_base_tool0 = G_i and
  T_cam_target = H_i, the unknown X = T_tool0_camera satisfies G_i X H_i = const
  (the target does not move). cv2.calibrateHandEye solves for X; its output
  R_cam2gripper, t_cam2gripper is T_tool0_camera directly. If this OpenCV build has no
  cv2.calibrateHandEye (e.g. OpenCV 5.0), built-in Tsai and Park implementations
  are used instead.

  Several methods are run and compared. If they disagree by more than 1 mm or 0.1 deg,
  the pose set is poorly conditioned (too little rotation variety, noisy detections).

EXAMPLES
  AprilTag:
    python3 hand_eye_calibration.py --images ./calib_imgs --poses ./poses.csv \
        --target apriltag --tag-size 0.02 --tag-family tag36h11 --fx 11714.5 --fy 11706.4 \
        --cx 639.77 --cy 515.83
  ChArUco:
    python3 hand_eye_calibration.py --images ./calib_imgs --poses ./poses.csv \
        --target charuco --squares-x 7 --squares-y 5 --square-length 0.04 \
        --marker-length 0.03 --aruco-dict DICT_5X5_100 --fx ... --fy ... --cx ... --cy ...
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------------------
# Defaults (same as the pick-and-place script)
# --------------------------------------------------------------------------------------
APRILTAG_RUNNER_DIR = "/home/kpatel2/ws_moveit/apriltag-reader-nsls-ii"
DEFAULT_FX = 11714.548810
DEFAULT_FY = 11706.389936
DEFAULT_CX = 639.771526
DEFAULT_CY = 515.834163

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

TRANS_AGREE_MM = 1.0   # methods should agree within this translation (mm)
ROT_AGREE_DEG = 0.1    # ... and within this rotation (deg)
MIN_TOTAL_ROT_DEG = 15.0   # largest rotation between any two poses should exceed this


# --------------------------------------------------------------------------------------
# Small math helpers
# --------------------------------------------------------------------------------------
def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def rotvec(R):
    return cv2.Rodrigues(np.asarray(R, dtype=np.float64))[0].reshape(3)


def rot_from_vec(v):
    return cv2.Rodrigues(np.asarray(v, dtype=np.float64).reshape(3, 1))[0]


def rot_angle_deg(R):
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def nearest_rotation(R):
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def quat_to_rot(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def format_T(T):
    rows = []
    for r in T[:3]:
        rows.append("[" + ", ".join(f"{v:.9f}" for v in r) + "]")
    rows.append("[0, 0, 0, 1]")
    return "np.array([" + ", ".join(rows) + "])"


# --------------------------------------------------------------------------------------
# Hand-eye solvers
# --------------------------------------------------------------------------------------
def _motion_pairs(Gs, Hs, min_rot_deg=2.0):
    """Relative motions A = G_j^-1 G_i (gripper), B = H_j H_i^-1 (camera), so A X = X B."""
    A, B = [], []
    n = len(Gs)
    for i in range(n):
        for j in range(i + 1, n):
            Aij = inv_T(Gs[j]) @ Gs[i]
            Bij = Hs[j] @ inv_T(Hs[i])
            if rot_angle_deg(Aij[:3, :3]) >= min_rot_deg and rot_angle_deg(Bij[:3, :3]) >= min_rot_deg:
                A.append(Aij)
                B.append(Bij)
    return A, B


def _solve_translation(A, B, R_X):
    # (R_A - I) t_X = R_X t_B - t_A, stacked over all pairs
    M = np.vstack([a[:3, :3] - np.eye(3) for a in A])
    v = np.concatenate([R_X @ b[:3, 3] - a[:3, 3] for a, b in zip(A, B)])
    t, *_ = np.linalg.lstsq(M, v, rcond=None)
    return t


def park_solve(Gs, Hs):
    A, B = _motion_pairs(Gs, Hs)
    if len(A) < 2:
        raise RuntimeError("Not enough pose pairs with rotation (need rotation between poses)")
    Hm = np.zeros((3, 3))
    for a, b in zip(A, B):
        Hm += np.outer(rotvec(b[:3, :3]), rotvec(a[:3, :3]))   # maps beta -> alpha
    U, _, Vt = np.linalg.svd(Hm)
    V = Vt.T
    D = np.diag([1, 1, np.linalg.det(V @ U.T)])
    R_X = V @ D @ U.T
    return make_T(R_X, _solve_translation(A, B, R_X))


def tsai_solve(Gs, Hs):
    A, B = _motion_pairs(Gs, Hs)
    if len(A) < 2:
        raise RuntimeError("Not enough pose pairs with rotation (need rotation between poses)")

    def p_vec(R):
        r = rotvec(R)
        th = np.linalg.norm(r)
        return np.zeros(3) if th < 1e-12 else 2.0 * np.sin(th / 2.0) * r / th

    lhs, rhs = [], []
    for a, b in zip(A, B):
        Pg, Pc = p_vec(a[:3, :3]), p_vec(b[:3, :3])
        lhs.append(skew(Pg + Pc))
        rhs.append(Pc - Pg)
    x, *_ = np.linalg.lstsq(np.vstack(lhs), np.concatenate(rhs), rcond=None)
    n2 = float(x @ x)
    Pcg = 2.0 * x / np.sqrt(1.0 + n2)
    p2 = float(Pcg @ Pcg)
    R_X = (1.0 - p2 / 2.0) * np.eye(3) + 0.5 * (
        np.outer(Pcg, Pcg) + np.sqrt(max(4.0 - p2, 0.0)) * skew(Pcg)
    )
    R_X = nearest_rotation(R_X)
    return make_T(R_X, _solve_translation(A, B, R_X))


def cv2_solve(Gs, Hs, method):
    R_g2b = [G[:3, :3].copy() for G in Gs]
    t_g2b = [G[:3, 3].reshape(3, 1).copy() for G in Gs]
    R_t2c = [H[:3, :3].copy() for H in Hs]
    t_t2c = [H[:3, 3].reshape(3, 1).copy() for H in Hs]
    R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
    return make_T(R, t)


def run_methods(Gs, Hs):
    """Return ({method name: 4x4 T_tool0_camera}, backend description)."""
    results = {}
    if hasattr(cv2, "calibrateHandEye"):
        backend = "cv2.calibrateHandEye"
        names = {
            "tsai": cv2.CALIB_HAND_EYE_TSAI,
            "park": cv2.CALIB_HAND_EYE_PARK,
            "horaud": cv2.CALIB_HAND_EYE_HORAUD,
            "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
            "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
        }
        for name, flag in names.items():
            try:
                results[name] = cv2_solve(Gs, Hs, flag)
            except cv2.error as e:
                print(f"[WARN] method {name} failed: {str(e).splitlines()[0]}")
    else:
        backend = (f"built-in Tsai/Park (OpenCV {cv2.__version__} has no cv2.calibrateHandEye)")
        for name, fn in (("tsai", tsai_solve), ("park", park_solve)):
            try:
                results[name] = fn(Gs, Hs)
            except RuntimeError as e:
                print(f"[WARN] method {name} failed: {e}")
    return results, backend


# --------------------------------------------------------------------------------------
# Quality checks
# --------------------------------------------------------------------------------------
def pose_diversity_report(Gs):
    """Print how much the tool0 orientation varies. Returns True if it looks sufficient."""
    R0 = Gs[0][:3, :3]
    rvs = np.array([rotvec(R0.T @ G[:3, :3]) for G in Gs])
    max_angle = max(
        rot_angle_deg(Gi[:3, :3].T @ Gj[:3, :3]) for i, Gi in enumerate(Gs) for Gj in Gs[i + 1:]
    )
    sv = np.linalg.svd(rvs - rvs.mean(axis=0), compute_uv=False)
    spread_deg = np.degrees(sv / np.sqrt(len(Gs)))   # rms spread along principal axes
    print(f"[INFO] Largest rotation between any two poses: {max_angle:.1f} deg")
    print("[INFO] RMS orientation spread along 3 principal axes [deg]: "
          + ", ".join(f"{s:.1f}" for s in spread_deg))
    ok = True
    if max_angle < MIN_TOTAL_ROT_DEG:
        print(f"[WARN] Largest rotation < {MIN_TOTAL_ROT_DEG:.0f} deg: rotation is too small for "
              "a reliable hand-eye result")
        ok = False
    if spread_deg[1] < 3.0:
        print("[WARN] Orientation varies mostly about ONE axis. Vary the rotation about at least "
              "two different axes (three is better).")
        ok = False
    return ok


def consistency_report(Gs, Hs, X):
    """The target is fixed, so G_i X H_i must be the same for every image. Return its scatter."""
    Ts = [G @ X @ H for G, H in zip(Gs, Hs)]
    t_mean = np.mean([T[:3, 3] for T in Ts], axis=0)
    # mean rotation via SVD of the averaged matrix
    R_mean = nearest_rotation(np.mean([T[:3, :3] for T in Ts], axis=0))
    dt = np.array([np.linalg.norm(T[:3, 3] - t_mean) * 1000.0 for T in Ts])        # mm
    dr = np.array([rot_angle_deg(R_mean.T @ T[:3, :3]) for T in Ts])               # deg
    return dt, dr, make_T(R_mean, t_mean)


def compare_methods(results, ref="park"):
    names = list(results)
    if len(names) < 2:
        return True
    if ref not in results:
        ref = names[0]
    print(f"\nMethod comparison (difference from '{ref}'):")
    print(f"  {'method':<12}{'d_trans [mm]':>14}{'d_rot [deg]':>14}")
    agree = True
    for n in names:
        D = inv_T(results[ref]) @ results[n]
        dtr = np.linalg.norm(results[n][:3, 3] - results[ref][:3, 3]) * 1000.0
        drot = rot_angle_deg(D[:3, :3])
        flag = ""
        if dtr > TRANS_AGREE_MM or drot > ROT_AGREE_DEG:
            flag = "  <-- exceeds tolerance"
            agree = False
        print(f"  {n:<12}{dtr:>14.3f}{drot:>14.3f}{flag}")
    return agree


# --------------------------------------------------------------------------------------
# Target detection
# --------------------------------------------------------------------------------------
class AprilTagDetector:
    def __init__(self, args):
        sys.path.append(str(Path(args.apriltag_runner_dir).expanduser()))
        from apriltag_runner import preprocess, detect_tags   # the user's existing module
        self.preprocess, self.detect_tags = preprocess, detect_tags
        self.args = args

    def detect(self, img):
        a = self.args
        gray = self.preprocess(img, use_clahe=False, use_sharpen=False, upscale=1.0)
        dets = self.detect_tags(
            gray, families=a.tag_family, nthreads=4, decimate=1.0, sigma=0.0,
            refine_edges=False, decode_sharpening=0.25, do_pose=True, tag_size=a.tag_size,
            fx=a.fx, fy=a.fy, cx=a.cx, cy=a.cy,
        )
        good = [d for d in dets
                if d.decision_margin >= a.min_margin and d.hamming <= 0
                and (a.tag_id is None or d.tag_id == a.tag_id)]
        if not good:
            return None, "no tag passed the quality gates"
        d = max(good, key=lambda x: x.decision_margin)
        T = make_T(np.asarray(d.pose_R, dtype=float).reshape(3, 3),
                   np.asarray(d.pose_t, dtype=float).reshape(3))
        pose_err = getattr(d, "pose_err", None)
        info = f"tag {int(d.tag_id)}, margin {d.decision_margin:.1f}"
        if pose_err is not None:
            info += f", pose_err {float(pose_err):.2e}"
        return T, info


class CharucoDetector:
    def __init__(self, args):
        dict_id = getattr(cv2.aruco, args.aruco_dict, None)
        if dict_id is None:
            raise SystemExit(f"Unknown ArUco dictionary: {args.aruco_dict}")
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        self.board = cv2.aruco.CharucoBoard(
            (args.squares_x, args.squares_y), args.square_length, args.marker_length, dictionary)
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.K = np.array([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1.0]])
        self.dist = np.array(args.dist, dtype=float) if args.dist else np.zeros(5)
        self.min_corners = args.min_corners

    def detect(self, img):
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.dtype == np.uint16:
            gray = (gray >> 8).astype(np.uint8)
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if corners is None or ids is None or len(ids) < self.min_corners:
            n = 0 if ids is None else len(ids)
            return None, f"only {n} ChArUco corners (need {self.min_corners})"
        obj, imgp = self.board.matchImagePoints(corners, ids)
        ok, rvec, tvec = cv2.solvePnP(obj, imgp, self.K, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, "solvePnP failed"
        proj, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - imgp.reshape(-1, 2)) ** 2, axis=1))))
        T = make_T(rot_from_vec(rvec), tvec.reshape(3))
        return T, f"{len(ids)} corners, reprojection rms {rms:.2f} px"


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
def load_poses(csv_path):
    poses = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        need = {"image", "x", "y", "z", "qx", "qy", "qz", "qw"}
        if reader.fieldnames is None or not need.issubset({h.strip() for h in reader.fieldnames}):
            raise SystemExit(f"Poses CSV must have the columns: {', '.join(sorted(need))}")
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items() if k}
            q = [float(row[k]) for k in ("qx", "qy", "qz", "qw")]
            T = make_T(quat_to_rot(*q), [float(row["x"]), float(row["y"]), float(row["z"])])
            poses[row["image"]] = T
    return poses


def find_pose(poses, path):
    for key in (path.name, path.stem):
        if key in poses:
            return poses[key]
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Eye-in-hand calibration: T_tool0_camera")
    p.add_argument("--images", required=True, help="Directory with the calibration images")
    p.add_argument("--poses", required=True, help="CSV: image,x,y,z,qx,qy,qz,qw (tool0 in base_link)")
    p.add_argument("--target", choices=["apriltag", "charuco"], default="apriltag")
    p.add_argument("--method", default="park", help="Method reported as the result "
                   "(tsai, park, horaud, andreff, daniilidis)")
    p.add_argument("--out", default=None, help="Optional output .npy path for the 4x4 matrix")
    p.add_argument("--drop", default="", help="Comma-separated image names to exclude")

    g = p.add_argument_group("camera intrinsics (pixels)")
    g.add_argument("--fx", type=float, default=DEFAULT_FX)
    g.add_argument("--fy", type=float, default=DEFAULT_FY)
    g.add_argument("--cx", type=float, default=DEFAULT_CX)
    g.add_argument("--cy", type=float, default=DEFAULT_CY)
    g.add_argument("--dist", type=float, nargs="+", default=None,
                   help="Distortion coefficients (ChArUco only; default none)")

    g = p.add_argument_group("AprilTag target")
    g.add_argument("--tag-size", type=float, default=0.005, help="Black-square edge length (m)")
    g.add_argument("--tag-family", default="tag36h11")
    g.add_argument("--tag-id", type=int, default=None)
    g.add_argument("--min-margin", type=float, default=20.0)
    g.add_argument("--apriltag-runner-dir", default=APRILTAG_RUNNER_DIR)

    g = p.add_argument_group("ChArUco target")
    g.add_argument("--squares-x", type=int, default=7)
    g.add_argument("--squares-y", type=int, default=5)
    g.add_argument("--square-length", type=float, default=0.04, help="Square edge (m)")
    g.add_argument("--marker-length", type=float, default=0.03, help="Marker edge (m)")
    g.add_argument("--aruco-dict", default="DICT_5X5_100")
    g.add_argument("--min-corners", type=int, default=12)
    return p.parse_args()


def main():
    args = parse_args()
    img_dir = Path(args.images).expanduser()
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"No images found in {img_dir}")
    poses = load_poses(Path(args.poses).expanduser())
    dropped = {s.strip() for s in args.drop.split(",") if s.strip()}
    detector = AprilTagDetector(args) if args.target == "apriltag" else CharucoDetector(args)

    names, Gs, Hs = [], [], []
    print(f"[INFO] {len(paths)} images in {img_dir}, target = {args.target}")
    for path in paths:
        if path.name in dropped or path.stem in dropped:
            print(f"  {path.name}: dropped by user")
            continue
        G = find_pose(poses, path)
        if G is None:
            print(f"  {path.name}: SKIPPED, no row in the poses CSV")
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  {path.name}: SKIPPED, cannot read image")
            continue
        H, info = detector.detect(img)
        if H is None:
            print(f"  {path.name}: SKIPPED, {info}")
            continue
        print(f"  {path.name}: OK, {info}, distance {np.linalg.norm(H[:3, 3]):.3f} m")
        names.append(path.name)
        Gs.append(G)
        Hs.append(H)

    n = len(Gs)
    print(f"\n[INFO] {n} usable poses")
    if n < 3:
        raise SystemExit("Need at least 3 usable poses (15-20 recommended)")
    if n < 10:
        print("[WARN] Fewer than 10 poses; 15-20 with varied orientation is recommended")

    diversity_ok = pose_diversity_report(Gs)

    results, backend = run_methods(Gs, Hs)
    print(f"[INFO] Solver backend: {backend}")
    if not results:
        raise SystemExit("All hand-eye methods failed")
    agree = compare_methods(results, ref="park")

    chosen = args.method.lower()
    if chosen not in results:
        fallback = "park" if "park" in results else next(iter(results))
        print(f"[WARN] Method '{chosen}' not available, using '{fallback}'")
        chosen = fallback
    X = results[chosen]

    dt, dr, T_base_target = consistency_report(Gs, Hs, X)
    print(f"\nConsistency of the target pose in base_link over all images (method '{chosen}'):")
    print("  (the target is fixed, so G_i * X * H_i should be identical for every image)")
    print(f"  {'image':<28}{'d_pos [mm]':>12}{'d_rot [deg]':>13}")
    for nm, a, b in zip(names, dt, dr):
        print(f"  {nm:<28}{a:>12.3f}{b:>13.3f}")
    print(f"  RMS: {np.sqrt(np.mean(dt ** 2)):.3f} mm, {np.sqrt(np.mean(dr ** 2)):.3f} deg   "
          f"worst: {dt.max():.3f} mm ({names[int(dt.argmax())]})")
    print(f"  Target in base_link [m]: {np.array2string(T_base_target[:3, 3], precision=5)}")

    print("\n" + "=" * 78)
    print(f"RESULT (method '{chosen}'): T_tool0_camera, camera expressed in tool0, meters")
    print("=" * 78)
    print(f"T_TOOL0_CAMERA = {format_T(X)}")
    print(f"\nCamera origin in tool0 [mm]: {np.array2string(X[:3, 3] * 1000, precision=2)}")

    if not agree or not diversity_ok:
        print("\n[WARN] The pose set looks poorly conditioned (see the warnings above). "
              "Add poses with larger and more varied rotation, and re-run before trusting this result.")
    elif np.sqrt(np.mean(dt ** 2)) > 2.0:
        print("\n[WARN] Large residuals: check for a bad detection (see the worst image) and use "
              "--drop to exclude it.")
    else:
        print("\n[INFO] Methods agree and the target is consistent across images.")

    if args.out:
        np.save(args.out, X)
        print(f"[INFO] Saved 4x4 matrix to {args.out}")


if __name__ == "__main__":
    main()
