#!/usr/bin/env python3
"""
Analyse a precision_test.py run.

Method
  The glass scale is a periodic grating with a KNOWN pitch (--pitch-um).  Each image is reduced to a
  1-D intensity profile along the image axis that the scale varies along.  The phase of the
  grating's fundamental frequency (a single-frequency DFT) gives the position of the scale to a
  small fraction of a pixel, in units of the pitch, i.e. directly in micrometres - no separate
  pixel calibration is needed.
  A phase only determines position modulo one pitch, so every image is unwrapped against the
  COMMANDED position (A -> 0, B -> step).  This is valid while |error| < pitch/2.

Outputs (in the run directory)
  summary.csv      one row per (axis, step)
  per_repeat.csv   one row per repeat
  Printed: camera noise/drift and the per-condition table.

Usage
  python3 analyze_precision.py precision_20260101_120000 --pitch-um 10 [--image-axis x] [--sign +1]
         [--roi x0 y0 x1 y1]
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Image -> position
# --------------------------------------------------------------------------- #
def load_gray(path):
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"cannot read {path}")
    except ImportError:
        from PIL import Image
        img = np.array(Image.open(path))
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    return img


def get_profile(img, roi, image_axis):
    if roi:
        x0, y0, x1, y1 = roi
        img = img[y0:y1, x0:x1]
    # scale lines vary along image X  -> average over rows ; along Y -> average over columns
    return img.mean(axis=0) if image_axis == "x" else img.mean(axis=1)


def _dft_mag(profile, freqs):
    n = np.arange(len(profile))
    w = np.hanning(len(profile))
    p = (profile - profile.mean()) * w
    return np.abs(np.exp(-2j * np.pi * np.outer(freqs, n)) @ p)


def estimate_frequency(profile, min_cycles=3):
    """Fundamental frequency (cycles/pixel) of the grating, refined to ~1e-6 cycles/pixel."""
    n = len(profile)
    pad = 16
    spec = np.abs(np.fft.rfft((profile - profile.mean()) * np.hanning(n), n * pad))
    lo = min_cycles * pad
    k = lo + int(np.argmax(spec[lo:]))
    f = k / (n * pad)
    half_width = 1.5 / (n * pad)
    for _ in range(3):   # each pass shrinks the search window 50x
        grid = np.linspace(f - half_width, f + half_width, 301)
        f = float(grid[np.argmax(_dft_mag(profile, grid))])
        half_width /= 50.0
    return f


def grating_phase(profile, f):
    n = np.arange(len(profile))
    d = np.sum((profile - profile.mean()) * np.hanning(len(profile)) * np.exp(-2j * np.pi * f * n))
    return float(np.angle(d)), float(np.abs(d))


def wrap(x, pitch):
    return (x + pitch / 2.0) % pitch - pitch / 2.0


def unwrap_to(wrapped, expected, pitch):
    return wrapped + pitch * round((expected - wrapped) / pitch)


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def stats(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return dict(n=0, mean=np.nan, std=np.nan, sem=np.nan, mn=np.nan, mx=np.nan, rms=np.nan, maxabs=np.nan)
    std = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    return dict(n=n, mean=float(np.mean(x)), std=std, sem=std / math.sqrt(n) if n > 1 else float("nan"),
                mn=float(np.min(x)), mx=float(np.max(x)), rms=float(np.sqrt(np.mean(x ** 2))),
                maxabs=float(np.max(np.abs(x))))


def read_rows(run_dir):
    with open(Path(run_dir) / "results.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["seq"] = int(r["seq"])
        r["repeat"] = int(r["repeat"])
        r["step_um"] = float(r["step_um"])
        r["t_epoch"] = float(r["t_epoch"])
        for k in ("fk_x_m", "fk_y_m", "fk_z_m"):
            r[k] = float(r[k])
    rows.sort(key=lambda r: r["seq"])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--pitch-um", type=float, required=True,
                    help="pitch (line-to-line distance) of the glass scale in micrometres")
    ap.add_argument("--image-axis", choices=["x", "y"], default="x",
                    help="image axis along which the scale lines repeat (default x)")
    ap.add_argument("--sign", type=int, choices=[1, -1], default=1,
                    help="+1 if +robot axis moves the scale towards +image axis, else -1. "
                         "Check with a step < pitch/2 (a positive bias ~ -2*step means the sign is wrong)")
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), default=None)
    ap.add_argument("--period-px", type=float, default=None, help="skip auto-detection of the line period")
    ap.add_argument("--robot-axis-index", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    P = args.pitch_um
    run = Path(args.run_dir)
    rows = [r for r in read_rows(run) if r["image"]]
    if not rows:
        sys.exit("No images in results.csv")

    # ---- profiles and the (single) grating frequency ----------------------
    profiles = {}
    for r in rows:
        profiles[r["seq"]] = get_profile(load_gray(run / r["image"]), args.roi, args.image_axis)
    ref_rows = [r for r in rows if r["phase"] == "noise"] or rows[:10]
    ref_profile = np.mean([profiles[r["seq"]] for r in ref_rows[:20]], axis=0)
    f = 1.0 / args.period_px if args.period_px else estimate_frequency(ref_profile)
    print(f"Grating period: {1.0 / f:.4f} px  ->  {P / (1.0 / f):.4f} um/px  (assuming pitch {P} um)")

    phase, amp = {}, {}
    for seq, prof in profiles.items():
        phase[seq], amp[seq] = grating_phase(prof, f)

    def rel_position_um(seq, ref_seq):
        """Wrapped shift of image `seq` relative to `ref_seq`, in um along the ROBOT axis."""
        dphi = phase[seq] - phase[ref_seq]
        shift_img = -dphi / (2 * math.pi) * P          # content moved by this much along +image axis
        return wrap(args.sign * shift_img, P)

    # ---- camera noise / drift --------------------------------------------
    noise = [r for r in rows if r["phase"] == "noise"]
    if len(noise) >= 3:
        pos = np.array([rel_position_um(r["seq"], noise[0]["seq"]) for r in noise])
        t = np.array([r["t_epoch"] for r in noise]) - noise[0]["t_epoch"]
        slope, icpt = np.polyfit(t, pos, 1)
        resid = pos - (slope * t + icpt)
        print("\nCAMERA NOISE / DRIFT  (stationary robot, {} images over {:.1f} min)".format(len(noise), t[-1] / 60))
        print(f"  std of position         : {np.std(pos, ddof=1):.4f} um   (peak-to-peak {np.ptp(pos):.4f} um)")
        print(f"  linear drift            : {slope * 60:+.4f} um/min  (total {slope * t[-1]:+.4f} um)")
        print(f"  std after drift removal : {np.std(resid, ddof=1):.4f} um   <- noise floor of one measurement")
        print(f"  => noise on a displacement (difference of two images) ~ {math.sqrt(2) * np.std(resid, ddof=1):.4f} um")
        if np.ptp(pos) > P / 4:
            print("  WARNING: scatter is a large fraction of the pitch; phase unwrapping is unreliable")
    else:
        print("\n(no/too few noise images in results.csv)")

    # ---- step conditions --------------------------------------------------
    groups = defaultdict(list)
    for r in rows:
        if r["phase"] == "step":
            groups[(r["axis"], r["step_um"])].append(r)

    summary, per_repeat, warnings = [], [], []
    for (axis, step), grp in sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        by = {(r["pose"], r["repeat"]): r for r in grp}
        if ("A", 0) not in by:
            continue
        a0 = by[("A", 0)]["seq"]
        pos = {}
        for (pose, k), r in by.items():
            expected = step if pose == "B" else 0.0
            pos[(pose, k)] = unwrap_to(rel_position_um(r["seq"], a0), expected, P)

        disp, ret, drift, robot_disp, b_pos = [], [], [], [], []
        axis_key = f"fk_{axis}_m"
        for k in range(1, max(kk for (_, kk) in by) + 1):
            if ("B", k) not in pos or ("A", k - 1) not in pos:
                continue
            d = pos[("B", k)] - pos[("A", k - 1)]
            disp.append(d)
            b_pos.append(pos[("B", k)])
            robot_disp.append((by[("B", k)][axis_key] - by[("A", k - 1)][axis_key]) * 1e6)
            row = {"axis": axis, "step_um": step, "repeat": k, "measured_disp_um": d,
                   "bias_um": d - step, "robot_fk_disp_um": robot_disp[-1]}
            if ("A", k) in pos:
                r_k = pos[("A", k)] - pos[("A", k - 1)]
                ret.append(r_k)
                drift.append(pos[("A", k)] - pos[("A", 0)])
                row["A_return_um"] = r_k
                row["A_vs_A0_um"] = drift[-1]
            per_repeat.append(row)

        sd, sr, sdr, sb, srb = stats(disp), stats(ret), stats(drift), stats(b_pos), stats(robot_disp)
        bias = sd["mean"] - step
        if abs(bias) > min(0.5 * step, P / 4):
            warnings.append(f"{axis.upper()} {step:g} um: |bias| {abs(bias):.2f} um is large - check --sign / "
                            f"--pitch-um / image axis (unwrapping only valid for |error| < pitch/2 = {P / 2:.1f} um)")
        summary.append({
            "axis": axis, "commanded_um": step, "n": sd["n"],
            "mean_measured_um": sd["mean"], "bias_um": bias, "bias_pct": 100.0 * bias / step,
            "std_disp_um": sd["std"], "sem_um": sd["sem"], "min_disp_um": sd["mn"], "max_disp_um": sd["mx"],
            "A_return_mean_um": sr["mean"], "A_return_std_um": sr["std"], "A_return_rms_um": sr["rms"],
            "A_return_max_abs_um": sr["maxabs"], "A_vs_A0_max_abs_um": sdr["maxabs"],
            "B_position_std_um": sb["std"], "robot_fk_mean_disp_um": srb["mean"],
        })

    # ---- report -----------------------------------------------------------
    print("\nRESULTS  (displacement = B_k - A_(k-1), along the commanded axis; all values in um)")
    hdr = (f"{'axis':>4} {'cmd':>7} {'n':>4} {'mean meas':>10} {'bias':>8} {'bias%':>7} {'std':>7} "
           f"{'A ret mean':>10} {'A ret std':>9} {'A ret max':>9} {'A-A0 max':>8}")
    print(hdr)
    print("-" * len(hdr))
    for s in summary:
        print(f"{s['axis'].upper():>4} {s['commanded_um']:>7g} {s['n']:>4} {s['mean_measured_um']:>10.3f} "
              f"{s['bias_um']:>+8.3f} {s['bias_pct']:>+7.2f} {s['std_disp_um']:>7.3f} "
              f"{s['A_return_mean_um']:>+10.3f} {s['A_return_std_um']:>9.3f} {s['A_return_max_abs_um']:>9.3f} "
              f"{s['A_vs_A0_max_abs_um']:>8.3f}")
    print("\n  bias      = mean measured displacement - commanded displacement")
    print("  A ret     = A_k - A_(k-1): did the robot come back to the same place (mean = systematic, std = scatter)")
    print("  A-A0 max  = largest excursion of A from the first A of that condition (includes drift)")
    for w in warnings:
        print("WARNING:", w)

    def dump(name, data):
        if not data:
            return
        keys = list(dict.fromkeys(k for d in data for k in d))
        with open(run / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(data)
        print(f"wrote {run / name}")

    dump("summary.csv", summary)
    dump("per_repeat.csv", per_repeat)


if __name__ == "__main__":
    main()
