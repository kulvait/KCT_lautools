#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Select contiguous angular range for near-180° endpoint-exclusive tomography.

Heuristic:
1) For each start index s:
   - Find first end index e_exceed where span(s, e_exceed) >= 180.
   - This defines base count N0 = e_exceed - s.
   - Consider counts N = N0, N0-1, ..., N0-extra_removals (N >= 3).
   - Choose N that minimizes endpoint error:
       | span(s, s+N-1) - ideal_span(N) |
     where ideal_span(N) = 180 * (N-1)/N.
   => exactly one candidate per start index.
2) Compare candidates across start indices:
   primary: absolute endpoint error
   secondary: interior sequence fit (RMS by default; L2 optional)
   tie-breaks: crop preference, keep more frames, earlier start.

This script reads raw metadata (nano log or micro h5), does not read/write ticks,
does not interpolate, and does not modify angles.
"""

import argparse
import json
import math
from pathlib import Path
import sys
from contextlib import redirect_stdout

import numpy as np
from lautools import NANOCT
from denpy import PETRA


def crop_preference(start: int, excess: int, mode: str) -> int:
    """Lower is better; used as tie-break after score terms."""
    if mode == "start":
        # Prefer cropping from start => larger start preferred.
        return excess - start
    if mode == "end":
        # Prefer cropping from end => smaller start preferred.
        return start
    # both: prefer balanced crop.
    return abs(start - (excess - start))


def compute_window_metrics(travel: np.ndarray, start: int, count: int) -> dict:
    """
    Metrics for one contiguous window [start, start+count).
    travel is monotonic increasing (direction-normalized).
    """
    stop = start + count
    rel = travel[start:stop] - travel[start]  # first point at 0
    span = float(rel[-1])  # measured endpoint-exclusive span

    ideal_step = 180.0 / count
    ideal = np.arange(count, dtype=np.float64) * ideal_step
    residual = rel - ideal

    l2 = float(np.sqrt(np.sum(residual ** 2)))
    rms = float(l2 / math.sqrt(count))
    max_abs = float(np.max(np.abs(residual)))

    ideal_span = float(180.0 * (count - 1) / count)
    signed_endpoint_error = span - ideal_span
    abs_endpoint_error = abs(signed_endpoint_error)

    increments = np.diff(rel)
    inc_min = float(np.min(increments)) if len(increments) else 0.0
    inc_max = float(np.max(increments)) if len(increments) else 0.0
    inc_med = float(np.median(increments)) if len(increments) else 0.0

    return {
        "span_deg": span,
        "ideal_span_deg": ideal_span,
        "signed_endpoint_error_deg": float(signed_endpoint_error),
        "absolute_endpoint_error_deg": float(abs_endpoint_error),
        "ideal_step_deg": float(ideal_step),
        "l2_deviation_deg": l2,
        "rms_deviation_deg": rms,
        "max_abs_deviation_deg": max_abs,
        "minimum_increment_deg": inc_min,
        "maximum_increment_deg": inc_max,
        "median_increment_deg": inc_med,
        "seam_gap_to_180_deg": float(180.0 - span),
    }


def choose_count_for_start(
    travel: np.ndarray,
    start: int,
    extra_removals: int,
) -> dict | None:
    """
    For a fixed start index:
    - find first index that reaches/exceeds 180°
    - test N0, N0-1, ..., N0-extra_removals
    - pick N minimizing endpoint error, then keep more frames, then better RMS
    Returns candidate dict or None if no valid window.
    """
    total = len(travel)
    # Find first e such that travel[e] - travel[start] >= 180
    e = start + 1
    while e < total and (travel[e] - travel[start]) < 180.0:
        e += 1

    if e >= total:
        # This start never reaches 180; cannot define N0 by "first exceed 180".
        return None

    n0 = e - start  # first count whose next sample would reach/exceed half-turn
    tested = []
    best = None
    best_key = None

    for removed in range(extra_removals + 1):
        n = n0 - removed
        if n < 3:
            continue
        stop = start + n
        if stop > total:
            continue

        m = compute_window_metrics(travel, start, n)
        tested.append({
            "count": int(n),
            "span_deg": m["span_deg"],
            "ideal_span_deg": m["ideal_span_deg"],
            "absolute_endpoint_error_deg": m["absolute_endpoint_error_deg"],
            "signed_endpoint_error_deg": m["signed_endpoint_error_deg"],
        })

        # Per-start selection: endpoint error first, then keep more frames,
        # then smoother interior.
        key = (
            round(m["absolute_endpoint_error_deg"], 12),
            -n,
            round(m["rms_deviation_deg"], 12),
        )
        if best_key is None or key < best_key:
            best_key = key
            best = {
                "fromidx": int(start),
                "count": int(n),
                "stop_exclusive": int(stop),
                "start_base_count_n0": int(n0),
                "removals_from_n0": int(removed),
                "counts_tested_for_start": tested,  # updated each loop
                **m,
            }

    if best is None:
        return None
    best["counts_tested_for_start"] = tested
    return best


def describe_selection(
    angles: np.ndarray,
    travel: np.ndarray,
    start: int,
    count: int,
    median_increment_full: float,
) -> dict:
    """Required output fields + expanded diagnostics for selected range."""
    stop = start + count
    rel = travel[start:stop] - travel[start]
    target_step = 180.0 / count
    ideal = np.arange(count, dtype=np.float64) * target_step
    residual = rel - ideal
    increments = np.diff(rel)

    worst_local = int(np.argmax(np.abs(residual)))
    worst_global = start + worst_local

    return {
        "selected_range_fromidx": int(start),
        "selected_range_count": int(count),
        "selected_range_span": float(rel[-1]),
        "selected_range_stop_exclusive": int(stop),
        "cropped_start_count": int(start),
        "cropped_end_count": int(len(angles) - stop),
        "selected_first_angle_deg": float(angles[start]),
        "selected_last_angle_deg": float(angles[stop - 1]),
        "diagnostics": {
            "ideal_step_deg": float(target_step),
            "ideal_span_deg": float(180.0 * (count - 1) / count),
            "span_minus_ideal_span_deg": float(
                rel[-1] - 180.0 * (count - 1) / count
            ),
            "max_abs_deviation_from_ideal_deg": float(np.max(np.abs(residual))),
            "rms_deviation_from_ideal_deg": float(np.sqrt(np.mean(residual ** 2))),
            "l2_deviation_from_ideal_deg": float(np.sqrt(np.sum(residual ** 2))),
            "worst_deviation_selected_index": int(worst_local),
            "worst_deviation_input_index": int(worst_global),
            "worst_signed_deviation_deg": float(residual[worst_local]),
            "minimum_increment_deg": float(np.min(increments)) if len(increments) else 0.0,
            "maximum_increment_deg": float(np.max(increments)) if len(increments) else 0.0,
            "median_increment_selected_deg": float(np.median(increments)) if len(increments) else 0.0,
            "median_increment_full_deg": float(median_increment_full),
            "max_increment_deviation_from_full_median_percent": float(
                100.0 * np.max(np.abs(increments - median_increment_full)) / median_increment_full
            ) if len(increments) and median_increment_full > 0 else 0.0,
            "gap_to_half_turn_deg": float(180.0 - rel[-1]),
        },
    }


def select_sequence(
    angles: np.ndarray,
    crop: str = "both",
    max_span_deg: float = 200.0,
    extra_removals: int = 2,
    score: str = "rms",
    boundary_tolerance_deg: float = 1e-6,
    top_candidates_limit: int = 10,
) -> dict:
    angles = np.asarray(angles, dtype=np.float64)
    if angles.ndim != 1 or len(angles) < 3:
        raise ValueError("Need at least three imaging angles.")
    if not np.isfinite(angles).all():
        raise ValueError("Angles contain NaN or infinity.")

    diffs = np.diff(angles)
    if np.all(diffs > 0):
        direction = 1
    elif np.all(diffs < 0):
        direction = -1
    else:
        raise ValueError(
            "Angles must be strictly monotonic and unwrapped; "
            "duplicates/reversals/wrapping are unsupported."
        )

    travel = direction * (angles - angles[0])
    increments = np.diff(travel)

    total = len(angles)
    input_span = float(travel[-1])
    full_ideal_span = float(180.0 * (total - 1) / total)

    median_increment = float(np.median(increments))
    mean_increment = float(np.mean(increments))
    std_increment = float(np.std(increments))
    min_increment = float(np.min(increments))
    max_increment = float(np.max(increments))

    report = {
        "mode": None,  # filled by caller
        "input_metadata": None,  # filled by caller
        "input_frame_count": int(total),
        "input_span_deg": input_span,
        "input_first_angle_deg": float(angles[0]),
        "input_last_angle_deg": float(angles[-1]),
        "direction": int(direction),
        "increment_stats_deg": {
            "median": median_increment,
            "mean": mean_increment,
            "std": std_increment,
            "min": min_increment,
            "max": max_increment,
        },
        "full_count_ideal_half_turn_span_deg": full_ideal_span,
        "maximum_processing_span_deg": float(max_span_deg),
        "crop_preference": crop,
        "score": score,
        "extra_removals": int(extra_removals),
        "angles_modified": False,
    }

    # Requested boundary behavior:
    # - below full-count ideal endpoint-exclusive span => suggest full range
    if input_span < full_ideal_span - boundary_tolerance_deg:
        report.update(describe_selection(
            angles, travel, start=0, count=total, median_increment_full=median_increment
        ))
        report["selection_reason"] = "full_range_below_endpoint_exclusive_half_turn"
        report["range_changed"] = False
        report["candidate_window_count"] = 1
        return report

    # - above max span (default 200) => suggest full range
    if input_span > max_span_deg + boundary_tolerance_deg:
        report.update(describe_selection(
            angles, travel, start=0, count=total, median_increment_full=median_increment
        ))
        report["selection_reason"] = "full_range_above_maximum_processing_span"
        report["range_changed"] = False
        report["candidate_window_count"] = 1
        return report

    # Build one candidate per start index using per-start count adaptation.
    candidates = []
    # We need at least 3 points and enough room to reach 180 with at least 3 points.
    for start in range(0, total - 2):
        c = choose_count_for_start(
            travel=travel,
            start=start,
            extra_removals=extra_removals,
        )
        if c is None:
            continue

        excess = total - c["count"]
        primary = c["rms_deviation_deg"] if score == "rms" else c["l2_deviation_deg"]

        # Global ranking across starts:
        # 1) interior fit by chosen score (RMS default)
        # 2) endpoint error as secondary safety/tie-break
        # 3) crop preference
        # 4) keep more frames
        # 5) earlier start
        c["_primary_score"] = float(primary)
        c["_key"] = (
            round(primary, 12),
            round(c["absolute_endpoint_error_deg"], 12),
            crop_preference(c["fromidx"], excess, crop),
            -c["count"],
            c["fromidx"],
        )
        candidates.append(c)

    if not candidates:
        raise ValueError(
            "No valid candidate found. "
            "Could not construct any start index that reaches/exceeds 180°."
        )

    candidates.sort(key=lambda x: x["_key"])
    best = candidates[0]

    report.update(describe_selection(
        angles,
        travel,
        start=best["fromidx"],
        count=best["count"],
        median_increment_full=median_increment,
    ))
    report["selection_reason"] = "selected_half_turn_range_per_start"
    report["range_changed"] = not (
        best["fromidx"] == 0 and best["count"] == total
    )
    report["selected_score"] = {
        "primary_metric": score,
        "primary_value_deg": float(best["_primary_score"]),
        "secondary_metric": "absolute_endpoint_error_deg",
        "secondary_value_deg": float(best["absolute_endpoint_error_deg"]),
        "ideal_span_deg": float(best["ideal_span_deg"]),
        "signed_endpoint_error_deg": float(best["signed_endpoint_error_deg"]),
        "rms_deviation_deg": float(best["rms_deviation_deg"]),
        "l2_deviation_deg": float(best["l2_deviation_deg"]),
        "max_abs_deviation_deg": float(best["max_abs_deviation_deg"]),
        "start_base_count_n0": int(best["start_base_count_n0"]),
        "removals_from_n0": int(best["removals_from_n0"]),
    }

    # Store best candidate summary per chosen count
    best_per_count = {}
    for c in candidates:
        k = str(c["count"])
        if k not in best_per_count:
            best_per_count[k] = {
                "fromidx": int(c["fromidx"]),
                "count": int(c["count"]),
                "absolute_endpoint_error_deg": float(c["absolute_endpoint_error_deg"]),
                "ideal_span_deg": float(c["ideal_span_deg"]),
                "signed_endpoint_error_deg": float(c["signed_endpoint_error_deg"]),
                "rms_deviation_deg": float(c["rms_deviation_deg"]),
                "l2_deviation_deg": float(c["l2_deviation_deg"]),
                "max_abs_deviation_deg": float(c["max_abs_deviation_deg"]),
                "span_deg": float(c["span_deg"]),
                "cropped_start": int(c["fromidx"]),
                "cropped_end": int(total - c["stop_exclusive"]),
                "start_base_count_n0": int(c["start_base_count_n0"]),
                "removals_from_n0": int(c["removals_from_n0"]),
            }
    report["best_per_count"] = best_per_count

    # Keep top candidates for diagnostics
    top = []
    limit = max(1, top_candidates_limit)
    for c in candidates[:limit]:
        top.append({
            "fromidx": int(c["fromidx"]),
            "count": int(c["count"]),
            "absolute_endpoint_error_deg": float(c["absolute_endpoint_error_deg"]),
            "ideal_span_deg": float(c["ideal_span_deg"]),
            "signed_endpoint_error_deg": float(c["signed_endpoint_error_deg"]),
            "rms_deviation_deg": float(c["rms_deviation_deg"]),
            "l2_deviation_deg": float(c["l2_deviation_deg"]),
            "max_abs_deviation_deg": float(c["max_abs_deviation_deg"]),
            "span_deg": float(c["span_deg"]),
            "cropped_start": int(c["fromidx"]),
            "cropped_end": int(total - c["stop_exclusive"]),
            "start_base_count_n0": int(c["start_base_count_n0"]),
            "removals_from_n0": int(c["removals_from_n0"]),
        })
    report["top_candidates"] = top
    report["candidate_window_count"] = int(len(candidates))

    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select optimal contiguous range approximating endpoint-exclusive 180° sequence."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation")

    for mode, input_name, help_text in (
        ("micro", "h5_file", "Read microCT HDF5 metadata"),
        ("nano", "log_file", "Read nanoCT log metadata"),
    ):
        sp = subparsers.add_parser(mode, help=help_text)
        sp.add_argument(input_name, type=Path, help=f"Path to input {mode} metadata")
        sp.add_argument(
            "--crop",
            choices=("start", "end", "both"),
            default="both",
            help="Preferred cropping side used as tie-break."
        )
        sp.add_argument(
            "--max-span-deg",
            type=float,
            default=200.0,
            help="If full range span exceeds this, suggest full range unchanged."
        )
        sp.add_argument(
            "--extra-removals",
            type=int,
            default=2,
            help="Per start index, also test N0-1..N0-N where N0 is first count that reaches/exceeds 180."
        )
        sp.add_argument(
            "--score",
            choices=("rms", "l2"),
            default="rms",
            help="Secondary ranking metric after endpoint error (default: rms)."
        )
        sp.add_argument(
            "--boundary-tolerance-deg",
            type=float,
            default=1e-6,
            help="Numerical tolerance for boundary decisions."
        )
        sp.add_argument(
            "--top-candidates-limit",
            type=int,
            default=10,
            help="Number of top candidates to include in output diagnostics."
        )
        sp.add_argument(
            "--output",
            type=Path,
            default=None,
            help="Output JSON path (default: stdout)."
        )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate numeric options
    if not math.isfinite(args.max_span_deg) or args.max_span_deg < 180.0:
        raise SystemExit("Error: --max-span-deg must be finite and >= 180.")
    if not math.isfinite(args.boundary_tolerance_deg) or args.boundary_tolerance_deg < 0:
        raise SystemExit("Error: --boundary-tolerance-deg must be finite and nonnegative.")
    if args.extra_removals < 0:
        raise SystemExit("Error: --extra-removals must be nonnegative.")
    if args.top_candidates_limit < 1:
        raise SystemExit("Error: --top-candidates-limit must be >= 1.")

    try:
        input_path = (args.h5_file if args.mode == "micro" else args.log_file).resolve()

        if args.output is not None and args.output.resolve() == input_path:
            raise ValueError("Output JSON must not overwrite input metadata.")

        # Keep stdout clean for JSON.
        with redirect_stdout(sys.stderr):
            if args.mode == "micro":
                df = PETRA.scanDataset(str(args.h5_file), includeCurrent=False)
            else:
                df = NANOCT.scanDataset(str(args.log_file))

        if "image_key" not in df.columns or "s_rot" not in df.columns:
            raise ValueError("Metadata must contain 'image_key' and 's_rot' columns.")

        df_img = df.loc[df["image_key"] == 0]
        angles = df_img["s_rot"].to_numpy(dtype=np.float64)

        report = select_sequence(
            angles=angles,
            crop=args.crop,
            max_span_deg=args.max_span_deg,
            extra_removals=args.extra_removals,
            score=args.score,
            boundary_tolerance_deg=args.boundary_tolerance_deg,
            top_candidates_limit=args.top_candidates_limit,
        )
        report["mode"] = args.mode
        report["input_metadata"] = str(input_path)

        out_text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(out_text)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(out_text, encoding="utf-8")

    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
