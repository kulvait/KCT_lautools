#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Current correction after dark-field correction.

Metadata sources:
    - PETRA HDF5
    - NanoCT LogScan.log, optionally matched against acquisition TIFFs

Examples:
    currentCorrect.py scan.h5 input.den output.den
    currentCorrect.py input.den output.den --input-h5 scan.h5
    currentCorrect.py input.den output.den \
        --nano-log-file LogScan.log --nano-image-dir /path/to/images

The selected metadata rows must follow the same acquisition order as the
frames stored in inputDen.

@author: Vojtěch Kulvait
"""

import argparse
import datetime
import logging
import multiprocessing
import os
import shutil
import sys
import time
from contextlib import nullcontext

# Create a logger specific to this module.
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s : %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S'
)
ch.setFormatter(formatter)

if not log.handlers:
    log.addHandler(ch)
log.propagate = False

import numpy as np
import pandas as pd
from denpy import DEN, PETRA, UTILS
from scipy.ndimage import convolve1d
from scipy.optimize import curve_fit
from skimage.restoration import denoise_tv_chambolle


# Info-object row definitions:
#  0: Time in seconds relative to the first selected frame
#  1: Original rotation angle in degrees
#  2: Current in mA
#  3: Original frame mean
#  4: Original frame median
#  5: Frame mean after current normalization
#  6: Frame median after current normalization
#  7: Original mean of the lowest 90% of pixels
#  8: Original median of the lowest 90% of pixels
#  9: Lower-quantile fraction
# 10: Final requested frame mean


class LoggingArgumentParser(argparse.ArgumentParser):
    """Log command-line errors while retaining argparse behavior."""

    def error(self, message):
        log.error("Invalid arguments: %s", message)
        super().error(message)


class MetadataError(RuntimeError):
    """Failure while loading or validating acquisition metadata."""


def format_elapsed(seconds):
    """Format elapsed seconds as hh:mm:ss, allowing hours above 23."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_parser():
    parser = LoggingArgumentParser(
        description=(
            "Correct DEN frame intensities using PETRA H5 or NanoCT log "
            "metadata, optionally followed by smoothing and detrending."
        )
    )

    # Optional leading positional preserves the historical H5 invocation.
    parser.add_argument(
        "h5file",
        nargs="?",
        default=None,
        help="Legacy positional H5 metadata file.",
    )
    parser.add_argument("inputDen", help="Input DEN image stack.")
    parser.add_argument("outputDen", help="Corrected output DEN image stack.")

    sources = parser.add_mutually_exclusive_group()
    sources.add_argument(
        "--input-h5",
        default=None,
        help="H5 metadata file instead of the legacy positional argument.",
    )
    sources.add_argument(
        "--nano-log-file",
        default=None,
        help="NanoCT acquisition log, for example LogScan.log.",
    )
    parser.add_argument(
        "--nano-image-dir",
        default=None,
        help=(
            "Original acquisition TIFF directory for matching NanoCT log "
            "records. Use the same selection/order as the input DEN."
        ),
    )
    parser.add_argument(
        "--type",
        default="IMG",
        const="IMG",
        nargs="?",
        choices=["IMG", "REF"],
        help="Select IMG projections or REF flat fields. Default: IMG.",
    )
    parser.add_argument(
        "--target-current-value",
        type=float,
        default=None,
        help=(
            "Target current in mA. Default: current associated with the "
            "first selected frame."
        ),
    )
    parser.add_argument(
        "--saveinfo",
        default=None,
        help="Save the 11-row statistics/correction info object as DEN.",
    )
    parser.add_argument(
        "--savefig",
        default=None,
        help="Save diagnostic figures as a multipage PDF.",
    )
    parser.add_argument(
        "--fix-corrupted-h5",
        action="store_true",
        help=(
            "On H5 metadata-loading failure, copy inputDen to outputDen "
            "without correction. Does not recover NanoCT or processing errors."
        ),
    )
    parser.add_argument(
        "--params-file",
        default="params",
        help=(
            "Parameter file. Default: params in the working directory. "
            "A missing file is treated as an empty parameter set."
        ),
    )

    offsets = parser.add_mutually_exclusive_group()
    offsets.add_argument(
        "--correct-time-offset",
        action="store_true",
        help="Apply time_offset_sec from params to H5 current measurements.",
    )
    offsets.add_argument(
        "--correct-insertion-offset",
        action="store_true",
        help=(
            "Apply time_insertion_offset_sec from params to H5 current "
            "measurements."
        ),
    )

    parser.add_argument(
        "--tv-denoise",
        action="store_true",
        help="TV-denoise the current-normalized mean before further processing.",
    )
    parser.add_argument(
        "--remove-short-term-noise",
        type=float,
        default=None,
        help=(
            "Gaussian smoothing sigma in degrees for IMG, or frame indices "
            "for REF. Smoothing is performed separately across time gaps."
        ),
    )
    parser.add_argument(
        "--remove-long-term-trend",
        type=float,
        default=None,
        help=(
            "Gaussian trend sigma in degrees for IMG, or frame indices for "
            "REF. Subtract the mean-centered smoothed trend."
        ),
    )
    parser.add_argument(
        "--connect-gaps",
        action="store_true",
        help="Align consecutive mean-intensity segments across time gaps.",
    )
    parser.add_argument(
        "--time-gap",
        type=float,
        default=1.0,
        help=(
            "Time gap in seconds used to split denoising/smoothing/alignment "
            "segments. Default: 1.0. Set above normal frame spacing."
        ),
    )

    detrending = parser.add_mutually_exclusive_group()
    detrending.add_argument(
        "--detrend-data",
        action="store_true",
        help="Remove an endpoint-constrained linear trend.",
    )
    detrending.add_argument(
        "--detrend-quadratic",
        action="store_true",
        help="Remove an endpoint-constrained quadratic trend.",
    )
    detrending.add_argument(
        "--detrend-exponential",
        action="store_true",
        help="Remove an endpoint-constrained exponential trend.",
    )

    parser.add_argument(
        "--perform-trigonometric-fit",
        type=int,
        default=None,
        help=(
            "Fit K sine/cosine harmonics plus a constant. K=0 estimates K "
            "from insertion_event_count in params or detected insertions. "
            "As in the original script, K is raised to at least 2 and even."
        ),
    )
    parser.add_argument(
        "--detrend-trigonometric-fit",
        action="store_true",
        help="Remove an endpoint linear trend before trigonometric fitting.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting existing output files.",
    )
    parser.add_argument(
        "-j",
        type=int,
        default=-1,
        help=(
            "Worker processes for frame statistics: 0 disables "
            "multiprocessing, -1 selects CPU count. DEN writing is serial."
        ),
    )
    return parser


def validate_arguments(parser, args):
    if args.h5file is not None and args.input_h5 is not None:
        parser.error("Use either positional H5 or --input-h5, not both.")

    args.h5file = args.input_h5 or args.h5file

    if (args.h5file is None) == (args.nano_log_file is None):
        parser.error("Specify exactly one H5 file or --nano-log-file.")

    if args.nano_image_dir is not None and args.nano_log_file is None:
        parser.error("--nano-image-dir requires --nano-log-file.")

    if args.fix_corrupted_h5 and args.h5file is None:
        parser.error("--fix-corrupted-h5 requires H5 input.")

    if args.j < -1:
        parser.error("-j must be -1 or a nonnegative integer.")

    if not np.isfinite(args.time_gap) or args.time_gap <= 0:
        parser.error("--time-gap must be finite and positive.")

    if args.target_current_value is not None:
        if (
            not np.isfinite(args.target_current_value)
            or args.target_current_value <= 0
        ):
            parser.error("--target-current-value must be finite and positive.")

    for name in ("remove_short_term_noise", "remove_long_term_trend"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(
                "--%s must be finite and positive." % name.replace("_", "-")
            )

    if (
        args.perform_trigonometric_fit is not None
        and args.perform_trigonometric_fit < 0
    ):
        parser.error("--perform-trigonometric-fit must be nonnegative.")

    if (
        args.detrend_trigonometric_fit
        and args.perform_trigonometric_fit is None
    ):
        parser.error(
            "--detrend-trigonometric-fit requires "
            "--perform-trigonometric-fit."
        )


def same_path(first, second):
    if os.path.realpath(first) == os.path.realpath(second):
        return True
    if os.path.exists(first) and os.path.exists(second):
        return os.path.samefile(first, second)
    return False


def validate_paths(args):
    """Prevent overwrite of inputs and collisions between outputs."""
    if not os.path.isfile(args.inputDen):
        raise FileNotFoundError(f"Input DEN does not exist: {args.inputDen}")

    inputs = [args.inputDen]
    inputs.extend(
        path for path in (
            args.h5file,
            args.nano_log_file,
            args.params_file,
        )
        if path is not None
    )
    outputs = [
        path for path in (args.outputDen, args.saveinfo, args.savefig)
        if path is not None
    ]

    for output in outputs:
        if any(same_path(output, source) for source in inputs):
            raise ValueError(f"Output would overwrite an input: {output}")
        if os.path.lexists(output) and not args.force:
            raise FileExistsError(
                f"Output exists: {output}; use --force to overwrite."
            )
        if os.path.isdir(output):
            raise IsADirectoryError(output)
        parent = os.path.dirname(os.path.abspath(output))
        if not os.path.isdir(parent):
            raise FileNotFoundError(f"Output directory does not exist: {parent}")

    for index, output in enumerate(outputs):
        if any(same_path(output, other) for other in outputs[index + 1:]):
            raise ValueError("Output DEN, info DEN and PDF paths must differ.")

    header = DEN.readHeader(args.inputDen)
    if len(header["dimspec"]) != 3:
        raise ValueError("Input DEN must have three dimensions.")
    if any(int(dim) <= 0 for dim in header["dimspec"]):
        raise ValueError("Input DEN dimensions must be positive.")
    return header


def read_params(args):
    if args.params_file is None:
        return {}

    if not os.path.exists(args.params_file):
        log.info("Parameter file %s not found; using defaults.", args.params_file)
        return {}

    params = UTILS.readParamsFile(args.params_file)
    if not isinstance(params, dict):
        raise TypeError("Parameter reader must return a dictionary.")
    return params


def normalize_current_dataset(data):
    """Normalize a current series without modifying its caller."""
    missing = {"time", "current"} - set(data.columns)
    if missing:
        raise ValueError(f"Current metadata is missing columns: {missing}")

    current = data[["time", "current"]].copy()
    current["time"] = pd.to_datetime(
        current["time"], utc=True, errors="raise"
    )
    current["current"] = pd.to_numeric(
        current["current"], errors="raise"
    ).astype(np.float64)

    if current.empty:
        raise ValueError("No current measurements available.")
    if current["time"].isna().any():
        raise ValueError("Current timestamps contain missing values.")
    if not np.all(np.isfinite(current["current"].to_numpy())):
        raise ValueError("Current measurements must be finite.")

    current = current.sort_values("time", kind="stable")
    conflicts = current.groupby("time")["current"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Conflicting currents at duplicate timestamps.")

    return current.drop_duplicates("time").reset_index(drop=True)


def adjust_insertion_events_to_slice(insertion_events, startind, endind):
    adjusted = []
    for event in insertion_events:
        if (
            startind <= event["start_index"] < endind
            and startind <= event["end_index"] < endind
        ):
            copied = event.copy()
            copied["start_index"] -= startind
            copied["end_index"] -= startind
            adjusted.append(copied)
    return adjusted


def load_metadata(args, params, image_count):
    """Return selected frames, current series and applied current offset."""
    offset = 0.0

    if args.h5file is not None:
        if not os.path.isfile(args.h5file):
            raise FileNotFoundError(args.h5file)

        key = None
        if args.correct_time_offset:
            key = "time_offset_sec"
        elif args.correct_insertion_offset:
            key = "time_insertion_offset_sec"

        if key is not None:
            if params.get(key) is None:
                log.warning("%s missing from params; using zero offset.", key)
            else:
                offset = float(params[key])

        if not np.isfinite(offset):
            raise ValueError("Current time offset must be finite.")

        log.info("Using H5 metadata with current offset %.6f s.", offset)
        scan = PETRA.scanDataset(
            args.h5file,
            includeCurrent=True,
            timeOffsetSec=offset,
        )
        current = PETRA.beamCurrentDataset(
            args.h5file,
            timeOffsetSec=offset,
        )
    else:
        from lautools import NANOCT

        if not os.path.isfile(args.nano_log_file):
            raise FileNotFoundError(args.nano_log_file)

        if args.correct_time_offset or args.correct_insertion_offset:
            log.warning(
                "Ignoring time-offset options for NanoCT: frame times and "
                "currents already share log timestamps."
            )

        full_scan = NANOCT.scanDataset(args.nano_log_file)
        if full_scan.empty:
            raise ValueError("NanoCT log contains no acquisition records.")

        # Retain all log current measurements, including rows whose images
        # may be absent from the selected raw-file collection.
        current = full_scan[["time", "current"]].copy()

        if args.nano_image_dir is None:
            scan = full_scan
            log.warning(
                "No --nano-image-dir supplied: assuming DEN frames follow "
                "the selected log rows in acquisition order."
            )
        else:
            scan = NANOCT.scanDataset(
                args.nano_log_file,
                args.nano_image_dir,
            )

        log.info("Using NanoCT log metadata with zero time offset.")

    if scan.empty:
        raise ValueError("No acquisition metadata available.")

    required = {"time", "image_key", "s_rot", "current"}
    missing = required - set(scan.columns)
    if missing:
        raise ValueError(f"Scan metadata is missing columns: {missing}")

    image_key = 1 if args.type == "REF" else 0
    frames = scan.loc[scan["image_key"] == image_key].copy()
    frames = frames.reset_index(drop=True)

    if len(frames) != image_count:
        raise ValueError(
            f"Metadata contains {len(frames)} {args.type} records; "
            f"DEN contains {image_count} frames."
        )

    frames["time"] = pd.to_datetime(
        frames["time"], utc=True, errors="raise"
    )
    if frames["time"].isna().any():
        raise ValueError("Frame timestamps contain missing values.")
    if not (frames["time"].diff().iloc[1:] > pd.Timedelta(0)).all():
        raise ValueError(
            "Frame timestamps must increase in DEN acquisition order. "
            "Metadata will not be silently sorted."
        )

    for column in ("s_rot", "current"):
        frames[column] = pd.to_numeric(
            frames[column], errors="raise"
        ).astype(np.float64)
        if not np.all(np.isfinite(frames[column].to_numpy())):
            raise ValueError(f"Invalid frame metadata in column {column}.")

    if np.any(frames["current"].to_numpy() <= 0):
        raise ValueError("Frame currents must be strictly positive.")

    frames["frame_ind"] = np.arange(image_count, dtype=np.int64)
    return frames, normalize_current_dataset(current), offset


def select_current_and_events(args, params, frames, current, offset):
    """Select the scan's current interval and optionally detect insertions."""
    zerotime = frames["time"].iloc[0]
    endtime = frames["time"].iloc[-1]

    needs_events = (
        args.savefig is not None
        or (
            args.perform_trigonometric_fit == 0
            and params.get("insertion_event_count") is None
        )
    )

    events = []
    if needs_events:
        if args.h5file is not None:
            # Existing PETRA H5 timestamps are timezone-naive UTC.
            events = PETRA.findInsertionEvents(
                args.h5file,
                timeOffsetSec=offset,
                zeroTime=zerotime.tz_localize(None),
            )
        else:
            # Requires the proposed DataFrame-capable PETRA implementation.
            events = PETRA.findInsertionEvents(
                current,
                timeOffsetSec=0.0,
                zeroTime=zerotime,
            )
            log.info(
                "NanoCT insertions are estimated from acquisition-associated "
                "current samples, not an independent dense current series."
            )

    # Include one neighbouring current sample on either side.
    start = max(
        0,
        int(current["time"].searchsorted(zerotime, side="right")) - 1,
    )
    end = min(
        len(current),
        int(current["time"].searchsorted(endtime, side="left")) + 1,
    )
    beam = current.iloc[start:end].reset_index(drop=True)
    valid_events = adjust_insertion_events_to_slice(events, start, end)

    # Automatic K counts only events whose midpoint lies in the scan.
    duration = (endtime - zerotime).total_seconds()
    event_count = sum(
        0.0 <= float(event["mid_time"]) <= duration
        for event in valid_events
    )

    log.info(
        "Selected %d %s frames; detected %d insertions within the scan.",
        len(frames), args.type, event_count,
    )
    return beam, valid_events, event_count


def processFrame(
    index, input_den, frame_time, angle, current, target_current,
    dimx, dimy, lower_quantile=0.9,
):
    """Compute statistics for one frame; usable as a multiprocessing worker."""
    image = DEN.getFrame(input_den, int(index))
    if image.shape != (dimy, dimx):
        raise ValueError(
            f"Frame {index} shape {image.shape}; expected {(dimy, dimx)}."
        )

    pixels = image.reshape(-1)
    if not np.all(np.isfinite(pixels)):
        raise ValueError(f"Frame {index} contains nonfinite pixels.")

    mean = float(np.mean(pixels, dtype=np.float64))
    median = float(np.median(pixels))
    if mean == 0:
        raise ValueError(f"Frame {index} has zero mean.")

    count = max(1, int(lower_quantile * pixels.size))
    count = min(count, pixels.size)
    lower = np.partition(pixels, count - 1)[:count]
    factor = target_current / current

    info = np.array([
        frame_time,
        angle,
        current,
        mean,
        median,
        mean * factor,
        median * factor,
        np.mean(lower, dtype=np.float64),
        np.median(lower),
        lower_quantile,
    ], dtype=np.float64)

    if not np.all(np.isfinite(info)):
        raise ValueError(f"Frame {index} produced invalid statistics.")
    return int(index), info


_WORKER_CONTEXT = None


def _initialize_statistics_worker(context):
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _statistics_worker(index):
    context = _WORKER_CONTEXT
    return processFrame(
        index,
        context["input_den"],
        context["times"][index],
        context["angles"][index],
        context["currents"][index],
        context["target"],
        context["dimx"],
        context["dimy"],
    )


def createInfoObject(args, frames, header):
    dimx, dimy, image_count = map(int, header["dimspec"])
    times = (
        frames["time"] - frames["time"].iloc[0]
    ).dt.total_seconds().to_numpy(dtype=np.float64)
    currents = frames["current"].to_numpy(dtype=np.float64)
    target = (
        float(currents[0])
        if args.target_current_value is None
        else args.target_current_value
    )

    context = {
        "input_den": args.inputDen,
        "times": times,
        "angles": frames["s_rot"].to_numpy(dtype=np.float64),
        "currents": currents,
        "target": target,
        "dimx": dimx,
        "dimy": dimy,
    }
    info = np.empty((10, image_count), dtype=np.float64)
    requested = multiprocessing.cpu_count() if args.j == -1 else args.j
    workers = min(max(1, requested), image_count)
    started = time.perf_counter()
    last_report = started

    log.info("Normalizing currents to %.6f mA.", target)

    if args.j == 0 or workers == 1:
        for index in range(image_count):
            _, values = processFrame(
                index, args.inputDen, times[index],
                context["angles"][index], currents[index], target,
                dimx, dimy,
            )
            info[:, index] = values
            now = time.perf_counter()
            if now - last_report >= 5:
                log.info(
                    "Statistics: %d/%d frames after %s.",
                    index + 1, image_count, format_elapsed(now - started),
                )
                last_report = now
    else:
        log.info("Starting %d statistics worker processes.", workers)
        with multiprocessing.Pool(
            processes=workers,
            initializer=_initialize_statistics_worker,
            initargs=(context,),
        ) as pool:
            results = pool.imap_unordered(
                _statistics_worker, range(image_count), chunksize=1
            )
            completed = 0
            while completed < image_count:
                try:
                    index, values = results.next(timeout=5.0)
                except multiprocessing.TimeoutError:
                    log.info(
                        "Statistics: %d/%d frames after %s.",
                        completed, image_count,
                        format_elapsed(time.perf_counter() - started),
                    )
                    continue

                info[:, index] = values
                completed += 1
                now = time.perf_counter()
                if now - last_report >= 5:
                    log.info(
                        "Statistics: %d/%d frames after %s.",
                        completed, image_count, format_elapsed(now - started),
                    )
                    last_report = now

    log.info(
        "Statistics completed for %d frames in %s.",
        image_count, format_elapsed(time.perf_counter() - started),
    )
    return info


def validate_series(signal, axis):
    signal = np.asarray(signal, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    if signal.ndim != 1 or axis.ndim != 1 or signal.shape != axis.shape:
        raise ValueError("Signal and sampling axis must be matching 1D arrays.")
    if signal.size == 0:
        raise ValueError("Signal must not be empty.")
    if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(axis)):
        raise ValueError("Signal and sampling axis must be finite.")
    if np.any(np.diff(axis) < 0):
        raise ValueError("Sampling axis must be nondecreasing.")
    return signal, axis


def normalized_axis(signal, axis):
    signal, axis = validate_series(signal, axis)
    if signal.size < 2 or axis[-1] <= axis[0]:
        raise ValueError("Fitting requires at least two distinct sample positions.")
    return signal, (axis - axis[0]) / (axis[-1] - axis[0])


def remove_trend(signal, trend):
    """Remove a trend while retaining the original arithmetic mean."""
    result = signal - trend
    result += np.mean(signal) - np.mean(result)
    return result


def linear_detrend_constrained(signal, sampling_axis):
    signal, normalized = normalized_axis(signal, sampling_axis)
    trend = signal[0] + (signal[-1] - signal[0]) * normalized
    return remove_trend(signal, trend), trend


def quadratic_detrend_constrained(signal, sampling_axis):
    """Stable one-parameter quadratic constrained at both endpoints."""
    signal, normalized = normalized_axis(signal, sampling_axis)
    line = signal[0] + (signal[-1] - signal[0]) * normalized
    basis = normalized * (normalized - 1.0)
    denominator = float(np.dot(basis, basis))
    coefficient = (
        float(np.dot(basis, signal - line)) / denominator
        if denominator > 0 else 0.0
    )
    trend = line + coefficient * basis
    return remove_trend(signal, trend), trend


def exponential_detrend_constrained(signal, sampling_axis):
    """Fit an endpoint-constrained exponential with a stable linear limit."""
    signal, normalized = normalized_axis(signal, sampling_axis)
    start, end = signal[0], signal[-1]

    def model(position, rate):
        if abs(rate) < 1e-6:
            # First-order expansion of expm1(-rate*x)/expm1(-rate).
            fraction = position + 0.5 * rate * position * (1.0 - position)
        else:
            fraction = np.expm1(-rate * position) / np.expm1(-rate)
        return start + (end - start) * fraction

    if signal.size < 3 or start == end:
        trend = start + (end - start) * normalized
    else:
        parameters, _ = curve_fit(
            model, normalized, signal,
            p0=[1.0], bounds=([-100.0], [100.0]), maxfev=10000,
        )
        trend = model(normalized, parameters[0])

    return remove_trend(signal, trend), trend


def segment_slices(time_array, gap=1.0):
    time_array = np.asarray(time_array, dtype=np.float64)
    if time_array.ndim != 1 or time_array.size == 0:
        raise ValueError("Segment times must be a nonempty 1D array.")
    if not np.all(np.isfinite(time_array)):
        raise ValueError("Segment times must be finite.")
    if np.any(np.diff(time_array) < 0):
        raise ValueError("Segment times must be nondecreasing.")

    starts = np.r_[0, np.flatnonzero(np.diff(time_array) > gap) + 1]
    ends = np.r_[starts[1:], len(time_array)]
    return [slice(int(start), int(end)) for start, end in zip(starts, ends)]


def align_segments(y_signal, time_array, gap=1.0, verbose=False):
    signal, times = validate_series(y_signal, time_array)
    aligned = signal.copy()
    previous = None

    for index, segment in enumerate(segment_slices(times, gap)):
        if previous is not None:
            offset = previous - aligned[segment.start]
            aligned[segment] += offset
            if verbose:
                log.info("Segment %d alignment offset: %.6f.", index + 1, offset)
        previous = aligned[segment.stop - 1]

    return aligned


def denoise_chambolle_segment(
    y_signal, time_array, weight=100, gap=1.0, verbose=False,
):
    signal, times = validate_series(y_signal, time_array)
    result = signal.copy()

    for index, segment in enumerate(segment_slices(times, gap)):
        values = signal[segment]
        if verbose:
            log.info("TV segment %d: %d samples.", index + 1, len(values))
        if len(values) > 1:
            result[segment] = denoise_tv_chambolle(values, weight=weight)

    return result


def smooth_intensity(
    x_angle, y_intensity, sigma_degrees,
    time_array=None, time_gap=1.0, verbose=False,
):
    """Gaussian smoothing using mean angular/frame-index spacing per segment.

    As in the original implementation, this approximates uniform sampling.
    Constant-angle segments are left unchanged with a warning.
    """
    signal, angles = validate_series(y_intensity, x_angle)
    if not np.isfinite(sigma_degrees) or sigma_degrees <= 0:
        raise ValueError("Smoothing sigma must be finite and positive.")

    if time_array is None:
        segments = [slice(0, len(signal))]
    else:
        _, times = validate_series(signal, time_array)
        segments = segment_slices(times, time_gap)

    result = signal.copy()
    for index, segment in enumerate(segments):
        positions = angles[segment]
        values = signal[segment]
        if len(values) < 2:
            continue

        differences = np.diff(positions)
        spacing = float(np.mean(differences))
        if spacing <= 0:
            log.warning(
                "Skipping smoothing of constant-angle segment %d.",
                index + 1,
            )
            continue

        if not np.allclose(differences, spacing, rtol=0.05, atol=1e-8):
            log.warning(
                "Segment %d is not uniformly sampled; Gaussian smoothing "
                "uses mean spacing %.6g.",
                index + 1, spacing,
            )

        sigma = sigma_degrees / spacing
        if not np.isfinite(sigma):
            raise ValueError("Smoothing sigma in samples is not finite.")

        kernel_size = max(3, int(6 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        positions_kernel = np.linspace(-3 * sigma, 3 * sigma, kernel_size)
        kernel = np.exp(-0.5 * (positions_kernel / sigma) ** 2)
        kernel /= np.sum(kernel)

        if verbose:
            log.info(
                "Smoothing segment %d: spacing=%.6g, sigma=%.6g samples, "
                "kernel=%d.",
                index + 1, spacing, sigma, kernel_size,
            )

        result[segment] = convolve1d(values, kernel, mode="reflect")

    return result


def effective_trig_components(K, enforce_even_K=True):
    if K < 1:
        raise ValueError("Trigonometric K must be positive.")
    if enforce_even_K:
        return max(2, K + K % 2)
    return K


def fit_trig_basis(
    time_axis, signal, K, detrend=False, enforce_even_K=True,
):
    signal, normalized = normalized_axis(signal, time_axis)
    original_K = K
    K = effective_trig_components(K, enforce_even_K)
    if K != original_K:
        log.info("Adjusted trigonometric K from %d to %d.", original_K, K)

    if detrend:
        signal, _ = linear_detrend_constrained(signal, time_axis)

    design = np.ones((len(signal), 2 * K + 1), dtype=np.float64)
    for harmonic in range(1, K + 1):
        design[:, 2 * harmonic - 1] = np.cos(
            2 * np.pi * harmonic * normalized
        )
        design[:, 2 * harmonic] = np.sin(
            2 * np.pi * harmonic * normalized
        )

    coefficients, _, rank, _ = np.linalg.lstsq(
        design, signal, rcond=None
    )
    if rank < design.shape[1]:
        log.warning(
            "Trigonometric fit is rank deficient: rank %d, %d coefficients.",
            rank, design.shape[1],
        )

    fitted = design @ coefficients
    return fitted, (
        coefficients[0],
        coefficients[1::2],
        coefficients[2::2],
    )


def unwrap_make_monotone(info_angle, wrap_increase=360, threshold=None):
    """Preserve the original increasing-angle unwrapping convention."""
    angles = np.asarray(info_angle, dtype=np.float64)
    if angles.ndim != 1 or angles.size == 0:
        raise ValueError("Angles must be a nonempty 1D array.")
    if not np.all(np.isfinite(angles)):
        raise ValueError("Angles must be finite.")

    threshold = 0.5 * wrap_increase if threshold is None else threshold
    corrected = np.empty_like(angles)
    corrected[0] = angles[0]
    offset = 0.0

    for index in range(1, len(angles)):
        if angles[index] - angles[index - 1] < -threshold:
            offset += wrap_increase
        corrected[index] = max(corrected[index - 1], angles[index] + offset)

    return corrected


def plot_series(pdf, times, title, series):
    """Save a diagnostic page; do nothing when no PDF was requested."""
    if pdf is None:
        return

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(15, 5))
    try:
        axis.set_title(title)
        for label, values in series:
            axis.scatter(times, values, s=5, alpha=0.7, label=label)
        axis.set_xlabel("Time [s]")
        axis.set_ylabel("Mean intensity")
        axis.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axis.legend()
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
    finally:
        plt.close(figure)


def plotInfoOverview(
    info, beamCurrentData, zerotime, insertion_events=None,
    mainLabel=None, pdf=None,
):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 4, figsize=(24, 15))
    try:
        if mainLabel:
            figure.suptitle(mainLabel)

        current_times = (
            beamCurrentData["time"] - zerotime
        ).dt.total_seconds().to_numpy()
        currents = beamCurrentData["current"].to_numpy()

        axes[0, 0].plot(
            current_times, currents, color="#DDCC77",
            label="Recorded current",
        )
        axes[0, 0].scatter(
            info[0], info[2], s=5, color="#332288",
            label="Frame-associated current",
        )
        for event in insertion_events or []:
            for key, color, label in (
                ("start_index", "green", "Insertion start"),
                ("end_index", "red", "Insertion end"),
            ):
                index = event[key]
                axes[0, 0].scatter(
                    current_times[index], currents[index],
                    color=color, marker="D", s=35, label=label,
                )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axes[0, 0].legend(unique.values(), unique.keys())
        axes[0, 0].set_title("Current vs time")
        axes[0, 0].set_xlabel("Time [s]")
        axes[0, 0].set_ylabel("Current [mA]")

        axes[0, 1].scatter(current_times, currents, s=3, color="red")
        intensity_axis = axes[0, 1].twinx()
        intensity_axis.plot(info[0], info[3], color="blue")
        intensity_axis.set_ylabel("Original mean intensity")
        axes[0, 1].set_title("Recorded current and frame intensity")
        axes[0, 1].set_xlabel("Time [s]")
        axes[0, 1].set_ylabel("Current [mA]")

        plots = [
            (axes[0, 2], info[0], info[1], "Angle vs time"),
            (axes[0, 3], info[6], info[5], "Corrected mean vs median"),
            (axes[1, 0], info[0], info[3], "Original mean vs time"),
            (axes[1, 1], info[0], info[4], "Original median vs time"),
            (axes[1, 2], info[0], info[5], "Current-normalized mean vs time"),
            (axes[1, 3], info[0], info[6], "Current-normalized median vs time"),
            (axes[2, 0], info[0], info[7], "Bottom 90% mean vs time"),
            (axes[2, 1], info[0], info[8], "Bottom 90% median vs time"),
            (axes[2, 2], info[2], info[5], "Corrected mean vs current"),
            (axes[2, 3], info[2], info[6], "Corrected median vs current"),
        ]
        for axis, x_values, y_values, title in plots:
            axis.scatter(x_values, y_values, s=4)
            axis.set_title(title)

        figure.tight_layout()
        if pdf is None:
            plt.show()
        else:
            pdf.savefig(figure, bbox_inches="tight")
    finally:
        plt.close(figure)


def process_mean(args, params, info, insertion_count, pdf=None):
    times = info[0] - info[0, 0]
    angles = (
        unwrap_make_monotone(info[1])
        if args.type == "IMG"
        else np.arange(info.shape[1], dtype=np.float64)
    )

    original = info[5].copy()
    processed = original.copy()
    preprocessing_series = [("Current-normalized mean", original)]

    segmented_processing = (
        args.tv_denoise
        or args.connect_gaps
        or args.remove_short_term_noise is not None
        or args.remove_long_term_trend is not None
    )
    if segmented_processing and len(times) > 1:
        differences = np.diff(times)
        if np.all(differences > args.time_gap):
            log.warning(
                "Every frame spacing exceeds --time-gap %.6g s. "
                "Denoising/smoothing segments will contain one frame each. "
                "Increase --time-gap if these frames belong to one acquisition.",
                args.time_gap,
            )

    if args.tv_denoise:
        processed = denoise_chambolle_segment(
            processed, times, weight=100,
            gap=args.time_gap, verbose=args.verbose,
        )
        preprocessing_series.append(("TV-denoised mean", processed.copy()))

    if args.connect_gaps:
        processed = align_segments(
            processed, times, gap=args.time_gap, verbose=args.verbose,
        )
        preprocessing_series.append(("Gap-aligned mean", processed.copy()))

    plot_series(
        pdf, times, "Preprocessing of current-normalized means",
        preprocessing_series,
    )

    if (
        args.remove_short_term_noise is not None
        or args.remove_long_term_trend is not None
    ):
        baseline = processed.copy()
        trend = np.zeros_like(processed)

        if args.remove_long_term_trend is not None:
            trend = smooth_intensity(
                angles, baseline, args.remove_long_term_trend,
                time_array=times, time_gap=args.time_gap,
                verbose=args.verbose,
            )

        if args.remove_short_term_noise is not None:
            processed = smooth_intensity(
                angles, baseline, args.remove_short_term_noise,
                time_array=times, time_gap=args.time_gap,
                verbose=args.verbose,
            )

        series = [
            ("Original current-normalized mean", original),
            ("Preprocessed mean", baseline),
            ("Short-term smoothed mean", processed.copy()),
        ]
        if args.remove_long_term_trend is not None:
            series.append(("Long-term trend", trend.copy()))
        plot_series(pdf, times, "Smoothed mean data", series)

        processed = processed - (trend - np.mean(trend))
        plot_series(
            pdf, times, "Trend-adjusted mean data",
            [("Preprocessed mean", baseline), ("Adjusted mean", processed)],
        )

    before_detrend = processed.copy()
    fitted_trend = None
    detrend_title = None

    if args.detrend_data:
        processed, fitted_trend = linear_detrend_constrained(processed, times)
        detrend_title = "Linear detrending"
    elif args.detrend_quadratic:
        axis = angles if args.connect_gaps else times
        processed, fitted_trend = quadratic_detrend_constrained(processed, axis)
        detrend_title = "Quadratic detrending"
    elif args.detrend_exponential:
        axis = angles if args.connect_gaps else times
        processed, fitted_trend = exponential_detrend_constrained(processed, axis)
        detrend_title = "Exponential detrending"

    if fitted_trend is not None:
        plot_series(
            pdf, times, detrend_title,
            [
                ("Mean before detrending", before_detrend),
                ("Fitted trend", fitted_trend),
                ("Detrended mean", processed),
            ],
        )

    if args.perform_trigonometric_fit is not None:
        K = args.perform_trigonometric_fit
        if K == 0:
            configured_count = params.get("insertion_event_count")
            count = (
                int(configured_count)
                if configured_count is not None
                else insertion_count
            )
            if count <= 0:
                raise ValueError(
                    "Cannot estimate trigonometric K: no insertion count "
                    "available. Set insertion_event_count in params or "
                    "specify a positive --perform-trigonometric-fit."
                )
            K = max(2, count // 2)

        effective_K = effective_trig_components(K)
        before_fit = processed.copy()
        processed, _ = fit_trig_basis(
            times, processed, K,
            detrend=args.detrend_trigonometric_fit,
            enforce_even_K=True,
        )
        plot_series(
            pdf, times, f"Trigonometric fit: {effective_K} harmonics",
            [("Mean before fit", before_fit), ("Fitted mean", processed)],
        )

    if not np.all(np.isfinite(processed)):
        raise ValueError("Final corrected means contain nonfinite values.")

    return processed


def correctFrame(
    index, args, frame_mean, frame_corrected_mean,
    lck=None, output_dtype=np.dtype("<f4"),
):
    if not np.isfinite(frame_mean) or frame_mean == 0:
        raise ValueError(f"Invalid original mean for frame {index}: {frame_mean}")
    if not np.isfinite(frame_corrected_mean):
        raise ValueError(f"Invalid corrected mean for frame {index}.")

    with lck if lck is not None else nullcontext():
        frame = DEN.getFrame(args.inputDen, int(index))

    factor = frame_corrected_mean / frame_mean
    # Calculate in float64, then explicitly match the output DEN dtype.
    corrected = np.asarray(frame, dtype=np.float64) * factor
    if not np.all(np.isfinite(corrected)):
        raise ValueError(f"Correction produced nonfinite pixels in frame {index}.")

    with np.errstate(over="raise", invalid="raise"):
        corrected = corrected.astype(output_dtype)
    if not np.all(np.isfinite(corrected)):
        raise ValueError(f"Output dtype overflow in frame {index}.")

    with lck if lck is not None else nullcontext():
        DEN.writeFrame(args.outputDen, int(index), corrected, force=True)

    if args.verbose:
        log.info(
            "Frame %d: %d pixels, correction factor %.8g.",
            index, corrected.size, factor,
        )
    return factor


def meanCorrect(args, info, header):
    dimx, dimy, image_count = map(int, header["dimspec"])
    if info.shape != (11, image_count):
        raise ValueError(
            f"Info shape {info.shape}; expected {(11, image_count)}."
        )
    if not np.all(np.isfinite(info)):
        raise ValueError("Info object contains nonfinite values.")
    if np.any(info[3] == 0):
        raise ValueError("Cannot correct frames with zero original mean.")

    # Preserve float64 input precision; other input types produce float32.
    source_dtype = np.dtype(header["type"])
    output_dtype = (
        np.dtype("<f8")
        if source_dtype.kind == "f" and source_dtype.itemsize == 8
        else np.dtype("<f4")
    )
    log.info("Writing corrected DEN with dtype %s.", output_dtype)

    if np.any(info[10] <= 0):
        log.warning("Some final frame means are nonpositive.")

    DEN.writeEmptyDEN(
        args.outputDen,
        (dimx, dimy, image_count),
        elementtype=output_dtype,
        force=args.force,
    )

    started = time.perf_counter()
    last_report = started
    for index in range(image_count):
        correctFrame(
            index, args, info[3, index], info[10, index],
            output_dtype=output_dtype,
        )
        now = time.perf_counter()
        if now - last_report >= 5:
            log.info(
                "Correction: %d/%d frames after %s.",
                index + 1, image_count, format_elapsed(now - started),
            )
            last_report = now

    log.info(
        "Corrected %d frames in %s.",
        image_count, format_elapsed(time.perf_counter() - started),
    )
    return info


def run(args):
    header = validate_paths(args)
    params = read_params(args)

    try:
        frames, current, offset = load_metadata(
            args, params, int(header["dimspec"][2])
        )
    except Exception as error:
        if args.fix_corrupted_h5 and args.h5file is not None:
            log.exception(
                "H5 metadata loading failed; --fix-corrupted-h5 requested."
            )
            shutil.copy2(args.inputDen, args.outputDen)
            log.warning(
                "Copied %s to %s without correction. "
                "No statistics or diagnostic PDF were generated.",
                args.inputDen, args.outputDen,
            )
            return 0
        raise MetadataError("Failed to load acquisition metadata.") from error

    beam, insertion_events, insertion_count = select_current_and_events(
        args, params, frames, current, offset,
    )
    info = createInfoObject(args, frames, header)

    if args.savefig is None:
        final_mean = process_mean(args, params, info, insertion_count)
    else:
        # Select the noninteractive backend before importing pyplot.
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages

        # The context manager closes the PDF on success or failure.
        with PdfPages(args.savefig) as pdf:
            source_label = "H5" if args.h5file is not None else "NanoCT log"
            plotInfoOverview(
                info,
                beam,
                frames["time"].iloc[0],
                insertion_events=insertion_events,
                mainLabel=(
                    f"{source_label} scan overview; "
                    f"current time offset={offset:.6f} s"
                ),
                pdf=pdf,
            )
            final_mean = process_mean(
                args, params, info, insertion_count, pdf=pdf
            )

    info = np.vstack([info, final_mean])
    meanCorrect(args, info, header)

    if args.saveinfo is not None:
        DEN.storeNdarrayAsDEN(args.saveinfo, info, force=args.force)
        log.info("Saved correction info to %s.", args.saveinfo)

    return 0


def main(argv=None):
    """Console-script entry point; return an exit status."""
    command_arguments = list(sys.argv[1:] if argv is None else argv)
    started = time.perf_counter()

    print("START currentCorrect %s" % " ".join(command_arguments), flush=True)
    print(
        "Date: %s" % datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        flush=True,
    )

    try:
        parser = build_parser()
        args = parser.parse_args(command_arguments)
        validate_arguments(parser, args)
        return run(args)
    except KeyboardInterrupt:
        log.error("currentCorrect interrupted by the user.")
        return 130
    except Exception:
        log.exception("Error in currentCorrect.")
        return 1
    finally:
        print(
            "END currentCorrect in %s"
            % format_elapsed(time.perf_counter() - started),
            flush=True,
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
