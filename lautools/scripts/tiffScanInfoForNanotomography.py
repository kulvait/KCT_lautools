#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-frame statistics for nanotomography acquisitions.

Metadata and TIFF matching come from lautools.NANOCT.scanDataset.
No H5 file, external beam-current interpolation, or time-offset
estimation is used.

@author: Vojtěch Kulvait
@year: 2026
@license: GPL3
"""
import argparse
import os
import io
from contextlib import redirect_stdout, redirect_stderr
import sys
import logging
from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer as timer

import matplotlib
import numpy as np
from PIL import Image

from denpy import DEN
from lautools import NANOCT

# Create a logger specific to this module
log = logging.getLogger(__name__)
log.setLevel(logging.INFO) # Set the logging level to INFO
# Create a console handler and set its level to INFO
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s : %(message)s', datefmt='%d.%m.%y %H:%M:%S')
ch.setFormatter(formatter)
# Add the handler to the logger
log.addHandler(ch)
log.propagate = False # Prevent log messages from being propagated to the root logger


def buildParser():
    parser = argparse.ArgumentParser(
        description="Compute image statistics from a nanoCT LogScan.log."
    )
    parser.add_argument(
        "logFile",
        help="Path to LogScan.log, replacing the original H5 input.",
    )
    parser.add_argument(
        "outputInfoDen",
        help="Output DEN file containing a (10, number_of_frames) info array.",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Raw TIFF directory; defaults to the directory containing logFile.",
    )
    parser.add_argument(
        "--type",
        default="IMG",
        choices=["IMG", "REF", "DAR"],
        help="IMG: projections; REF: flat fields; DAR: dark fields.",
    )
    parser.add_argument(
        "--ref-is-dar",
        action="store_true",
        help="With --type DAR, use this measurement's reference images as darks.",
    )
    parser.add_argument(
        "--input-den",
        default=None,
        help="Read pixel data from this DEN file instead of matched TIFF files.",
    )
    parser.add_argument(
        "--target-current-value",
        type=float,
        default=None,
        help="Normalization current in mA; defaults to the first selected frame.",
    )
    parser.add_argument(
        "--no-current-correction",
        action="store_true",
        help="Disable current normalization. Always disabled for DAR.",
    )
    parser.add_argument(
        "--lower-quantile",
        type=float,
        default=0.9,
        help="Fraction of lowest-valued pixels used for statistics (0 < q <= 1).",
    )
    parser.add_argument(
        "--no-check-block-order",
        action="store_true",
        help="Disable NANOCT's check of file-time ordering versus counters.",
    )
    parser.add_argument(
        "--read-info",
        action="store_true",
        help="Read the existing output info DEN instead of recomputing it.",
    )
    parser.add_argument(
        "--savefig",
        default=None,
        help="Save the overview to a PDF instead of displaying it.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "-j",
        type=int,
        default=-1,
        help="Thread count: 0 for sequential processing, -1 for automatic.",
    )
    return parser


def loadFrameTable(ARG):
    rawDir = (
        os.path.realpath(ARG.raw_dir)
        if ARG.raw_dir is not None
        else os.path.dirname(os.path.realpath(ARG.logFile))
    )

    scanData = NANOCT.scanDataset(
        ARG.logFile,
        imgDir=rawDir,
        drop_missing=True,
        check_block_order=not ARG.no_check_block_order,
    )

    if scanData.empty:
        raise ValueError(f"No matched frames found in {ARG.logFile!r}.")

    image_key = {"IMG": 0, "REF": 1, "DAR": 2}[ARG.type]
    if ARG.ref_is_dar:
        image_key = 1

    df = scanData.loc[scanData["image_key"] == image_key].copy()
    if df.empty:
        return df  # No frames of the requested type; return an empty DataFrame.

    if ARG.ref_is_dar:
        df["image_key"] = 2

    df = df.reset_index(drop=True)
    df["frame_ind"] = np.arange(len(df))

    # Preserve acquisition order rather than sorting metadata independently
    # of the frame ordering.
    if df["time"].isna().any():
        raise ValueError("Selected frames contain missing timestamps.")

    if not df["time"].is_monotonic_increasing:
        raise ValueError("Selected frame timestamps are not nondecreasing.")

    return df


def getFrame(ARG, df, index):
    if ARG.input_den is not None:
        return np.asarray(
            DEN.getFrame(ARG.input_den, index, dtype=np.float32),
            dtype=np.float32,
        )

    # NANOCT returns absolute paths: do not strip their leading slash or
    # prepend rawDir again.
    filename = os.fsdecode(df["image_file"].iloc[index])
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Frame file does not exist: {filename}")

    with Image.open(filename) as image:
        return np.array(image, dtype=np.float32)


def correctionFactors(ARG, df):
    if ARG.type == "DAR" or ARG.no_current_correction:
        return np.ones(len(df), dtype=np.float64)

    currents = df["current"].to_numpy(dtype=np.float64)
    invalid = ~np.isfinite(currents) | (currents <= 0)
    if invalid.any():
        indices = np.flatnonzero(invalid).tolist()
        raise ValueError(
            f"Current normalization requires finite positive currents; "
            f"invalid frame indices: {indices}. "
            "Use --no-current-correction to retain raw statistics."
        )

    target = (
        currents[0]
        if ARG.target_current_value is None
        else ARG.target_current_value
    )
    if not np.isfinite(target) or target <= 0:
        raise ValueError("--target-current-value must be finite and positive.")

    return target / currents


def processFrame(i, ARG, df, shape, elapsed, factors):
    image = getFrame(ARG, df, i)
    if image.shape != shape:
        raise ValueError(
            f"Frame {i} shape {image.shape} differs from expected {shape}."
        )

    pixels = image.ravel()
    if not np.isfinite(pixels).all():
        raise ValueError(f"Frame {i} contains NaN or infinite pixel values.")

    mean = np.mean(pixels, dtype=np.float64)
    median = float(np.median(pixels))

    # Support q=1 and very small images without an invalid partition index.
    k = max(1, int(ARG.lower_quantile * pixels.size))
    lower = np.partition(pixels, k - 1)[:k]

    return np.array(
        [
            elapsed[i],                              # 0: relative time [s]
            float(df["s_rot"].iloc[i]),               # 1: rotation [degrees]
            float(df["current"].iloc[i]),             # 2: logged current [mA]
            mean,                                    # 3: raw mean
            median,                                  # 4: raw median
            mean * factors[i],                       # 5: corrected mean
            median * factors[i],                     # 6: corrected median
            np.mean(lower, dtype=np.float64),         # 7: lower-fraction mean
            float(np.median(lower)),                 # 8: lower-fraction median
            ARG.lower_quantile,                      # 9: fraction
        ],
        dtype=np.float64,
    )


def createInfoObject(ARG, df):
    imageCount = len(df)

    if ARG.input_den is not None:
        if not os.path.isfile(ARG.input_den):
            raise FileNotFoundError(ARG.input_den)

        header = DEN.readHeader(ARG.input_den)
        dims = header["dimspec"]
        if len(dims) != 3:
            raise ValueError(
                f"Input DEN must have three dimensions, got {len(dims)}."
            )
        if dims[2] != imageCount:
            raise ValueError(
                f"Input DEN contains {dims[2]} frames, but the selected "
                f"matched dataset contains {imageCount} frames."
            )
        shape = (int(dims[1]), int(dims[0]))
    else:
        first = getFrame(ARG, df, 0)
        if first.ndim != 2 or first.size == 0:
            raise ValueError(
                f"Expected a nonempty 2D grayscale frame, got {first.shape}."
            )
        shape = first.shape

    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"Invalid frame shape: {shape}.")

    elapsed = (
        (df["time"] - df["time"].iloc[0])
        .dt.total_seconds()
        .to_numpy(dtype=np.float64)
    )
    factors = correctionFactors(ARG, df)
    info = np.empty((10, imageCount), dtype=np.float64)

    def worker(index):
        return processFrame(index, ARG, df, shape, elapsed, factors)

    started = timer()

    if ARG.j == 0:
        for i in range(imageCount):
            info[:, i] = worker(i)
    else:
        threads = os.cpu_count() or 1 if ARG.j == -1 else ARG.j
        threads = min(threads, imageCount)

        if ARG.verbose:
            print(f"Processing {imageCount} frames with {threads} threads.")

        with ThreadPoolExecutor(max_workers=threads) as executor:
            # map preserves frame order and propagates worker exceptions.
            for i, result in enumerate(executor.map(worker, range(imageCount))):
                info[:, i] = result
                if ARG.verbose and (
                    (i + 1) % 100 == 0 or i + 1 == imageCount
                ):
                    print(
                        f"{i + 1}/{imageCount} frames processed "
                        f"after {timer() - started:.1f}s."
                    )

    if ARG.verbose:
        print(f"Statistics computed in {timer() - started:.1f}s.")

    return info


def readInfoObject(ARG, frameCount):
    if not os.path.isfile(ARG.outputInfoDen):
        raise FileNotFoundError(
            f"--read-info requested, but {ARG.outputInfoDen!r} does not exist."
        )

    info = np.asarray(DEN.getNumpyArray(ARG.outputInfoDen))

    # Compatibility with the original script's stacked info objects:
    # use the first, unshifted statistics object.
    if info.ndim == 3 and info.shape[0] > 0:
        info = info[0]

    if info.shape != (10, frameCount):
        raise ValueError(
            f"Info shape is {info.shape}; expected (10, {frameCount})."
        )

    return info


def plotInfoOverview(info, mainLabel, currentCorrection, pdf=None):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 4, figsize=(24, 15))
    figure.suptitle(mainLabel, fontsize=16)

    t = info[0]
    normalized = "Current-normalized" if currentCorrection else "Unscaled"

    axes[0, 0].scatter(t, info[2], s=5, color="#332288")
    axes[0, 0].set_title("Logged beam current vs time")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Current [mA]")

    current_axis = axes[0, 1]
    current_axis.scatter(
        t, info[2], s=5, color="green", label="Logged current"
    )
    current_axis.set_title("Logged current and raw mean vs time")
    current_axis.set_xlabel("Time [s]")
    current_axis.set_ylabel("Current [mA]")
    intensity_axis = current_axis.twinx()
    intensity_axis.plot(t, info[3], color="blue", label="Raw mean")
    intensity_axis.set_ylabel("Mean intensity")
    current_axis.legend(loc="upper left")
    intensity_axis.legend(loc="upper right")

    axes[0, 2].plot(t, info[1])
    axes[0, 2].set_title("Angle vs time")
    axes[0, 2].set_xlabel("Time [s]")
    axes[0, 2].set_ylabel("Angle [degrees]")

    axes[0, 3].scatter(info[6], info[5], s=5)
    axes[0, 3].set_title(f"{normalized} mean vs median")
    axes[0, 3].set_xlabel("Median intensity")
    axes[0, 3].set_ylabel("Mean intensity")

    q = 100 * info[9, 0]
    time_plots = [
        (axes[1, 0], 3, "Raw mean vs time"),
        (axes[1, 1], 4, "Raw median vs time"),
        (axes[1, 2], 5, f"{normalized} mean vs time"),
        (axes[1, 3], 6, f"{normalized} median vs time"),
        (axes[2, 0], 7, f"Raw mean of bottom {q:g}% vs time"),
        (axes[2, 1], 8, f"Raw median of bottom {q:g}% vs time"),
    ]
    for axis, row, title in time_plots:
        axis.scatter(t, info[row], s=5)
        axis.set_title(title)
        axis.set_xlabel("Time [s]")
        axis.set_ylabel("Intensity")

    for axis, row, statistic in [
        (axes[2, 2], 5, "mean"),
        (axes[2, 3], 6, "median"),
    ]:
        axis.scatter(info[2], info[row], s=5)
        axis.set_title(f"{normalized} {statistic} vs current")
        axis.set_xlabel("Logged current [mA]")
        axis.set_ylabel("Intensity")

    for axis in axes.flat:
        axis.grid(True, alpha=0.3)

    figure.tight_layout(rect=(0, 0, 1, 0.96))

    if pdf is not None:
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def main(argv=None):
    # Redirect stdout and stderr to capture argparse help messages
    parser = buildParser()
    try:
        _out = io.StringIO()
        _err = io.StringIO()
        with redirect_stdout(_out), redirect_stderr(_err):
            arg_list = sys.argv[1:] if argv is None else argv
            if not arg_list:
                arg_list = ["--help"]
            ARG = parser.parse_args(arg_list)
    except SystemExit as err:
        print("Program to produce Zarr file from raw TIFF structure. Usage:")
        sys.stderr.write(_err.getvalue())
        sys.stdout.write(_out.getvalue())
        return err.code
    print("START tiffScanInfoForNanotomography %s" % " ".join(sys.argv[1:]))

    if not np.isfinite(ARG.lower_quantile) or not 0 < ARG.lower_quantile <= 1:
        parser.error("--lower-quantile must satisfy 0 < q <= 1.")
    if ARG.j < -1:
        parser.error("-j must be -1, 0, or a positive integer.")
    if ARG.ref_is_dar and ARG.type != "DAR":
        parser.error("--ref-is-dar requires --type DAR.")

    currentCorrection = ARG.type != "DAR" and not ARG.no_current_correction
    if ARG.target_current_value is not None and not currentCorrection:
        parser.error(
            "--target-current-value cannot be used when current "
            "normalization is disabled."
        )

    if ARG.savefig:
        # Select the headless backend before importing pyplot.
        matplotlib.use("Agg")

    df = loadFrameTable(ARG)
    if df.empty:
        log.warning(f"No frames of type {ARG.type} found in {ARG.logFile!r}, will not produce outputs and exit cleanly.")
        print("END tiffScanInfoForNanotomography, no frames of requested %s found." % ARG.type)
        sys.exit(0)

    if ARG.verbose:
        duration = (
            df["time"].iloc[-1] - df["time"].iloc[0]
        ).total_seconds()
        print(
            f"Processing {ARG.type}: {len(df)} matched frames "
            f"over {duration:.3f}s from {ARG.logFile}."
        )

    if ARG.read_info:
        info = readInfoObject(ARG, len(df))
    else:
        if os.path.exists(ARG.outputInfoDen) and not ARG.force:
            raise FileExistsError(
                f"{ARG.outputInfoDen!r} already exists; use --force."
            )
        info = createInfoObject(ARG, df)
        DEN.storeNdarrayAsDEN(ARG.outputInfoDen, info, force=ARG.force)

    title = f"Nanotomography {ARG.type}: {os.path.basename(ARG.logFile)}"
    if ARG.ref_is_dar:
        title += " — references used as dark fields"
    if ARG.read_info:
        title += " — existing statistics"
        # Labels describe the requested mode. Existing info should have been
        # computed with the same correction settings.

    if ARG.savefig:
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(ARG.savefig) as pdf:
            plotInfoOverview(info, title, currentCorrection, pdf=pdf)
    else:
        plotInfoOverview(info, title, currentCorrection)
    print("END tiffScanInfoForNanotomography")
    sys.exit(0)

if __name__ == "__main__":
    raise SystemExit(main())
