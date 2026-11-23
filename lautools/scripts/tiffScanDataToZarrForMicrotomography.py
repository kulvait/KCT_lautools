#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Feb 2026, rewritten 11 2026 on denpy.TIFF

Zarr store or DEN files of a PETRA III microtomography scan from its raw TIFF files and NeXus HDF5
metadata. Replaces syscripts rawTiffStructureToZarr.py and rawTiffStructureToDen.py:

	rawTiffStructureToZarr.py h5 out.zar --zip ...  ->  tiffScanDataToZarrForMicrotomography h5 out.zar --zip ...
	rawTiffStructureToDen.py h5 DIR ...             ->  tiffScanDataToZarrForMicrotomography h5 DIR --den --dtype float32 ...

Zarr: arrays dar, ref, img, the group params with the scan description (PETRA.ScanDescription, the
canonical form of the HDF5 entry) and the scan table; the root attributes are the experiment info.
As rawTiffStructureToZarr.py, except that single strings of older HDF5 files are strings, not
one-element arrays, and image codecs get no additional zstd compressor.
DEN: dar.den, ref.den, img.den in the directory (e.g. the working directory; other files there are
not touched) in the dtype of the TIFF files (--dtype float32 as rawTiffStructureToDen.py), the
metadata in params.json.

--spec-extension adds processing to the conversion: virtual stacks and arrays computed with lambdas
or reductions (see the TIFF stack specification, docs/TIFFSTACK_SPEC.md of denpy), e.g. a dark median
and dark corrected projections:

	{"virtualstacks": {"dar_median": {"stack": "dar", "reduce": "median", "dtype": "uint16"}},
	 "arrays": {"dar_median": {"stack": "dar_median"},
	            "img": {"stack": "img", "dtype": "uint16", "clip": true, "args": ["dar_median"],
	                    "lambda": "lambda x, d: x.astype('int32') - d"}}}

This command is the example of how to use the API:

	1. metadata -> scan dataset     scanDescription = PETRA.ScanDescriptionFactory.fromH5(h5)
	                                df = scanDescription.scanDataset()  (time, image_key, image_file, ...)
	2. scan dataset -> spec         spec = TOMOGRAPHY.scanTiffStackConversionSpecs(df, output, rawDir=...)
	                                spec = TIFF.extend_spec(spec, extension)  (optional)
	3. spec -> arrays               report = TIFF.execute(TIFF.validate(spec))
	4. metadata -> the same store   scanDescription.writeZarrGroup(group, scanTable=df)  (mode "a")

Steps 1 and 4 are specific to the source of the metadata (NANOCT.scanDataset for nanotomography);
steps 2 and 3 are the same for every scan. --write-spec saves the spec, which can be run again with
convertTiffStack.

@author: Vojtěch Kulvait
@license: GNU General Public License v3.0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import zarr

from lautools import PETRA
from denpy import TIFF

from lautools import TOMOGRAPHY

COMPRESSIONS = ['none', 'zstd', 'lz4', 'gzip', 'blosc', 'blosc-blosclz', 'blosc-lz4', 'blosc-lz4hc', 'blosc-snappy',
				'blosc-zlib', 'blosc-zstd', 'avif', 'jpeg2k', 'htj2k', 'jpegxl', 'jpegxr', 'sz3', 'zfp']

# Arguments of the image codecs for 12 bit grayscale detector frames; one codec thread each, the
# frames are compressed in parallel by the writing threads
CODEC_ARGUMENTS = {
	"jpeg2k": {"bitspersample": 12, "colorspace": "GRAY", "mct": False, "numthreads": 1},
	"htj2k": {"rgb": False},
	"avif": {"bitspersample": 12, "pixelformat": "YUV400 ", "numthreads": 1},
	"jpegxl": {"bitspersample": 12, "photometric": "GRAY", "numthreads": 1},
	"jpegxr": {"photometric": "GRAY", "hasalpha": False},
}


def build_parser():
	parser = argparse.ArgumentParser(
		prog="tiffScanDataToZarrForMicrotomography",
		description="Zarr store or DEN files of a PETRA III microtomography scan from its raw TIFF files and "
		"NeXus HDF5 metadata: arrays dar, ref, img and group params. Replaces rawTiffStructureToZarr.py and "
		"(with --den --dtype float32) rawTiffStructureToDen.py.")
	parser.add_argument("inputh5")
	parser.add_argument("outputZarr", help="Output zarr store; with --den the directory of dar.den, ref.den, img.den.")
	parser.add_argument("--raw-dir", default=None, type=str,
						help="Directory of the TIFF files. [defaults to the directory of inputh5]")
	parser.add_argument("--fix-corrupted-h5", action="store_true",
						help="If the HDF5 file lists no projections, img are all TIFF files of the raw directory that are not darks or refs.")
	parser.add_argument("--fill-missing", type=float, default=None, metavar="VALUE",
						help="Frames listed in the HDF5 file whose TIFF file is missing are written as VALUE (null frames, "
						"listed in the report). [defaults to an error]")
	parser.add_argument("--force", action="store_true", help="Replace the output if it exists.")
	parser.add_argument("--compression", type=str, choices=COMPRESSIONS, default="blosc-zstd",
						help="Compression type (default: blosc-zstd).")
	parser.add_argument("--clevel", type=float, default=5, help="Compression level (default: 5).")
	parser.add_argument("-j", "--threads", default=-1, type=int, dest="j",
						help="Number of threads. [defaults to -1: a quarter of the CPUs available to the process, at least "
						"%d and at most %d; 0 is one thread]" % (TIFF.DEFAULT_MIN_THREADS, TIFF.DEFAULT_MAX_THREADS))
	parser.add_argument("--zip", action="store_true", help="Zip store instead of a directory store.")
	parser.add_argument("--v2", action="store_true", help="Zarr v2 instead of v3.")
	parser.add_argument("--den", action="store_true",
						help="DEN files dar.den, ref.den, img.den in the directory outputZarr instead of zarr; the metadata "
						"go to params.json there. The directory may exist (e.g. the working directory), its other files are "
						"not touched; existing DEN files of the arrays need --force.")
	parser.add_argument("--dtype", default=None,
						help="dtype of the arrays dar, ref, img (e.g. float32 as rawTiffStructureToDen.py). [defaults to the "
						"dtype of the TIFF files; int32 for zfp, float32 for sz3, the dtypes these codecs support]")
	parser.add_argument("--spec-extension", action="append", default=[], metavar="EXTENSION.json",
						help="Extend the generated spec by virtual stacks and arrays (lambdas, reductions; null removes an "
						"array), see denpy TIFF.extend_spec; can be repeated.")
	parser.add_argument("--dry-run", action="store_true",
						help="Validate and print the arrays to write (and save --write-spec), write nothing.")
	parser.add_argument("--write-spec", default=None, help="Save the TIFF stack specification (JSON).")
	parser.add_argument("--report", default=None, help="Save the report of the conversion (JSON).")
	parser.add_argument("--check-headers", action="store_true", help="Compare shape and dtype of every TIFF before writing.")
	parser.add_argument("--verbose", action="store_true")
	return parser


# ---------------------------------------------------------------- metadata

def scan_for_tiff_files(directory, exclude_files):
	"""All TIFF files below directory (relative paths, sorted) that are not in exclude_files."""
	tiff_files = []
	for root, _, files in os.walk(directory):
		for file in files:
			if file.lower().endswith('.tiff') or file.lower().endswith('.tif'):
				relative_path = os.path.relpath(os.path.join(root, file), directory)
				if relative_path not in exclude_files:
					tiff_files.append(relative_path)
	tiff_files.sort()
	return pd.DataFrame({'image_file': tiff_files, 'image_key': 0})


def scan_dataset(scanDescription, raw_dir, fix_corrupted):
	"""Step 1: the frames of the scan in acquisition order."""
	df = scanDescription.scanDataset(includeCurrent=False)
	if df.empty:
		raise ValueError(f"No frames in {scanDescription.name}")
	if fix_corrupted and not (df["image_key"] == 0).any():
		print("HDF5 file lists no projections: img are the TIFF files of the raw directory that are not darks or refs.")
		listed = df.loc[df["image_key"].isin([1, 2])]
		found = scan_for_tiff_files(raw_dir, set(listed["image_file"]))
		df = pd.concat([df, found], ignore_index=True)
	return df


def write_params(output, scanDescription, df):
	"""Step 4: the group params (scan description and scan table) into the finished store."""
	from denpy import ZARZS
	store = ZARZS.ZipStore(output["path"], mode="a") if output["store"] == "zip" else output["path"]
	try:
		scanDescription.writeZarrGroup(zarr.open_group(store, mode="a"), scanTable=df)
	finally:
		if output["store"] == "zip":
			store.close()


def main(argv=None):
	"""Run the conversion; argv defaults to sys.argv[1:]. Returns the exit status."""
	parser = build_parser()
	argv = sys.argv[1:] if argv is None else list(argv)
	arg = parser.parse_args(argv or ["--help"])
	start_time = time.time()
	print(f"START tiffScanDataToZarrForMicrotomography {' '.join(argv)}", flush=True)
	h5file = os.path.realpath(arg.inputh5)
	output_path = os.path.abspath(arg.outputZarr)
	raw_dir = arg.raw_dir if arg.raw_dir is not None else os.path.dirname(h5file)
	threads = 1 if arg.j == 0 else arg.j

	# 1. metadata -> scan dataset
	scanDescription = PETRA.ScanDescriptionFactory.fromH5(h5file)
	df = scan_dataset(scanDescription, raw_dir, arg.fix_corrupted_h5)

	# 2. scan dataset -> spec
	mode = "w" if arg.force else "w-"
	if arg.den:
		output = {"format": "den", "path": output_path, "mode": mode}
	else:
		output = {"format": "zarr", "store": "zip" if arg.zip else "directory", "path": output_path,
				  "zarr_format": 2 if arg.v2 else 3, "mode": mode}
	try:
		spec = TOMOGRAPHY.scanTiffStackConversionSpecs(df, output, rawDir=raw_dir, fillValue=arg.fill_missing)
	except TIFF.SpecError as e:
		print(f"ERROR: {e}", file=sys.stderr)
		if "fillValue" in str(e):
			print("Use --fill-missing VALUE to write the frames without a file as VALUE.", file=sys.stderr)
		return 1
	any_stack = next(iter(spec["tiffstacks"].values()))
	dimy, dimx = any_stack["frame_shape"]
	dtype = np.dtype(any_stack["dtype"])
	if dtype != np.uint16:
		print(f"ERROR: the TIFF images are {dtype}, expected uint16", file=sys.stderr)
		return 1
	if arg.dtype is not None:
		try:
			outtype = np.dtype(arg.dtype)
		except TypeError:
			print(f"ERROR: invalid --dtype {arg.dtype}", file=sys.stderr)
			return 1
	elif arg.compression == "zfp" and not arg.den:
		outtype = np.dtype(np.int32)  # zfp: int32, int64, float32, float64 only; uint16 fits int32 exactly
	elif arg.compression == "sz3" and not arg.den:
		outtype = np.dtype(np.float32)  # sz3: floating point only
	else:
		outtype = dtype
	defaults = {}
	if outtype != dtype:
		defaults["dtype"] = str(outtype)
	if not arg.den:
		clevel = int(arg.clevel) if float(arg.clevel).is_integer() else arg.clevel
		defaults["zarr"] = {"codec": {"name": arg.compression, "clevel": clevel, **CODEC_ARGUMENTS.get(arg.compression, {})}}
		counts = {name: len(stack["frames"]) for name, stack in spec["tiffstacks"].items()}
		spec["output"]["attributes"] = scanDescription.rootAttributes(export={
			"dimx": dimx, "dimy": dimy, "dtype": str(outtype),
			"img_count": counts["img"], "ref_count": counts["ref"], "dar_count": counts["dar"],
			"h5_path": h5file, "output_zarr_path": output_path,
			"compression": {"name": arg.compression, "clevel": arg.clevel}})
	if defaults:
		spec["defaults"] = defaults
	params_json = os.path.join(output_path, "params.json")
	try:
		for extension in arg.spec_extension:
			spec = TIFF.extend_spec(spec, extension)
		if arg.den and os.path.lexists(params_json) and not arg.force:
			raise TIFF.SpecError(f"{params_json} exists; use --force to replace it")
	except TIFF.SpecError as e:
		print(f"ERROR: {e}", file=sys.stderr)
		return 1
	if arg.write_spec:
		with open(arg.write_spec, "w") as f:
			json.dump(spec, f, indent=1)
			f.write("\n")

	# 3. spec -> arrays
	try:
		plan = TIFF.validate(spec, check_headers=arg.check_headers)
		print(plan.describe(), flush=True)
		if arg.dry_run:
			return 0
		report = TIFF.execute(plan, threads=threads, verbose=arg.verbose, log=lambda m: print(m, flush=True))
	except (TIFF.SpecError, TIFF.FrameError) as e:
		print(f"ERROR: {e}", file=sys.stderr)
		return 1

	# 4. metadata -> the same store
	t0 = time.time()
	if arg.den:
		params = {"experiment": scanDescription.rootAttributes(),
				  "scan": json.loads(df.to_json(orient="split", date_format="iso"))}
		with open(params_json, "w") as f:
			json.dump(params, f, indent=1)
	else:
		write_params(spec["output"], scanDescription, df)
	print(f"TIME metadata: {time.time() - t0:.2f}s", flush=True)

	for name, a in report["arrays"].items():
		null = f", {len(a['null_frames'])} null frames {a['null_frames'][:10]}" if a["null_frames"] else ""
		print(f"SIZE {name}: {a['stored_bytes'] / 2**30:.3f} GiB stored, {a['ratio_to_raw']} of raw, "
			  f"TIFF input {a['input_bytes'] / 2**30:.3f} GiB{null}", flush=True)
	if arg.report:
		with open(arg.report, "w") as f:
			json.dump(report, f, indent=1)
			f.write("\n")
	elapsed = int(time.time() - start_time)
	print(f"END tiffScanDataToZarrForMicrotomography in {elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}s")
	return 0


if __name__ == "__main__":
	sys.exit(main())
