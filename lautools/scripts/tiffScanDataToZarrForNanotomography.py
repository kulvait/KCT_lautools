#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert nanoCT TIFF acquisitions into img/ref/dar Zarr stacks.

Metadata comes from lautools.NANOCT:
- scanDataset for the main acquisition;
- scanDatasetWithForeignDarkFields when foreign references supply darks.

Requires Zarr-Python 3, even when writing Zarr format 2.

@author: Vojtěch Kulvait
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import tempfile
from timeit import default_timer as timer

import numpy as np
from PIL import Image
import zarr

from denpy import ZAR
from lautools import NANOCT


COMPRESSION_TYPES = [
	"none", "zstd", "lz4", "gzip", "blosc",
	"blosc-blosclz", "blosc-lz4", "blosc-lz4hc",
	"blosc-snappy", "blosc-zlib", "blosc-zstd",
	"avif", "jpeg2k", "htj2k", "jpegxl", "jpegxr",
	"sz3", "zfp",
]

SERIALIZER_CODECS = {
	"avif", "jpeg2k", "htj2k", "jpegxl", "jpegxr", "sz3", "zfp",
}


def build_parser():
	parser = argparse.ArgumentParser(
		description="Convert nanoCT LogScan.log and raw TIFF files to Zarr."
	)
	parser.add_argument("logFile", help="Main acquisition's LogScan.log.")
	parser.add_argument("outputZarr", help="Output directory or ZIP file.")
	parser.add_argument(
		"--raw-dir",
		default=None,
		help="Main TIFF directory; defaults to the directory of logFile.",
	)
	parser.add_argument(
		"--foreign-log-file",
		default=None,
		help="Separate measurement whose reference images supply dark fields.",
	)
	parser.add_argument(
		"--foreign-raw-dir",
		default=None,
		help="Foreign TIFF directory; defaults to the foreign log directory.",
	)
	parser.add_argument(
		"--no-check-block-order",
		action="store_true",
		help="Disable NANOCT's time-order versus counter-order validation.",
	)
	parser.add_argument(
		"--zero-based-local-indices",
		action="store_true",
		help="Interpret the last index in multi-number TIFF names as zero-based.",
	)
	parser.add_argument(
		"--compression",
		choices=COMPRESSION_TYPES,
		default="blosc-zstd",
	)
	parser.add_argument(
		"--clevel",
		type=float,
		default=5,
		help="Codec-specific compression setting; default 5.",
	)
	parser.add_argument(
		"--bitspersample",
		type=int,
		default=None,
		help="Optional image-codec bit depth. No 12-bit assumption is made.",
	)
	parser.add_argument(
		"-j", "--threads",
		dest="j",
		type=int,
		default=-1,
		help="Threads: -1 automatic, 0 sequential, positive integer explicit.",
	)
	parser.add_argument("--zip", action="store_true")
	parser.add_argument("--v2", action="store_true")
	parser.add_argument("--force", action="store_true")
	parser.add_argument("--verbose", action="store_true")
	return parser


def inputFile(path):
	resolved = Path(path).resolve(strict=True)
	if not resolved.is_file():
		raise ValueError(f"Not a file: {resolved}")
	return str(resolved)


def inputDirectory(path):
	resolved = Path(path).resolve(strict=True)
	if not resolved.is_dir():
		raise ValueError(f"Not a directory: {resolved}")
	return str(resolved)


def loadDataset(ARG):
	options = {
		"one_based": not ARG.zero_based_local_indices,
		"drop_missing": True,
		"check_block_order": not ARG.no_check_block_order,
	}

	if ARG.foreign_log_file is None:
		df = NANOCT.scanDataset(
			ARG.logFile, imgDir=ARG.raw_dir, **options
		)
	else:
		loader = getattr(NANOCT, "scanDatasetWithForeignDarkFields", None)
		if loader is None:
			raise RuntimeError(
				"Installed lautools.NANOCT does not provide "
				"scanDatasetWithForeignDarkFields. Update lautools "
				"or omit --foreign-log-file."
			)

		df = loader(
			ARG.logFile,
			ARG.raw_dir,
			ARG.foreign_log_file,
			ARG.foreign_raw_dir,
			**options,
		)

	if df.empty:
		raise ValueError("No matched TIFF frames available for export.")

	required = {"time", "image_key", "image_file"}
	missing = required.difference(df.columns)
	if missing:
		raise ValueError(f"Dataset lacks required columns: {sorted(missing)}")

	df = df.copy().reset_index(drop=True)

	if df["image_file"].isna().any():
		raise ValueError("Dataset contains unmatched TIFF paths.")

	df["image_file"] = [
		os.fsdecode(path) for path in df["image_file"]
	]

	invalid_keys = ~df["image_key"].isin([0, 1, 2])
	if invalid_keys.any():
		raise ValueError(
			"Unsupported image_key values: "
			f"{df.loc[invalid_keys, 'image_key'].unique().tolist()}"
		)

	if df["time"].isna().any():
		raise ValueError("Dataset contains missing frame timestamps.")

	for filename in df["image_file"]:
		if not os.path.isabs(filename):
			raise ValueError(
				f"NANOCT must return absolute TIFF paths: {filename!r}"
			)
		if not os.path.isfile(filename):
			raise FileNotFoundError(filename)

	# Foreign measurements can introduce duplicate local counters/block IDs.
	# Preserve them unchanged; these explicit indices describe storage order.
	df["zarr_array"] = df["image_key"].map(
		{0: "img", 1: "ref", 2: "dar"}
	)
	df["zarr_index"] = df.groupby(
		"zarr_array", sort=False
	).cumcount()

	return df


def readTiff(filename):
	try:
		with Image.open(filename) as image:
			array = np.array(image)
	except Exception as error:
		raise ValueError(f"Could not read TIFF {filename!r}: {error}") from error

	if array.ndim != 2 or array.size == 0:
		raise ValueError(
			f"Expected nonempty 2D grayscale TIFF: "
			f"{filename!r}, shape={array.shape}"
		)

	return array


def codecOptions(ARG):
	options = {}

	if ARG.compression in {"jpeg2k", "avif", "jpegxl"}:
		if ARG.bitspersample is not None:
			options["bitspersample"] = ARG.bitspersample
		options["numthreads"] = 1

	if ARG.compression == "jpeg2k":
		options.update(colorspace="GRAY", mct=False)
	elif ARG.compression == "htj2k":
		options["rgb"] = False
	elif ARG.compression == "avif":
		options["pixelformat"] = "YUV400 "
	elif ARG.compression == "jpegxl":
		options["photometric"] = "GRAY"
	elif ARG.compression == "jpegxr":
		options.update(photometric="GRAY", hasalpha=False)

	return options


def createImageArray(group, name, count, shape, dtype, codec, ARG):
	kwargs = {
		"shape": (count, *shape),
		"chunks": (1, *shape),
		"dtype": dtype,
	}

	if ARG.v2:
		kwargs["compressor"] = codec
	elif ARG.compression in SERIALIZER_CODECS:
		if not codec or len(codec) != 1:
			raise ValueError(
				f"Expected one serializer for {ARG.compression!r}."
			)
		kwargs["serializer"] = codec[0]
		# Do not add the default compressor after the image serializer.
		kwargs["compressors"] = []
	else:
		kwargs["compressors"] = codec

	return group.create_array(name, **kwargs)


def writeZarrArray(df, array, source_dtype, threads, verbose):
	if len(df) != array.shape[0]:
		raise ValueError(
			f"{array.path}: metadata count {len(df)} "
			f"differs from stack length {array.shape[0]}."
		)

	# Empty categories are valid and already have an allocated empty stack.
	if df.empty:
		return

	expected_shape = tuple(array.shape[1:])
	paths = df["image_file"].tolist()

	def writeFrame(index):
		filename = paths[index]
		image = readTiff(filename)

		if image.shape != expected_shape:
			raise ValueError(
				f"Shape mismatch in {filename!r}: "
				f"{image.shape}, expected {expected_shape}."
			)
		if image.dtype != source_dtype:
			raise ValueError(
				f"Dtype mismatch in {filename!r}: "
				f"{image.dtype}, expected {source_dtype}."
			)

		# Only a prevalidated safe output conversion is permitted.
		array[index, :, :] = image.astype(array.dtype, copy=False)
		return index

	started = timer()

	if threads == 0:
		for index in range(len(paths)):
			writeFrame(index)
			if verbose and (
				(index + 1) % 100 == 0 or index + 1 == len(paths)
			):
				print(f"{array.path}: {index + 1}/{len(paths)} frames")
	else:
		workers = min(threads, len(paths))
		# Each task writes one distinct, whole-frame chunk.
		with ThreadPoolExecutor(max_workers=workers) as executor:
			for completed, _ in enumerate(
				executor.map(writeFrame, range(len(paths))), start=1
			):
				if verbose and (
					completed % 100 == 0 or completed == len(paths)
				):
					print(f"{array.path}: {completed}/{len(paths)} frames")

	if verbose:
		print(
			f"{array.path}: finished in {timer() - started:.1f}s; "
			f"shape={array.shape}, dtype={array.dtype}"
		)


def removePath(path):
	if path.is_symlink() or path.is_file():
		path.unlink()
	elif path.exists():
		shutil.rmtree(path)


def main(argv=None):
	parser = build_parser()
	print("START tiffScanDataToZarrForMicrotomography %s" % " ".join(sys.argv[1:]))
	parser = buildParser()
	ARG = parser.parse_args(argv)

	if ARG.j < -1:
		parser.error("-j must be -1, 0, or a positive integer.")
	if ARG.foreign_raw_dir and not ARG.foreign_log_file:
		parser.error("--foreign-raw-dir requires --foreign-log-file.")
	if ARG.bitspersample is not None and ARG.bitspersample <= 0:
		parser.error("--bitspersample must be positive.")

	ARG.logFile = inputFile(ARG.logFile)
	ARG.raw_dir = inputDirectory(
		ARG.raw_dir or str(Path(ARG.logFile).parent)
	)

	if ARG.foreign_log_file:
		ARG.foreign_log_file = inputFile(ARG.foreign_log_file)
		ARG.foreign_raw_dir = inputDirectory(
			ARG.foreign_raw_dir or str(Path(ARG.foreign_log_file).parent)
		)

	output = Path(ARG.outputZarr).absolute()
	if output.is_symlink():
		raise ValueError("Refusing to overwrite a symlink output.")

	# Prevent --force from deleting an input file or an input directory's
	# ancestor after successful conversion.
	resolved_output = output.resolve()
	protected = [Path(ARG.logFile), Path(ARG.raw_dir)]
	if ARG.foreign_log_file:
		protected.extend([
			Path(ARG.foreign_log_file),
			Path(ARG.foreign_raw_dir),
		])
	for source in protected:
		if resolved_output == source or resolved_output in source.parents:
			raise ValueError(f"Output would overwrite an input: {output}")

	if output.exists() and not ARG.force:
		raise FileExistsError(f"{output} exists; use --force.")

	df = loadDataset(ARG)

	first = readTiff(df["image_file"].iloc[0])
	shape = first.shape
	source_dtype = first.dtype
	output_dtype = source_dtype

	# Match the original converter's small-integer upcasting, but use
	# signed int32 so uint16 values remain representable without wrapping.
	if ARG.compression in {"zfp", "sz3"}:
		if np.issubdtype(source_dtype, np.integer) and source_dtype.itemsize < 4:
			output_dtype = np.dtype("int32")

	if not np.can_cast(source_dtype, output_dtype, casting="safe"):
		raise ValueError(
			f"Unsafe conversion: {source_dtype} -> {output_dtype}"
		)

	codec = ZAR.get_compressor(
		ARG.compression,
		ARG.clevel,
		zarrv2=ARG.v2,
		dtype=output_dtype,
		**codecOptions(ARG),
	)

	subsets = {
		name: df.loc[df["zarr_array"] == name].copy()
		for name in ("img", "ref", "dar")
	}

	# Preserve source order, including darks from multiple measurements.
	# Do not require global timestamp ordering or lexical filename ordering.
	counts = {name: len(table) for name, table in subsets.items()}

	export = {
		"dimx": int(shape[1]),
		"dimy": int(shape[0]),
		"dtype": np.dtype(output_dtype).name,
		"source_dtype": source_dtype.name,
		"img_count": counts["img"],
		"ref_count": counts["ref"],
		"dar_count": counts["dar"],
		"log_path": ARG.logFile,
		"raw_dir": ARG.raw_dir,
		"foreign_log_path": ARG.foreign_log_file,
		"foreign_raw_dir": ARG.foreign_raw_dir,
		"output_zarr_path": str(output),
		"zarr_format": 2 if ARG.v2 else 3,
		"compression": {
			"name": ARG.compression,
			"clevel": ARG.clevel,
			"codec_kwargs": codecOptions(ARG),
		},
	}

	root_attrs = {
		"acquisition_type": "nanotomography",
		"metadata_source": "LogScan.log",
		"export": export,
	}

	# Same params attribute structure as the original converter.
	df_json = json.loads(
		df.to_json(orient="split", date_format="iso", date_unit="ns")
	)

	is_zip = ARG.zip or str(output).lower().endswith(".zip")
	threads = (os.cpu_count() or 1) if ARG.j == -1 else ARG.j

	# Conservative ZIP policy: all writes are sequential.
	if is_zip:
		threads = 0

	output.parent.mkdir(parents=True, exist_ok=True)

	# Build separately so a failed read/codec/write does not destroy an
	# existing output. Publication happens only after all writes succeed.
	workdir = Path(tempfile.mkdtemp(
		prefix=f".{output.name}.partial-",
		dir=output.parent,
	))
	temporary_output = workdir / ("data.zip" if is_zip else "data.zarr")
	zip_store = None
	started = timer()

	try:
		if is_zip:
			zip_store = zarr.storage.ZipStore(
				str(temporary_output), mode="w"
			)
			store = zip_store
		else:
			store = str(temporary_output)

		try:
			group = zarr.open_group(
				store=store,
				mode="w",
				zarr_format=2 if ARG.v2 else 3,
				attributes=root_attrs,
			)

			# There is no H5 hierarchy to copy for a nanoCT acquisition.
			group.create_group("params", attributes=df_json)

			for name in ("ref", "dar", "img"):
				array = createImageArray(
					group, name, counts[name], shape,
					output_dtype, codec, ARG,
				)
				writeZarrArray(
					subsets[name], array, source_dtype,
					threads, ARG.verbose,
				)
		finally:
			if zip_store is not None:
				zip_store.close()

		if output.exists():
			if not ARG.force:
				raise FileExistsError(f"Output appeared during export: {output}")
			removePath(output)

		temporary_output.rename(output)
	finally:
		shutil.rmtree(workdir, ignore_errors=True)

	print(
		f"Written {output}: "
		f"IMG={counts['img']}, REF={counts['ref']}, DAR={counts['dar']}; "
		f"elapsed={timer() - started:.1f}s"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
