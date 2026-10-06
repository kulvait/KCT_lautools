#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Feb 2026

@author: Vojtěch Kulvait
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from multiprocessing import Value
from multiprocessing.dummy import Lock, Pool  # threads
from pathlib import Path
from timeit import default_timer as timer

import h5py
import numpy as np
import pandas as pd
import zarr
from PIL import Image
from termcolor import colored

from denpy import PETRA
from denpy import ZAR


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser()
	parser.add_argument("inputh5")
	parser.add_argument("outputZarr")
	parser.add_argument(
		"--raw-dir",
		default=None,
		type=str,
		help="Provide raw directory where to find files, by default parrent directory of inputh5.",
	)
	parser.add_argument("--fix-corrupted-h5", action="store_true", help="Fix corrupted HDF5 file by scanning for TIFF files.")
	parser.add_argument("--force", action="store_true")
	parser.add_argument(
		"--compression",
		type=str,
		choices=["none", "zstd", "lz4", "gzip", "blosc", "blosc-blosclz", "blosc-lz4", "blosc-lz4hc", "blosc-snappy", "blosc-zlib", "blosc-zstd", "avif", "jpeg2k", "htj2k", "jpegxl", "jpegxr", "sz3", "zfp"],
		default="blosc-zstd",
		help="Compression type (default: blosc-zstd).",
	)
	parser.add_argument("--clevel", type=float, default=5, help="Compression level (default: 5).")
	parser.add_argument(
		"-j",
		"--threads",
		default=-1,
		type=int,
		dest="j",
		help="Number of threads to use. [defaults to -1 which is mp.cpu_count(), 0 without threading]",
	)
	parser.add_argument("--zip", action="store_true", help="Use zip store for Zarr output instead of directory store.")
	parser.add_argument("--v2", action="store_true", help="Use Zarr v2 format instead of v3 (default: False, use v3).")
	parser.add_argument("--verbose", action="store_true")
	return parser


def _init_writer_worker(lock):
	# Kept for multiprocessing initializer compatibility.
	# The current implementation does not need a shared writer lock.
	return None


def insertToDf(df, dat, name):
	time_values = dat["%s/time" % (name)]
	value_values = dat["%s/value" % (name)]
	for i in range(len(value_values)):
		t = time_values[i]
		v = value_values[i]
		df.loc[t][name] = v


def tiffImageToArrayIndex(tiffFile, outArray, kIndex):
	"""Read a single TIFF image and write it to the specified index in the Zarr/numpy array."""
	if outArray is None or not hasattr(outArray, "shape") or not hasattr(outArray, "dtype"):
		raise TypeError("outArray must have 'shape' and 'dtype' attributes (NumPy/Zarr-like).")

	if len(outArray.shape) != 3:
		raise ValueError(f"outArray must be 3D (Z, Y, X); got shape={outArray.shape}")
	dimz, dimy, dimx = outArray.shape
	if not (0 <= kIndex < dimz):
		raise IndexError(f"kIndex {kIndex} out of bounds for array with shape {outArray.shape}")
	target_dtype = np.dtype(outArray.dtype)

	try:
		with Image.open(tiffFile) as im:
			img = np.array(im)
	except Exception as e:
		raise ValueError(f"Failed to read TIFF file '{tiffFile}': {e}")

	if img.shape != (dimy, dimx):
		raise ValueError(f"Shape mismatch in TIFF '{tiffFile}': got {img.shape}, expected {(dimy, dimx)}")

	if img.dtype != target_dtype:
		try:
			img = img.astype(target_dtype, copy=False)
		except Exception as e:
			raise ValueError(f"Failed to convert image dtype from {img.dtype} to {target_dtype}: {e}")

	outArray[kIndex, :, :] = img


def tiffImageToArrayIndex_worker(tiffFile, outArray, kIndex):
	"""Worker function for multiprocessing that wraps tiffImageToArrayIndex and captures exceptions."""
	try:
		tiffImageToArrayIndex(tiffFile, outArray, kIndex)
		return {"tiffFile": tiffFile, "kIndex": kIndex, "n_img": outArray.shape[0], "error": None}
	except Exception:
		return {"tiffFile": tiffFile, "kIndex": kIndex, "n_img": outArray.shape[0], "error": traceback.format_exc()}


def progress_callback(result, progress, verbose):
	with progress.get_lock():
		progress.value += 1
		count = progress.value
	if verbose and (count % 100 == 0 or result["error"] is not None):
		print(
			f"Written {count}/{result['n_img']} frames, current: {result['kIndex'] + 1}/{result['n_img']} "
			f"({os.path.basename(result['tiffFile'])})"
		)


def writeZarrArray(df, zarrArray, inputDir, threads, verbose):
	progress = Value("i", 0)
	if df.empty:
		raise ValueError("Dataframe is empty, can not write to Zarr")
	if "time" in df.columns:
		if not df["time"].is_monotonic_increasing:
			raise ValueError("Dataframe is not sorted by time, can not write to Zarr")
	if "image_file" not in df.columns:
		raise ValueError("Dataframe does not contain image_file column, can not write to Zarr")
	if not df["image_file"].is_monotonic_increasing:
		raise ValueError("Dataframe is not sorted by image_file, can not write to Zarr")

	inputTifFiles = [x.decode("utf-8") if isinstance(x, bytes) else x for x in df["image_file"]]
	tifFilesBasename = [os.path.basename(f) for f in inputTifFiles]
	if not all(tifFilesBasename[i] <= tifFilesBasename[i + 1] for i in range(len(tifFilesBasename) - 1)):
		raise ValueError("Dataframe is not sorted by image_file, can not write to Zarr")
	else:
		print(tifFilesBasename[0], tifFilesBasename[-1])

	inputTifFiles = [os.path.join(inputDir, f.lstrip("/")) for f in inputTifFiles]
	n_images = len(inputTifFiles)

	if threads == 0:
		for i, f in enumerate(inputTifFiles):
			start = timer()
			tiffImageToArrayIndex(f, zarrArray, i)
			if verbose and (i % 100 == 0 or i == n_images - 1):
				print(f"Written frame {i + 1}/{n_images} ({os.path.basename(f)}) in {timer() - start:.3f}s")
	else:
		results = []
		with Pool(threads, initializer=_init_writer_worker, initargs=(Lock(),)) as pool:
			for i, f in enumerate(inputTifFiles):
				res = pool.apply_async(
					tiffImageToArrayIndex_worker,
					args=(f, zarrArray, i),
					callback=lambda r: progress_callback(r, progress, verbose),
				)
				results.append(res)
			pool.close()
			pool.join()

		errors = []
		for res in results:
			r = res.get()
			if r["error"] is not None:
				errors.append((r["tiffFile"], r["kIndex"], r["error"]))
		if len(errors) > 0:
			print(colored(f"Encountered {len(errors)} errors during TIFF processing:", "red"))
			for tiffFile, kIndex, error in errors:
				print(colored(f"Error processing '{tiffFile}' at index {kIndex}:\n{error}", "red"))
			raise RuntimeError(f"{len(errors)} errors occurred during TIFF processing. See above for details.")

	if verbose:
		print(colored(f"Zarrarray written to {zarrArray.path} with shape {zarrArray.shape} and dtype {zarrArray.dtype}", "green"))


def scanForTiffFiles(directory, exclude_files):
	tiff_files = []
	for root, _, files in os.walk(directory):
		for file in files:
			if file.lower().endswith(".tiff") or file.lower().endswith(".tif"):
				filepath = os.path.join(root, file)
				relative_path = os.path.relpath(filepath, directory)
				if relative_path not in exclude_files:
					tiff_files.append(relative_path)
	tiff_files.sort()
	return pd.DataFrame({"image_file": tiff_files})


def sanitize_for_json(obj):
	"""
	Recursively convert obj so it can be JSON-serialized:
	- numpy scalars -> Python scalars
	- numpy arrays -> lists
	- dict/list/tuple -> recurse
	- leave Python scalars/None/str as-is
	"""
	if isinstance(obj, (np.generic,)):
		return obj.item()
	if isinstance(obj, np.ndarray):
		return obj.tolist()
	if isinstance(obj, dict):
		return {str(k): sanitize_for_json(v) for k, v in obj.items()}
	if isinstance(obj, (list, tuple)):
		return [sanitize_for_json(x) for x in obj]
	if isinstance(obj, (bytes, bytearray, np.bytes_)):
		try:
			return obj.decode("utf-8")
		except UnicodeDecodeError:
			return obj.decode("utf-8", errors="replace")
	return obj


def copy_h5_arrays_into_group(
	h5,
	zarr_params_group: zarr.Group,
	compressor=None,
	prefer_src_chunks: bool = True,
	chunk_bytes: int = 64 * 1024 * 1024,
	verbose: bool = True,
) -> None:
	"""
	Copy all HDF5 N-D datasets from `h5` into the given Zarr group.
	Preserves shapes, dtypes, attributes and subgroup structure.
	"""
	attributes = dict(h5.attrs)
	for entry in list(h5):
		if verbose:
			print(f"Processing HDF5 entry: {h5.name}/{entry}  type={type(h5[entry])}")
		if isinstance(h5[entry], h5py.Dataset):
			if h5[entry].shape == ():
				val = h5[entry][()]
				if isinstance(val, (bytes, np.bytes_)):
					val = val.decode("utf-8")
				if isinstance(val, np.generic):
					val = val.item()
				attributes[entry] = val
			elif h5[entry].dtype == np.dtype("O") or h5[entry].dtype.kind == "S":
				str_array = h5[entry][:]
				if str_array.flatten().size > 0:
					val = str_array.flatten()[0]
					if isinstance(val, (bytes, np.bytes_)):
						str_array = np.char.decode(str_array.astype("S"), "utf-8", errors="replace")
				za = zarr_params_group.create_array(entry, shape=str_array.shape, dtype="string")
				za[:] = str_array
			else:
				za = zarr_params_group.create_array(
					entry,
					shape=h5[entry].shape,
					dtype=h5[entry].dtype,
					compressors=compressor,
				)
				za[:] = h5[entry][:]
				if verbose:
					print(
						f"[dataset] {h5.name}/{entry}  shape={h5[entry].shape}\tdtype={h5[entry].dtype}  "
						f"-> zarr dtype={za.dtype}  chunks={za.chunks}"
					)
		elif isinstance(h5[entry], h5py.Group):
			subgroup = zarr_params_group.require_group(entry)
			copy_h5_arrays_into_group(h5[entry], subgroup)
		else:
			raise ValueError(f"HDF5 entry '{entry}' is neither a group nor a dataset, cannot copy.")

	for k, v in attributes.items():
		if isinstance(v, (bytes, np.bytes_)):
			v = v.decode("utf-8")
		if isinstance(v, np.generic):
			v = v.item()
		zarr_params_group.attrs[k] = v

	if verbose:
		print("Finished copying HDF5 arrays into Zarr group:", zarr_params_group.path)


def main(argv=None):
	parser = build_parser()
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
	print("START tiffScanDataToZarrForMicrotomography %s" % " ".join(sys.argv[1:]))
	start_time = time.time()
	ARG.inputh5 = str(Path(ARG.inputh5).resolve(strict=True))
	ARG.outputZarr = str(Path(ARG.outputZarr).resolve(strict=False))

	if ARG.j < 0:
		ARG.j = mp.cpu_count()
		print("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()" % (ARG.j))
	elif ARG.j == 0:
		print("No threading will be used ARG.j=0.")
	else:
		print("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()=%d" % (ARG.j, mp.cpu_count()))

	if ARG.raw_dir is not None:
		inputDir = ARG.raw_dir
	else:
		inputDir = os.path.dirname(os.path.realpath(ARG.inputh5))

	if os.path.exists(ARG.outputZarr):
		if ARG.force:
			if os.path.isfile(ARG.outputZarr):
				os.remove(ARG.outputZarr)
			else:
				shutil.rmtree(ARG.outputZarr)
		else:
			raise IOError(f"File {ARG.outputZarr} exists, use --force to overwrite")

	df = PETRA.scanDataset(ARG.inputh5, includeCurrent=True)
	experimentInfo = PETRA.getExperimentInfo(ARG.inputh5)
	export = {}

	dtype = None
	if df.empty:
		raise ValueError("Dataframe is empty, no data to write to Zarr file %s" % (os.path.realpath(ARG.outputZarr)))
	else:
		tiff_file_str = df["image_file"].iloc[0]
		if isinstance(tiff_file_str, bytes):
			tiff_file_str = tiff_file_str.decode("utf-8")
		tiff_file_path = os.path.join(inputDir, tiff_file_str.lstrip("/"))
		if not os.path.exists(tiff_file_path):
			raise FileNotFoundError(
				f"TIFF file {tiff_file_path} does not exist, cannot determine image dimensions and dtype."
			)
		with Image.open(tiff_file_path) as img:
			img_array = np.array(img)
			dimy, dimx = img_array.shape
			dtype = img_array.dtype
			print(f"Determined image dimensions: {dimx}x{dimy}, dtype: {dtype}")

	if dtype is None:
		raise ValueError("Could not determine dtype of TIFF images, cannot proceed.")

	if dtype != np.uint16:
		raise ValueError(f"Warning: TIFF images are of type {dtype}, expected uint16. This may lead to unexpected behavior.")

	if ARG.compression in ["zfp", "sz3"]:
		if np.issubdtype(dtype, np.integer) and dtype.itemsize < 4:
			outtype = np.uint32
		else:
			outtype = dtype
	else:
		outtype = dtype

	codec_kwargs = {}
	if ARG.compression == "jpeg2k":
		codec_kwargs["bitspersample"] = 12
		codec_kwargs["colorspace"] = "GRAY"
		codec_kwargs["mct"] = False
		codec_kwargs["numthreads"] = 1
	elif ARG.compression == "htj2k":
		codec_kwargs["rgb"] = False
	elif ARG.compression == "avif":
		codec_kwargs["bitspersample"] = 12
		codec_kwargs["pixelformat"] = "YUV400 "
		codec_kwargs["numthreads"] = 1
	elif ARG.compression == "jpegxl":
		codec_kwargs["bitspersample"] = 12
		codec_kwargs["photometric"] = "GRAY"
		codec_kwargs["numthreads"] = 1
	elif ARG.compression == "jpegxr":
		codec_kwargs["photometric"] = "GRAY"
		codec_kwargs["hasalpha"] = False

	dark = df.loc[df["image_key"] == 2]
	white = df.loc[df["image_key"] == 1]
	scan = df.loc[df["image_key"] == 0]

	df_json = json.loads(df.to_json(orient="split", date_format="iso"))

	chunk_shape = (1, dimy, dimx)
	dar_count = len(dark)
	ref_count = len(white)
	img_count = len(scan)

	export["dimx"] = dimx
	export["dimy"] = dimy
	export["dtype"] = str(outtype)
	export["img_count"] = img_count
	export["ref_count"] = ref_count
	export["dar_count"] = dar_count
	export["h5_path"] = ARG.inputh5
	export["output_zarr_path"] = ARG.outputZarr
	export["compression"] = {}
	export["compression"]["name"] = ARG.compression
	export["compression"]["clevel"] = ARG.clevel

	experimentInfo["export"] = export
	experimentInfo_sanitized = sanitize_for_json(experimentInfo)

	if ARG.zip or ARG.outputZarr.endswith(".zip"):
		store = zarr.storage.ZipStore(ARG.outputZarr, mode="w")
	else:
		store = ARG.outputZarr

	zarr_top_level = zarr.open_group(
		store=store,
		mode="w",
		attributes=experimentInfo_sanitized,
		zarr_format=2 if ARG.v2 else 3,
	)

	codec = ZAR.get_compressor(ARG.compression, ARG.clevel, zarrv2=ARG.v2, dtype=outtype, **codec_kwargs)
	try:
		if not ARG.v2:
			if ARG.compression in ["avif", "jpeg2k", "htj2k", "jpegxl", "jpegxr", "sz3", "zfp"]:
				zarr_array_ref = zarr_top_level.create_array("ref", shape=(ref_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, serializer=codec[0])
				zarr_array_dar = zarr_top_level.create_array("dar", shape=(dar_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, serializer=codec[0])
				zarr_array_img = zarr_top_level.create_array("img", shape=(img_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, serializer=codec[0])
			else:
				zarr_array_ref = zarr_top_level.create_array("ref", shape=(ref_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, compressors=codec)
				zarr_array_dar = zarr_top_level.create_array("dar", shape=(dar_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, compressors=codec)
				zarr_array_img = zarr_top_level.create_array("img", shape=(img_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, compressors=codec)
		else:
			zarr_array_ref = zarr.open_array(store=store, path="ref", shape=(ref_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, compressor=codec, zarr_format=2, mode="w")
			zarr_array_dar = zarr.open_array(store=store, path="dar", shape=(dar_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, compressor=codec, zarr_format=2, mode="w")
			zarr_array_img = zarr.open_array(store=store, path="img", shape=(img_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, compressor=codec, zarr_format=2, mode="w")
	except zipfile.BadZipFile as e:
		print(
			f"Zarr store {ARG.outputZarr} is not a valid zip file or is corrupted. This can happen when two processes try to write to the same zip file simultaneously. Please delete the file and try again."
		)
		print(f"Error details: {e}")
		return 1

	zarr_params = zarr_top_level.create_group(name="params", attributes=df_json)

	with h5py.File(ARG.inputh5, "r") as h5:
		if "entry" in h5:
			h5 = h5["entry"]
		copy_h5_arrays_into_group(h5, zarr_params, verbose=ARG.verbose)

	writeZarrArray(dark, zarr_array_dar, inputDir, threads=ARG.j, verbose=ARG.verbose)
	writeZarrArray(white, zarr_array_ref, inputDir, threads=ARG.j, verbose=ARG.verbose)

	if ARG.fix_corrupted_h5 and scan.empty:
		print("HDF5 file is corrupted. Scanning directory for TIFF files to create img.den...")
		dark_files = set(dark["image_file"])
		white_files = set(white["image_file"])
		exclude_files = dark_files.union(white_files)
		scan = scanForTiffFiles(inputDir, exclude_files)

	writeZarrArray(scan, zarr_array_img, inputDir, threads=ARG.j, verbose=ARG.verbose)

	elapsed_seconds = int(time.time() - start_time)
	hours = elapsed_seconds // 3600
	minutes = (elapsed_seconds % 3600) // 60
	seconds = elapsed_seconds % 60
	print(f"END tiffScanDataToZarrForMicrotomography in {hours:02d}:{minutes:02d}:{seconds:02d}s")
	return 0

if __name__ == "__main__":
	sys.exit(main())
