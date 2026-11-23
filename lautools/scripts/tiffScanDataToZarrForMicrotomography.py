#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Feb 2026

Zarr store of a PETRA III scan from its raw TIFF files (dar, ref, img) and NeXus HDF5 metadata (params).

Same conversion as syscripts rawTiffStructureToZarr.py, as a lautools command (main()). Step 1 of the
cleanup: the code is that script's, with the parsing and the conversion moved into build_parser()
and main(); the writing and metadata handling are to become a shared API next.

@author: Vojtěch Kulvait
"""
import h5py
import numpy as np
import pandas as pd
#from libtiff import TIFF
#pd.set_option('display.max_columns', 100) to display untruncated columns
from PIL import Image
from PIL.TiffTags import TAGS
try:
	import tifffile  # about twice as fast as PIL and releases the GIL, so the reading threads scale
except ImportError:
	tifffile = None


def read_tiff(path):
	"""The image of a TIFF file as a numpy array (tifffile if available, else PIL)."""
	if tifffile is not None:
		return tifffile.imread(path)
	with Image.open(path) as im:
		return np.array(im)
import sys
import os
import argparse
from denpy import DEN
from denpy import ZAR
from denpy import PETRA
from denpy import ZARZS  # zip store that threads write in parallel
import io
from contextlib import redirect_stdout
from contextlib import redirect_stderr
import zarr
# Threads, not processes: all writers share one store. A ZipStore must not be used from several
# processes (each would reopen the zip with mode "w", truncating it); its lock serializes the writes.
from multiprocessing.dummy import Pool, Lock  # threads
import multiprocessing as mp
from multiprocessing import Value
import traceback
from termcolor import colored
import shutil
from timeit import default_timer as timer
import json
import numcodecs
import zipfile
from numcodecs import Blosc, GZip, VLenUTF8
from pathlib import Path
import time
from datetime import timedelta
import logging

# Create a logger specific to this module
log = logging.getLogger(__name__)
log.setLevel(logging.INFO) # Set the logging level to INFO
# Create a console handler and set its level to INFO
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s : %(message)s', datefmt='%d.%m.%Y %H:%M:%S')
ch.setFormatter(formatter)
# Add the handler to the logger
log.addHandler(ch)
log.propagate = False # Prevent log messages from being propagated to the root logger



def build_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("inputh5")
	parser.add_argument("outputZarr")
	parser.add_argument(
		"--raw-dir",
		default=None,
		type=str,
		help=
		"Provide raw directory where to find files, by default parrent directory of inputh5."
	)
	parser.add_argument("--fix-corrupted-h5", action="store_true", help="Fix corrupted HDF5 file by scanning for TIFF files.")
	parser.add_argument("--force", action="store_true")
	parser.add_argument('--compression', type=str,
						choices=['none', 'zstd', 'lz4', 'gzip', 'blosc', 'blosc-blosclz', 'blosc-lz4', 'blosc-lz4hc', 'blosc-snappy', 'blosc-zlib', 'blosc-zstd', 'avif', 'jpeg2k', 'htj2k', 'jpegxl', 'jpegxr', 'sz3', 'zfp'],
						default='blosc-zstd',
						help="Compression type (default: blosc-zstd).")
	parser.add_argument('--clevel', type=float, default=5,
						help="Compression level (default: 5).")
	parser.add_argument("-j","--threads", default=-1, type=int, help="Number of threads to use. [defaults to -1: a quarter of the CPUs available to the process (in a SLURM job its allocation), at least 16 and at most 32, but not more than available; 0 without threading]", dest="j")
	parser.add_argument("--zip", action="store_true", help="Use zip store for Zarr output instead of directory store.")
	parser.add_argument("--v2", action="store_true", help="Use Zarr v2 format instead of v3 (default: False, use v3).")
	parser.add_argument("--verbose", action="store_true")
	return parser


# Time of the previous phase, see report_phase()
_phase_time = [time.time()]


def report_phase(name):
	"""Print the time spent since the previous phase, to see where a conversion spends its time."""
	now = time.time()
	print(f"TIME {name}: {now - _phase_time[0]:.1f}s", flush=True)
	_phase_time[0] = now

_write_lock = None

def _init_writer_worker(lock):
	global _write_lock
	_write_lock = lock
#END threading setup

#To create dataframe with given columns
def insertToDf(df, dat, name):
	time = dat["%s/time" % (name)]
	value = dat["%s/value" % (name)]
	for i in range(len(value)):
		t = time[i]
		v = value[i]
		df.loc[t][name] = v

def tiffImageToArrayIndex(tiffFile, outArray, kIndex):
	"""Read a single TIFF image and write it to the specified index in the Zarr/numpy array."""
	# ---- Basic array checks ----
	if outArray is None or not hasattr(outArray, "shape") or not hasattr(outArray, "dtype"):
		raise TypeError("outArray must have 'shape' and 'dtype' attributes (NumPy/Zarr-like).")

	if len(outArray.shape) != 3:
		raise ValueError(f"outArray must be 3D (Z, Y, X); got shape={outArray.shape}")
	dimz, dimy, dimx = outArray.shape
	if not (0 <= kIndex < dimz):
		raise IndexError(f"kIndex {kIndex} out of bounds for array with shape {outArray.shape}")
	target_dtype = np.dtype(outArray.dtype)
	# ---- Read TIFF ----
	try:
		img = read_tiff(tiffFile)
	except Exception as e:
		raise ValueError(f"Failed to read TIFF file '{tiffFile}': {e}")
	# ---- Check shape ----
	if img.shape != (dimy, dimx):
		raise ValueError(
			f"Shape mismatch in TIFF '{tiffFile}': got {img.shape}, expected {(dimy, dimx)}"
		)
	# ---- Cast to output dtype ALWAYS ----
	if img.dtype != target_dtype:
		try:
			img = img.astype(target_dtype, copy=False)
		except Exception as e:
			raise ValueError(
				f"Failed to convert image dtype from {img.dtype} to {target_dtype}: {e}"
			)
	outArray[kIndex, :, :] = img

def tiffImageToArrayIndex_worker(tiffFile, outArray, kIndex):
	"""Worker function for multiprocessing that wraps tiffImageToArrayIndex and captures exceptions."""
	try:
		tiffImageToArrayIndex(tiffFile, outArray, kIndex)
		return {"tiffFile": tiffFile, "kIndex": kIndex, "n_img": outArray.shape[0], "error": None}
	except Exception as e:
		return {"tiffFile": tiffFile, "kIndex": kIndex, "n_img": outArray.shape[0], "error": traceback.format_exc()}

progress = Value('i', 0)   # atomic int

def progress_callback(result):
	# result is what the worker returned
	with progress.get_lock():
		progress.value += 1
		count = progress.value
	if ARG.verbose and (count % 100 == 0 or result["error"] is not None):
		print(f"Written {count}/{result['n_img']} frames, current: {result['kIndex']+1}/{result['n_img']} "
			  f"({os.path.basename(result['tiffFile'])})")

# Sizes of the input TIFF files per array (img, ref, dar), for the size report
INPUT_TIFF_BYTES = {}


def stored_bytes(output, prefixes):
	"""Bytes stored in the Zarr output (directory or zip) under each of the top level prefixes."""
	sizes = dict.fromkeys(prefixes, 0)
	if os.path.isfile(output):
		with zipfile.ZipFile(output) as z:
			for info in z.infolist():
				top = info.filename.split("/")[0]
				if top in sizes:
					sizes[top] += info.compress_size
	else:
		for top in prefixes:
			for root, _, files in os.walk(os.path.join(output, top)):
				sizes[top] += sum(os.path.getsize(os.path.join(root, f)) for f in files)
	return sizes


def total_bytes(output):
	if os.path.isfile(output):
		return os.path.getsize(output)
	return sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(output) for f in files)


def gib(n):
	return n / 1024**3


def report_sizes(output, codec_name, clevel):
	"""Print the stored size of img, ref and dar compared with their input TIFF files."""
	stored = stored_bytes(output, list(INPUT_TIFF_BYTES))
	print(f"SIZE compression {codec_name} (clevel {clevel}):")
	for name, tiff in INPUT_TIFF_BYTES.items():
		ratio = tiff / stored[name] if stored[name] else float("nan")
		print(f"SIZE   {name}: TIFF {gib(tiff):.3f} GiB -> Zarr {gib(stored[name]):.3f} GiB, ratio {ratio:.2f}x ({100 * stored[name] / tiff:.1f}% of TIFF)")
	tiff, out = sum(INPUT_TIFF_BYTES.values()), total_bytes(output)
	print(f"SIZE   total: TIFF {gib(tiff):.3f} GiB -> output {gib(out):.3f} GiB, ratio {tiff / out:.2f}x ({100 * out / tiff:.1f}% of TIFF, output includes the params metadata)")


#To write dataframe to 
def writeZarrArray(df, zarrArray, inputDir):
	progress = Value('i', 0)   # atomic int
	if df.empty:
		raise ValueError("Dataframe is empty, can not write to Zarr")
	if "time" in df.columns:
		#Check it is sorted by time and raise error if not
		if not df["time"].is_monotonic_increasing:
			raise ValueError("Dataframe is not sorted by time, can not write to Zarr")
	#Check image_file column exists and dataset is sorted by image_file
	if "image_file" not in df.columns:
		raise ValueError("Dataframe does not contain image_file column, can not write to Zarr")
	if not df["image_file"].is_monotonic_increasing:
		raise ValueError("Dataframe is not sorted by image_file, can not write to Zarr")
	inputTifFiles = [ x.decode("utf-8") if isinstance(x, bytes) else x for x in df["image_file"] ]
	tifFilesBasename = [ os.path.basename(f) for f in inputTifFiles ]
	if not all(tifFilesBasename[i] <= tifFilesBasename[i+1] for i in range(len(tifFilesBasename)-1)):
		raise ValueError("Dataframe is not sorted by image_file, can not write to Zarr")
	else:
		print(tifFilesBasename[0], tifFilesBasename[-1])
	inputTifFiles = [ os.path.join(inputDir, f.lstrip("/")) for f in inputTifFiles ]
	INPUT_TIFF_BYTES[zarrArray.basename] = sum(os.path.getsize(f) for f in inputTifFiles)
	# Write images
	if ARG.j == 0:
		for i, f in enumerate(inputTifFiles):
			start = timer()
			tiffImageToArrayIndex(f, zarrArray, i)
			if ARG.verbose and (i % 100 == 0 or i == len(inputTifFiles) - 1):
				print(f"Written frame {i+1}/{len(inputTifFiles)} ({os.path.basename(f)}) in {timer()-start:.3f}s")
	else:
		results = []
		with Pool(ARG.j, initializer=_init_writer_worker, initargs=(Lock(),)) as pool:
			for i, f in enumerate(inputTifFiles):
				res = pool.apply_async(tiffImageToArrayIndex_worker, args=(f, zarrArray, i), callback=progress_callback)
				results.append(res)
			# Wait for all tasks to complete and check for exceptions
			pool.close()
			pool.join()
		errors = []
		for res in results:
			r = res.get()  # This will re-raise any exception from the worker
			if r["error"] is not None:
				errors.append((r["tiffFile"], r["kIndex"], r["error"]))
		if len(errors) > 0:
			print(colored(f"Encountered {len(errors)} errors during TIFF processing:", "red"))
			for tiffFile, kIndex, error in errors:
				print(colored(f"Error processing '{tiffFile}' at index {kIndex}:\n{error}", "red"))
			raise RuntimeError(f"{len(errors)} errors occurred during TIFF processing. See above for details.")
	if ARG.verbose:
		print(colored(f"Zarrarray written to {zarrArray.path} with shape {zarrArray.shape} and dtype {zarrArray.dtype}", "green"))


# Function to scan directory for TIFF files and create a DataFrame
def scanForTiffFiles(directory, exclude_files):
	tiff_files = []
	for root, _, files in os.walk(directory):
		for file in files:
			if file.lower().endswith('.tiff') or file.lower().endswith('.tif'):
				filepath = os.path.join(root, file)
				relative_path = os.path.relpath(filepath, directory)
				if relative_path not in exclude_files:
					tiff_files.append(relative_path)
	tiff_files.sort()
	return pd.DataFrame({'image_file': tiff_files})

def sanitize_for_json(obj):
	"""
	Convert metadata to JSON-compatible values.

	Recursively collapse singleton arrays/lists/tuples to their underlying
	value. Preserve empty sequences and sequences with multiple elements.
	- numpy scalars -> Python scalars
	- numpy arrays	-> lists
	- dict/list/tuple -> recurse
	- leave Python scalars/None/str as-is
	"""
	# NumPy arrays -> lists (be careful with very large arrays; attrs should stay small)
	if isinstance(obj, np.ndarray):
		return sanitize_for_json(obj.tolist())
	# NumPy scalars -> Python scalars
	if isinstance(obj, np.generic):
		return sanitize_for_json(obj.item())
	if isinstance(obj, (bytes, bytearray)):
		try:
			return obj.decode("utf-8")
		except UnicodeDecodeError:
			return obj.decode("utf-8", errors="replace")
	# Mappings -> recurse
	if isinstance(obj, dict):
		return {str(k): sanitize_for_json(v) for k, v in obj.items()}
	# Sequences -> recurse element-wise
	if isinstance(obj, (list, tuple)):
		if len(obj) == 1:
			return sanitize_for_json(obj[0])
		return [sanitize_for_json(value) for value in obj]
	# Everything else: assume already JSON-safe (str, int, float, bool, None)
	return obj

def h5_group_attributes(h5, verbose=True):
	"""JSON-safe attributes of an HDF5 group: its attributes and its scalar datasets."""
	attributes = {}
	for entry in list(h5):
		if isinstance(h5[entry], h5py.Dataset) and h5[entry].shape == ():
			# Scalar datasets are stored as attributes to preserve their value
			val = h5[entry][()]
			if isinstance(val, (bytes, np.bytes_)):
				val = val.decode("utf-8")
			if isinstance(val, np.generic):
				val = val.item()
			attributes[entry] = val
	attributes.update(dict(h5.attrs))  # attributes of the group win over scalar datasets of the same name
	sanitized = {}
	for k, v in attributes.items():
		if verbose:
			print(f"Processing attribute {h5.name}@{k}", flush=True)
		try:
			value = sanitize_for_json(v)
			json.dumps(value)
		except (TypeError, ValueError, OverflowError) as exc:
			raise ValueError(f"Cannot convert HDF5 attribute {h5.name}@{k} to JSON: type={type(v).__name__}, value={v!r}") from exc
		sanitized[k] = value
	return sanitized


def copy_h5_arrays_into_group(
	h5,
	zarr_params_group: zarr.Group,
	compressor = None,
	prefer_src_chunks: bool = True,
	chunk_bytes: int = 64 * 1024 * 1024,  # ~64 MB streaming slabs
	verbose: bool = True,
) -> None:
	"""
	Copy all HDF5 N-D datasets and subgroups of `h5` into the given Zarr group, which must already
	carry the attributes of `h5` (h5_group_attributes). Subgroups are created with their attributes,
	so that no group metadata is rewritten afterwards (in a zip store every rewrite adds an entry).
	Preserves shapes, dtypes, attributes and subgroup structure.
	"""
	for entry in list(h5):
		if verbose:
			print(f"Processing HDF5 entry: {h5.name}/{entry}  type={type(h5[entry])}")
		if isinstance(h5[entry], h5py.Dataset):
			if h5[entry].shape == ():
				continue  # scalar dataset, an attribute of the group (h5_group_attributes)
			elif h5[entry].dtype == np.dtype("O") or h5[entry].dtype.kind == 'S':
				# Handle object/string arrays by converting to UTF-8 strings in Zarr
				str_array = h5[entry][:]
				if str_array.flatten().size > 0:
					val = str_array.flatten()[0]
					if isinstance(val, (bytes, np.bytes_)):
						str_array = np.char.decode(str_array.astype('S'), 'utf-8', errors='replace')
				za = zarr_params_group.create_array(
					entry,
					shape=str_array.shape,
					dtype="string",
				)
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
					print(f"[dataset] {h5.name}/{entry}  shape={h5[entry].shape}	dtype={h5[entry].dtype}  -> zarr dtype={za.dtype}  chunks={za.chunks}")
		elif isinstance(h5[entry], h5py.Group):
			subgroup = zarr_params_group.create_group(entry, attributes=h5_group_attributes(h5[entry], verbose=verbose))
			copy_h5_arrays_into_group(h5[entry], subgroup, compressor=compressor, verbose=verbose)
		else:
			raise ValueError(f"HDF5 entry '{entry}' is neither a group nor a dataset, cannot copy.")
	if verbose:
		print("Finished copying HDF5 arrays into Zarr group:", zarr_params_group.path)


def main(argv=None):
	"""Run the conversion; argv defaults to sys.argv[1:]. Returns the exit status."""
	global ARG  # the functions above read the arguments from ARG
	parser = build_parser()
	start_time = time.time()
	_phase_time[0] = start_time
	INPUT_TIFF_BYTES.clear()  # sizes for the report of this run
	argv = sys.argv[1:] if argv is None else list(argv)
	script_name = sys.argv[0]
	script_params = " ".join(argv)

	print(f"START {script_name} {script_params}")


	try:
		_out = io.StringIO()
		_err = io.StringIO()
		with redirect_stdout(_out):
			with redirect_stderr(_err):
				ARG = parser.parse_args(argv or ['--help'])
	except SystemExit as err:
		print("Program to produce Zarr file from raw TIFF structure. Usage:")
		#possibly parser.print_help()
		sys.stderr.write(_err.getvalue())
		sys.stdout.write(_out.getvalue())
		return err.code
	#ARG = parser.parse_args()
	#ARG = parser.parse_args(["/home/user/desy_example_data/syn0101_17L_Ti_12w_000_nexus.h5", "/tmp/desy", "--force"])

	ARG.inputh5 = str(Path(ARG.inputh5).resolve(strict=True))
	ARG.outputZarr = str(Path(ARG.outputZarr).resolve(strict=False))

	#Set up threading
	# Default thread count: a quarter of the CPUs this process may run on (a SLURM job's allocation, not
	# the whole node), kept between DEFAULT_MIN_THREADS and DEFAULT_MAX_THREADS. Measured on Maxwell with
	# 5120x3840 frames: reading the TIFFs limits the speed from about 16-32 threads on, 64 threads were
	# slower; on 40-CPU nodes HTJ2K needs 16 threads (CPU-bound). zarr runs the codecs in a pool of at
	# most 32 threads, and every thread holds frames in memory.
	DEFAULT_MIN_THREADS, DEFAULT_MAX_THREADS = 16, 32
	try:
		AVAILABLE_CPUS = len(os.sched_getaffinity(0))
	except AttributeError:  # not on Linux
		AVAILABLE_CPUS = mp.cpu_count()
	if ARG.j < 0:
		ARG.j = min(AVAILABLE_CPUS, max(DEFAULT_MIN_THREADS, min(AVAILABLE_CPUS // 4, DEFAULT_MAX_THREADS)))
		print("Starting threadpool of %d threads (%d CPUs available; default a quarter of them, %d to %d)" % (ARG.j, AVAILABLE_CPUS, DEFAULT_MIN_THREADS, DEFAULT_MAX_THREADS))
	elif ARG.j == 0:
		print("No threading will be used ARG.j=0.")
	else:
		print("Starting threadpool of %d threads, %d CPUs available" % (ARG.j, AVAILABLE_CPUS))


	#Parse input data
	if ARG.raw_dir is not None:
		inputDir = ARG.raw_dir
	else:
		inputDir = os.path.dirname(os.path.realpath(ARG.inputh5))

	# Check if the Zarr file already exists and handle based on force flag
	if os.path.exists(ARG.outputZarr):
		if ARG.force:
			if os.path.isfile(ARG.outputZarr):
				os.remove(ARG.outputZarr)
			else:
				shutil.rmtree(ARG.outputZarr)
		else:
			raise IOError(f"File {ARG.outputZarr} exists, use --force to overwrite")

	report_phase("setup")
	df = PETRA.scanDataset(ARG.inputh5, includeCurrent=False)
	experimentInfo = PETRA.getExperimentInfo(ARG.inputh5)
	export = {}

	dtype = None
	#Test if df is not empty and determine numpy type of image_file TIFF and its dimensions
	if df.empty:
		raise ValueError("Dataframe is empty, no data to write to Zarr file %s" % (os.path.realpath(ARG.outputZarr)))
	else:
		tiff_file_str = df["image_file"].iloc[0]#In some versions this is string but in some bytes
		if isinstance(tiff_file_str, bytes):
			tiff_file_str = tiff_file_str.decode("utf-8")
		tiff_file_path = os.path.join(inputDir, tiff_file_str.lstrip("/"))
		if not os.path.exists(tiff_file_path):
			raise FileNotFoundError(f"TIFF file {tiff_file_path} does not exist, cannot determine image dimensions and dtype.")
		img_array = read_tiff(tiff_file_path)
		dimy, dimx = img_array.shape
		dtype = img_array.dtype
		print(f"Determined image dimensions: {dimx}x{dimy}, dtype: {dtype}")

	if dtype is None:
		raise ValueError("Could not determine dtype of TIFF images, cannot proceed.")

	if dtype != np.uint16:
		raise ValueError(f"Warning: TIFF images are of type {dtype}, expected uint16. This may lead to unexpected behavior.")

	if ARG.compression in ["zfp", "sz3"]:
		# For integer types smaller then int32, zfp requires at least 32 bits, so we need to upcast to int32
		if np.issubdtype(dtype, np.integer) and dtype.itemsize < 4:
			outtype = np.uint32
		else:
			outtype = dtype
	else:
		outtype = dtype  # Keep the same dtype as the original TIFF images

	codec_kwargs = {}
	if ARG.compression == "jpeg2k":
		codec_kwargs["bitspersample"] = 12 # Set bit depth to 12 for JPEG2000 which matches the original TIFF bit depth
		codec_kwargs["colorspace"] = "GRAY" # Set color space to GRAY for grayscale images
		codec_kwargs["mct"] = False # Disable multi-component transform for grayscale images
		codec_kwargs["numthreads"] = 1 #Better having here 1 since we are using multiprocessing to write the images
	elif ARG.compression == "htj2k":
		codec_kwargs["rgb"] = False # Set RGB to False for grayscale images
	elif ARG.compression == "avif":
		codec_kwargs["bitspersample"] = 12 # Set bit depth to 12 for AVIF which matches the original TIFF bit depth
		codec_kwargs["pixelformat"] = "YUV400 " #AVIF_PIXEL_FORMAT_YUV400
		#codec_kwargs["matrix"] = "IDENTITY" #AVIF_MATRIX_COEFFICIENTS_IDENTITY
		codec_kwargs["numthreads"] = 1 #Better having here 1 since we are using multiprocessing to write the images
	elif ARG.compression == "jpegxl":
		codec_kwargs["bitspersample"] = 12
		codec_kwargs["photometric"] = "GRAY"
		codec_kwargs["numthreads"] = 1 #Better having here 1 since we are using multiprocessing to write the images
	elif ARG.compression == "jpegxr":
		codec_kwargs["photometric"] = "GRAY"
		codec_kwargs["hasalpha"] = False

	dark = df.loc[df["image_key"] == 2]
	white = df.loc[df["image_key"] == 1]
	scan = df.loc[df["image_key"] == 0]

	if ARG.fix_corrupted_h5 and scan.empty:
		# before the arrays are created, so that img gets the number of the found images
		print("HDF5 file is corrupted. Scanning directory for TIFF files to create img.den...")
		exclude_files = set(dark["image_file"]).union(set(white["image_file"]))
		scan = scanForTiffFiles(inputDir, exclude_files)

	df_json = json.loads(df.to_json(orient="split", date_format="iso"))

	chunk_shape = (1, dimy, dimx)  # Chunking by individual images
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
		store = ZARZS.ZipStore(ARG.outputZarr, mode="w")  # threads write their chunks in parallel
	else:
		store = ARG.outputZarr

	zarr_top_level = zarr.open_group(
				store=store,
				mode='w',
				attributes=experimentInfo_sanitized,
				zarr_format=2 if ARG.v2 else 3,
			)

	codec = ZAR.get_compressor(ARG.compression, ARG.clevel, zarrv2=ARG.v2, dtype=outtype, **codec_kwargs)
	try:
		if not ARG.v2:
			# In Zarr v3 we can set the compressor at the group level and it will be inherited by all arrays
			if ARG.compression in ['avif', 'jpeg2k', 'htj2k', 'jpegxl', 'jpegxr', 'sz3', 'zfp']:
				zarr_array_ref = zarr_top_level.create_array('ref', shape=(ref_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, serializer = codec[0])
				zarr_array_dar = zarr_top_level.create_array('dar', shape=(dar_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, serializer = codec[0])
				zarr_array_img = zarr_top_level.create_array('img', shape=(img_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, serializer = codec[0])
			else:
				zarr_array_ref = zarr_top_level.create_array('ref', shape=(ref_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, compressors=codec)
				zarr_array_dar = zarr_top_level.create_array('dar', shape=(dar_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, compressors=codec)
				zarr_array_img = zarr_top_level.create_array('img', shape=(img_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, compressors=codec)
		else:
			# In Zarr v2 we have to set the compressor on each array individually
			zarr_array_ref = zarr.open_array(store=store, path='ref', shape=(ref_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, compressor=codec, zarr_format=2, mode='w')
			zarr_array_dar = zarr.open_array(store=store, path='dar', shape=(dar_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, compressor=codec, zarr_format=2, mode='w')
			zarr_array_img = zarr.open_array(store=store, path='img', shape=(img_count, dimy, dimx), dtype=outtype, chunks=chunk_shape, compressor=codec, zarr_format=2, mode='w')
	except zipfile.BadZipFile as e:
		print(f"Zarr store {ARG.outputZarr} is not a valid zip file or is corrupted. This can happen when two processes try to write to the same zip file simultaneously. Please delete the file and try again.")
		print(f"Error details: {e}")
		return 1

	h5 = h5py.File(ARG.inputh5, "r")
	if "entry" in h5:
		h5 = h5["entry"]
	# df_json and the HDF5 attributes (these win, as before) are written once, with the group
	zarr_params = zarr_top_level.create_group(name="params", attributes={**df_json, **h5_group_attributes(h5, verbose=ARG.verbose)})
	copy_h5_arrays_into_group(h5, zarr_params, verbose=ARG.verbose)
	report_phase("HDF5 scan table, arrays and metadata")

	writeZarrArray(dark, zarr_array_dar, inputDir)
	report_phase(f"dar ({dar_count} frames)")
	writeZarrArray(white, zarr_array_ref, inputDir)
	report_phase(f"ref ({ref_count} frames)")

	writeZarrArray(scan, zarr_array_img, inputDir)
	report_phase(f"img ({img_count} frames)")

	if isinstance(store, zarr.storage.ZipStore):
		store.close()
		report_phase("closing the zip")

	h5.file.close()
	report_sizes(ARG.outputZarr, ARG.compression, ARG.clevel)

	elapsed_seconds = int(time.time() - start_time)

	hours = elapsed_seconds // 3600
	minutes = (elapsed_seconds % 3600) // 60
	seconds = elapsed_seconds % 60

	print(f"END {script_name} in {hours:02d}:{minutes:02d}:{seconds:02d}s")

	return 0


if __name__ == "__main__":
	sys.exit(main())
