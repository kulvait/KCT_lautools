#!/usr/bin/env python
"""
2023-2026

@author: Vojtech Kulvait
@license: GNU GPL v3

This script removes transient bright pixels (e.g. hot pixels or cosmic ray hits) 
from a 3D image stack (time or depth series) stored in the DEN format. It works 
by applying an iterative median filter to each frame, identifying outlier pixels 
that deviate from their local neighborhood by more than a specified number of 
standard deviations, and replacing them with the corresponding median value.

Hot pixel / zinger removal tool for 2D/3D image stacks.
	
	INPUT / OUTPUT FORMAT
	---------------------
	The file format is inferred from the presence of ':' in the path.
	
	Zarr format:
	  If the argument contains a colon ':' it is interpreted as:
	
	      /path/to/zarr_container:/path/inside/zarr
	
	  Examples:
	      data.zarr:/volume
	      /data/experiment.zarr:/group/subgroup/array
	
	  - Left side  → filesystem path to Zarr store
	  - Right side → internal array path inside the store
	
	  Zarr storage type is inferred from the left-side suffix:
	
	      *.zarr        → directory-based Zarr store
	      *.zip, *.zar  → zipped Zarr store (read/write as archive)
	
	DEN format:
	  If the argument does NOT contain ':' it is interpreted as a DEN file:
	
	      /path/to/file.den
	
	Notes:
	- Zarr supports parallel writes (no locking required).
	- DEN writes are synchronized (thread-locked).
	
	FILTERING OVERVIEW
	------------------
	Filtering is based on comparing each pixel to a local filtered estimate.
	
	Two filtering approaches are commonly used:
	
	1) Median filtering:
	   Uses a local median as the reference value.
	   This is the most common and robust approach for general-purpose
	   outlier (hot pixel) removal.
	
	2) Zinger filtering:
	   Uses a directional kernel instead of a full neighborhood.
	   Designed for impulse-like artifacts (so-called "zingers"),
	   as described in:
	   https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-12-17849
	
	   This approach is also used in the ALgotom framework, which uses relative thresholding and suggests range [0.05, 0.1].
	
	Thresholding strategies
	-----------------------
	Outlier detection is based on deviation from the filtered estimate.
	
	1) Absolute thresholding:
	   |pixel - filtered|
	
	   Commonly used with median filtering.
	   Best suited for:
	     - detector artifacts
	     - low-intensity regions
	
	2) Relative thresholding:
	   |pixel - filtered| / |filtered|
	
	   Used in ALgotom and similar methods.
	   Best suited for:
	     - high dynamic range data
	     - intensity-dependent noise
	
	Design note:
	------------
	Traditionally:
	- Median filtering → absolute thresholds
	- Zinger / ALgotom methods → relative thresholds
	
	Here, both thresholding strategies are decoupled from the filtering
	method. This allows:
	- applying relative thresholding to median filtering
	- reproducing ALgotom behavior when desired
	- combining absolute and relative criteria for robustness
	
	Rule of thumb:
	--------------
	- Median + absolute std threshold → general-purpose cleanup
	- Zinger + relative threshold → high dynamic range / Algotom-style correction
	- Median + relative threshold → intensity-dependent noise scenarios
	
"""

import argparse
from denpy import DEN 
from denpy import ZAR
import zarr
import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage import convolve
from multiprocessing.dummy import Process, Lock, Pool
import multiprocessing
import time
import traceback
import warnings
from lautools import remove_hot_pixels



# Global write_lock for workers
write_lock = None

def init_worker(l):
	global write_lock
	write_lock = l 

# Lock is protected by finally clause to avoid deadlocks
def processFrame(ARG, k):
	try:
		# Read (no lock unless DEN requires one)
		if inputIsZarr:
			f = inputArray[k]
		else:
			f = DEN.getFrame(ARG.inputFile, k)
		f_filtered, f_corrupted_pixel, corrected_pixels = remove_hot_pixels(f, iterations=ARG.iterations, filter_size=ARG.filter_size, correct_threshold_abs_sigma=ARG.filter_threshold, zinger_algorithm=ARG.zinger_algorithm)
		# Write (locked)
		if outputIsZarr == True:
			outputArray[k] = f_filtered.astype(outputType)
		if ARG.output_mask is not None and maskIsZarr == True:
			maskOutputArray[k] = f_corrupted_pixel.astype(np.uint8)
		# --- Determine if we need synchronization ---
		needs_sync = (not outputIsZarr) or (
			ARG.output_mask is not None and not maskIsZarr
		)
		if write_lock and needs_sync:
			write_lock.acquire()
		try:
			if not outputIsZarr:
				DEN.writeFrame(ARG.outputFile, k, f_filtered, force=True)
			if ARG.output_mask is not None and not maskIsZarr:
				DEN.writeFrame(ARG.output_mask, k, f_corrupted_pixel.astype(np.uint8), force=True)
		finally:
			if write_lock:
				write_lock.release()
		return {"k": k, "pixels": corrected_pixels, "error": None}
	except Exception:
		return {"k": k, "pixels": 0, "error": traceback.format_exc()}

class FakeAsyncResult:
	def __init__(self, value):
		self._value = value
	def get(self):
		return self._value

def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	    description="Hot pixel / zinger removal tool for 2D/3D image stacks."
	)
	
	parser.add_argument("inputFile", help="Input array to filter, DEN or Zarr format. For Zarr use /path/to/store:/array/path, without ':' it is interpreted as DEN file.")
	parser.add_argument("outputFile", help="Output array, DEN or Zarr format. For Zarr use /path/to/store:/array/path, without ':' it is interpreted as DEN file.")
	parser.add_argument("--output-mask", help="Output array for the binary mask of detected outliers. Output is np.uint8 array where 1 indicates processed pixel while 0 is unchanged pixel. For Zarr use /path/to/store:/array/path, without ':' it is interpreted as DEN file.", default=None)
	parser.add_argument("--filter-size", type=int, default=5, help="Size parameter to the scipy.ndimage.median_filter or Zinger kernel size.")
	parser.add_argument("--filter-threshold-abs", type=float, default=3.0, help="Absolute threshold to substitute data.")
	parser.add_argument("--filter-threshold-rel", type=float, default=None, help="Relative threshold to substitute data.")
	parser.add_argument("--filter-threshold-abs-sigma", type=float, default=None, help="Number of standard deviations of absolute difference to substitute data.")
	parser.add_argument("--filter-threshold-rel-sigma", type=float, default=None, help="Number of standard deviations of relative difference to substitute data.")
	parser.add_argument("--filter-iterations", type=int, default=1, help="Number of iterations.")
	parser.add_argument("--zinger-algorithm", help="Use Zinger correction algorithm described in https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-12-17849", action="store_true")
	parser.add_argument('--zarr-compression', type=str,
						choices=['none', 'zstd', 'lz4', 'gzip', 'blosc', 'blosc-blosclz',
								 'blosc-lz4', 'blosc-lz4hc', 'blosc-snappy', 'blosc-zlib', 'blosc-zstd'],
						default='blosc-zstd',
						help="Compression type if output Zarr is used (default: blosc-zstd).")
	parser.add_argument('--zarr-clevel', type=int, default=5,
						help="Compression level if output Zarr is used (default: 5).")
	parser.add_argument("-j","--threads", default=-1, type=int, help="Number of threads to use. [defaults to -1 which is mp.cpu_count(), 0 without threading]", dest="j")
	parser.add_argument("--keep-input-dtype", help="Keep input data type in output, otherwise output is float32.", action="store_true")
	parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
	ARG = parser.parse_args()
	
	inputIsZarr = False
	outputIsZarr = False
	maskIsZarr = False
	
	if ":" in ARG.inputFile:
		zarTokens = ARG.inputFile.split(":", 1)
		zarStorePath = zarTokens[0]
		zarPath = zarTokens[1]
		if zarPath == "/":
			zarPath = ""
		if zarStorePath.endswith(".zip") or zarStorePath.endswith(".zar"):
			zarInputStore = zarr.storage.ZipStore(zarStorePath)
		else:
			zarInputStore = zarr.storage.LocalStore(zarStorePath)
		print(f"Opening Zarr array from store '{zarStorePath}' with path '{zarPath}'")
		inputArray = zarr.open_array(zarInputStore, mode="r", path=zarPath)
		inputIsZarr = True
		if len(inputArray.shape) != 3:
			raise ValueError("Input Zarr array must be 3D, but got shape %s"%(inputArray.shape,))
		zdim, ydim, xdim = inputArray.shape
		dimspec = (xdim, ydim, zdim)
		inputType = inputArray.dtype
	else:
		header = DEN.readHeader(ARG.inputFile)
		dimspec = header["dimspec"]
		xdim = np.uint64(dimspec[0])
		ydim = np.uint64(dimspec[1])
		zdim = np.uint64(dimspec[2])
		inputType = header["type"]
	
	if ARG.keep_input_dtype:
		outputType = inputType
	else:
		outputType = np.float32
	
	frameSize = xdim * ydim
	totalSize = frameSize * zdim
	
	print(f"Starting processing file '{ARG.inputFile}' containing {zdim} frames of size {xdim}x{ydim} to produce '{ARG.outputFile}'")
	print(f"Filter size: {ARG.filter_size}, Threshold: {ARG.filter_threshold}, Iterations: {ARG.iterations}")
	
	if ARG.j < 0:
		ARG.j = multiprocessing.cpu_count()
		print("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()"%(ARG.j))
	elif ARG.j == 0:
		print("No threading will be used ARG.j=0.")
	else:
		print("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()=%d"%(ARG.j, multiprocessing.cpu_count()))
	
	if ":" in ARG.outputFile:
		zarTokens = ARG.outputFile.split(":", 1)
		zarStorePath = zarTokens[0]
		zarPath = zarTokens[1]
		if zarPath == "/":
			zarPath = ""
		if zarStorePath.endswith(".zip") or zarStorePath.endswith(".zar"):
			zarOutputStore = zarr.storage.ZipStore(zarStorePath)
		else:
			zarOutputStore = zarr.storage.LocalStore(zarStorePath)
		print(f"Creating Zarr array in store '{zarStorePath}' with path '{zarPath}'")
		codec = ZAR.get_compressor(ARG.zarr_compression, clevel=ARG.zarr_clevel, zarrv2=False, dtype=outputType)
		outputArray = zarr.create_array(
				store=zarOutputStore,
				shape=(zdim, ydim, xdim),
				chunks=(1, ydim, xdim),
				dtype=outputType,
				compressors=codec,
				zarr_format=3,
				overwrite=True,
			)
		outputIsZarr = True
	else:
		DEN.writeEmptyDEN(ARG.outputFile, dimspec, force=True)
	
	if ARG.output_mask is not None:
		if ":" in ARG.output_mask:
			maskZarTokens = ARG.output_mask.split(":", 1)
			maskZarStorePath = maskZarTokens[0]
			maskZarPath = maskZarTokens[1]
			if maskZarPath == "/":
				maskZarPath = ""
			if maskZarStorePath.endswith(".zip") or maskZarStorePath.endswith(".zar"):
				maskZarOutputStore = zarr.storage.ZipStore(maskZarStorePath)
			else:
				maskZarOutputStore = zarr.storage.LocalStore(maskZarStorePath)
			print(f"Creating Zarr array for mask in store '{maskZarStorePath}' with path '{maskZarPath}'")
			codec = ZAR.get_compressor(ARG.zarr_compression, clevel=ARG.zarr_clevel, zarrv2=False, dtype=np.uint8)
			maskOutputArray = zarr.create_array(
					store=maskZarOutputStore,
					shape=(zdim, ydim, xdim),
					chunks=(1, ydim, xdim),
					dtype=np.uint8,
					compressors=codec,
					zarr_format=3,
					overwrite=True,
				)
			maskIsZarr = True
	else:
		DEN.writeEmptyDEN(ARG.output_mask, dimspec, force=True, elementtype=np.dtype('<u1'))
	results = []
	if ARG.j == 0:
		for k in range(zdim):
			res = processFrame(ARG, k)
			results.append(FakeAsyncResult(res))
	else:
		lock = Lock()
		tp = Pool(processes=ARG.j, initializer=init_worker, initargs=(lock,))
		for k in range(zdim):
			res = tp.apply_async(processFrame, args=(ARG, k))
			results.append(res)
		tp.close()
		tp.join()
	
	errors = []
	total_pixels_corrected = 0
	total_frames_sucessful = 0
	for result in results:
		r = result.get()
		k = r["k"]
		if r["error"] is not None:
			errors.append((r["k"], r["error"]))
		else:
			total_pixels_corrected += r["pixels"]
			total_frames_sucessful += 1
			if ARG.verbose:
				corrected_fraction = r["pixels"] / frameSize
				print("Frame %d: %d pixels corrected, fraction: %.2f%%"%(r["k"], r["pixels"], corrected_fraction*100))
	
	if len(errors) > 0:
		print("The following frames raised exceptions:")
		for (k, error) in errors:
			print(f"Frame {k} exception:\n{error}")
		print(f"{len(errors)} frames raised exceptions.")
		if total_frames_sucessful > 0:
			total_pixels_fraction = total_pixels_corrected / (total_frames_sucessful*frameSize)
			print("From total %d frames corrected in '%s' with %d pixels corrected, fraction: %.2f%%"%(total_frames_sucessful, ARG.outputFile, total_pixels_corrected, total_pixels_fraction*100))
		else:
			total_pixels_fraction = 0.0
			print("Processing failed for all frames, no pixels corrected in '%s'."%(ARG.outputFile))
	else:
		total_pixels_fraction = total_pixels_corrected / totalSize
		print("Sucessfully created '%s' with %d pixels corrected, fraction: %.2f%%"%(ARG.outputFile, total_pixels_corrected, total_pixels_fraction*100))
	
	print("END destar.py")
