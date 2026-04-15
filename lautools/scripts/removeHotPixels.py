#!/usr/bin/env python
"""
This script removes transient bright pixels (e.g. hot pixels or cosmic ray hits) 
from a 3D image stack (time or depth series) stored in the DEN format. It works 
by applying an iterative median filter to each frame, identifying outlier pixels 
that deviate from their local neighborhood by more than a specified number of 
standard deviations, and replacing them with the corresponding median value.

Created: 03/2026, use core functionality of destar.py from syscripts

@author: Vojtech Kulvait
@license: GNU GPL v3


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
	
		  *.zarr		→ directory-based Zarr store
		  *.zip, *.zar	→ zipped Zarr store (read/write as archive)
	
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
from lautools import remove_hot_pixels
from denpy import DEN 
from denpy import ZAR
import zarr
import numpy as np
from multiprocessing.dummy import Process, Lock, Pool
import multiprocessing
import time
import sys
import traceback
import logging
import asyncio


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


# Global variables for controling I/O and synchronization in workers
inputIsZarr = False
outputIsZarr = False
maskIsZarr = False
inputArray = None
outputArray = None
maskOutputArray = None
outputType = np.float32

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
		# Signature is def remove_hot_pixels(frame, iterations, filter_size, correct_threshold_abs_sigma=3.0, correct_threshold_abs=None, correct_threshold_rel_sigma=None, correct_threshold_rel=None, zinger_algorithm=False, filter_large_components=False, large_component_minpixcount=10, epsilon=1e-6):
		f_filtered, f_corrupted_pixel, corrected_pixels = remove_hot_pixels(frame=f, iterations=ARG.filter_iterations, filter_size=ARG.filter_size, correct_threshold_abs_sigma=ARG.filter_threshold_abs_sigma, correct_threshold_abs=ARG.filter_threshold_abs, correct_threshold_rel_sigma=ARG.filter_threshold_rel_sigma, correct_threshold_rel=ARG.filter_threshold_rel, correct_threshold_local=ARG.filter_threshold_loc, correct_threshold_local_sigma=ARG.filter_threshold_loc_sigma, zinger_algorithm=ARG.zinger_algorithm)
		f_filtered = f_filtered.astype(outputType)
		f_corrupted_pixel = f_corrupted_pixel.astype(np.uint8)
		# Write (locked)
		if outputIsZarr == True:
			outputArray[k] = f_filtered
		if ARG.output_mask is not None and maskIsZarr == True:
			maskOutputArray[k] = f_corrupted_pixel
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
		if corrected_pixels == 0:
			log.info("Frame %d in '%s' has no detected outliers."%(k, ARG.outputFile))
		return {"k": k, "pixels": corrected_pixels, "error": None}
	except Exception:
		return {"k": k, "pixels": 0, "error": traceback.format_exc()}

class FakeAsyncResult:
	def __init__(self, value):
		self._value = value
	def get(self):
		return self._value

def validate_thresholds(ARG):
	# Check if at least one threshold is set
	if not any([ARG.filter_threshold_abs, ARG.filter_threshold_rel,
				ARG.filter_threshold_abs_sigma, ARG.filter_threshold_rel_sigma, ARG.filter_threshold_loc, ARG.filter_threshold_loc_sigma]):
		log.warning("No threshold specified, defaulting to --filter-threshold-abs-sigma 3.0")
		ARG.filter_threshold_abs_sigma = 3.0

def main():
	start_time = time.time()
	log.info("START removeHotPixels %s"%(" ".join(sys.argv[1:])))
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
		description="Hot pixel / zinger removal tool for 2D/3D image stacks.", epilog="Example usage:\n\n"
		"  removeHotPixels.py input.den output.den --filter-size 5 --filter-threshold-abs-sigma 3 --filter-iterations 1\n"
		"  removeHotPixels.py input.zarr:/volume output.zarr:/volume --filter-size 5 --filter-threshold-rel 0.1 --zinger-algorithm\n"
		"  removeHotPixels.py input.den output.zarr:/volume --filter-size 5 --filter-threshold-rel-sigma 3\n"
	)
	
	parser.add_argument("inputFile", help="Input array to filter, DEN or Zarr format. For Zarr use /path/to/store:/array/path, without ':' it is interpreted as DEN file.")
	parser.add_argument("outputFile", help="Output array, DEN or Zarr format. For Zarr use /path/to/store:/array/path, without ':' it is interpreted as DEN file.")
	parser.add_argument("--output-mask", help="Output array for the binary mask of detected outliers. Output is np.uint8 array where 1 indicates processed pixel while 0 is unchanged pixel. For Zarr use /path/to/store:/array/path, without ':' it is interpreted as DEN file.", default=None)
	parser.add_argument("--filter-size", type=int, default=5, help="Size parameter to the scipy.ndimage.median_filter or Zinger kernel size.")
	parser.add_argument("--filter-threshold-abs", type=float, default=None, help="Absolute threshold to substitute data.")
	parser.add_argument("--filter-threshold-rel", type=float, default=None, help="Relative threshold to substitute data.")
	parser.add_argument("--filter-threshold-loc", type=float, default=None, help="Local threshold of median weighted differences")
	parser.add_argument("--filter-threshold-abs-sigma", type=float, default=None, help="Number of standard deviations of absolute difference to substitute data.")
	parser.add_argument("--filter-threshold-rel-sigma", type=float, default=None, help="Number of standard deviations of relative difference to substitute data.")
	parser.add_argument("--filter-threshold-loc-sigma", type=float, default=None, help="Number of standard deviations of local median weighted differences to substitute data.")
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
	validate_thresholds(ARG)
	
	global inputArray
	global inputIsZarr
	global outputArray
	global outputIsZarr
	global maskOutputArray
	global maskIsZarr
	global outputType
	
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
		log.info(f"Opening Zarr array from store '{zarStorePath}' with path '{zarPath}'")
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
		xdim = int(dimspec[0])
		ydim = int(dimspec[1])
		zdim = int(dimspec[2])
		inputType = header["type"]
	
	if ARG.keep_input_dtype:
		outputType = inputType
	else:
		outputType = np.float32
	
	frameSize = xdim * ydim
	totalSize = frameSize * zdim
	
	outputZarStorePath = None
	if ":" in ARG.outputFile:
		zarTokens = ARG.outputFile.split(":", 1)
		outputZarStorePath = zarTokens[0]
		zarPath = zarTokens[1]
		if zarPath == "/":
			zarPath = ""
		if outputZarStorePath.endswith(".zip") or outputZarStorePath.endswith(".zar"):
			zarOutputStore = zarr.storage.ZipStore(outputZarStorePath, mode="a")
			zarOutputStore._sync_open() # After this fix might be removed https://github.com/zarr-developers/zarr-python/issues/3846
		else:
			zarOutputStore = zarr.storage.LocalStore(outputZarStorePath, mode="a")
		outputIsZarr = True
		try:
			outputArray = zarr.open_array(zarOutputStore, mode="r+", path=zarPath)
			if outputArray.shape != (zdim, ydim, xdim) or outputArray.dtype != outputType:
				asyncio.run(zarOutputStore.delete(zarPath))
				log.warning(f"Existing Zarr array in store '{outputZarStorePath}' with path '{zarPath}' has incompatible shape {outputArray.shape} or dtype {outputArray.dtype}, expected shape {(zdim, ydim, xdim)} and dtype {outputType}. It will be overwritten and for some stores (e.g. ZipStore) this leads to NotImplementedError")	
				outputArrayExists = False
			else:
				log.info(f"Using existing Zarr array in store '{outputZarStorePath}' with path '{zarPath}'")
				outputArrayExists = True
		except zarr.errors.ArrayNotFoundError:
			outputArrayExists = False
		if not outputArrayExists:
			log.info(f"Creating Zarr array in store '{outputZarStorePath}' with path '{zarPath}'")
			codec = ZAR.get_compressor(ARG.zarr_compression, clevel=ARG.zarr_clevel, zarrv2=False, dtype=outputType)
			outputArray = zarr.create_array(
					store=zarr.storage.StorePath(zarOutputStore, zarPath),
					shape=(zdim, ydim, xdim),
					chunks=(1, ydim, xdim),
					dtype=outputType,
					compressors=codec,
					zarr_format=3,
					overwrite=True,
				)
	else:
		DEN.writeEmptyDEN(ARG.outputFile, dimspec, force=True, elementtype=outputType)
	
	if ARG.output_mask is not None:
		if ":" in ARG.output_mask:
			maskZarTokens = ARG.output_mask.split(":", 1)
			maskZarStorePath = maskZarTokens[0]
			maskZarPath = maskZarTokens[1]
			if outputZarStorePath is not None and maskZarStorePath == outputZarStorePath:
				maskZarOutputStore = zarOutputStore
			else:
				if maskZarPath == "/":
					maskZarPath = ""
				if maskZarStorePath.endswith(".zip") or maskZarStorePath.endswith(".zar"):
					maskZarOutputStore = zarr.storage.ZipStore(maskZarStorePath, mode="a")
					maskZarOutputStore._sync_open() # After this fix might be removed https://github.com/zarr-developers/zarr-python/issues/3846
				else:
					maskZarOutputStore = zarr.storage.LocalStore(maskZarStorePath, mode="a")
			maskIsZarr = True
			log.info(f"Creating Zarr array for mask in store '{maskZarStorePath}' with path '{maskZarPath}'")
			try:
				maskOutputArray = zarr.open_array(maskZarOutputStore, mode="r+", path=maskZarPath)
				if maskOutputArray.shape != (zdim, ydim, xdim) or maskOutputArray.dtype != np.uint8:
					asyncio.run(maskZarOutputStore.delete(maskZarPath))
					maskOutputArrayExists = False
				else:
					log.info(f"Using existing Zarr array for mask in store '{maskZarStorePath}' with path '{maskZarPath}'")
					maskOutputArrayExists = True
			except zarr.errors.ArrayNotFoundError:
				maskOutputArrayExists = False
				log.info(f"No existing Zarr array for mask found in store '{maskZarStorePath}' with path '{maskZarPath}', will create new one.")
			if not maskOutputArrayExists:
					codec = ZAR.get_compressor(ARG.zarr_compression, clevel=ARG.zarr_clevel, zarrv2=False, dtype=np.uint8)
					maskOutputArray = zarr.create_array(
							store=zarr.storage.StorePath(maskZarOutputStore, maskZarPath),
							shape=(zdim, ydim, xdim),
							chunks=(1, ydim, xdim),
							dtype=np.uint8,
							compressors=codec,
							zarr_format=3,
							overwrite=True,
						)
			if maskOutputArray.shape != (zdim, ydim, xdim) or maskOutputArray.dtype != np.uint8:
				raise ValueError(f"Mask Zarr array at '{maskZarStorePath}:{maskZarPath}' has incompatible shape {maskOutputArray.shape} or dtype {maskOutputArray.dtype}, expected shape {(zdim, ydim, xdim)} and dtype uint8.")
		else:
			DEN.writeEmptyDEN(ARG.output_mask, dimspec, force=True, elementtype=np.dtype('<u1'))
	
	log.info(f"Starting processing file '{ARG.inputFile}' containing {zdim} frames of size {xdim}x{ydim} to produce '{ARG.outputFile}'")
	log.info(f"Filter size: {ARG.filter_size}, iterations: {ARG.filter_iterations}, zinger algorithm: {ARG.zinger_algorithm}")
	# Threshold information
	if ARG.filter_threshold_abs is not None:
		log.info(f"Applying absolute threshold with value: {ARG.filter_threshold_abs}")
	if ARG.filter_threshold_rel is not None:
		log.info(f"Applying relative threshold with value: {ARG.filter_threshold_rel}")
	if ARG.filter_threshold_abs_sigma is not None:
		log.info(f"Applying absolute threshold based on {ARG.filter_threshold_abs_sigma} standard deviations.")
	if ARG.filter_threshold_rel_sigma is not None:
		log.info(f"Applying relative threshold based on {ARG.filter_threshold_rel_sigma} standard deviations.")
	if ARG.filter_threshold_loc is not None:
		log.info(f"Applying local threshold with value: {ARG.filter_threshold_loc}")
	if ARG.filter_threshold_loc_sigma is not None:
		log.info(f"Applying local threshold based on {ARG.filter_threshold_loc_sigma} standard deviations.")
	
	if ARG.j < 0:
		ARG.j = multiprocessing.cpu_count()
		log.info("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()"%(ARG.j))
	elif ARG.j == 0:
		log.info("No threading will be used ARG.j=0.")
	else:
		log.info("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()=%d"%(ARG.j, multiprocessing.cpu_count()))
	
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
				log.info("Frame %d: %d pixels corrected, fraction: %.2f%%"%(r["k"], r["pixels"], corrected_fraction*100))
	
	if len(errors) > 0:
		log.error(f"{zdim - total_frames_sucessful}/{zdim} frames raised the following exceptions:")
		for (k, error) in errors:
			log.error(f"Frame {k} exception:\n{error}")
		if total_frames_sucessful > 0:
			total_pixels_fraction = total_pixels_corrected / (total_frames_sucessful*frameSize)
			log.warning("From total %d/%d frames corrected in '%s' with %d pixels corrected, fraction: %.2f%%"%(total_frames_sucessful, zdim, ARG.outputFile, total_pixels_corrected, total_pixels_fraction*100))
		else:
			total_pixels_fraction = 0.0
			log.error("Processing failed for all frames, no pixels corrected in '%s'."%(ARG.outputFile))
	else:
		total_pixels_fraction = total_pixels_corrected / totalSize
		log.info("Sucessfully created '%s' with %d pixels corrected, fraction: %.2f%%"%(ARG.outputFile, total_pixels_corrected, total_pixels_fraction*100))
	end_time = time.time()
	elapsed_time = end_time - start_time
	seconds = elapsed_time % 60
	minutes = int(elapsed_time // 60)
	hours = int(minutes // 60)
	minutes = minutes % 60
	formatted_time = f"{hours}h {minutes}m {seconds:.2f}s" if hours > 0 else (f"{minutes}m {seconds:.2f}s" if minutes > 0 else f"{seconds:.2f}s")
	log.info("END removeHotPixels, elapsed time: %s."%(formatted_time,))
