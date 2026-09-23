#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Vojtěch Kulvait
@year: 2022-2026
@license: GNU GPL v3
"""
import argparse
from denpy import DEN
import os
#os.environ["OMP_NUM_THREADS"] = "16"
#os.environ["OPENBLAS_NUM_THREADS"] = "16"  # export OPENBLAS_NUM_THREADS=4
#os.environ["MKL_NUM_THREADS"] = "16"  # export MKL_NUM_THREADS=6
#os.environ["VECLIB_MAXIMUM_THREADS"] = "16"  # export VECLIB_MAXIMUM_THREADS=4
#os.environ["NUMEXPR_NUM_THREADS"] = "16"  # export NUMEXPR_NUM_THREADS=6
import numpy as np
import sys
import time
import statistics

import traceback
import multiprocessing as mp
from multiprocessing.dummy import Pool, Lock

# -------------------------------------------------------------------------
# Global write lock
# -------------------------------------------------------------------------
write_lock = None

def init_worker(lock):
	"""
	Initialize each worker with the shared write lock.
	"""
	global write_lock
	write_lock = lock

def processFrame(k, fileIn, fileOut, ydim_red, xdim_red, ydim_bin, xdim_bin, bin_y, bin_x, newshape, process_average=False):
	try:
		start = time.time()
		# Read frame k from input file
		f = DEN.getFrame(fileIn, k, row_to=ydim_red, col_to=xdim_red)
		# Reshape the frame to 4D array for binning
		f.shape = (ydim_bin, bin_y, xdim_bin, bin_x)
		f_reshaped = f.swapaxes(1, 2)
		f_reshaped = f_reshaped.reshape(newshape)
		# Compute median or average along the last axis
		if process_average:
			f_cor = np.mean(f_reshaped, axis=-1)
		else: #median
			f_cor = np.median(f_reshaped, axis=-1)
		f_cor = f_cor.astype(np.float32, copy=False)
		# Write the processed frame to output file using lock to ensure thread safety
		if write_lock:
			write_lock.acquire()
		try:
			DEN.writeFrame(fileOut, k, f_cor, force=True)
		finally:
			if write_lock:
				write_lock.release()
		end = time.time()
		return {"k": k, "elapsed_time": end - start}
	except Exception as e:
		print(f"Error processing frame {k}: {e}")
		return {"k": k, "error": traceback.format_exc()}

def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)

def main(argv=None):
	print("START lautools:binDenFile %s" % (" ".join(sys.argv[1:])), flush=True)
	
	parser = argparse.ArgumentParser(description="Binning of a DEN file in X and Y dimensions.")
	parser.add_argument("inputDen")
	parser.add_argument("outputDen")
	parser.add_argument("--bin-x",
						type=int,
						default=1,
						help="X dimension of binning box.")
	parser.add_argument("--bin-y",
						type=int,
						default=1,
						help="Y dimension of binning box.")
	parser.add_argument("--force", action="store_true")
	parser.add_argument("--threads", default=-1, type=int, help="Number of threads to use. [defaults to -1 which is mp.cpu_count(), 0 without threading]")
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--average', action='store_true')
	group.add_argument('--median', action='store_true')
	
	ARG = parser.parse_args(argv)
	
	if ARG.threads == -1:
		ARG.threads = mp.cpu_count()
		print("Using %d threads estimated using mp.cpu_count()" % ARG.threads)
	
	#First read dimensions of the DEN file
	header = DEN.readHeader(ARG.inputDen)
	if not header["dimcount"] in [2, 3]:
		print("File %s shall have dimension two or three" %
			  (os.path.basename(ARG.inputDen)))
		sys.exit(-1)
	if header["dimcount"] == 2:
		zdim = 1
	else:
		zdim = header["dimspec"][2]
	xdim = header["dimspec"][0]
	ydim = header["dimspec"][1]
	bin_x = ARG.bin_x
	bin_y = ARG.bin_y
	xdim_red = xdim - xdim % ARG.bin_x
	ydim_red = ydim - ydim % ARG.bin_y
	xdim_out = xdim_red // ARG.bin_x
	ydim_out = ydim_red // ARG.bin_y
	if xdim_red < 1 or ydim_red < 1 or xdim_out < 1 or ydim_out < 1:
		print("Dimensions are zero after reduction!")
		sys.exit(-1)
	if xdim % ARG.bin_x != 0:
		print("dimx=%d is not divisible by bin_x=%d, reducing to %d" % (xdim, ARG.bin_x, xdim_red))
	if ydim % ARG.bin_y != 0:
		print("dimy=%d is not divisible by bin_y=%d, reducing to %d" % (ydim, ARG.bin_y, ydim_red))
	process_average = ARG.average
	DEN.writeEmptyDEN(ARG.outputDen, [xdim_out, ydim_out, zdim],
					  header["type"],
					  force=ARG.force)
	print("Input file %s dimensions [dimx, dimy, dimz] = [%d, %d, %d]" % (os.path.basename(ARG.inputDen), xdim, ydim, zdim), flush=True)
	print("Output file %s dimensions [dimx, dimy, dimz] = [%d, %d, %d]" % (os.path.basename(ARG.outputDen), xdim_out, ydim_out, zdim), flush=True)
	
	boxsize = bin_x * bin_y
	arraySize_red = ydim_red * xdim_red
	arraySize_bin = arraySize_red // boxsize
	ydim_bin = ydim_red // bin_y
	xdim_bin = xdim_red // bin_x
	newshape = (ydim_bin, xdim_bin, boxsize)
	results = []
	global_start_time = time.time()
	if ARG.threads >= 1:
		lock = Lock()
		pool = Pool(ARG.threads, initializer=init_worker, initargs=(lock,))
		for k in range(zdim):
			res = pool.apply_async(processFrame, args=(k, ARG.inputDen, ARG.outputDen, ydim_red, xdim_red, ydim_bin, xdim_bin, bin_y, bin_x, newshape, process_average))
			results.append(res)
		pool.close() # No more jobs can be submitted.
		pool.join() # Wait for all worker processes to finish.
	else:
		for k in range(zdim):
			res = processFrame(k, ARG.inputDen, ARG.outputDen, ydim_red, xdim_red, ydim_bin, xdim_bin, bin_y, bin_x, newshape, process_average)
			results.append(res)
	global_end_time = time.time()
	
	# -------------------------------------------------------------------------
	# Collect errors
	# -------------------------------------------------------------------------
	errors = []
	for res in results:
		try:
			if isinstance(res, dict):
				r = res
			else:
				r = res.get()
			if "error" in r:
				errors.append((r["k"], r["error"]))
		except Exception:
			errors.append(("unknown", traceback.format_exc()))
	
	# -------------------------------------------------------------------------
	# Report errors and elapsed time
	# -------------------------------------------------------------------------
	global_elapsed_time = global_end_time - global_start_time
	#Format to hh:mm:ss
	hours, rem = divmod(global_elapsed_time, 3600)
	minutes, seconds = divmod(rem, 60)
	if hours > 0:
		formatted_time = "%02d:%02d:%05.2f" % (hours, minutes, seconds)
	elif minutes > 0:
		formatted_time = "%02d:%05.2f" % (minutes, seconds)
	else:
		formatted_time = "%05.2f" % seconds
	print("Total elapsed time: %s, average time per frame: %0.2fs" %
		  (formatted_time, global_elapsed_time / zdim))
	if len(errors) > 0:
		eprint("\nErrors occurred in the following frames:", flush=True)
		for k, err in errors:
			eprint("\nFrame %s error:\n%s" % (k, err), flush=True)
		raise RuntimeError("Errors occurred during processing. See above for details.")
	else:
		print("All %d frames processed successfully." % zdim, flush=True)
	
	print("END lautools:binDenFile", flush=True)

if __name__ == "__main__":
	main()
