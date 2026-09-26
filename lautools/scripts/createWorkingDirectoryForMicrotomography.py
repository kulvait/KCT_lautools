#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a working directory with selected samples.

Scripts for processing microtomography data from P05/P07. The input `rawDir` is expected to contain one subdirectory per sample. Each
sample directory must contain the raw data together with a single `.h5` file
describing the scan. The script reads the metadata from these sample
directories and selects the ones that should be prepared for further work.

For every selected sample, the script creates a corresponding subdirectory in
`workingDir`. Each created directory contains symbolic links to the original
raw data, the `.h5` file, and optionally the matching processed directory. It
also writes a `params` file with metadata needed for downstream processing.

The purpose of this script is to prepare a smaller, dedicated working area
containing only the samples chosen for reconstruction or further analysis.

@author: Vojtech Kulvait
@year: 2023-2026
@license: GNU GPL v3.0

"""
import argparse
import glob
import io
import os
import sys
import shutil
import re
import h5py
import random
import traceback
import argparse
import sys
import datetime
import logging
from pathlib import Path
from denpy import DICOM
from denpy import PETRA
from denpy import UTILS

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


def getInfo(directory):
	if not os.path.isdir(directory):
		return {}
	h5files = glob.glob(os.path.join(directory, "*.h5"))
	if len(h5files) != 1:
		print("\nExcluding directory %s because there is %d h5 files." %
			  (directory, len(h5files)))
		return {}
	h5 = h5files[0]
	log_buffer = io.StringIO()
	old_stdout = sys.stdout
	sys.stdout = log_buffer
	try:
		out = {}
		scanData = PETRA.scanDataset(h5, includeCurrent=False)
		out["h5"] = os.path.realpath(h5)
		out["rawdir"] = os.path.realpath(directory)
		out["scanData"] = scanData
		for f in scanData["image_file"]:
			f = f.lstrip("/")
			if not os.path.exists(os.path.join(directory, f)):
				print("\nExcluding directory %s as file %s does not exist." %
					  (directory, os.path.join(directory, f)))
				return {}
		return out
	except Exception as e:
		print("\nExcluding directory %s because there was error parsing h5 file %s. \nException: %s"
			  % (directory, h5, traceback.format_exc()))
		return {}
	finally:
		sys.stdout = old_stdout
		out["log"] = log_buffer.getvalue()
		log_buffer.close()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("rawDir")
	parser.add_argument("workingDir", nargs="?", default=None, help="Directory where the working directories will be created. Required unless --list or --dry-run is used.")
	parser.add_argument("--processed-dir", default=None)
	parser.add_argument("--pattern", default=None, help="Regex patern to match against scanned directories [defults to None].")
	selection_group = parser.add_mutually_exclusive_group()
	selection_group.add_argument("--random-item-count", default=None, type=int, help="Maximum count of items to process, chosen randomly [defaults to None].")
	selection_group.add_argument("--samples", nargs="+", default=None, help="List of sample names to process [defaults to None].")
	selection_group.add_argument("--samples-file", default=None, help="File containing list of sample names to process [defaults to None].")
	parser.add_argument("--processed-only", action="store_true")
	parser.add_argument("--params-update", action="store_true")
	parser.add_argument("--force", action="store_true")
	parser.add_argument("--singledir", action="store_true")
	modifiers_group = parser.add_mutually_exclusive_group()
	modifiers_group.add_argument("--list", action="store_true", help="List the directories that would be processed and exit # is used to comment out lines in the list.")
	modifiers_group.add_argument("--dry-run", action="store_true")
	parser.add_argument("--verbose", action="store_true")
	ARG = parser.parse_args()
	
	if ARG.workingDir is None and not ARG.list and not ARG.dry_run:
		print("Error: workingDir must be specified unless --list or --dry-run is used.")
		sys.exit(1)
	
	rawDir = ARG.rawDir
	
	if ARG.singledir:
		subDirs = [rawDir]
	else:
		subDirs = next(os.walk(rawDir))[1]
		subDirs = [os.path.join(rawDir, x) for x in subDirs]
	
	if ARG.list:
		print("# List of directories that would be processed:")
		SAMPLE_NAMES = []
		for d in subDirs:
			h5files = glob.glob(os.path.join(d, "*.h5"))
			if len(h5files) == 1:
				SAMPLE_NAMES.append(os.path.basename(d))
		SAMPLE_NAMES.sort()
		for name in SAMPLE_NAMES:
			print(name)
		sys.exit(0)
	
	# Option --list shall not print the start and end messages, but --dry-run shall, so we move the print statements here.
	print("START createWorkingDirectoryForMicrotomography %s" % " ".join(sys.argv[1:]))
	print("Date: %s" % datetime.datetime.now().strftime("%d.%m.%y %H:%M:%S"))
	
	subDirsLen = len(subDirs)
	if ARG.verbose:
		print("There is %d item in subDirs list to be processed." % subDirsLen)
	
	if ARG.pattern is not None:
		regexp = re.compile(ARG.pattern)
		subDirs = [
			x for x in subDirs if regexp.search(os.path.basename(x)) is not None
		]
		if len(subDirs) < subDirsLen:
			subDirsLen = len(subDirs)
			if ARG.verbose:
				print("There is %d item in subDirs matching pattern %s." % (subDirsLen, ARG.pattern))
	
	if ARG.random_item_count is not None and len(subDirs) > ARG.random_item_count:
		subDirs = random.choices(subDirs, k=ARG.random_item_count)
	
	selectedSamples = set()
	if ARG.samples is not None:
		selectedSamples.update(ARG.samples)
	if ARG.samples_file is not None:
		with open(ARG.samples_file, "r") as f:
			for line in f:
				line = line.split("#", 1)[0].strip()
				if len(line) > 0:
					selectedSamples.add(line)
	
	if ARG.samples is not None or ARG.samples_file is not None:
		subDirs = [x for x in subDirs if os.path.basename(x) in selectedSamples]
		if len(subDirs) < subDirsLen:
			subDirsLen = len(subDirs)
			if ARG.verbose:
				print("There is %d item in subDirs matching samples list." % subDirsLen)
	
	subDirs.sort()
	processed_dir = ARG.processed_dir
	if processed_dir is None:
		#Try if rawdir/../processed exists
		processed_path = os.path.join(rawDir, "..", "processed")
		if os.path.exists(processed_path):
			processed_dir = os.path.realpath(processed_path)
	processed_count = 0
	
	for d in subDirs:
		info = getInfo(d)
		if len(info) == 0:
			print("Skipping directory %s as there was an error." % d)
			continue
		basename = os.path.basename(info["rawdir"])
		params = {}
		if not ARG.dry_run:
			workdir = os.path.join(ARG.workingDir, basename)
			params["h5"] = info["h5"]
			params["raw"] = info["rawdir"]
			params["workdir"] = os.path.realpath(workdir)
		# Attempt to process file
		try:
			print("\nProcessing file %s in %s" % (info["h5"], info["rawdir"]))
			# Check if processed_dir is specified and exists
			if processed_dir is not None:
				processeddir = os.path.join(processed_dir, basename)
				if os.path.exists(processeddir):
					print("Found processed dir %s" % processeddir)
					params["processed"] = os.path.realpath(processeddir)
			
			# Check if we need to skip based on processed_only
			if "processed" in params:
				logfile = glob.glob(os.path.join(params["processed"], '**/reco.log'), recursive=True)
				if len(logfile) == 0 and ARG.processed_only:
					continue
				elif len(logfile) != 0:
					params["logfile"] = os.path.realpath(logfile[0])
					with open(params["logfile"], "r") as logfile:
						logcontent = dict(map(str.strip, l.split(':', 1)) for l in logfile.readlines())
						if "rotation_axis_offset_reco" in logcontent:
							params["rotation_axis_offset_reco"] = logcontent["rotation_axis_offset_reco"]
						if "effective_pixel_size" in logcontent:
							params["effective_pixel_size"] = logcontent["effective_pixel_size"].rsplit(" ")[0]
						if "sample_detector_distance" in logcontent:
							params["sample_detector_distance"] = logcontent["sample_detector_distance"].rsplit(" ")[0]
			else:
				if ARG.processed_only:
					print("Skipping %s as it has not related entry in %d dir and --processed-only is set." % (info["rawdir"], processed_dir))
					continue
			
			# Open the HDF5 file and retrieve camera information
			with h5py.File(info["h5"], 'r') as h5:
				if "/entry/hardware/camera1" in h5:
					cam = "camera1"
				elif "/entry/hardware/camera" in h5:
					cam = "camera"
				else:
					print("Skiping %s as there is no camera entry in h5 file." % info["h5"])
					continue
				
				# Extract camera pixel size and magnification
				if "entry/hardware/%s/pixelsize" % cam in h5:
					params["pixel_size_camera"] = h5["entry/hardware/%s/pixelsize" % cam][0]
				elif "entry/hardware/%s/camera/px_size" % cam in h5:
					params["pixel_size_camera"] = h5["entry/hardware/%s/camera/px_size" % cam][0]
				else:
					print("Skiping %s as there is no pixel_size_camera entry in entry/hardware/%s/pixelsize or entry/hardware/%s/camera/px_size." % (info["h5"], cam, cam))
					continue
				if "entry/hardware/%s/magnification" % cam in h5:
					params["pixel_size_magnification"] = h5["entry/hardware/%s/magnification" % cam][0]
				elif "entry/hardware/%s/calibration/magnification" % cam in h5:
					params["pixel_size_magnification"] = h5["entry/hardware/%s/calibration/magnification" % cam][0]
				else:
					print("Skiping %s as there is no pixel_size_magnification entry in entry/hardware/%s/magnification or entry/hardware/%s/calibration/magnification." % (info["h5"], cam, cam))
					continue
				
				# Check for zero magnification and raise an informative error
				if float(params["pixel_size_magnification"]) == 0:
					print("Skiping %s as there is zero pixel_size_magnification in entry/hardware/%s/magnification." % (info["h5"], cam))
					continue
				
				# Calculate pixel sizes
				params["pixel_size_x"] = float(params["pixel_size_camera"]) / float(params["pixel_size_magnification"])
				params["pixel_size_y"] = float(params["pixel_size_camera"]) / float(params["pixel_size_magnification"])
				
				# Optionally, get sensor size if available
				pdimx = 0
				pdimy = 0
				if "entry/hardware/%s/sensorsize_x" % cam in h5:
					pdimx = int(h5["entry/hardware/%s/sensorsize_x" % cam][0])
				elif "entry/hardware/%s/camera/senzor_xsize" % cam in h5:
					pdimx = int(h5["entry/hardware/%s/camera/senzor_xsize" % cam][0])
				if pdimx != 0:
					params["pdimx"] = "%d" % pdimx
					params["projection_size_x"] = pdimx * float(params["pixel_size_x"])
				if "entry/hardware/%s/sensorsize_y" % cam in h5:
					pdimy = int(h5["entry/hardware/%s/sensorsize_y" % cam][0])
				elif "entry/hardware/%s/camera/senzor_ysize" % cam in h5:
					pdimy = int(h5["entry/hardware/%s/camera/senzor_ysize" % cam][0])
				if pdimy != 0:
					params["pdimy"] = "%d" % pdimy
					params["projection_size_y"] = pdimy * float(params["pixel_size_y"])
			
			# Handle output processing, logging, and directory setup
			if ARG.params_update:
				if os.path.exists(workdir):
					UTILS.writeParamsFile(params, os.path.join(workdir, "params"))
			elif not ARG.dry_run:
				if os.path.exists(workdir):
					if ARG.force:
						print("Removing existing %s" % workdir)
						shutil.rmtree(workdir)
					else:
						print("Skipping existing but updating params file %s" % workdir)
						UTILS.writeParamsFile(params, os.path.join(workdir, "params"))
						continue
				Path(workdir).mkdir(parents=True, exist_ok=True)
				if "processed" in params:
					os.symlink(params["processed"], os.path.join(workdir, "processed"))
				os.symlink(params["h5"], os.path.join(workdir, "h5"))
				os.symlink(params["raw"], os.path.join(workdir, "raw"))
				UTILS.writeParamsFile(params, os.path.join(workdir, "params"))
				#Finally write any log in info["log"]
				if "log" in info and len(info["log"]) > 0:
					os.makedirs(os.path.join(workdir, "log"), exist_ok=True)
					with open(os.path.join(workdir, "log", "createWorkingDirectory.log"), "w") as logf:
						logf.write(info["log"])
					lines = info["log"].strip().split("\n")
					last_line = lines[-1]
					print("%d log lines captured, see log/createWorkingDirectory.log. Last line: %s" % (len(lines), last_line))
			
			# Increment processed count
			processed_count += 1
			print("Successfully processed file %s in %s" % (info["h5"], info["rawdir"]), flush=True)
		except Exception as e:
			# Log error with line number
			import traceback
			tb = traceback.format_exc()
			print("Error processing file %s in %s: %s\n%s" % (info.get("h5", "unknown"), info.get("rawdir", "unknown"), str(e), tb), flush=True)
	
	# Summary of processing
	print("Total successfully processed subdirectories:", processed_count)
	print("END createWorkingDirectoryForMicrotomography")

if __name__ == "__main__":
    main()
