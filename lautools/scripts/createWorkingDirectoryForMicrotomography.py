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
from pprint import pprint
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
formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s : %(message)s', datefmt='%d.%m.%y %H:%M:%S')
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

def matlabLogParse(path):
	"""Parse a MATLAB reco log and keep only valid key:value entries."""
	content = {}
	with open(path, "r", errors="replace") as f:
		for line in f:
			if ":" not in line:
				continue
			key, value = (part.strip() for part in line.split(":", 1))
			if key and value and key not in content:
				content[key] = value
	return content


def matlabLogDirectoryGetFirstItem(content, *keys):
	"""Return the value of the first key in keys that is in content, else None."""
	for k in keys:
		if k in content and content[k] != "":
			return content[k]
	return None

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
	print("Date: %s" % datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
	
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
					params["processed"] = os.path.realpath(processeddir)
			
			# Check if we need to skip based on processed_only
			if "processed" in params:
				logfile = glob.glob(os.path.join(params["processed"], 'reco*/**/reco_*.log'), recursive=True)
				print(f"Found processed dir {params['processed']} with {len(logfile)} Matlab reco.log files.")
				if len(logfile) == 0 and ARG.processed_only:
					continue
				elif len(logfile) != 0:
					params["jm_reco_logfile"] = os.path.realpath(logfile[0])
					with open(params["jm_reco_logfile"], "r") as logfile:
						print("Reading Matlab log file %s" % params["jm_reco_logfile"])
						logcontent = matlabLogParse(params["jm_reco_logfile"])
						jm_offset =  matlabLogDirectoryGetFirstItem(logcontent, "rot_axis_offset_reco", "rot_axis_offset")
						if jm_offset is not None:
							params["jm_rotation_axis_offset_binned"] = jm_offset
						jm_binning = matlabLogDirectoryGetFirstItem(logcontent, "raw_binning_factor", "raw_bin", "reco_binning_factor")
						if jm_binning is not None:
							params["jm_binning"] = jm_binning
			else:
				if ARG.processed_only:
					print("Skipping %s as it has not related entry in %d dir and --processed-only is set." % (info["rawdir"], processed_dir))
					continue
			info_petra = PETRA.getExperimentInfo(info["h5"])
			if "pix_size" in info_petra:
				params["pixel_size_x"] = info_petra["pix_size"]
				params["pixel_size_y"] = info_petra["pix_size"]
			if "fresnel_number" in info_petra:
				params["fresnel_number"] = info_petra["fresnel_number"]
			if "energy_keV" in info_petra:
				params["energy_keV"] = info_petra["energy_keV"]
			if "propagation_distance_mm" in info_petra:
				params["propagation_distance_mm"] = info_petra["propagation_distance_mm"]
			if "field_of_view_x" in info_petra["camera"]:
				params["field_of_view_x"] = info_petra["field_of_view_x"]
			if "field_of_view_y" in info_petra["camera"]:
				params["field_of_view_y"] = info_petra["field_of_view_y"]
			if "camera" in info_petra:
				if "magnification" in info_petra["camera"]:
					params["camera_magnification"] = info_petra["camera"]["magnification"]
				if "pixelsize" in info_petra["camera"]:
					params["camera_pixel_size"] = info_petra["camera"]["pixelsize"]
				if "sensorsize_x" in info_petra["camera"]:
					params["senzorsize_x"] = info_petra["camera"]["sensorsize_x"]
				if "sensorsize_y" in info_petra["camera"]:
					params["senzorsize_y"] = info_petra["camera"]["sensorsize_y"]
				if "roi_width" in info_petra["camera"]:
					params["pdimx"] = "%d" % info_petra["camera"]["roi_width"]
				if "roi_height" in info_petra["camera"]:
					params["pdimy"] = "%d" % info_petra["camera"]["roi_height"]
			print("Parameters:")
			pprint(params)
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
