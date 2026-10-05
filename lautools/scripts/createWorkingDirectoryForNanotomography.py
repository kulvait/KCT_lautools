#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2026

@author: Vojtech Kulvait

This scripts creates soft links in the directory structure
"""
import logging
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
from pathlib import Path
from denpy import DICOM
from denpy import PETRA
from denpy import UTILS
import argparse

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


parser = argparse.ArgumentParser()
parser.add_argument("rawDir")
parser.add_argument("workingDir")
parser.add_argument("--processed-dir", default=None)
parser.add_argument(
	"--pattern",
	default=None,
	help="Regex patern to match against scanned directories [defults to None].")
parser.add_argument(
	"--random-item-count",
	default=None,
	type=int,
	help="Maximum count of items to process, chosen randomly [defaults to None]."
)
parser.add_argument("--processed-only", action="store_true")
parser.add_argument("--params-update", action="store_true")
parser.add_argument("--force", action="store_true")
parser.add_argument("--singledir", action="store_true")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--verbose", action="store_true")
ARG = parser.parse_args()
rawDir = ARG.rawDir

if ARG.singledir:
	subDirs = [rawDir]
else:
	subDirs = next(os.walk(rawDir))[1]
	subDirs = [os.path.join(rawDir, x) for x in subDirs]

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
			print(
				"There is %d item in subDirs list to be processed after filtering."
				% subDirsLen)

if ARG.random_item_count is not None and len(subDirs) > ARG.random_item_count:
	subDirs = random.choices(subDirs, k=ARG.random_item_count)

subDirs.sort()

def getConfigFile(directory, filename):
	# 1) preferred exact filename
	dirname = os.path.basename(directory)
	standardFileLocation = os.path.join(directory, f"{dirname}__{filename}")
	if os.path.isfile(standardFileLocation):
		return standardFileLocation
	# 2) fallback: glob
	matches = glob.glob(os.path.join(directory, f"*{filename}"))
	if len(matches) == 1:
		return matches[0]
	if len(matches) == 0:
		log.info(f"Missing {filename} in {directory}")
	else:
		log.info(f"Ambiguous {filename} in {directory}: {matches}")
	return None


#We will attempt to locate the following files in the directory:
# LogBeam.log
# LogMotors.log
# LogScan.log
# LogScript.py.log
# ScanParam.txt

def getInfo(directory):
	if not os.path.isdir(directory):
		return {}
	out = {}
	out["rawdir"] = os.path.realpath(directory)
	out["basename"] = os.path.basename(directory)
	out["LogBeam"] = getConfigFile(directory, "LogBeam.log")
	out["LogMotors"] = getConfigFile(directory, "LogMotors.log")
	out["LogScan"] = getConfigFile(directory, "LogScan.log")
	out["LogScript"] = getConfigFile(directory, "LogScript.py.log")
	out["ScanParam"] = getConfigFile(directory, "ScanParam.txt")
	return out

processed_dir = ARG.processed_dir
if processed_dir is None:
	#Try if rawdir/../processed exists
	processed_path = os.path.join(rawDir, "..", "processed")
	if os.path.exists(processed_path):
		processed_dir = os.path.realpath(processed_path)
processed_count = 0 

for d in subDirs:
	info = getInfo(d)
	if info is None:
		log.info("Skipping directory %s as it is not a valid directory." % d)
		continue
	logBeam = info.get("LogBeam", None)
	logMotors = info.get("LogMotors", None)
	logScan = info.get("LogScan", None)
	logScript = info.get("LogScript", None)
	scanParam = info.get("ScanParam", None)
	if logBeam is None:
		print("Skipping directory %s as LogBeam.log was not found." % d)
		continue
	if logMotors is None:
		print("Skipping directory %s as LogMotors.log was not found." % d)
		continue
	if logScan is None:
		print("Skipping directory %s as LogScan.log was not found." % d)
		continue
	if logScript is None:
		print("Skipping directory %s as LogScript.py.log was not found." % d)
		continue
	if scanParam is None:
		print("Skipping directory %s as ScanParam.txt was not found." % d)
		continue
	basename = os.path.basename(info["rawdir"])
	workdir = os.path.join(ARG.workingDir, basename)
	params = {}
	params["LogBeam"] = os.path.realpath(info["LogBeam"])
	params["LogMotors"] = os.path.realpath(info["LogMotors"])
	params["LogScan"] = os.path.realpath(info["LogScan"])
	params["LogScript"] = os.path.realpath(info["LogScript"])
	params["ScanParam"] = os.path.realpath(info["ScanParam"])
	params["workdir"] = os.path.realpath(workdir)
	params["raw"] = os.path.realpath(info["rawdir"])
	# Attempt to process file
	try:
		log.info("Processing directory %s" % info["basename"])
		# Check if processed_dir is specified and exists
		if processed_dir is not None:
			processeddir = os.path.join(processed_dir, basename)
			if os.path.exists(processeddir):
				print("Found processed dir %s" % processeddir)
				params["processed"] = os.path.realpath(processeddir)
		
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
			os.symlink(params["LogBeam"], os.path.join(workdir, "LogBeam.log"))
			os.symlink(params["LogMotors"], os.path.join(workdir, "LogMotors.log"))
			os.symlink(params["LogScan"], os.path.join(workdir, "LogScan.log"))
			os.symlink(params["LogScript"], os.path.join(workdir, "LogScript.py.log"))
			os.symlink(params["ScanParam"], os.path.join(workdir, "ScanParam.txt"))
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
		log.info("Successfully processed directory %s" % d)
	except Exception as e:
		# Log error with line number
		import traceback
		tb = traceback.format_exc()
		print("Error processing file %s in %s: %s\n%s" % (info.get("h5", "unknown"), info.get("rawdir", "unknown"), str(e), tb), flush=True)

# Summary of processing
print("Total successfully processed subdirectories:", processed_count)
