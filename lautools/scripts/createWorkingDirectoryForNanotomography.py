#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a working directory with selected nanotomography samples.

This script scans raw nanotomography sample directories and prepares matching
working directories with symbolic links to the original files and a params file
for downstream processing.

@author: Vojtech Kulvait
@year: 2026
@license: GNU GPL v3.0
"""

import argparse
import datetime
import glob
import logging
import os
import random
import re
import shutil
import sys
import traceback
from pathlib import Path

from denpy import UTILS


# Create a logger specific to this module
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter(
	'%(asctime)s - %(name)s:%(lineno)d - %(levelname)s : %(message)s',
	datefmt='%d.%m.%Y %H:%M:%S'
)
ch.setFormatter(formatter)

if not log.handlers:
	log.addHandler(ch)
log.propagate = False

def getConfigFile(directory, filename, logging=True):
	# 1) preferred exact filename
	dirname = os.path.basename(directory)
	standardFileLocation = os.path.join(directory, f"{dirname}__{filename}")
	if os.path.isfile(standardFileLocation):
		return standardFileLocation
	# 2) fallback: glob
	matches = glob.glob(os.path.join(directory, f"*{filename}"))
	if len(matches) == 1:
		return matches[0]
	if logging:
		if len(matches) == 0:
			log.info("Missing %s in %s", filename, directory)
		else:
			log.info("Ambiguous %s in %s: %s", filename, directory, matches)
	return None

def getInfo(directory, logging=True):
	if not os.path.isdir(directory):
		return {}
	out = {}
	out["rawdir"] = os.path.realpath(directory)
	out["basename"] = os.path.basename(directory)
	out["LogBeam"] = getConfigFile(directory, "LogBeam.log", logging=logging)
	out["LogMotors"] = getConfigFile(directory, "LogMotors.log", logging=logging)
	out["LogScan"] = getConfigFile(directory, "LogScan.log", logging=logging)
	out["LogScript"] = getConfigFile(directory, "LogScript.py.log", logging=logging)
	out["ScanParam"] = getConfigFile(directory, "ScanParam.txt", logging=logging)
	return out

def list_candidate_directories(subDirs):
	print("# List of directories that would be processed:")
	sample_names = []
	for d in subDirs:
		info = getInfo(d, logging=False)
		if len(info) == 0:
			continue
		if (
			info.get("LogBeam") is not None
			and info.get("LogMotors") is not None
			and info.get("LogScan") is not None
			and info.get("LogScript") is not None
			and info.get("ScanParam") is not None
		):
			sample_names.append(os.path.basename(d))
	sample_names.sort()
	for name in sample_names:
		print(name)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("rawDir")
	parser.add_argument(
		"workingDir",
		nargs="?",
		default=None,
		help="Directory where the working directories will be created. Required unless --list or --dry-run is used."
	)
	parser.add_argument("--processed-dir", default=None)
	parser.add_argument(
		"--pattern",
		default=None,
		help="Regex patern to match against scanned directories [defults to None]."
	)

	selection_group = parser.add_mutually_exclusive_group()
	selection_group.add_argument(
		"--random-item-count",
		default=None,
		type=int,
		help="Maximum count of items to process, chosen randomly [defaults to None]."
	)
	selection_group.add_argument(
		"--samples",
		nargs="+",
		default=None,
		help="List of sample names to process [defaults to None]."
	)
	selection_group.add_argument(
		"--samples-file",
		default=None,
		help="File containing list of sample names to process [defaults to None]."
	)

	parser.add_argument("--processed-only", action="store_true")
	parser.add_argument("--params-update", action="store_true")
	parser.add_argument("--force", action="store_true")
	parser.add_argument("--singledir", action="store_true")

	modifiers_group = parser.add_mutually_exclusive_group()
	modifiers_group.add_argument(
		"--list",
		action="store_true",
		help="List the directories that would be processed and exit # is used to comment out lines in the list."
	)
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
		list_candidate_directories(subDirs)
		sys.exit(0)

	print("START createWorkingDirectoryForNanotomography %s" % " ".join(sys.argv[1:]))
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
				print(
					"There is %d item in subDirs matching pattern %s."
					% (subDirsLen, ARG.pattern)
				)

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
		# Try if rawdir/../processed exists
		processed_path = os.path.join(rawDir, "..", "processed")
		if os.path.exists(processed_path):
			processed_dir = os.path.realpath(processed_path)
	processed_count = 0
	for d in subDirs:
		info = getInfo(d)
		if len(info) == 0:
			print("Skipping directory %s as there was an error." % d)
			continue

		logBeam = info.get("LogBeam")
		logMotors = info.get("LogMotors")
		logScan = info.get("LogScan")
		logScript = info.get("LogScript")
		scanParam = info.get("ScanParam")

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
		params = {}

		if not ARG.dry_run:
			workdir = os.path.join(ARG.workingDir, basename)
			params["LogBeam"] = os.path.realpath(info["LogBeam"])
			params["LogMotors"] = os.path.realpath(info["LogMotors"])
			params["LogScan"] = os.path.realpath(info["LogScan"])
			params["LogScript"] = os.path.realpath(info["LogScript"])
			params["ScanParam"] = os.path.realpath(info["ScanParam"])
			params["workdir"] = os.path.realpath(workdir)
			params["raw"] = os.path.realpath(info["rawdir"])

		try:
			print("\nProcessing directory %s" % info["rawdir"])
			log.info("Processing directory %s", info["basename"])

			if processed_dir is not None:
				processeddir = os.path.join(processed_dir, basename)
				if os.path.exists(processeddir):
					print("Found processed dir %s" % processeddir)
					params["processed"] = os.path.realpath(processeddir)
				elif ARG.processed_only:
					print(
						"Skipping %s as it has no related entry in %s and --processed-only is set."
						% (info["rawdir"], processed_dir)
					)
					continue
			elif ARG.processed_only:
				print(
					"Skipping %s as no processed directory base was found and --processed-only is set."
					% info["rawdir"]
				)
				continue

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

				if "log" in info and len(info["log"]) > 0:
					os.makedirs(os.path.join(workdir, "log"), exist_ok=True)
					with open(
						os.path.join(workdir, "log", "createWorkingDirectory.log"),
						"w"
					) as logf:
						logf.write(info["log"])
					lines = info["log"].strip().split("\n")
					last_line = lines[-1]
					print(
						"%d log lines captured, see log/createWorkingDirectory.log. Last line: %s"
						% (len(lines), last_line)
					)

			processed_count += 1
			print("Successfully processed directory %s" % info["rawdir"], flush=True)
			log.info("Successfully processed directory %s", d)

		except Exception:
			tb = traceback.format_exc()
			print(
				"Error processing directory %s: %s\n%s"
				% (info.get("rawdir", "unknown"), "exception raised", tb),
				flush=True,
			)

	print("Total successfully processed subdirectories:", processed_count)
	print("END createWorkingDirectoryForNanotomography")


if __name__ == "__main__":
	main()
