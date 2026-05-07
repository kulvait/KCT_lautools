import pandas as pd
import numpy as np
import re
from glob import glob
import os
from datetime import datetime, timedelta
import pytz
import logging

# Create a logger specific to this module
log = logging.getLogger(__name__)
log.setLevel(logging.INFO) # Set the logging level to INFO
# Create a console handler and set its level to INFO
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# Add the handler to the logger
log.addHandler(ch)


def find_and_sort_tiff_files(imgDir, prefix="img"):
	if not os.path.isdir(imgDir):
		raise FileNotFoundError(f"Image directory '{imgDir}' does not exist.")
	# Find all .tif and .tiff files starting with prefix
	pattern = os.path.join(imgDir, f"{prefix}*.tif*")
	files = glob(pattern)  # returns absolute/relative paths depending on imgDir
	files = [os.path.realpath(f) for f in files]  # convert to absolute paths
	# Function to extract numeric part for sorting
	def numeric_key(f):
		name = os.path.basename(f)
		nums = re.findall(r'\d+', name)
		return int(nums[-1]) if nums else -1  # use last number in filename
	files.sort(key=numeric_key)
	return files

# Function to process data from P05 nanoCT log file
# LogScan shall be location of LogScan.log file
# imgDir shall be location of the directory where the acquisition images are stored
def scanDataset(LogScan, imgDir=None):
	with open(LogScan, 'r') as file:
		lines = file.readlines()
	start_time = None
	for line in lines:
		line = line.strip()
		if line.startswith("#starttime"):
			parts = line.split("=", maxsplit=1)
			if len(parts) == 2:
				time_str = parts[1].strip()
				try:
					start_time = int(float(time_str))  # Convert to float first to handle cases like 1.761064e+09
					dt_utc = pd.to_datetime(start_time, unit='s', utc=True)
					# Convert to Central European Time with daylight saving automatically
					cet_tz = pytz.timezone("Europe/Berlin")
					dt_cet = dt_utc.tz_convert(cet_tz)
					formatted_time = dt_cet.strftime("%d.%m.%Y %H:%M")
					log.info(f"Experiment start: {formatted_time} (CET)")
				except ValueError:
					log.log(logging.WARNING, "Invalid start time format in the log file, %s"%(time_str))
				break
	if start_time is None:
		log.log(logging.WARNING, "Start time not found in the log file.")
		start_time = 0.0  # Default to 0 if not found
	# Test if imgDir is a valid directory
	if imgDir is not None:
		if not os.path.isdir(imgDir):
			log.error(f"Image directory '{imgDir}' does not exist.")
			raise FileNotFoundError(f"Image directory '{imgDir}' does not exist.")
		# List all files in the image directory starting with "img" and those starting with "ref"
		img_files = find_and_sort_tiff_files(imgDir, prefix="img")
		ref_files = find_and_sort_tiff_files(imgDir, prefix="ref")
	data_lines = []
	for line in lines:
		if not line.startswith("#") and line.strip():
			data_lines.append(line.strip())
	# Create a DataFrame from the data lines
	# Parse log lines
	parsed = []
	ref_counter = 0
	img_counter = 0
	for line in data_lines:
		parts = line.split()
		image_type = parts[0]
		timestamp = float(parts[5])  # #05 column
		absolute_time = start_time + timestamp
		dt = pd.to_datetime(absolute_time, unit='s', utc=True)
		current = float(parts[6]) # #06 column
		s_rot = float(parts[7]) # #07 column
		if image_type == "ref":
			image_key = 1
			if imgDir is None:
				ref_counter += 1
				image_path = f""
			elif ref_counter < len(ref_files):
				image_path = ref_files[ref_counter]
				ref_counter += 1
			else:
				raise ValueError("Not enough reference images in the directory for the log entries.")
		elif image_type == "img":
			image_key = 0
			if imgDir is None:
				img_counter += 1
				image_path = f""
			elif img_counter < len(img_files):
				image_path = img_files[img_counter]
				img_counter += 1
			else:
				raise ValueError("Not enough image files in the directory for the log entries.")
		parsed.append({
			"time": dt,
			"image_key": image_key,
			"image_file": image_path,
			"s_rot": s_rot,
			"s_stage_x": 0.0,
			"s_stage_z": 0.0,
			"current": current
		})
	df = pd.DataFrame(parsed)
	return df
