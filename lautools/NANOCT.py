import pandas as pd
import numpy as np
import re
from glob import glob
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import logging

# Create a logger specific to this module
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)  # Set the logging level to INFO

# Add handler only once to avoid duplicate log lines on repeated imports
if not log.handlers:
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	ch.setFormatter(formatter)
	log.addHandler(ch)


IMAGE_KEYS = {"img": 0, "ref": 1, "dar": 2, "dark": 2}
_RAW_FILE_RE = re.compile(r'^(img|ref|dar|dark)_(\d+)(?:_(\d+))?\.tiff?$', re.IGNORECASE)


def find_and_sort_tiff_files(imgDir, prefix="img"):
	"""Find TIFF files by prefix and sort by the last numeric token in the filename.

	This function is kept for backward compatibility. The new scanDataset()
	implementation does not rely on this ordering for matching.
	"""
	if not os.path.isdir(imgDir):
		raise FileNotFoundError(f"Image directory '{imgDir}' does not exist.")
	pattern = os.path.join(imgDir, f"{prefix}*.tif*")
	files = glob(pattern)
	files = [os.path.realpath(f) for f in files]

	def numeric_key(f):
		name = os.path.basename(f)
		nums = re.findall(r'\d+', name)
		return int(nums[-1]) if nums else -1

	files.sort(key=numeric_key)
	return files


def _file_time(path):
	"""Return filesystem time used for acquisition ordering.

	Prefer birth/creation time where available. Otherwise use modification time.
	On Linux, st_ctime is inode change time, so st_mtime is safer.
	"""
	st = os.stat(path)
	bt = getattr(st, "st_birthtime", None)
	return bt if bt is not None else st.st_mtime


def _parse_raw_files(imgDir, one_based=True):
	"""Parse raw TIFF files and infer log counter (#04) from file names.

	Supported patterns:
	- ref_00020_7.tif   -> counter = 20 + (7 - 1) when one_based=True
	- img_00427.tif     -> counter = 427
	- dark_00000_1.tiff -> counter = 0

	Returns list of dicts sorted by filesystem time.
	"""
	records = []
	for fname in os.listdir(imgDir):
		match = _RAW_FILE_RE.match(fname)
		if match is None:
			continue

		ftype = match.group(1).lower()
		first = int(match.group(2))
		local = int(match.group(3)) if match.group(3) is not None else None

		if local is not None:
			counter = first + (local - 1 if one_based else local)
			block_id = first
		else:
			counter = first
			block_id = None

		path = os.path.realpath(os.path.join(imgDir, fname))
		records.append({
			"path": path,
			"type": ftype,
			"block_id": block_id,
			"local_idx": local,
			"counter": counter,
			"ftime": _file_time(path),
		})

	records.sort(key=lambda r: (r["ftime"], r["counter"], os.path.basename(r["path"])))
	return records


def _validate_file_block_order(block):
	"""Check that sorting by time gives the same order as sorting by counter.

	Missing counters are allowed. Duplicate counters or reversed order are treated
	as errors because block matching becomes ambiguous or suspicious.
	"""
	if not block:
		return

	label = f"{block[0]['type']} block (block_id={block[0]['block_id']})"

	seen = {}
	for record in block:
		counter = record["counter"]
		if counter in seen:
			raise ValueError(
				f"Duplicate counter {counter} in {label}: "
				f"{seen[counter]['path']} and {record['path']}. "
				"Block matching is ambiguous; inspect the raw files."
			)
		seen[counter] = record

	time_order = sorted(block, key=lambda record: (record["ftime"], record["counter"], os.path.basename(record["path"])))
	counter_order = sorted(block, key=lambda record: record["counter"])

	for position, (timed, counted) in enumerate(zip(time_order, counter_order), start=1):
		if timed["path"] != counted["path"]:
			raise ValueError(
				f"Time order disagrees with counter order in {label} at position {position}: "
				f"time order has {timed['path']} "
				f"(counter={timed['counter']}, time={timed['ftime']!r}), "
				f"but counter order expects {counted['path']} "
				f"(counter={counted['counter']}, time={counted['ftime']!r}). "
				"Block matching aborted; inspect timestamps and filenames."
			)


def _split_file_blocks(records, check_block_order=False):
	"""Split time-ordered raw files into blocks by file type and block_id.

	A new block starts when file type changes or when block_id changes.
	If check_block_order=True, each block must have the same order when sorted
	by filesystem time and by inferred counter.
	"""
	blocks = []
	cur = []

	def finish_block():
		if cur:
			if check_block_order:
				_validate_file_block_order(cur)
			blocks.append(cur.copy())

	for record in records:
		if cur:
			prev = cur[-1]
			new_block = (
				record["type"] != prev["type"]
				or record["block_id"] != prev["block_id"]
			)
			if new_block:
				finish_block()
				cur.clear()
		cur.append(record)

	finish_block()
	return blocks


def _parse_log(LogScan):
	"""Parse LogScan.log into start time and structured rows."""
	with open(LogScan, 'r') as file:
		lines = file.readlines()

	start_time = None
	rows = []

	for line in lines:
		line = line.strip()
		if not line:
			continue

		if line.startswith("#"):
			if line.startswith("#starttime") and start_time is None:
				parts = line.split("=", maxsplit=1)
				if len(parts) == 2:
					time_str = parts[1].strip()
					try:
						start_time = float(time_str)
						dt_utc = pd.to_datetime(start_time, unit='s', utc=True)
						dt_cet = dt_utc.tz_convert(ZoneInfo("Europe/Berlin"))
						formatted_time = dt_cet.strftime("%d.%m.%Y %H:%M")
						log.info(f"Experiment start: {formatted_time} (CET)")
					except ValueError:
						log.warning("Invalid start time format in the log file: %s", time_str)
			continue

		parts = line.split()
		if len(parts) < 8:
			log.warning("Skipping malformed log line: %s", line)
			continue

		try:
			rows.append({
				"type": parts[0].lower(),
				"infostr": parts[1],
				"n02": int(parts[2]),
				"n03": int(parts[3]),
				"counter": int(parts[4]),
				"timestamp": float(parts[5]),
				"current": float(parts[6]),
				"s_rot": float(parts[7]),
			})
		except ValueError:
			log.warning("Skipping unparsable log line: %s", line)

	if start_time is None:
		log.warning("Start time not found in the log file.")
		start_time = 0.0

	return start_time, rows

def _split_log_blocks(rows):
	"""Split log rows into acquisition blocks.

	A new block starts when:
	- image type changes;
	- counter (#04) does not strictly increase;
	- reference acquisition identifiers (#02, #03) change.

	Reference identifiers distinguish consecutive reference acquisitions
	even when their counters keep increasing.
	"""
	blocks = []
	current = []
	for index, row in enumerate(rows):
		if current:
			previous = rows[current[-1]]
			reference_acquisition_changed = (
				row["type"] == previous["type"] == "ref"
				and (
					row["n02"] != previous["n02"]
					or row["n03"] != previous["n03"]
				)
			)
			new_block = (
				row["type"] != previous["type"]
				or row["counter"] <= previous["counter"]
				or reference_acquisition_changed
			)
			if new_block:
				blocks.append(current)
				current = []
		current.append(index)
	if current:
		blocks.append(current)
	return blocks


def _match_blocks(rows, log_blocks, file_blocks):
	"""Match raw file blocks to log blocks by type and inferred counter overlap.

	Returns:
	- assignment: dict mapping log row index -> raw file path
	"""
	assignment = {}
	used_log_blocks = set()
	log_block_counters = [{rows[i]["counter"]: i for i in block} for block in log_blocks]

	for file_block in file_blocks:
		ftype = file_block[0]["type"]
		fcounters = [r["counter"] for r in file_block]

		best_block = None
		best_overlap = 0

		for block_index, log_block in enumerate(log_blocks):
			if block_index in used_log_blocks:
				continue
			if rows[log_block[0]]["type"] != ftype:
				continue

			overlap = sum(1 for counter in fcounters if counter in log_block_counters[block_index])
			if overlap > best_overlap:
				best_overlap = overlap
				best_block = block_index

		block_desc = f"{os.path.basename(file_block[0]['path'])} .. {os.path.basename(file_block[-1]['path'])}"

		if best_block is None:
			log.warning("File block %s (%d files) has no matching log block; ignored.", block_desc, len(file_block))
			continue

		used_log_blocks.add(best_block)
		counter_map = log_block_counters[best_block]
		unmatched = []

		for record in file_block:
			row_index = counter_map.get(record["counter"])
			if row_index is None:
				unmatched.append(os.path.basename(record["path"]))
			else:
				assignment[row_index] = record["path"]

		if unmatched:
			log.warning(
				"File block %s: %d file(s) without log entry: %s",
				block_desc,
				len(unmatched),
				", ".join(unmatched)
			)

	return assignment


# Function to process data from P05 nanoCT log file
# LogScan shall be location of LogScan.log file
# imgDir shall be location of the directory where the acquisition images are stored
def scanDataset(LogScan, imgDir=None, one_based=True, drop_missing=True, check_block_order=True):
	"""Process data from P05 nanoCT log file.

	Parameters
	----------
	LogScan : str
		Path to LogScan.log file.
	imgDir : str or None
		Path to directory containing acquisition TIFF files.
		If None, no raw file matching is attempted.
	one_based : bool
		Interpret filenames like ref_00020_7.tif as local index 7 meaning
		counter = 20 + 6. This matches the observed one-based naming.
	drop_missing : bool
		If True and imgDir is provided, rows without a matched raw file are
		omitted from the returned DataFrame.
	check_block_order : bool
		If True, each raw file block must have the same ordering when sorted
		by filesystem time and by inferred counter, otherwise ValueError is raised.

	Returns
	-------
	pandas.DataFrame
		Columns include:
		- time
		- image_key
		- image_file
		- file_exists
		- s_rot
		- s_stage_x
		- s_stage_z
		- current
		- log_counter
		- log_block
	"""
	start_time, rows = _parse_log(LogScan)
	log_blocks = _split_log_blocks(rows)
	assignment = {}

	if imgDir is not None:
		if not os.path.isdir(imgDir):
			log.error(f"Image directory '{imgDir}' does not exist.")
			raise FileNotFoundError(f"Image directory '{imgDir}' does not exist.")

		records = _parse_raw_files(imgDir, one_based=one_based)
		file_blocks = _split_file_blocks(records, check_block_order=check_block_order)

		log.info(
			"Log: %d rows in %d blocks; raw: %d files in %d blocks",
			len(rows), len(log_blocks), len(records), len(file_blocks)
		)
		assignment = _match_blocks(rows, log_blocks, file_blocks)
		matched_paths = set(assignment.values())
		unmatched_files = [record["path"] for record in records if record["path"] not in matched_paths]
		if unmatched_files:
			raise ValueError(
				"Some TIFF files were not matched to any log block: "
				+ ", ".join(unmatched_files)
			)
		for block_index, log_block in enumerate(log_blocks):
			missing = [rows[i]["counter"] for i in log_block if i not in assignment]
			if missing:
				image_type = rows[log_block[0]]["type"]
				missing_text = missing if len(missing) <= 20 else f"{missing[:20]}..."
				log.warning(
					"Log block %d (%s, #04 %d..%d): %d of %d file(s) missing, #04 = %s",
					block_index,
					image_type,
					rows[log_block[0]]["counter"],
					rows[log_block[-1]]["counter"],
					len(missing),
					len(log_block),
					missing_text
				)

	parsed = []
	for block_index, log_block in enumerate(log_blocks):
		for row_index in log_block:
			row = rows[row_index]
			image_path = assignment.get(row_index)
			file_exists = image_path is not None

			if imgDir is not None and drop_missing and not file_exists:
				continue

			absolute_time = start_time + row["timestamp"]
			dt = pd.to_datetime(absolute_time, unit='s', utc=True)

			parsed.append({
				"time": dt,
				"image_key": IMAGE_KEYS.get(row["type"], -1),
				"image_file": image_path if imgDir is not None else "",
				"file_exists": file_exists,
				"s_rot": row["s_rot"],
				"s_stage_x": 0.0,
				"s_stage_z": 0.0,
				"current": row["current"],
				"log_counter": row["counter"],
				"log_block": block_index
			})

	df = pd.DataFrame(parsed)
	return df

def scanDatasetWithForeignDarkFields(
	LogScan,
	imgDir,
	foreignLogScan,
	foreignRawDir,
	one_based=True,
	drop_missing=True,
	check_block_order=True,
):
	"""Append reference images from another measurement as dark fields.

	Both measurements are processed independently using scanDataset.
	Only foreign rows with image_key=1 are appended, with image_key=2.
	All original rows are preserved, including any existing dark fields.

	Original rows come first, followed by foreign dark-field rows.
	Timestamps, file paths, counters, and block identifiers retain their
	values from their respective measurements. Block identifiers are
	therefore local to each measurement, not globally unique.

	The parsing and matching errors from either measurement propagate.
	Raises ValueError if no foreign reference rows remain after parsing
	and applying drop_missing.
	"""
	options = {
		"one_based": one_based,
		"drop_missing": drop_missing,
		"check_block_order": check_block_order,
	}

	dataset = scanDataset(LogScan, imgDir=imgDir, **options)
	foreign_dataset = scanDataset(
		foreignLogScan,
		imgDir=foreignRawDir,
		**options,
	)

	# scanDataset currently returns a columnless DataFrame when empty.
	if foreign_dataset.empty:
		raise ValueError(
			f"No foreign reference images available from '{foreignLogScan}'."
		)

	dark_fields = foreign_dataset.loc[
		foreign_dataset["image_key"] == 1
	].copy()

	if dark_fields.empty:
		raise ValueError(
			f"No foreign reference images (image_key=1) available "
			f"from '{foreignLogScan}'."
		)

	dark_fields["image_key"] = 2

	log.info(
		"Appending %d foreign reference images as dark fields.",
		len(dark_fields),
	)

	return pd.concat([dataset, dark_fields], ignore_index=True)
