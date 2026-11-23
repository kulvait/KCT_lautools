#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metadata of PETRA III scans (P05, P07): NeXus HDF5 files of the beamline layouts and the params group
of zarr stores are parsed into one canonical ScanDescription (ScanDescriptionFactory); the analyses
(getExperimentInfo, scanDataset, imageDataset, beamCurrentDataset, findInsertionEvents) work on it,
and it is written to zarr (writeZarrGroup) or to HDF5 in the 2025 layout (writeH5). The module
functions take an HDF5 file, a zarr store or a ScanDescription.

Reimplementation of denpy PETRA with the same module functions: replace "from denpy import PETRA"
by "from lautools import PETRA". test_PETRA checks that both give the same results.

@author: Vojtech Kulvait
@date: 2022-2026
"""
import h5py
import json
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

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


class BeamCurrentError(ValueError):
	"""The beam current of a scan file is missing or unusable (entry/hardware/beam_current/current)."""


def _scalar_number(value):
	"""value as a finite float if it is a single number (not bool), else None."""
	if isinstance(value, np.ndarray) and value.size == 1:
		value = value.reshape(()).item()
	if isinstance(value, np.generic):
		value = value.item()
	if isinstance(value, bool) or not isinstance(value, (int, float)):
		return None
	value = float(value)
	return value if np.isfinite(value) else None


def findInsertionEvents(
	h5file,
	timeOffsetSec=None,
	zeroTime=None,
	*,
	peakDistance=5,
	minDurationSec=1.0,
	maxDurationSec=10.0,
	minCurrentIncrease=0.5,
):
	"""Find current insertion events from scan metadata (HDF5 file, zarr store, ScanDescription) or a DataFrame.

	DataFrame input must contain datetime 'time' and numeric 'current'
	columns. Event indices refer to row positions in that DataFrame.

	With zeroTime, event times are seconds relative to zeroTime.
	Otherwise, event times are pandas timestamps.
	"""
	offset = 0.0 if timeOffsetSec is None else float(timeOffsetSec)
	if not np.isfinite(offset):
		raise ValueError("timeOffsetSec must be finite.")
	if peakDistance < 1:
		raise ValueError("peakDistance must be at least 1.")
	if not 0 <= minDurationSec < maxDurationSec:
		raise ValueError("Invalid insertion duration bounds.")
	if not np.isfinite(minCurrentIncrease) or minCurrentIncrease < 0:
		raise ValueError("minCurrentIncrease must be finite and nonnegative.")
	if isinstance(h5file, pd.DataFrame):
		missing = {"time", "current"} - set(h5file.columns)
		if missing:
			raise ValueError(f"Current dataset is missing columns: {missing}")
		data = h5file[["time", "current"]].copy()
		# Do not modify the caller's DataFrame.
		if pd.api.types.is_numeric_dtype(data["time"].dtype):
			raise TypeError("DataFrame time values must be datetimes.")
		data["time"] = pd.to_datetime(data["time"], utc=True, errors="raise")
		data["time"] += pd.to_timedelta(offset, unit="s")
	else:
		# an HDF5 file, a zarr store or a ScanDescription
		data = beamCurrentDataset(h5file, timeOffsetSec=offset).copy()
		data["time"] = pd.to_datetime( data["time"], utc=True, errors="raise")
	data = data.reset_index(drop=True)
	values = pd.to_numeric(
		data["current"], errors="raise"
	).to_numpy(dtype=np.float64)
	if data["time"].isna().any() or not np.all(np.isfinite(values)):
		raise ValueError("Current timestamps and values must be valid.")
	if len(data) < 2:
		return []
	if not (data["time"].diff().iloc[1:] > pd.Timedelta(0)).all():
		raise ValueError("Current timestamps must be strictly increasing.")
	times = list(data["time"])
	origin = (
		None if zeroTime is None
		else pd.to_datetime(zeroTime, utc=True, errors="raise")
	)
	if origin is not None and pd.isna(origin):
		raise ValueError("zeroTime must be a valid datetime.")
	min_indices, _ = find_peaks(-values, distance=peakDistance)
	max_indices, _ = find_peaks(values, distance=peakDistance)
	events = []
	i = j = 0
	while i < len(min_indices) and j < len(max_indices):
		start = int(min_indices[i])
		end = int(max_indices[j])
		if end <= start:
			j += 1
			continue
		duration = (times[end] - times[start]).total_seconds()
		increase = float(values[end] - values[start])
		segment = values[start:end + 1]
		if (
			np.all(np.diff(segment) >= 0)
			and minDurationSec < duration < maxDurationSec
			and increase > minCurrentIncrease
		):
			if origin is None:
				start_time = times[start]
				end_time = times[end]
			else:
				start_time = (times[start] - origin).total_seconds()
				end_time = (times[end] - origin).total_seconds()
			events.append({
				"start_index": start,
				"end_index": end,
				"start_time": start_time,
				"end_time": end_time,
				"mid_time": start_time + (end_time - start_time) / 2,
				"current_increase": increase,
				"duration_sec": duration,
			})
		i += 1
	return events




TIMESTAMP_FLAT_BLOCKS = ("Timestamp issue: found known structure of wrongly positioned flat field blocks within "
						 "acquisition")
TIMESTAMP_REPEATED = "Timestamp issue: found repeated value with %d items offset from original"
TIMESTAMP_UNKNOWN = "Timestamp issue: unknown issue encountered proceeding with default correction"


def fix_timestamps(old_ts, keys=None, fixKnownTimestampIssues=True):
	"""(strictly increasing timestamps, issues) for frames logged with timestamps that repeat or drop
	back; the frames keep their order. issues: [(kind, (first row, last row), detail)] with kind
	"flat_blocks", "repeated" or "unknown"; timestamp_issue_messages() makes the log lines.

	Default correction: from a zero or negative step on, the frames are moved 1 ms behind the frame
	before them keeping their steps, until an original time fits again (is later than the corrected
	time before it); so only frames with a wrong time are re-timed. Each such run is classified:
	- flat_blocks (known structure, needs keys = image_key): the run is exactly a flat field block
	  (key 1) whose first time is the first time of the block before it (seen in P05/P07 files: the
	  flats after the darks and the flats after the projections); fixed by centring it between the
	  frames before and after it, or at the end of the scan by putting it after the frame before it
	  with the pause of the transition into that frame's block (moving the object out as long as in);
	- repeated (known): a single frame with exactly the time of the frame n rows before it, the frames
	  around it on time (seen with n = 20); fixed by putting it halfway between its neighbours;
	- unknown: the default correction.
	With fixKnownTimestampIssues=False the known issues are reported but get the default correction.
	"""
	old_ts = np.asarray(old_ts, dtype=np.int64)
	keys = None if keys is None else np.asarray(keys)
	new_ts = old_ts.copy()
	offset = 0
	for i in range(1, len(old_ts)):
		if old_ts[i] > new_ts[i - 1]:
			offset = 0   # the original time fits again
		elif old_ts[i] + offset <= new_ts[i - 1]:
			offset = new_ts[i - 1] + 1 - old_ts[i]   # a repeat or a drop
		new_ts[i] = old_ts[i] + offset
	changed = np.nonzero(new_ts != old_ts)[0]
	issues = []
	if len(changed) == 0:
		return new_ts, issues

	def block_start(row):
		while row > 0 and keys[row - 1] == keys[row]:
			row -= 1
		return row

	for run in np.split(changed, np.nonzero(np.diff(changed) != 1)[0] + 1):
		first, last = int(run[0]), int(run[-1])
		at_end = last + 1 == len(old_ts)
		before = new_ts[first - 1]
		kind = "unknown"
		if keys is not None and keys[first] == 1 and keys[first] != keys[first - 1] and \
				(at_end or keys[last + 1] != keys[last]) and (keys[first:last + 1] == 1).all():
			previous = block_start(first - 1)
			if abs(int(old_ts[first]) - int(old_ts[previous])) <= 1:
				kind = "flat_blocks"
		elif len(run) == 1 and not at_end and old_ts[first + 1] > before:
			copies = [n for n in range(1, first + 1) if old_ts[first - n] == old_ts[first]]
			if copies:
				kind, distance = "repeated", copies[0]
		if kind != "unknown" and fixKnownTimestampIssues:
			if kind == "repeated":
				new_ts[first] = before + (new_ts[first + 1] - before) // 2
			elif not at_end:
				span = new_ts[last] - new_ts[first]
				free = new_ts[last + 1] - before
				if free - span >= 2:
					new_ts[run] += before + (free - span) // 2 - new_ts[first]
			else:
				steps = np.diff(new_ts[run])
				gap = int(np.median(steps)) if len(steps) else 1
				previous = block_start(first - 1)
				if previous > 0 and new_ts[previous] - new_ts[previous - 1] > gap:
					gap = int(new_ts[previous] - new_ts[previous - 1])
				new_ts[run] += before + gap - new_ts[first]
		rows = "row %d" % first if first == last else "rows %d-%d" % (first, last)
		detail = "%s %+d ms" % (rows, int(new_ts[first] - old_ts[first]))
		if kind == "repeated":
			kind = "repeated %d" % distance
			detail += " (time of row %d)" % (first - distance)
		issues.append((kind, (first, last), detail))
	return new_ts, issues


def timestamp_issue_messages(issues, fixKnownTimestampIssues=True):
	"""One line per kind of issue of fix_timestamps, with its rows."""
	lines = []
	for kind in dict.fromkeys(k for k, _, _ in issues):
		details = ", ".join(d for k, _, d in issues if k == kind)
		if kind == "flat_blocks":
			text = TIMESTAMP_FLAT_BLOCKS
		elif kind.startswith("repeated"):
			text = TIMESTAMP_REPEATED % int(kind.split()[1])
		else:
			text = TIMESTAMP_UNKNOWN
		text += ": " + details
		if kind != "unknown":
			text += "; fixed" if fixKnownTimestampIssues else "; default correction (fixKnownTimestampIssues=False)"
		lines.append(text)
	return lines


def normalize_timestamps_preserve_increase(old_ts, keys=None, fixKnownTimestampIssues=True):
	"""Strictly increasing timestamps, the frames in their order: see fix_timestamps."""
	return fix_timestamps(old_ts, keys, fixKnownTimestampIssues)[0]



# ---------------------------------------------------------------- scan description
#
# The metadata of a P05 scan exist as NeXus HDF5 files in two layouts (2021: attributes and strings
# as (1,) object arrays; 2025: plain string attributes, scalar strings, empty (0,) logs) and as the
# params group of zarr stores. They are parsed into one canonical form, the ScanDescription, and
# every function of this module works on it, so there is one parser per format and one
# implementation of each analysis.
#
# The canonical form follows the zarr data model, a tree of groups (ScanGroup), each with
#	- attributes: JSON values; the HDF5 attributes and the scalar datasets, numpy values as python
#	  values, bytes as str, one-element sequences as their element
#	- arrays: numpy arrays with their dtype; strings as object arrays of str
#	- groups
# A string array of one element is a str attribute (an artefact of the writing software, e.g.
# entry/beamline/name of 2021 files), except in logs (NXlog groups, or groups with time and value),
# whose time and value arrays stay arrays whatever their length.

SCAN_TABLE_KEYS = ("columns", "index", "data")  # pandas orient="split" scan table on the params group
H5_ATTRIBUTES = ("NX_class", )  # attributes that are HDF5 attributes in the 2025 layout, others are scalar datasets


def sanitizeForJson(obj):
	"""Metadata to JSON values: numpy to python, bytes to str, arrays to lists; one-element sequences
	collapse to their element, empty and longer ones are kept."""
	if isinstance(obj, np.ndarray):
		return sanitizeForJson(obj.tolist())
	if isinstance(obj, np.generic):
		return sanitizeForJson(obj.item())
	if isinstance(obj, (bytes, bytearray)):
		try:
			return obj.decode("utf-8")
		except UnicodeDecodeError:
			return obj.decode("utf-8", errors="replace")
	if isinstance(obj, dict):
		return {str(k): sanitizeForJson(v) for k, v in obj.items()}
	if isinstance(obj, (list, tuple)):
		if len(obj) == 1:
			return sanitizeForJson(obj[0])
		return [sanitizeForJson(value) for value in obj]
	return obj


def _isStringDtype(dtype):
	return dtype.kind in "OSUT"


def _stringArray(values):
	"""An object array of str (from bytes, numpy strings, str), the canonical string array."""
	values = np.asarray(values)
	flat = [sanitizeForJson(v) if isinstance(v, (bytes, bytearray, np.generic)) else v for v in values.ravel()]
	result = np.empty(len(flat), dtype=object)
	result[:] = [v if isinstance(v, str) else str(v) for v in flat]
	return result.reshape(values.shape)


class _LazyH5Array:
	"""A dataset of an HDF5 file not read yet: shape and dtype known, values read on first use."""

	def __init__(self, filename, name, shape, dtype, string):
		self.filename, self.name, self.shape, self.dtype, self.string = filename, name, shape, dtype, string
		self.size = int(np.prod(shape))

	def read(self, h5):
		values = h5[self.name][()]
		return _stringArray(values) if self.string else np.asarray(values)


class _LazyH5Group:
	"""A group of an HDF5 file not read yet (its subgroups are read when used in turn)."""

	def __init__(self, filename, name):
		self.filename, self.name = filename, name

	def read(self, h5):
		return _groupFromH5(h5[self.name])


class _Members(dict):
	"""The arrays or the groups of a ScanGroup; members of an HDF5 file are read when first used, all
	not yet read members of the group at once (one opening of the file)."""

	def _load(self):
		lazy = [(k, v) for k, v in dict.items(self) if isinstance(v, (_LazyH5Array, _LazyH5Group))]
		if lazy:
			with h5py.File(lazy[0][1].filename, "r") as h5:
				for key, value in lazy:
					dict.__setitem__(self, key, value.read(h5))

	def __getitem__(self, key):
		if isinstance(dict.__getitem__(self, key), (_LazyH5Array, _LazyH5Group)):
			self._load()
		return dict.__getitem__(self, key)

	def get(self, key, default=None):
		return self[key] if key in self else default

	def values(self):
		self._load()
		return dict.values(self)

	def items(self):
		self._load()
		return dict.items(self)


class ScanGroup:
	"""A group of a ScanDescription: attrs (JSON values), arrays (numpy), groups (ScanGroup).

	Paths are relative and separated by "/". A path names a group, an array or an attribute (a
	scalar of the source); value(path) gives the scalar wherever it is stored."""

	def __init__(self, attrs=None, arrays=None, groups=None):
		self.attrs = dict(attrs or {})
		self.arrays = _Members(arrays or {})
		self.groups = _Members(groups or {})

	@classmethod
	def canonical(cls, attrs, arrays, groups, where="/"):
		"""The group in the canonical form: string arrays as object arrays of str, one-element string
		arrays outside logs as str attributes, attributes JSON values (an attribute wins over a scalar of
		the same name)."""
		attrs = {str(k): sanitizeForJson(v) for k, v in attrs.items()}
		for key, value in attrs.items():
			try:
				json.dumps(value)
			except (TypeError, ValueError, OverflowError) as exc:
				raise ValueError(f"Attribute {where}@{key} is no JSON value: type={type(value).__name__}, value={value!r}") from exc
		names = set(arrays) | set(groups)
		log = attrs.get("NX_class") == "NXlog" or {"time", "value"} <= names
		canonical = {}
		for name, array in arrays.items():
			if isinstance(array, _LazyH5Array):  # its form is known from shape and dtype
				if array.string and array.size == 1 and not log:
					with h5py.File(array.filename, "r") as h5:
						attrs.setdefault(name, array.read(h5).ravel()[0])
					continue
				canonical[name] = array
				continue
			array = np.asarray(array)
			if _isStringDtype(array.dtype):
				array = _stringArray(array)
				if array.size == 1 and not log:
					attrs.setdefault(name, array.ravel()[0])
					continue
			canonical[name] = array
		return cls(attrs, canonical, groups)

	def _split(self, path):
		parts = [p for p in str(path).split("/") if p]
		if not parts:
			return self, None
		group = self
		for part in parts[:-1]:
			group = group.groups.get(part)
			if group is None:
				return None, None
		return group, parts[-1]

	def group(self, path):
		"""The group at path or None."""
		parent, name = self._split(path)
		if parent is None:
			return None
		return parent if name is None else parent.groups.get(name)

	def array(self, path):
		"""The array at path or None."""
		parent, name = self._split(path)
		return None if parent is None or name is None else parent.arrays.get(name)

	def __contains__(self, path):
		parent, name = self._split(path)
		return parent is not None and (name is None or name in parent.groups or name in parent.arrays or name in parent.attrs)

	def __getitem__(self, path):
		parent, name = self._split(path)
		if parent is not None:
			if name is None:
				return parent
			for members in (parent.groups, parent.arrays, parent.attrs):
				if name in members:
					return members[name]
		raise KeyError(path)

	def value(self, path, default=None):
		"""The scalar at path: an attribute, or the element of a one-element array; None for an empty
		array; a longer array as it is; default if path does not exist or is a group."""
		parent, name = self._split(path)
		if parent is None or name is None:
			return default
		if name in parent.attrs:
			return parent.attrs[name]
		if name in parent.arrays:
			array = parent.arrays[name]
			if array.size == 0:
				return None
			if array.size == 1:
				return sanitizeForJson(array.ravel()[0])
			return array
		return default

	def data(self, path):
		"""The values at path as a 1-D numpy array (an array raveled, an attribute as one element), or None."""
		parent, name = self._split(path)
		if parent is None or name is None:
			return None
		if name in parent.arrays:
			return np.asarray(parent.arrays[name]).ravel()
		if name in parent.attrs:
			return np.asarray(parent.attrs[name]).ravel()
		return None

	def scalars(self):
		"""The values of the arrays and attributes of the group (not NX_class, not subgroups), as value()."""
		values = {name: self.value(name) for name in self.arrays}
		values.update({k: v for k, v in self.attrs.items() if k not in H5_ATTRIBUTES})
		return values

	def differences(self, other, path=""):
		"""Differences to another group (attributes, arrays with dtype and values, groups), for tests and checks."""
		problems = []
		where = path or "/"
		if self.attrs != other.attrs:
			keys = sorted(k for k in set(self.attrs) | set(other.attrs) if self.attrs.get(k, "<missing>") != other.attrs.get(k, "<missing>"))
			problems.append(f"{where}: attributes differ in {keys}")
		for kind in ("arrays", "groups"):
			mine, theirs = getattr(self, kind), getattr(other, kind)
			if set(mine) != set(theirs):
				problems.append(f"{where}: {kind} {sorted(set(mine) ^ set(theirs))} only in one")
		for name in sorted(set(self.arrays) & set(other.arrays)):
			a, b = self.arrays[name], other.arrays[name]
			if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
				problems.append(f"{path}/{name}: {a.dtype}{a.shape} differs from {b.dtype}{b.shape}")
		for name in sorted(set(self.groups) & set(other.groups)):
			problems.extend(self.groups[name].differences(other.groups[name], f"{path}/{name}"))
		return problems

	def __eq__(self, other):
		return isinstance(other, ScanGroup) and not self.differences(other)


def _camera_values(camera_group, camera, overrideMagnification=None):
	"""Camera numbers under the names of the newer files, None where missing, empty or not a number.

	Older names: px_size/pixel_size, senzor_xsize/senzor_ysize, roi_xsize/roi_ysize. The
	magnification is taken from overrideMagnification, the camera group or its calibration subgroup;
	a magnification of 0 counts as missing.
	"""
	def first(*keys):
		for key in keys:
			value = _scalar_number(camera.get(key))
			if value is not None:
				return value
		return None
	values = {
		"pixelsize": first("pixel_size", "px_size", "pixelsize"),
		"sensorsize_x": first("senzor_xsize", "sensorsize_x"),
		"sensorsize_y": first("senzor_ysize", "sensorsize_y"),
		"roi_width": first("roi_xsize", "roi_width"),
		"roi_height": first("roi_ysize", "roi_height"),
	}
	magnification = _scalar_number(overrideMagnification) if overrideMagnification is not None else first("magnification")
	if magnification is None and overrideMagnification is None:
		magnification = _scalar_number(camera_group.value("calibration/magnification"))
	values["magnification"] = magnification if magnification else None
	return values


def _insert_label(df, data, label_name, key_col="image_key"):
	"""Insert the log data/label_name (time and value) into df by its timestamps: the whole table or
	the frames without key 2 or without keys 1 and 2."""
	time_ds, value_ds = data.array(f"{label_name}/time"), data.array(f"{label_name}/value")
	if time_ds is None or value_ds is None:
		raise ValueError(f"{label_name} has no time and value arrays.")
	if time_ds.ndim != 1 or value_ds.ndim != 1:
		raise ValueError(f"{label_name} time and value must be 1-D, got shapes {time_ds.shape} and {value_ds.shape}.")
	if len(time_ds) != len(value_ds):
		raise ValueError(f"{label_name} has {len(time_ds)} times but {len(value_ds)} values.")
	if len(time_ds) == 0:
		raise ValueError(f"{label_name} has empty time and value datasets.")
	if len(time_ds) == len(df):
		if not np.all(time_ds == df["time"].values):
			raise ValueError(f"{label_name}/time does not match df['time']")
		df[label_name] = value_ds
		return df
	filters = [
		(df[key_col] != 2),  # exclude key==2
		((df[key_col] != 2) & (df[key_col] != 1))  # exclude key 1 and 2
	]
	for mask in filters:
		if mask.sum() == len(time_ds):
			if not np.all(time_ds == df.loc[mask, "time"].values):
				raise ValueError(f"{label_name}/time shape matches subset ({mask.sum()} rows) "
								 "but timestamps do not align exactly.")
			df.loc[mask, label_name] = value_ds
			return df
	raise ValueError(f"Could not insert {label_name}: no matching subset found.")


class ScanDescription:
	"""The metadata of a P05 scan in the canonical form (see above), from an HDF5 file or the params
	group of a zarr store (ScanDescriptionFactory), with the analyses of the scan and the writers to
	zarr and to HDF5 (2025 layout).

	root: the ScanGroup of /entry (HDF5) or params (zarr).
	source: the file or store it was read from; h5Path: the HDF5 file of the scan (for a zarr
	store as recorded there), reported by getExperimentInfo.
	From an HDF5 file, groups and arrays are read when first used (the members of a group at a time;
	P05 files have about 150 groups and store arrays in chunks of one element, slow to read), so the
	file has to stay where it is until then; load() reads everything.
	"""

	def __init__(self, root, source=None, h5Path=None):
		self.root = root
		self.source = source
		self.h5Path = h5Path if h5Path is not None else source

	@property
	def name(self):
		return self.source or "<scan description>"

	def load(self):
		"""Read all arrays not read yet (from an HDF5 file); returns self."""
		todo = [self.root]
		while todo:
			group = todo.pop()
			group.arrays.values()
			todo.extend(group.groups.values())
		return self

	def __eq__(self, other):
		return isinstance(other, ScanDescription) and self.root == other.root

	def differences(self, other):
		return self.root.differences(other.root)

	# ------------------------------------------------ analyses

	def _string(self, path):
		if path not in self.root:
			log.warning("The path 'entry/%s' does not exist in %s." % (path, self.name))
			return None
		value = self.root.value(path)
		if value is None:
			log.warning("The path 'entry/%s' is empty in %s." % (path, self.name))
			return None
		return value if isinstance(value, str) else str(value)

	def _camera_group(self):
		"""hardware/camera, else hardware/camera1, else None."""
		for path in ("hardware/camera", "hardware/camera1"):
			group = self.root.group(path)
			if group is not None:
				return group
		return None

	def getExperimentInfo(self, overrideMagnification=None):
		"""Experiment, beamline, times and frame rates, setup (distance, energy), camera (pixel size,
		field of view) and the Fresnel number of the scan."""
		info = {}
		h5 = os.path.realpath(self.h5Path) if self.h5Path else None
		info["h5"] = h5
		info["h5_name"] = os.path.basename(h5) if h5 else None
		experimentString = self._string("beamline/experiment")
		if experimentString is not None:
			info["experiment"] = experimentString
		beamlineString = self._string("beamline/name")
		if beamlineString is not None:
			info["beamline"] = beamlineString
		image_file_time = self.root.data("scan/data/image_file/time")
		if image_file_time is not None and len(image_file_time) > 0:
			image_file_time = pd.to_datetime(image_file_time.astype(np.int64), unit="ms")
			start, end = image_file_time.min(), image_file_time.max()
			info["start_date_time"] = start.strftime("%d.%m.%Y %H:%M:%S")
			info["end_date_time"] = end.strftime("%d.%m.%Y %H:%M:%S")
			info["scan_frame_count"] = len(image_file_time)
			info["scan_duration_sec"] = (end - start).total_seconds()
			info["scan_fps"] = len(image_file_time) / info["scan_duration_sec"] if info["scan_duration_sec"] > 0 else None
		image_key = self.root.data("scan/data/image_key/value")
		image_key_time = self.root.data("scan/data/image_key/time")
		if image_key is not None and image_key_time is not None:
			if len(image_key) != len(image_key_time):
				log.warning("image_key has %d values but %d times in %s; sample frame statistics skipped." % (len(image_key), len(image_key_time), self.name))
			else:
				image_times = pd.to_datetime(image_key_time[image_key == 0].astype(np.int64), unit="ms")
				if len(image_times) > 0:
					info["sample_frame_count"] = len(image_times)
					info["sample_duration_sec"] = (image_times.max() - image_times.min()).total_seconds()
					info["sample_fps"] = len(image_times) / info["sample_duration_sec"] if info["sample_duration_sec"] > 0 else None
		setup = {}
		camera = {}
		setup_group = self.root.group("scan/setup")
		if setup_group is not None:
			setup = setup_group.scalars()
			for key in ("pos_o_ccd_dist", "o_ccd_dist"):
				distance = _scalar_number(setup.get(key))
				if distance is not None:
					info["propagation_distance_mm"] = round(distance, 0)
					break
			for key in ("pos_p05_energy", "pos_p07_energy", "p05_energy", "p07_energy"):
				energy = _scalar_number(setup.get(key))
				if energy is not None:
					info["energy_keV"] = round(energy / 1000, 3)
					break
		else:
			log.warning("The path 'entry/scan/setup' does not exist in %s." % self.name)
		camera_group = self._camera_group()
		if camera_group is None:
			log.warning("Neither 'entry/hardware/camera' nor 'entry/hardware/camera1' exists in %s." % self.name)
		else:
			camera = camera_group.scalars()
			values = _camera_values(camera_group, camera, overrideMagnification)
			if overrideMagnification is not None:
				camera["magnification"] = overrideMagnification
			elif values["magnification"] is not None:
				camera["magnification"] = values["magnification"]
			# Names of older files, as the newer ones
			for key in ("pixelsize", "sensorsize_x", "sensorsize_y", "roi_width", "roi_height"):
				if values[key] is not None:
					camera[key] = values[key]
			# Without a ROI, the whole sensor
			for roi, sensor in (("roi_width", "sensorsize_x"), ("roi_height", "sensorsize_y")):
				if values[roi] is None and values[sensor] is not None:
					camera[roi] = values[sensor]
			if values["magnification"] is not None and values["pixelsize"] is not None:
				info["pix_size"] = values["pixelsize"] / values["magnification"]
				width = values["roi_width"] if values["roi_width"] is not None else values["sensorsize_x"]
				height = values["roi_height"] if values["roi_height"] is not None else values["sensorsize_y"]
				if width is not None:
					info["field_of_view_x"] = info["pix_size"] * width
				if height is not None:
					info["field_of_view_y"] = info["pix_size"] * height
			else:
				log.warning("No usable camera pixelsize and magnification in %s; pix_size not set." % self.name)
		if "propagation_distance_mm" in info and "energy_keV" in info and "pix_size" in info and info["energy_keV"] > 0 \
				and info["propagation_distance_mm"] > 0 and info["pix_size"] > 0:
			wavelength_m = 1.2398419843320026e-9 / info["energy_keV"]
			propagation_distance_m = info["propagation_distance_mm"] / 1000.0
			pix_size_m = info["pix_size"] / 1000.0
			info["fresnel_number"] = (pix_size_m ** 2) / (wavelength_m * propagation_distance_m)
		info["setup"] = setup
		info["camera"] = camera
		return info

	def getBasicDatasetInfo(self):
		"""DataFrame of the frames: time (ms), image_key, image_file, as logged; warns about times
		with several frames."""
		data = self.root.group("scan/data")
		if data is None:
			raise ValueError(f"No entry/scan/data in {self.name}")
		arrays = {}
		for name in ("image_key/time", "image_key/value", "image_file/time", "image_file/value"):
			arrays[name] = data.array(name)
			if arrays[name] is None:
				raise ValueError(f"No entry/scan/data/{name} in {self.name}")
		key_time, key_value = arrays["image_key/time"], arrays["image_key/value"]
		file_time, file_value = arrays["image_file/time"], arrays["image_file/value"]
		lengths = [len(key_time), len(key_value), len(file_time), len(file_value)]
		if len(set(lengths)) != 1:
			raise ValueError(f"Length mismatch: key_time={len(key_time)}, key_value={len(key_value)}, "
							 f"file_time={len(file_time)}, file_value={len(file_value)}")
		if not np.array_equal(key_time, file_time):
			raise ValueError("Time arrays for image_key and image_file do not match.")
		df = pd.DataFrame({"time": key_time, "image_key": key_value, "image_file": file_value})
		duplicates = df[df.duplicated(subset=['time'], keep=False)].copy()
		if not duplicates.empty:
			duplicates["orig_index"] = duplicates.index
			n_duplicated_times = duplicates['time'].nunique()
			log.warning("There is %d duplicated entries in %d distinct times in file %s." %
						(len(duplicates), n_duplicated_times, self.name))
			times_to_duplicates = duplicates.groupby('time')['image_key'].apply(list).to_dict()
			duplicates = duplicates.sort_values(by=['time', 'image_key'])
			duplicates["key_set"] = duplicates["time"].map(lambda t: tuple(sorted(set(times_to_duplicates[t]))))
			for key_set, group in duplicates.groupby('key_set'):
				first_time = group['time'].iloc[0]
				affected = group[group['time'] == first_time][['image_file', 'orig_index']]
				formatted_key_set = ", ".join(str(k) for k in key_set)
				affected_str = ", ".join(f"{f} (id={i})" for f, i in zip(affected['image_file'], affected['orig_index']))
				log.warning(f"key_set=[{formatted_key_set}], n_times={group['time'].nunique()}, "
							f"first_time={first_time}, affected_files={affected_str}")
		return df

	def beamCurrentDataset(self, timeOffsetSec=None):
		"""DataFrame of the beam current readings (mA): index the UNIX time in ms (shifted by
		timeOffsetSec), columns time and current; BeamCurrentError if there are none."""
		timestampadjustment = np.int64(0) if timeOffsetSec is None else np.int64(timeOffsetSec * 1000)
		beam_current_path = "hardware/beam_current/current"
		parts = {}
		for part in ("time", "value"):
			parts[part] = self.root.data(f"{beam_current_path}/{part}")
			if parts[part] is None:
				raise BeamCurrentError(f"No beam current in {self.name}: 'entry/{beam_current_path}/{part}' does not exist.")
		ID = parts["time"].astype(np.uint64)
		value = parts["value"].astype(np.float64)
		if len(ID) != len(value):
			raise BeamCurrentError(f"Beam current in {self.name} has {len(ID)} timestamps but {len(value)} values.")
		valid = np.isfinite(value)
		if not valid.all():
			log.warning("Ignoring %d of %d beam current readings without a finite value in %s." % ((~valid).sum(), len(value), self.name))
			ID, value = ID[valid], value[valid]
		df = pd.DataFrame({"time": pd.to_datetime(ID.astype(np.int64), unit="ms"), "current": value}, index=ID)
		df = df[~df.index.duplicated(keep='first')]  # duplicate timestamps: the first reading
		df.drop(df[df.index == 0].index, inplace=True)  # timestamp 0 is invalid
		df.sort_index(inplace=True)
		if timestampadjustment != 0:
			adjusted = df.index.to_numpy(dtype=np.int64) + timestampadjustment
			if np.any(adjusted < 0):
				raise ValueError(f"Time offset {timeOffsetSec}s seconds results in negative timestamps. Invalid uint64 index.")
			df.index = pd.Index(adjusted.astype(np.uint64), name=df.index.name)
			df["time"] = pd.to_datetime(df.index, unit="ms")
		return df

	def scanDataset(self, includeCurrent=False, timeOffsetSec=None, fixKnownTimestampIssues=True):
		"""DataFrame of the frames of a scan: time, image_key, image_file and the logs of entry/scan/data.

		With includeCurrent, the column current holds the beam current interpolated at the frame times
		(constant before the first and after the last reading); raises BeamCurrentError if there is no
		usable beam current. timeOffsetSec is the offset applied to the current readings to be in sync
		with the frame times.
		"""
		if timeOffsetSec is None:
			timeOffsetSec = 0.0
		df = self.getBasicDatasetInfo()
		data = self.root.group("scan/data")
		labels = sorted(set(data.groups) | set(data.arrays))
		if len(labels) < 1:
			raise ValueError("Error: labels count is %d!" % (len(labels)))
		for lab in labels:
			if lab in ("image_key", "image_file"):
				continue
			if lab not in data.groups:
				log.warning(f"Label '{lab}' not inserted into DataFrame from file {self.name}: not a group with time and value.")
				continue
			try:
				df = _insert_label(df.copy(deep=True), data, lab, key_col="image_key")  # on a copy, no partial insertion
			except ValueError as ve:
				log.warning(f"Label '{lab}' not inserted into DataFrame from file {self.name}: {ve}")
		time_values = df["time"].values.astype(np.int64)
		if not all(np.diff(time_values) > 0):
			normalized, issues = fix_timestamps(time_values, df["image_key"].to_numpy(), fixKnownTimestampIssues)
			for text in timestamp_issue_messages(issues, fixKnownTimestampIssues):
				log.warning("%s (%s)" % (text, self.name))
			df["time"] = normalized
		df.index = df["time"].values
		df["time"] = pd.to_datetime(df["time"].astype(np.int64), unit="ms")
		if includeCurrent:
			currentFrame = self.beamCurrentDataset(timeOffsetSec=timeOffsetSec)
			if currentFrame.empty:
				raise BeamCurrentError(f"No beam current readings in {self.name}, can not include the current.")
			current_times = currentFrame.index.to_numpy().astype(np.float64)
			frame_times = df.index.to_numpy().astype(np.float64)
			outside = int(np.sum(frame_times < current_times[0]) + np.sum(frame_times > current_times[-1]))
			if outside > 0:
				log.warning("%d of %d frames of %s are outside the beam current readings; the first/last reading is used for them." % (outside, len(df), self.name))
			df["current"] = np.interp(frame_times, current_times, currentFrame["current"].to_numpy(dtype=np.float64))
		df = df.sort_values("time", ascending=True)
		return df

	def imageDataset(self, image_key=0, includeCurrent=True, includePixelShift=False, overrideMagnification=None,
					 fixKnownTimestampIssues=True):
		"""The frames of one image_key with frame_ind and optionally pixel_shift (from s_stage_x)."""
		df = self.scanDataset(includeCurrent, fixKnownTimestampIssues=fixKnownTimestampIssues)
		df = df.loc[df["image_key"] == image_key]
		df = df.assign(frame_ind=np.arange(len(df)))
		if includePixelShift:
			if "s_stage_x" not in df.columns:
				raise ValueError("No s_stage_x log in %s, can not compute the pixel shift." % self.name)
			camera_group = self._camera_group()
			if camera_group is None:
				raise ValueError("There is no entry/hardware/camera or entry/hardware/camera1 entry in %s." % self.name)
			values = _camera_values(camera_group, camera_group.scalars(), overrideMagnification)
			if values["pixelsize"] is None or values["magnification"] is None:
				raise ValueError("No usable camera pixelsize and magnification in %s, can not compute the pixel shift." % self.name)
			pix_size = values["pixelsize"] / values["magnification"]
			zeropos = 0.5 * (df["s_stage_x"].max() + df["s_stage_x"].min())
			df = df.assign(pixel_shift=(df["s_stage_x"] - zeropos) / pix_size)
		return df

	def findInsertionEvents(self, timeOffsetSec=None, zeroTime=None, **kwargs):
		"""Current insertion events from the beam current, see the module function findInsertionEvents."""
		return findInsertionEvents(self, timeOffsetSec, zeroTime, **kwargs)

	# ------------------------------------------------ writers

	def rootAttributes(self, export=None, overrideMagnification=None):
		"""JSON root attributes of a zarr store of the scan: the experiment info and, if given, export
		(a dict describing the conversion)."""
		info = self.getExperimentInfo(overrideMagnification=overrideMagnification)
		if export is not None:
			info["export"] = export
		return sanitizeForJson(info)

	def writeZarrGroup(self, zarrGroup, name="params", scanTable=True, compressor=None):
		"""Create the group name in zarrGroup with the description (attributes, arrays, groups); with
		scanTable (True: scanDataset(), or a DataFrame) also the scan table as pandas "split"
		attributes columns, index, data. Returns the created group."""
		attributes = {}
		if scanTable is not False and scanTable is not None:
			table = self.scanDataset() if scanTable is True else scanTable
			attributes.update(json.loads(table.to_json(orient="split", date_format="iso")))
		attributes.update(self.root.attrs)
		group = zarrGroup.create_group(name, attributes=attributes)
		_writeZarrMembers(self.root, group, compressor)
		return group

	def writeH5(self, path, chunk=1):
		"""Write the description as the /entry of an HDF5 file in the 2025 P05 layout: NX_class as
		plain string attributes, the other attributes as scalar datasets (strings as variable-length
		ASCII, UTF-8 if not ASCII), arrays chunked (chunk, ...) with an unlimited first dimension and
		no compression."""
		with h5py.File(path, "w") as h5:
			_writeH5Group(self.root, h5.create_group("entry"), chunk)


def _writeZarrMembers(group, zarrGroup, compressor):
	for name in sorted(group.groups):
		member = group.groups[name]
		_writeZarrMembers(member, zarrGroup.create_group(name, attributes=member.attrs), compressor)
	for name in sorted(group.arrays):
		array = group.arrays[name]
		if array.dtype == object:
			zarrGroup.create_array(name, shape=array.shape, dtype="string")[...] = array
		else:
			zarrGroup.create_array(name, shape=array.shape, dtype=array.dtype, compressors=compressor)[...] = array


def _h5StringDtype(values):
	"""Variable-length ASCII bytes as the beamline writes them, UTF-8 if a value is not ASCII."""
	return h5py.string_dtype("ascii" if all(v.isascii() for v in values) else "utf-8")


def _h5Encode(values, dtype):
	return [v.encode(h5py.check_string_dtype(dtype).encoding) for v in values]


def _writeH5Scalar(hgroup, name, value):
	if isinstance(value, str):
		dtype = _h5StringDtype([value])
		hgroup.create_dataset(name, data=_h5Encode([value], dtype)[0], dtype=dtype)
	elif isinstance(value, (bool, int, float)):
		hgroup.create_dataset(name, data=np.array(value))
	elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
		dtype = _h5StringDtype(value)
		hgroup.create_dataset(name, data=np.array(_h5Encode(value, dtype), dtype=object), dtype=dtype)
	elif isinstance(value, list) and all(isinstance(v, (bool, int, float)) for v in value):
		hgroup.create_dataset(name, data=np.array(value))
	else:
		# None, dictionaries and mixed lists have no HDF5 dataset form; kept as JSON text
		text = json.dumps(value)
		log.warning(f"{hgroup.name}/{name} is stored as JSON text: {text[:100]}")
		hgroup.create_dataset(name, data=text.encode(), dtype=_h5StringDtype([text]))


def _writeH5Group(group, hgroup, chunk):
	for key, value in group.attrs.items():
		if key in H5_ATTRIBUTES:
			hgroup.attrs[key] = value if isinstance(value, str) else json.dumps(value)
		else:
			_writeH5Scalar(hgroup, key, value)
	for name in sorted(group.groups):
		_writeH5Group(group.groups[name], hgroup.create_group(name), chunk)
	for name in sorted(group.arrays):
		data = group.arrays[name]
		if data.dtype == object:
			values = [str(v) for v in data.ravel()]
			dtype = _h5StringDtype(values)
			data = np.array(_h5Encode(values, dtype), dtype=object).reshape(data.shape)
		else:
			dtype = data.dtype
		hgroup.create_dataset(name, data=data, dtype=dtype, maxshape=(None, ) + data.shape[1:],
							  chunks=(chunk, ) + tuple(max(1, n) for n in data.shape[1:]))


def _groupFromH5(hgroup):
	attrs, arrays, groups = {}, {}, {}
	for name in sorted(hgroup):
		item = hgroup.get(name)
		if isinstance(item, h5py.Group):  # read when first used
			groups[name] = _LazyH5Group(item.file.filename, item.name)
		elif isinstance(item, h5py.Dataset):
			if item.shape == ():
				attrs[name] = sanitizeForJson(item[()])
			else:  # read when first used
				arrays[name] = _LazyH5Array(item.file.filename, item.name, item.shape, item.dtype, _isStringDtype(item.dtype))
		else:
			log.warning(f"HDF5 entry {hgroup.name}/{name} is neither a group nor a dataset (a broken link?), skipped.")
	attrs.update({k: sanitizeForJson(v) for k, v in hgroup.attrs.items()})  # attributes win over scalar datasets
	return ScanGroup.canonical(attrs, arrays, groups, hgroup.name)


def _groupFromZarr(zgroup, top=False):
	import zarr
	attrs = dict(zgroup.attrs)
	if top:
		for key in SCAN_TABLE_KEYS:
			attrs.pop(key, None)
	arrays, groups = {}, {}
	for name, member in sorted(zgroup.members()):
		if isinstance(member, zarr.Group):
			groups[name] = _groupFromZarr(member)
		else:
			arrays[name] = np.asarray(member[...])
	return ScanGroup.canonical(attrs, arrays, groups, "/" + zgroup.path)


class ScanDescriptionFactory:
	"""ScanDescription from an HDF5 file (fromH5), the params group of a zarr store (fromZarr), or
	whichever of them a source is (create)."""

	@staticmethod
	def fromH5(source):
		"""From an HDF5 file name, h5py.File or h5py.Group (its entry group if it has one)."""
		if isinstance(source, (str, os.PathLike)):
			path = os.path.realpath(os.fspath(source))
			with h5py.File(path, "r") as h5:
				return ScanDescriptionFactory.fromH5(h5)
		path = os.path.realpath(source.file.filename)
		group = source["entry"] if "entry" in source else source
		return ScanDescription(_groupFromH5(group), source=path, h5Path=path)

	@staticmethod
	def fromZarr(source, path="params"):
		"""From a zarr store (directory or zip file name) or zarr.Group: its group path (the group itself
		if it has no such member). The HDF5 file of the scan is taken from the root attributes (h5)."""
		import zarr
		if isinstance(source, (str, os.PathLike)):
			location = os.path.realpath(os.fspath(source))
			if os.path.isfile(location):
				from denpy import ZARZS
				store = ZARZS.ZipStore(location, mode="r")
			else:
				store = location
			try:
				root = zarr.open_group(store, mode="r")
				description = ScanDescriptionFactory.fromZarr(root, path)
			finally:
				if not isinstance(store, str):
					store.close()
			description.source = location
			if description.h5Path is None:
				description.h5Path = location
			return description
		group = source[path] if path and path in source else source
		if not isinstance(group, zarr.Group):
			raise ValueError(f"{path} of the zarr store is no group")
		h5 = source.attrs.get("h5")
		return ScanDescription(_groupFromZarr(group, top=True), source=None, h5Path=h5 if isinstance(h5, str) else None)

	@staticmethod
	def create(source):
		"""A ScanDescription as it is, or from an HDF5 file / h5py object, or a zarr store / zarr.Group."""
		if isinstance(source, ScanDescription):
			return source
		if isinstance(source, (h5py.File, h5py.Group)):
			return ScanDescriptionFactory.fromH5(source)
		if isinstance(source, (str, os.PathLike)):
			path = os.fspath(source)
			if os.path.isfile(path) and h5py.is_hdf5(path):
				return ScanDescriptionFactory.fromH5(path)
			if os.path.exists(path):
				return ScanDescriptionFactory.fromZarr(path)
			raise FileNotFoundError(path)
		try:
			import zarr
			if isinstance(source, zarr.Group):
				return ScanDescriptionFactory.fromZarr(source)
		except ImportError:
			pass
		raise TypeError(f"Can not describe a scan from {type(source).__name__}")


# ---------------------------------------------------------------- module functions
# The source of these functions is an HDF5 file (name or h5py object), a zarr store with a params
# group (name or zarr.Group) or a ScanDescription.

def getExperimentInfo(h5file, overrideMagnification=None):
	return ScanDescriptionFactory.create(h5file).getExperimentInfo(overrideMagnification)


def getBasicDatasetInfo(h5file):
	return ScanDescriptionFactory.create(h5file).getBasicDatasetInfo()


def beamCurrentDataset(h5file, timeOffsetSec=None):
	"""Beam current readings (mA) of a scan, see ScanDescription.beamCurrentDataset."""
	return ScanDescriptionFactory.create(h5file).beamCurrentDataset(timeOffsetSec)


def scanDataset(h5file, includeCurrent=False, timeOffsetSec=None, fixKnownTimestampIssues=True):
	"""DataFrame of the frames of a scan, see ScanDescription.scanDataset."""
	return ScanDescriptionFactory.create(h5file).scanDataset(includeCurrent, timeOffsetSec, fixKnownTimestampIssues)


def imageDataset(h5file, image_key=0, includeCurrent=True, includePixelShift=False, overrideMagnification=None,
				 fixKnownTimestampIssues=True):
	return ScanDescriptionFactory.create(h5file).imageDataset(image_key, includeCurrent, includePixelShift, overrideMagnification,
															  fixKnownTimestampIssues)


def scanDatasetNanotomography(logfile, folder, includeCurrent=False):
	"""
	Build a scan DataFrame from LogScan.log and a directory of TIFF files.

	Files must follow:
		img_<batch>_<frame>.tif[f]
		ref_<batch>_<frame>.tif[f]

	Within each identifier, files are sorted numerically by (batch, frame)
	and paired with log rows in their original order. Log image numbers
	are preserved as metadata but are not used to construct filenames.

	Only log rows whose infostr is 'start' are used.

	Returns:
		DataFrame with an increasing UNIX-millisecond index and columns:
		time, image_key, image_file, s_rot, time_sec,
		image_identifier, infostr, image_number_i, image_number_ii,
		current_number, and optionally current.

	image_key:
		img -> 0 (projection)
		ref -> 1 (reference/flat field)

	No timeOffsetSec is needed: beam current is recorded in each log row.
	"""
	import io
	import re
	from decimal import Decimal

	folder = os.path.abspath(os.fspath(folder))
	logfile = os.path.abspath(os.fspath(logfile))
	if not os.path.isdir(folder):
		raise NotADirectoryError(folder)

	start_time = None
	rows = []
	with open(logfile, "r", encoding="utf-8") as stream:
		for line_number, line in enumerate(stream, 1):
			line = line.strip()
			if not line:
				continue
			if line.startswith("#"):
				match = re.fullmatch(
					r"#\s*starttime\s*=\s*(\S+)", line
				)
				if match:
					if start_time is not None:
						raise ValueError("Multiple #starttime headers.")
					start_time = Decimal(match.group(1))
				continue
			if len(line.split()) != 8:
				raise ValueError(
					f"{logfile}:{line_number}: expected 8 columns."
				)
			rows.append(line)

	if start_time is None or not start_time.is_finite():
		raise ValueError(f"Missing or invalid #starttime in {logfile}.")
	if not rows:
		raise ValueError(f"No data rows in {logfile}.")

	columns = [
		"image_identifier", "infostr",
		"image_number_i", "image_number_ii", "current_number",
		"time_sec", "current", "s_rot",
	]
	df = pd.read_csv(
		io.StringIO("\n".join(rows)),
		sep=r"\s+",
		names=columns,
		dtype=str,
	)
	df = df.loc[df["infostr"] == "start"].copy().reset_index(drop=True)
	if df.empty:
		raise ValueError(f"No 'start' rows in {logfile}.")

	key_map = {"img": 0, "ref": 1}
	unknown = set(df["image_identifier"]) - set(key_map)
	if unknown:
		raise ValueError(f"Unsupported image identifiers: {sorted(unknown)}.")

	# Convert epoch seconds + relative seconds directly to milliseconds.
	# Decimal avoids first rounding the epoch through a Python float.
	relative_times = [Decimal(value) for value in df["time_sec"]]
	if any(not value.is_finite() for value in relative_times):
		raise ValueError("Log timestamps must be finite.")
	timestamps = np.asarray(
		[int((start_time + value) * 1000) for value in relative_times],
		dtype=np.int64,
	)

	for column in ("image_number_i", "image_number_ii", "current_number"):
		df[column] = pd.to_numeric(df[column], errors="raise")
		if not np.isfinite(df[column].to_numpy()).all():
			raise ValueError(f"Non-finite values in {column}.")
		if not (df[column] == np.floor(df[column])).all():
			raise ValueError(f"Non-integer values in {column}.")
		df[column] = df[column].astype(np.int64)

	for column in ("time_sec", "current", "s_rot"):
		df[column] = pd.to_numeric(df[column], errors="raise")
		if not np.isfinite(df[column].to_numpy()).all():
			raise ValueError(f"Non-finite values in {column}.")

	pattern = re.compile(
		r"^(img|ref)_(\d+)_(\d+)\.tiff?$", re.IGNORECASE
	)
	files = {"img": [], "ref": []}
	with os.scandir(folder) as entries:
		for entry in entries:
			match = pattern.fullmatch(entry.name)
			if match and entry.is_file():
				identifier = match.group(1).lower()
				order = (int(match.group(2)), int(match.group(3)))
				files[identifier].append((order, entry.path))

	df["image_file"] = ""
	for identifier, candidates in files.items():
		candidates.sort(key=lambda item: item[0])
		orders = [order for order, _ in candidates]
		if len(set(orders)) != len(orders):
			raise ValueError(
				f"Ambiguous filenames: duplicate numeric {identifier} "
				f"(batch, frame) identifiers in {folder}."
			)

		mask = df["image_identifier"] == identifier
		expected = int(mask.sum())
		if len(candidates) != expected:
			raise ValueError(
				f"{identifier}: {expected} log rows but "
				f"{len(candidates)} matching TIFF files in {folder}. "
				"Cannot safely pair files by acquisition order."
			)
		if expected:
			df.loc[mask, "image_file"] = [
				path for _, path in candidates
			]

	df["image_key"] = df["image_identifier"].map(key_map).astype(np.int64)

	# Follow scanDataset's convention for duplicate/non-increasing times.
	# time_sec retains the original relative timestamps.
	if np.any(np.diff(timestamps) <= 0):
		timestamps, issues = fix_timestamps(timestamps)
		for text in timestamp_issue_messages(issues):
			log.warning("%s (%s)", text, logfile)

	df.index = timestamps
	df["time"] = pd.to_datetime(timestamps, unit="ms")
	if not includeCurrent:
		df.drop(columns=["current"], inplace=True)
	return df
