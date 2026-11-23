#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created 11 2026

Tomography scans independent of the instrument: functions on the scan dataset, the DataFrame of the
frames of a scan that PETRA.scanDataset (microtomography, NeXus HDF5) and NANOCT.scanDataset
(nanotomography, LogScan.log) create. Its columns used here: image_key (0 projection, 1 flat field,
2 dark field) and image_file; optional file_exists. The rows are in acquisition order.

	df = PETRA.scanDataset(h5file)                             # or NANOCT.scanDataset(...)
	spec = TOMOGRAPHY.scanTiffStackConversionSpecs(df, output, rawDir=rawDir)
	report = TIFF.execute(TIFF.validate(spec))                 # denpy.TIFF

@author: Vojtěch Kulvait
@license: GNU General Public License v3.0
"""
import os

import numpy as np

from denpy import TIFF

# image_key -> name of the tiff stack and array, the names the pipelines read
SCAN_ARRAY_NAMES = {2: "dar", 1: "ref", 0: "img"}


def scanTiffStackConversionSpecs(scanDataset, output, rawDir=None, names=None, fillValue=None, frameShape=None,
								 dtype=None, defaults=None, attributes=None):
	"""TIFF stack specification (denpy docs/TIFFSTACK_SPEC.md) of a scan: one tiff stack and one array
	per image_key, the frames in the order of scanDataset.

	scanDataset: DataFrame with columns image_key and image_file (optional file_exists).
	output: the output object of the spec, e.g. {"format": "zarr", "store": "zip", "path": "raw.zar"}.
	rawDir: image_file is relative to rawDir (a leading "/" is ignored, as in P05 files); None:
		image_file are paths as they are (NANOCT: absolute).
	names: image_key -> name; default SCAN_ARRAY_NAMES {2: "dar", 1: "ref", 0: "img"}, in this
		order; a key without rows gives a stack of 0 frames (a scan without darks).
	fillValue: None: a row without a file (empty image_file, file_exists false, not on disk) is a
		denpy.TIFF.SpecError; a number: such rows become null frames written as fillValue and
		listed in the report.
	frameShape, dtype: of the frames; default from the first existing file.
	defaults: defaults of the arrays, e.g. {"zarr": {"codec": {"name": "htj2k", "clevel": 0}}}.
	attributes: attributes of the root group (zarr).
	The result is an ordinary spec: it can be extended (virtual stacks, more arrays), saved as JSON
	and run with denpy.TIFF.execute or convertTiffStack.
	"""
	names = dict(SCAN_ARRAY_NAMES if names is None else names)
	for column in ("image_key", "image_file"):
		if column not in scanDataset.columns:
			raise TIFF.SpecError(f"the scan dataset has no column {column}")
	unknown = sorted(set(int(k) for k in scanDataset["image_key"].unique()) - set(names))
	if unknown:
		raise TIFF.SpecError(f"the scan dataset has image_key {unknown} without a name (names={names})")
	if "file_exists" in scanDataset.columns:
		exists = [bool(v) for v in scanDataset["file_exists"].tolist()]
	else:
		exists = [True] * len(scanDataset)
	paths = []
	for value, ok in zip(scanDataset["image_file"].tolist(), exists):
		if isinstance(value, (bytes, bytearray)):
			value = value.decode("utf-8")
		if not ok or not isinstance(value, str) or not value:
			paths.append(None)
		elif rawDir is None:
			paths.append(value)
		else:
			paths.append(os.path.join(os.fspath(rawDir), value.lstrip("/")))
	missing = {i for i, p in enumerate(paths) if p is None or not os.path.isfile(p)}
	if missing and fillValue is None:
		shown = [paths[i] or f"row {i} without a file" for i in sorted(missing)[:5]]
		raise TIFF.SpecError(f"{len(missing)} frame(s) of the scan dataset have no file on disk, e.g. {shown}; "
							 "give fillValue to write them as null frames")
	if frameShape is None or dtype is None:
		present = [p for i, p in enumerate(paths) if i not in missing]
		if not present:
			raise TIFF.SpecError("no frame of the scan dataset has a file, give frameShape and dtype")
		shape, first = TIFF.tiff_header(present[0])
		frameShape = list(shape) if frameShape is None else frameShape
		dtype = first if dtype is None else dtype
	frameShape, dtype = [int(v) for v in frameShape], str(np.dtype(dtype))
	base = None if rawDir is None else os.path.abspath(os.fspath(rawDir))
	keys = [int(k) for k in scanDataset["image_key"].tolist()]
	tiffstacks = {}
	for key, name in names.items():
		frames = [None if i in missing else (os.path.relpath(os.path.abspath(paths[i]), base) if base else paths[i])
				  for i, k in enumerate(keys) if k == key]
		stack = {"frames": frames, "framecount": len(frames), "frame_shape": frameShape, "dtype": dtype}
		if None in frames:
			stack["fill_value"] = fillValue
		tiffstacks[name] = stack
	spec = {"spec": TIFF.SPEC, "version": TIFF.VERSION}
	if base is not None:
		spec["base_dir"] = base
	spec.update({"tiffstacks": tiffstacks,
				 "arrays": {name: {"stack": name} for name in names.values()},
				 "output": dict(output)})
	if defaults:
		spec["defaults"] = defaults
	if attributes:
		spec["output"]["attributes"] = attributes
	return spec
