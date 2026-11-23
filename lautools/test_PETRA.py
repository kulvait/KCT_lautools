#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Created 2026

Test of lautools PETRA (ScanDescription): reading P05/P07 scan HDF5 files in the layouts of the
beamline files and their border cases (missing or empty beam current, empty or scalar camera values,
incomplete logs); the canonical description, its round trips through zarr and HDF5; the same results
as denpy PETRA, the previous implementation.

Real beamline files: set LAUTOOLS_PETRA_TEST_FILES to a directory with *.h5 scan files (e.g.
syscripts_update with exampleA.h5, exampleB.h5, example03-05.h5); without it these tests are skipped.

  2021 layout: group attributes are 1-element object arrays, strings are (1,) datasets
  2025 layout: group attributes are plain strings, strings are scalar datasets, some values are
               empty (0,) datasets

@author: Vojtech Kulvait
"""
import os
import pathlib
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

from lautools import PETRA

KEYS = [2, 2, 1, 1, 1, 0, 0, 0, 0, 1]
T0 = 1_758_209_568_000


def make_scan(path, style="2025", current=True, mutate=None):
	"""Write a small scan file; mutate(h5) changes it before it is closed. Returns the path."""
	old = style == "2021"
	with h5py.File(path, "w") as h5:
		def attr(obj, value):
			if old:
				obj.attrs.create("NX_class", np.array([value], dtype=object), dtype=h5py.string_dtype())
			else:
				obj.attrs["NX_class"] = value

		def group(name, nx_class="NXcollection"):
			g = h5.require_group(name)
			attr(g, nx_class)
			return g

		def text(name, value):
			if old:
				h5.create_dataset(name, data=np.array([value], dtype=object), dtype=h5py.string_dtype())
			else:
				h5.create_dataset(name, data=value.encode(), dtype=h5py.string_dtype("ascii"))

		group("entry", "NXentry")
		group("entry/beamline")
		text("entry/beamline/experiment", "microtomography")
		text("entry/beamline/name", "P05 IBL EH2")
		camera = group("entry/hardware/camera")
		for name, value in (("pixelsize", 6.5), ("magnification", 10.0), ("roi_width", 100.0), ("roi_height", 50.0)):
			camera.create_dataset(name, data=np.array([value]))
		setup = group("entry/scan/setup")
		setup.create_dataset("pos_o_ccd_dist", data=np.array([200.0]))
		setup.create_dataset("pos_p05_energy", data=np.array([20000.0], dtype=np.float32))
		data = group("entry/scan/data")
		times = T0 + np.arange(len(KEYS), dtype=np.uint64) * 100
		files = ["scan_%06d_%s.tif" % (i, {0: "img", 1: "ref", 2: "dar"}[k]) for i, k in enumerate(KEYS)]
		data.create_dataset("image_key/value", data=np.array(KEYS, dtype=np.int16))
		data.create_dataset("image_key/time", data=times)
		data.create_dataset("image_file/value", data=np.array([f.encode() for f in files], dtype=object),
							dtype=h5py.string_dtype("utf-8" if old else "ascii"))
		data.create_dataset("image_file/time", data=times)
		moving = np.array(KEYS) != 2
		data.create_dataset("s_rot/value", data=np.linspace(0, 180, moving.sum()).astype(np.float32))
		data.create_dataset("s_rot/time", data=times[moving])
		data.create_dataset("s_stage_x/value", data=np.linspace(1.0, 2.0, moving.sum()).astype(np.float32))
		data.create_dataset("s_stage_x/time", data=times[moving])
		for name in ("image_key", "image_file", "s_rot", "s_stage_x"):
			attr(data[name], "NXlog")
		if current:
			log = group("entry/hardware/beam_current/current", "NXlog")
			# readings every 250 ms from 100 ms before the scan, 100 mA falling by 1 mA/s
			current_times = T0 - 100 + np.arange(6, dtype=np.uint64) * 250
			log.create_dataset("time", data=current_times)
			log.create_dataset("value", data=100.0 - (current_times.astype(np.float64) - T0) / 1000.0)
		if mutate is not None:
			mutate(h5)
	return path


def replace(h5, path, data, dtype=None):
	del h5[path]
	h5.create_dataset(path, data=data, dtype=dtype)


class TestPETRA(unittest.TestCase):

	def setUp(self):
		self.tmp = tempfile.TemporaryDirectory()

	def tearDown(self):
		self.tmp.cleanup()

	def scan(self, name="scan", **kwargs):
		return make_scan(os.path.join(self.tmp.name, name + ".h5"), **kwargs)

	def test_both_layouts(self):
		for style in ("2021", "2025"):
			with self.subTest(style=style):
				path = self.scan(style=style)
				info = PETRA.getExperimentInfo(path)
				self.assertEqual((info["experiment"], info["beamline"]), ("microtomography", "P05 IBL EH2"))
				self.assertAlmostEqual(info["pix_size"], 0.65)
				self.assertAlmostEqual(info["field_of_view_x"], 65.0)
				self.assertEqual((info["energy_keV"], info["propagation_distance_mm"]), (20.0, 200.0))
				self.assertEqual(info["sample_frame_count"], 4)
				df = PETRA.scanDataset(path)
				self.assertEqual(list(df["image_key"]), KEYS)
				self.assertEqual(df["image_file"].iloc[0], "scan_000000_dar.tif")
				self.assertTrue(df["s_rot"].iloc[:2].isna().all() and df["s_rot"].iloc[2:].notna().all())
				images = PETRA.imageDataset(path, includeCurrent=False, includePixelShift=True)
				self.assertEqual(len(images), 4)
				self.assertIn("pixel_shift", images.columns)

	def test_current_interpolation(self):
		df = PETRA.scanDataset(self.scan(), includeCurrent=True)
		# linear between readings, the readings fall by 1 mA/s
		expected = 100.0 - (df.index.to_numpy().astype(np.float64) - T0) / 1000.0
		np.testing.assert_allclose(df["current"].to_numpy(), expected)
		# a time offset shifts the readings
		shifted = PETRA.scanDataset(self.scan(), includeCurrent=True, timeOffsetSec=0.5)
		# the readings start 400 ms after the scan now: the first 4 frames get the first reading
		np.testing.assert_allclose(shifted["current"].to_numpy()[4:], expected[4:] + 0.5)
		np.testing.assert_allclose(shifted["current"].to_numpy()[:4], 100.1)
		# a pathlib.Path works as a file name
		PETRA.scanDataset(pathlib.Path(self.scan()), includeCurrent=True)

	def test_current_outside_readings_and_nan(self):
		def few(h5):  # one reading in the middle of the scan, one NaN reading
			replace(h5, "entry/hardware/beam_current/current/time", np.array([T0 + 400, T0 + 500], dtype=np.uint64))
			replace(h5, "entry/hardware/beam_current/current/value", np.array([90.0, np.nan]))
		df = PETRA.scanDataset(self.scan(mutate=few), includeCurrent=True)
		np.testing.assert_allclose(df["current"].to_numpy(), 90.0)
		self.assertEqual(len(PETRA.beamCurrentDataset(self.scan(mutate=few))), 1)

	def test_missing_current_is_a_clean_error(self):
		def empty(h5):
			replace(h5, "entry/hardware/beam_current/current/time", np.array([], dtype=np.uint64))
			replace(h5, "entry/hardware/beam_current/current/value", np.array([], dtype=np.float64))

		def all_nan(h5):
			replace(h5, "entry/hardware/beam_current/current/value", np.full(6, np.nan))

		def mismatch(h5):
			replace(h5, "entry/hardware/beam_current/current/value", np.zeros(5))
		cases = {"no current": dict(current=False), "empty": dict(mutate=empty), "all NaN": dict(mutate=all_nan),
				 "length mismatch": dict(mutate=mismatch)}
		for case, kwargs in cases.items():
			with self.subTest(case=case):
				path = self.scan(**kwargs)
				with self.assertRaises(PETRA.BeamCurrentError):
					PETRA.scanDataset(path, includeCurrent=True)
				with self.assertRaises(PETRA.BeamCurrentError):
					PETRA.imageDataset(path)
				# without the current the file is fine
				self.assertEqual(len(PETRA.scanDataset(path)), len(KEYS))
		self.assertTrue(issubclass(PETRA.BeamCurrentError, ValueError))
		self.assertEqual(PETRA.findInsertionEvents(self.scan(mutate=empty)), [])
		with self.assertRaises(PETRA.BeamCurrentError):
			PETRA.findInsertionEvents(self.scan(current=False))

	def test_camera_values(self):
		def empty_magnification(h5):
			replace(h5, "entry/hardware/camera/magnification", np.array([], dtype=np.float64))

		def zero_magnification(h5):
			replace(h5, "entry/hardware/camera/magnification", np.array([0.0]))

		def scalars(h5):
			replace(h5, "entry/hardware/camera/pixelsize", 6.5)
			replace(h5, "entry/hardware/camera/magnification", 5.0)

		def calibration(h5):
			replace(h5, "entry/hardware/camera/magnification", np.array([], dtype=np.float64))
			h5.create_dataset("entry/hardware/camera/calibration/magnification", data=np.array([20.0]))

		def old_names(h5):
			del h5["entry/hardware/camera"]
			camera = h5.create_group("entry/hardware/camera1")
			camera.create_dataset("px_size", data=np.array([6.5]))
			camera.create_dataset("magnification", data=np.array([10.0]))
			camera.create_dataset("senzor_xsize", data=np.array([200.0]))
		for style in ("2021", "2025"):
			for mutate in (empty_magnification, zero_magnification):
				with self.subTest(style=style, case=mutate.__name__):
					path = self.scan(style=style, mutate=mutate)
					info = PETRA.getExperimentInfo(path)
					self.assertNotIn("pix_size", info)
					self.assertNotIn("fresnel_number", info)
					with self.assertRaises(ValueError):
						PETRA.imageDataset(path, includeCurrent=False, includePixelShift=True)
					# an explicit magnification is used instead
					self.assertAlmostEqual(PETRA.getExperimentInfo(path, overrideMagnification=13.0)["pix_size"], 0.5)
					PETRA.imageDataset(path, includeCurrent=False, includePixelShift=True, overrideMagnification=13.0)
			for mutate, pix_size in ((scalars, 1.3), (calibration, 0.325), (old_names, 0.65)):
				with self.subTest(style=style, case=mutate.__name__):
					path = self.scan(style=style, mutate=mutate)
					self.assertAlmostEqual(PETRA.getExperimentInfo(path)["pix_size"], pix_size)
					images = PETRA.imageDataset(path, includeCurrent=False, includePixelShift=True)
					self.assertTrue(np.isfinite(images["pixel_shift"]).all())
		info = PETRA.getExperimentInfo(self.scan(mutate=old_names))
		self.assertAlmostEqual(info["field_of_view_x"], 130.0)  # whole sensor without a ROI
		self.assertEqual(info["camera"]["roi_width"], 200.0)

	def test_empty_strings_and_missing_groups(self):
		def empty_name(h5):
			replace(h5, "entry/beamline/name", np.array([], dtype=object), dtype=h5py.string_dtype())

		def no_setup(h5):
			del h5["entry/scan/setup"]
		info = PETRA.getExperimentInfo(self.scan(mutate=empty_name))
		self.assertNotIn("beamline", info)
		info = PETRA.getExperimentInfo(self.scan(mutate=no_setup))
		self.assertNotIn("energy_keV", info)
		self.assertAlmostEqual(info["pix_size"], 0.65)

	def test_incomplete_logs_are_skipped(self):
		def string_log(h5):
			data = h5["entry/scan/data"]
			data.create_dataset("comment/time", data=data["image_key/time"][:])
			data.create_dataset("comment/value", data=np.array([b"c"] * len(KEYS), dtype=object), dtype=h5py.string_dtype())

		def broken_logs(h5):
			data = h5["entry/scan/data"]
			data.create_dataset("no_value/time", data=data["image_key/time"][:3])
			data.create_dataset("not_a_group", data=np.arange(3))
			data.create_dataset("short_value/time", data=data["image_key/time"][:])
			data.create_dataset("short_value/value", data=np.arange(3.0))
		df = PETRA.scanDataset(self.scan(mutate=string_log))
		self.assertEqual(list(df["comment"]), ["c"] * len(KEYS))
		df = PETRA.scanDataset(self.scan(mutate=broken_logs))
		self.assertEqual(list(df.columns), ["time", "image_key", "image_file", "s_rot", "s_stage_x"])

	def test_truncated_image_file(self):
		def truncated(h5):  # an interrupted write: image_file shorter than image_key
			for part in ("value", "time"):
				v = h5[f"entry/scan/data/image_file/{part}"][:-3]
				replace(h5, f"entry/scan/data/image_file/{part}", v,
						dtype=h5py.string_dtype() if part == "value" else None)
		path = self.scan(mutate=truncated)
		info = PETRA.getExperimentInfo(path)  # from the image_key log
		self.assertEqual(info["sample_frame_count"], 4)
		with self.assertRaisesRegex(ValueError, "Length mismatch"):
			PETRA.scanDataset(path)

	def test_non_increasing_times(self):
		def duplicate(h5):
			for name in ("image_key", "image_file"):
				t = h5[f"entry/scan/data/{name}/time"][:]
				t[6] = t[5]
				replace(h5, f"entry/scan/data/{name}/time", t)
			del h5["entry/scan/data/s_rot"]
			del h5["entry/scan/data/s_stage_x"]
		df = PETRA.scanDataset(self.scan(mutate=duplicate), includeCurrent=True)
		self.assertTrue(np.all(np.diff(df.index.to_numpy().astype(np.int64)) > 0))
		self.assertTrue(isinstance(df["time"].iloc[0], pd.Timestamp))


def h5_layout_differences(a, b):
	"""Differences between the /entry of two HDF5 files: groups, attributes, datasets with shape, dtype
	(strings by their character set), values, chunks and maxshape."""
	problems = []

	def norm(dataset):
		value = dataset[()]
		if h5py.check_string_dtype(dataset.dtype) is not None:
			return [v.decode() if isinstance(v, bytes) else v for v in np.asarray(value, dtype=object).ravel()]
		return value

	def kind(dataset):
		string = h5py.check_string_dtype(dataset.dtype)
		return ("str", string.encoding) if string is not None else dataset.dtype.str

	def walk(x, y):
		if dict(x.attrs) != dict(y.attrs):
			problems.append(f"{x.name}: attributes {dict(x.attrs)} != {dict(y.attrs)}")
		if set(x) != set(y):
			problems.append(f"{x.name}: members {sorted(set(x) ^ set(y))} only in one")
		for name in sorted(set(x) & set(y)):
			p, q = x[name], y[name]
			if isinstance(p, h5py.Group) and isinstance(q, h5py.Group):
				walk(p, q)
			elif isinstance(p, h5py.Dataset) and isinstance(q, h5py.Dataset):
				same = (p.shape, kind(p), p.chunks, p.maxshape) == (q.shape, kind(q), q.chunks, q.maxshape)
				vp, vq = norm(p), norm(q)
				if not same or not (vp == vq if isinstance(vp, list) else np.array_equal(vp, vq)):
					problems.append(f"{p.name}: {p.shape} {kind(p)} {p.chunks} {p.maxshape} != {q.shape} {kind(q)} {q.chunks} {q.maxshape} or values")
			else:
				problems.append(f"{p.name}: group and dataset")

	with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
		walk(fa["entry"], fb["entry"])
	return problems


def beamline_h5(path):
	"""The 2025 layout as the beamline writes it, for writeH5 to reproduce: (1,)-chunked datasets with
	unlimited first dimension, scalar strings."""
	def mutate(h5):
		def rewrite(name, obj):
			if isinstance(obj, h5py.Dataset) and obj.shape != ():
				data, dtype = obj[()], obj.dtype
				del h5[name]
				h5.create_dataset(name, data=data, dtype=dtype, maxshape=(None, ) + data.shape[1:], chunks=(1, ) + data.shape[1:])
		names = []
		h5.visititems(lambda name, obj: names.append((name, obj)) if isinstance(obj, h5py.Dataset) else None)
		for name, obj in names:
			rewrite(name, obj)
	return make_scan(path, style="2025", mutate=mutate)


class TestScanDescription(unittest.TestCase):
	"""One canonical description of a scan from both HDF5 layouts and from zarr; the writers back to
	zarr and to HDF5 (2025 layout)."""

	def setUp(self):
		self.tmp = tempfile.TemporaryDirectory()

	def tearDown(self):
		self.tmp.cleanup()

	def path(self, name):
		return os.path.join(self.tmp.name, name)

	def test_canonical_form(self):
		old = PETRA.ScanDescriptionFactory.fromH5(make_scan(self.path("a.h5"), style="2021"))
		new = PETRA.ScanDescriptionFactory.fromH5(make_scan(self.path("b.h5"), style="2025"))
		self.assertEqual(old.differences(new), [])  # the two layouts are one description
		root = new.root
		self.assertEqual(root.attrs, {"NX_class": "NXentry"})
		self.assertEqual(root["beamline/name"], "P05 IBL EH2")  # (1,) and scalar strings: a str attribute
		self.assertEqual(old.root["beamline"].attrs["name"], "P05 IBL EH2")
		self.assertEqual(root.value("hardware/camera/pixelsize"), 6.5)  # a (1,) number stays an array
		self.assertEqual(root.array("hardware/camera/pixelsize").shape, (1, ))
		files = root.array("scan/data/image_file/value")  # in a log, strings stay an array
		self.assertEqual((files.dtype, files[0]), (np.dtype(object), "scan_000000_dar.tif"))
		self.assertIn("scan/data/s_rot/time", root)
		self.assertIsNone(root.group("no/such"))
		self.assertEqual(old.root.group("scan/setup").scalars(), {"pos_o_ccd_dist": 200.0, "pos_p05_energy": 20000.0})

	def test_one_element_string_log(self):
		# a log with a single string reading stays an array, outside logs it becomes a str
		def mutate(h5):
			log = h5.create_group("entry/scan/data/comment")
			log.create_dataset("time", data=np.array([T0], dtype=np.uint64))
			log.create_dataset("value", data=np.array([b"start"], dtype=object), dtype=h5py.string_dtype("ascii"))
			h5.create_dataset("entry/scan/note", data=np.array([b"x"], dtype=object), dtype=h5py.string_dtype())
		d = PETRA.ScanDescriptionFactory.fromH5(make_scan(self.path("s.h5"), mutate=mutate))
		self.assertEqual(list(d.root.array("scan/data/comment/value")), ["start"])
		self.assertEqual(d.root["scan"].attrs["note"], "x")

	def test_zarr_round_trip_and_analyses(self):
		import zarr
		for style in ("2021", "2025"):
			h5 = make_scan(self.path(style + ".h5"), style=style)
			d = PETRA.ScanDescriptionFactory.fromH5(h5)
			for store in ("dir.zarr", "z.zar"):
				with self.subTest(style=style, store=store):
					target = self.path(style + store)
					if store.endswith(".zar"):
						from denpy import ZARZS
						s = ZARZS.ZipStore(target, mode="w")
					else:
						s = target
					root = zarr.open_group(s, mode="w", attributes=d.rootAttributes(export={"h5_path": h5}))
					d.writeZarrGroup(root)
					if not isinstance(s, str):
						s.close()
					z = PETRA.ScanDescriptionFactory.create(target)  # from the store name
					self.assertEqual(d.differences(z), [])
					self.assertEqual(z.h5Path, os.path.realpath(h5))
					# the analyses are the same from either source
					pd.testing.assert_frame_equal(PETRA.scanDataset(target, includeCurrent=True), PETRA.scanDataset(h5, includeCurrent=True))
					pd.testing.assert_frame_equal(PETRA.imageDataset(target, includePixelShift=True), PETRA.imageDataset(h5, includePixelShift=True))
					pd.testing.assert_frame_equal(PETRA.beamCurrentDataset(target, 0.5), PETRA.beamCurrentDataset(h5, 0.5))
					self.assertEqual(PETRA.getExperimentInfo(target), PETRA.getExperimentInfo(h5))
					self.assertEqual(PETRA.findInsertionEvents(target), PETRA.findInsertionEvents(h5))
					# the scan table on the params group is not part of the description
					group = zarr.open_group(s if isinstance(s, str) else ZARZS.ZipStore(target, mode="r"), mode="r")
					self.assertEqual(group["params"].attrs["columns"][:3], ["time", "image_key", "image_file"])

	def test_write_h5(self):
		# the 2025 layout is reproduced exactly, the 2021 layout gives the same description
		source = beamline_h5(self.path("beamline.h5"))
		d = PETRA.ScanDescriptionFactory.fromH5(source)
		d.writeH5(self.path("rewritten.h5"))
		self.assertEqual(h5_layout_differences(source, self.path("rewritten.h5")), [])
		old = PETRA.ScanDescriptionFactory.fromH5(make_scan(self.path("2021.h5"), style="2021"))
		old.writeH5(self.path("2021_as_2025.h5"))
		self.assertEqual(h5_layout_differences(source, self.path("2021_as_2025.h5")), [])
		self.assertEqual(PETRA.ScanDescriptionFactory.fromH5(self.path("2021_as_2025.h5")).differences(old), [])
		pd.testing.assert_frame_equal(PETRA.scanDataset(self.path("2021_as_2025.h5")), PETRA.scanDataset(self.path("2021.h5")))


class TestOneReader(unittest.TestCase):
	"""All functions read through one opening of the file and take an open h5py object; group
	attributes are values as the scalar datasets (as lautools PETRA, the canonical description)."""

	def test_open_file_and_attributes(self):
		def mutate(h5):
			h5["entry/hardware/camera"].attrs["model"] = np.array(["test camera"], dtype=object)
			h5["entry/hardware/camera"].attrs["sensor_gain"] = np.array([2.0])
			calibration = h5.create_group("entry/hardware/camera/calibration")
			calibration.attrs["magnification"] = 5.0
			del h5["entry/hardware/camera/magnification"]
			del h5["entry/beamline/experiment"]
			h5["entry/beamline"].attrs["experiment"] = "microtomography"
		with tempfile.TemporaryDirectory() as tmp:
			path = make_scan(os.path.join(tmp, "scan.h5"), style="2021", mutate=mutate)
			info = PETRA.getExperimentInfo(path)
			self.assertEqual(info["camera"]["model"], "test camera")
			self.assertEqual(info["camera"]["sensor_gain"], 2.0)
			self.assertEqual(info["camera"]["magnification"], 5.0)  # from the attribute of calibration
			self.assertEqual(info["experiment"], "microtomography")  # an attribute of beamline
			self.assertNotIn("NX_class", info["camera"])
			with h5py.File(path, "r") as h5:
				self.assertEqual(PETRA.getExperimentInfo(h5), info)
				pd.testing.assert_frame_equal(PETRA.scanDataset(h5, includeCurrent=True), PETRA.scanDataset(path, includeCurrent=True))
				pd.testing.assert_frame_equal(PETRA.imageDataset(h5["entry"], includePixelShift=True), PETRA.imageDataset(path, includePixelShift=True))
				pd.testing.assert_frame_equal(PETRA.getBasicDatasetInfo(h5), PETRA.getBasicDatasetInfo(path))
				self.assertTrue(bool(h5))  # not closed by the functions


def real_files():
	directory = os.environ.get("LAUTOOLS_PETRA_TEST_FILES")
	if not directory or not os.path.isdir(directory):
		return []
	return sorted(os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".h5"))


class TestAgainstDenpyPETRA(unittest.TestCase):
	"""lautools PETRA gives the results of denpy PETRA, on synthetic scans and real beamline files."""

	ANALYSES = (("scanDataset", (True, )), ("scanDataset", (False, 0.5)), ("getExperimentInfo", ()),
				("getBasicDatasetInfo", ()), ("beamCurrentDataset", ()), ("imageDataset", ()),
				("imageDataset", (0, False, True)), ("findInsertionEvents", ()))

	def compare(self, path):
		import logging
		from denpy import PETRA as denpyPETRA
		logging.disable(logging.WARNING)
		try:
			for name, args in self.ANALYSES:
				with self.subTest(file=os.path.basename(path), analysis=name, args=args):
					try:
						expected = getattr(denpyPETRA, name)(path, *args)
					except Exception as e:  # the same error (each module has its own BeamCurrentError)
						with self.assertRaises(Exception) as raised:
							getattr(PETRA, name)(path, *args)
						self.assertEqual(type(raised.exception).__name__, type(e).__name__)
						self.assertEqual(isinstance(raised.exception, ValueError), isinstance(e, ValueError))
						continue
					result = getattr(PETRA, name)(path, *args)
					if isinstance(expected, pd.DataFrame):
						pd.testing.assert_frame_equal(result, expected)
					else:
						self.assertEqual(result, expected)
			description = PETRA.ScanDescriptionFactory.fromH5(path)
			with tempfile.TemporaryDirectory() as tmp:
				description.writeH5(os.path.join(tmp, "rewritten.h5"))
				self.assertEqual(PETRA.ScanDescriptionFactory.fromH5(os.path.join(tmp, "rewritten.h5")).differences(description), [])
		finally:
			logging.disable(logging.NOTSET)

	def test_synthetic(self):
		with tempfile.TemporaryDirectory() as tmp:
			for style in ("2021", "2025"):
				for current in (True, False):
					self.compare(make_scan(os.path.join(tmp, f"{style}{current}.h5"), style=style, current=current))

	@unittest.skipUnless(real_files(), "set LAUTOOLS_PETRA_TEST_FILES to a directory with scan HDF5 files")
	def test_real_files(self):
		for path in real_files():
			self.compare(path)


if __name__ == "__main__":
	unittest.main()
