#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Created 2026

Test of TOMOGRAPHY.scanTiffStackConversionSpecs: scan datasets as PETRA.scanDataset and
NANOCT.scanDataset create them, to TIFF stack specifications run with denpy.TIFF.

@author: Vojtech Kulvait
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import tifffile
import zarr

from denpy import DEN, TIFF, ZAR
from lautools import TOMOGRAPHY

Y, X = 24, 32


def quiet(message):
	pass


class TestScanTiffStackConversionSpecs(unittest.TestCase):

	def setUp(self):
		self.tmp = tempfile.TemporaryDirectory()
		self.dir = self.tmp.name
		rng = np.random.default_rng(1)
		self.data = {}
		for prefix, n, base in (("dar", 5, 100), ("ref", 6, 3000), ("img", 10, 1500)):
			frames = (base + rng.integers(0, 200, (n, Y, X))).astype(np.uint16)
			for i, frame in enumerate(frames):
				tifffile.imwrite(self.path(f"{prefix}_{i:05d}.tif"), frame)
			self.data[prefix] = frames
		os.remove(self.path("img_00004.tif"))  # listed in the metadata, absent on disk
		ZAR.register_codecs()

	def tearDown(self):
		self.tmp.cleanup()

	def path(self, name):
		return os.path.join(self.dir, name)

	def scan_dataset(self, absolute=False):
		"""Darks, refs, projections, refs again: the order of a P05 scan."""
		rows = [(2, f"dar_{i:05d}.tif") for i in range(5)] + [(1, f"ref_{i:05d}.tif") for i in range(3)] + \
			[(0, f"img_{i:05d}.tif") for i in range(10)] + [(1, f"ref_{i:05d}.tif") for i in range(3, 6)]
		files = [self.path(f) if absolute else "/" + f for _, f in rows]  # PETRA: names with a leading /
		files[0] = files[0].encode()  # older PETRA files give bytes
		return pd.DataFrame({"image_key": [k for k, _ in rows], "image_file": files})

	def run_spec(self, spec):
		return TIFF.execute(TIFF.validate(spec), threads=4, log=quiet)

	def test_petra_style(self):
		df = self.scan_dataset()
		output = {"format": "den", "path": self.path("den")}
		with self.assertRaisesRegex(TIFF.SpecError, "1 frame\\(s\\).*img_00004.tif.*give fillValue"):
			TOMOGRAPHY.scanTiffStackConversionSpecs(df, output, rawDir=self.dir)
		spec = TOMOGRAPHY.scanTiffStackConversionSpecs(df, output, rawDir=self.dir, fillValue=0)
		self.assertEqual(spec["base_dir"], self.dir)
		self.assertEqual(spec["tiffstacks"]["img"]["frames"][3:6], ["img_00003.tif", None, "img_00005.tif"])
		self.assertEqual(spec["tiffstacks"]["ref"]["framecount"], 6)
		self.assertEqual(spec["tiffstacks"]["dar"]["frame_shape"], [Y, X])
		self.assertEqual(spec["tiffstacks"]["dar"]["dtype"], "uint16")
		# extended with a dark median before running
		spec["virtualstacks"] = {"dar_median": {"stack": "dar", "reduce": "median", "dtype": "uint16"}}
		spec["arrays"]["dar_median"] = {"stack": "dar_median"}
		report = self.run_spec(spec)
		self.assertEqual(report["arrays"]["img"]["null_frames"], [4])
		np.testing.assert_array_equal(DEN.getNumpyArray(self.path("den/ref.den")), self.data["ref"])
		np.testing.assert_array_equal(DEN.getNumpyArray(self.path("den/dar_median.den")),
									  np.rint(np.median(self.data["dar"], axis=0)).astype(np.uint16))
		expected = self.data["img"].copy()
		expected[4] = 0
		np.testing.assert_array_equal(DEN.getNumpyArray(self.path("den/img.den")), expected)

	def test_nanoct_style(self):
		# absolute paths, file_exists, no darks, own names
		df = self.scan_dataset(absolute=True)
		df = df[df["image_key"] != 2].reset_index(drop=True)
		df["file_exists"] = [f != self.path("img_00004.tif") for f in df["image_file"]]
		spec = TOMOGRAPHY.scanTiffStackConversionSpecs(
			df, {"format": "zarr", "store": "zip", "path": self.path("n.zar")}, names={2: "dar", 1: "fla", 0: "img"},
			fillValue=0, defaults={"zarr": {"codec": {"name": "htj2k", "clevel": 0}}}, attributes={"source": "test"})
		self.assertNotIn("base_dir", spec)
		self.assertEqual(spec["tiffstacks"]["dar"]["frames"], [])
		self.run_spec(spec)
		group = zarr.open_group(zarr.storage.ZipStore(self.path("n.zar"), mode="r"), mode="r")
		self.assertEqual(group.attrs["source"], "test")
		self.assertEqual(group["dar"].shape, (0, Y, X))
		np.testing.assert_array_equal(group["fla"][:], self.data["ref"])
		self.assertEqual(group["img"][4].max(), 0)
		with self.assertRaisesRegex(TIFF.SpecError, "image_key \\[0\\] without a name"):
			TOMOGRAPHY.scanTiffStackConversionSpecs(df, {"format": "den", "path": "x"}, names={1: "fla"}, fillValue=0)


if __name__ == "__main__":
	unittest.main()
