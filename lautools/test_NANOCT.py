#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of NANOCT.ScanDescriptionNanotomography (CT.ScanDescription of LogScan.log and TIFF files)
on a synthetic scan: flats, projections, flats.
"""
import os
import tempfile
import unittest

import numpy as np
import tifffile

from lautools import CT, NANOCT

START = 1_700_000_000.0  # #starttime of the log, UNIX seconds


def make_scan(directory):
	"""LogScan.log and raw/ with 3 flats (counters 0-2), 5 projections (3-7), 3 flats (8-10);
	TIFF files of 6 x 4 pixels, file times in acquisition order."""
	rows, files = [], []
	for counter in range(11):
		kind = "img" if 3 <= counter <= 7 else "ref"
		block = 0 if counter < 3 else 1
		rows.append(f"{kind} start {block} 0 {counter} {0.5 * counter:.3f} {100.0 - counter:.1f} {18.0 * (counter - 3) if kind == 'img' else 0.0:.1f}")
		if kind == "img":
			files.append(f"img_{counter:05d}.tif")
		else:
			first = 0 if counter < 3 else 8
			files.append(f"ref_{first:05d}_{counter - first + 1}.tif")
	log = os.path.join(directory, "LogScan.log")
	with open(log, "w") as f:
		f.write(f"#starttime={START}\n" + "\n".join(rows) + "\n")
	raw = os.path.join(directory, "raw")
	os.makedirs(raw)
	for i, name in enumerate(files):
		path = os.path.join(raw, name)
		tifffile.imwrite(path, np.full((6, 4), i, dtype=np.uint16))
		os.utime(path, (START + i, START + i))
	return log, raw


class TestScanDescriptionNanotomography(unittest.TestCase):

	def setUp(self):
		self.tmp = tempfile.TemporaryDirectory()
		self.log, self.raw = make_scan(self.tmp.name)

	def tearDown(self):
		self.tmp.cleanup()

	def test_scan_dataset(self):
		scan = NANOCT.ScanDescriptionNanotomography(self.log, self.raw)
		self.assertIsInstance(scan, CT.ScanDescription)
		df = CT.validateScanDataset(scan.scanDataset(), name=self.log)
		self.assertEqual(list(df["image_key"]), [1] * 3 + [0] * 5 + [1] * 3)
		np.testing.assert_array_equal(df.index, int(START * 1000) + 500 * np.arange(11))
		self.assertNotIn("current", df.columns)
		self.assertIn("current", scan.scanDataset(includeCurrent=True).columns)
		images = scan.imageDataset()
		np.testing.assert_array_equal(images["s_rot"], [0.0, 18.0, 36.0, 54.0, 72.0])
		np.testing.assert_array_equal(images["frame_ind"], np.arange(5))
		self.assertTrue(all(images["s_stage_x"] == 0.0))

	def test_detector_geometry(self):
		det = NANOCT.ScanDescriptionNanotomography(self.log, self.raw).detectorGeometry()
		self.assertEqual((det.sizex, det.sizey, det.pixel_sizex), (4, 6, 1.0))
		det = NANOCT.ScanDescriptionNanotomography(self.log, self.raw, pixelSize_mm=0.00005).detectorGeometry()
		self.assertEqual((det.pixel_sizex, det.pixel_sizey), (0.00005, 0.00005))
		with self.assertRaisesRegex(ValueError, "No image directory"):
			NANOCT.ScanDescriptionNanotomography(self.log).detectorGeometry()

	def test_beam_current(self):
		current = NANOCT.ScanDescriptionNanotomography(self.log, self.raw).beamCurrentDataset()
		np.testing.assert_array_equal(current.index, int(START * 1000) + 500 * np.arange(11))
		np.testing.assert_array_equal(current["current"], 100.0 - np.arange(11))


if __name__ == "__main__":
	unittest.main()
