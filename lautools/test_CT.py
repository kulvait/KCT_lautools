#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of lautools.CT: DetectorGeometry2D/1D, FrameGeometry/1D, validateScanDataset and the common part of ScanDescription.
The implementations are tested in test_PETRA (microtomography) and test_NANOCT (nanotomography).
"""
import unittest

import numpy as np
import pandas as pd

from lautools import CT


def frames(keys=(2, 1, 0, 0, 0, 1), start=1_700_000_000_000):
	n = len(keys)
	index = start + 100 * np.arange(n, dtype=np.int64)
	return pd.DataFrame({"time": pd.to_datetime(index, unit="ms"), "image_key": list(keys),
						 "image_file": [f"f{i}.tif" for i in range(n)], "s_rot": np.linspace(0, 10, n),
						 "s_stage_x": np.zeros(n), "current": np.full(n, 100.0)}, index=index)


class ListScan(CT.ScanDescription):
	"""A scan from a DataFrame, for the common methods."""

	def __init__(self, df):
		self.df = df

	def scanDataset(self, includeCurrent=False):
		return self.df if includeCurrent else self.df.drop(columns=["current"])

	def detectorGeometry(self):
		return CT.DetectorGeometry2D(sizex=8, sizey=4)

	def beamCurrentDataset(self):
		return self.df[["time", "current"]]


class TestDetectorGeometry(unittest.TestCase):

	def test_detector(self):
		det = CT.DetectorGeometry2D(sizex=5120, sizey=1001, pixel_sizex=0.00128, pixel_sizey=0.00128)
		# world coordinates: origin + pixel size * (size - 1) / 2
		self.assertAlmostEqual(det.geometric_centerx(), 0.00128 * 2559.5)
		self.assertAlmostEqual(det.geometric_centery(), 0.00128 * 500.0)
		moved = det.set_origin(-3.0, 1.0)
		self.assertEqual((moved.originx, moved.originy, det.originx), (-3.0, 1.0, 0.0))
		self.assertAlmostEqual(moved.geometric_centerx(), -3.0 + 0.00128 * 2559.5)
		self.assertEqual(CT.DetectorGeometry1D(7, 0.5).set_origin(2.0).geometric_centerx(), 3.5)
		self.assertEqual((CT.DetectorGeometry2D(4, 4).pixel_sizex, CT.DetectorGeometry1D(7).geometric_centerx()), (1.0, 3.0))
		for kwargs in ({"sizex": 0}, {"pixel_sizex": 0.0}, {"pixel_sizey": -1.0}):
			with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
				CT.DetectorGeometry2D(**{"sizex": 10, "sizey": 10, **kwargs})
		with self.assertRaises(Exception):
			det.sizex = 20

	def test_frame(self):
		det = CT.DetectorGeometry2D(sizex=10, sizey=6, pixel_sizex=0.001, pixel_sizey=0.002)
		frame = CT.FrameGeometry2D(det, crop_left=2, bin_x=4, bin_y=2)
		self.assertEqual((frame.sizex, frame.sizey, frame.pixel_sizex, frame.pixel_sizey), (2, 3, 0.004, 0.004))
		close = lambda a, b: np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
		# world coordinates, the frame origin is the center of the frame pixel 0, i.e. of detector pixels 2..5 in x
		close((frame.originx, frame.originy), (0.001 * 3.5, 0.002 * 0.5))
		close(frame.geometric_center, (0.002, 0.004))
		# the center of the frame is the center of the cropped detector pixels 2..9
		close(frame.detector_coordinate_of_geometric_center(), (0.001 * 5.5, 0.002 * 2.5))
		close(frame.frame_coordinate_to_detector_coordinate(0, 0), (0.0035, 0.001))
		# detector coordinates of the frame pixel (1, 2), frame coordinates divided by the frame pixel size
		fx, fy = frame.detector_coordinate_to_frame_coordinate(0.001 * 7.5, 0.002 * 4.5)
		close((fx / frame.pixel_sizex, fy / frame.pixel_sizey), (1.0, 2.0))
		close(frame.geometric_center_offset(), (0.001 * (2 - 0) / 2, 0.0))
		# the offset does not depend on the origin of the detector
		close(CT.FrameGeometry2D(det.set_origin(5.0, -2.0), crop_left=2, bin_x=4, bin_y=2).geometric_center_offset(),
			  frame.geometric_center_offset())
		self.assertEqual(det.get_frame_geometry(crop_left=2, bin_x=4, bin_y=2), frame)
		for kwargs in ({"bin_x": 3}, {"crop_top": 6}, {"crop_left": -1}, {"bin_y": 0}):
			with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
				CT.FrameGeometry2D(det, **kwargs)
		line = CT.FrameGeometry1D(CT.DetectorGeometry1D(10, 0.001), crop_right=2, bin_x=2)
		self.assertEqual((line.sizex, line.pixel_sizex), (4, 0.002))
		close(line.geometric_center_offset(), 0.001 * (0 - 2) / 2)
		close(line.frame_coordinate_to_detector_coordinate(line.geometric_center), 0.001 * 3.5)
		with self.assertRaises(ValueError):
			CT.FrameGeometry1D(CT.DetectorGeometry1D(10), bin_x=3)


class TestValidateScanDataset(unittest.TestCase):

	def test_valid(self):
		df = frames()
		self.assertIs(CT.validateScanDataset(df), df)

	def test_violations(self):
		df = frames()
		cases = {
			"missing columns \\['s_stage_x'\\]": df.drop(columns=["s_stage_x"]),
			"not strictly increasing": df.set_axis(df.index[::-1]),
			"not the UNIX time": df.set_axis(df.index.astype(np.float64)),
			"image_key values \\[3\\]": df.assign(image_key=[2, 1, 0, 3, 0, 1]),
		}
		for message, bad in cases.items():
			with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
				CT.validateScanDataset(bad)


class TestScanDescription(unittest.TestCase):

	def test_abstract(self):
		with self.assertRaises(TypeError):
			CT.ScanDescription()

	def test_image_dataset(self):
		scan = ListScan(frames())
		images = scan.imageDataset()
		self.assertEqual(list(images["image_file"]), ["f2.tif", "f3.tif", "f4.tif"])
		np.testing.assert_array_equal(images["frame_ind"], [0, 1, 2])
		self.assertNotIn("current", images.columns)
		flats = scan.imageDataset(1, includeCurrent=True)
		self.assertEqual(list(flats["image_file"]), ["f1.tif", "f5.tif"])
		self.assertIn("current", flats.columns)


if __name__ == "__main__":
	unittest.main()
