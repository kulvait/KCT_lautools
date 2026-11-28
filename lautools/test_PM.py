#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of lautools.PM: the matrices are compared with the formulas of createProjectionMatrices.py
(binning center adjustment, pixel shifts of s_stage_x, fitted matrix of a skewed rotation axis) and
with the position of the image of the rotation axis stated in the module documentation.
"""
import unittest

import numpy as np
import pandas as pd

from lautools import CT
from lautools import PM


def detector(sizex=2049, sizey=64, pixel_size=0.00325):
	return CT.DetectorGeometry2D(sizex=sizex, sizey=sizey, pixel_sizex=pixel_size, pixel_sizey=pixel_size)


class ListScan(CT.ScanDescription):
	"""A scan from projections s_rot, s_stage_x and a detector."""

	def __init__(self, s_rot, s_stage_x, det):
		n = len(s_rot)
		index = 1_700_000_000_000 + 100 * np.arange(n, dtype=np.int64)
		self.df = pd.DataFrame({"time": pd.to_datetime(index, unit="ms"), "image_key": np.zeros(n, dtype=int),
								"image_file": [f"f{i}.tif" for i in range(n)], "s_rot": np.asarray(s_rot, dtype=float),
								"s_stage_x": np.asarray(s_stage_x, dtype=float)}, index=index)
		self.det = det

	def scanDataset(self, includeCurrent=False):
		return self.df

	def detectorGeometry(self):
		return self.det

	def beamCurrentDataset(self):
		raise NotImplementedError


def legacy_matrices(s_rot, s_stage_x, det, bin_x, bin_y, N, M, offset_pix, petra, material, flat=False,
					fit_b_pix=None, fit_a_pix=None):
	"""createProjectionMatrices.py for an uncropped frame binned from the top left corner, offsets in
	unbinned camera pixels (--rotation-center-file values *_pix), detector_center_offsetvx/vy = 0."""
	cam = det.pixel_sizex
	PX, PY = cam * bin_x, cam * bin_y
	binadj = 0.5 * (det.sizex % bin_x) * cam
	rco = offset_pix * cam
	s = np.asarray(s_stage_x, dtype=float)
	pixel_shift = (s - 0.5 * (s.max() + s.min())) / cam
	total = binadj + rco
	slope = 0.0
	if fit_b_pix is not None:
		# --rotation-center-file-fitted-matrix: offset at the center row (M-1)/2 and its slope
		unbinned_row = lambda y: y * bin_y + 0.5 * (bin_y - 1)
		total = binadj + (fit_a_pix + fit_b_pix * unbinned_row(0.5 * (M - 1))) * cam
		slope = fit_b_pix * cam * bin_y / PX
	out = []
	for i, deg in enumerate(s_rot):
		t = np.pi * deg / 180
		if petra:
			t = -t - 0.5 * np.pi
		if material:
			a = np.array([np.cos(t) / PX, np.sin(t) / PX, 0.0])
			b = np.array([0.0, 0.0, 1.0 / PY])
		else:
			a = np.array([np.cos(t) / PX, -np.sin(t) / PX, 0.0])
			b = np.array([0.0, 0.0, -1.0 / PY])
		px0 = N * 0.5 - 0.5 + total / PX - pixel_shift[i] / bin_x
		py0 = M * 0.5 - 0.5
		if fit_b_pix is not None:
			a[2] = slope * b[2]
		out.append([np.append(a, px0)] if flat else [np.append(a, px0), np.append(b, py0)])
	return np.array(out)


class TestPM(unittest.TestCase):

	def setUp(self):
		self.det = detector()
		self.s_rot = np.linspace(0.0, 180.0, 37)
		# wackle scan: the stage moves by up to 20 pixels
		rng = np.random.default_rng(3)
		self.s_stage_x = 1.234 + self.det.pixel_sizex * rng.uniform(-10, 10, len(self.s_rot))
		self.scan = ListScan(self.s_rot, self.s_stage_x, self.det)

	def test_2D_matches_createProjectionMatrices(self):
		# bin 2 of 2049 columns drops the last one: crop_right = 1
		for petra in (False, True):
			for material in (False, True):
				frame = CT.FrameGeometry2D(self.det, crop_right=1, bin_x=2, bin_y=2)
				P = PM.projection_matrices2D(self.scan, crop_right=1, bin_x=2, bin_y=2, rotation_center_offset_pix=7.25,
											 petra_compatibility_transform=petra, material_ct_convention=material)
				L = legacy_matrices(self.s_rot, self.s_stage_x, self.det, 2, 2, frame.sizex, frame.sizey, 7.25,
									petra, material)
				self.assertEqual(P.shape, (len(self.s_rot), 2, 4))
				np.testing.assert_allclose(P, L, rtol=0, atol=1e-9)

	def test_1D_fitted_matrix_matches_createProjectionMatrices(self):
		# Legacy 2x4 fitted matrix evaluated at z: the row r = py(z) of the 1x4 matrix gives the same column
		frame = CT.FrameGeometry2D(self.det, crop_right=1, bin_x=2, bin_y=2)
		A, B = 3.5, 0.02
		P = PM.projection_matrices1D(self.scan, crop_right=1, bin_x=2, bin_y=2, rotation_center_offset_pix=A, rotation_center_slope=B,
									 material_ct_convention=True)
		L = legacy_matrices(self.s_rot, self.s_stage_x, self.det, 2, 2, frame.sizex, frame.sizey, 0.0, False, True,
							fit_a_pix=A, fit_b_pix=B)
		rng = np.random.default_rng(5)
		for i in range(len(self.s_rot)):
			for x, y, z in rng.uniform(-2, 2, (5, 3)):
				px_legacy = L[i, 0, :3] @ [x, y, z] + L[i, 0, 3]
				r = L[i, 1, :3] @ [x, y, z] + L[i, 1, 3]
				px = P[i, 0, :3] @ [x, y, r] + P[i, 0, 3]
				self.assertAlmostEqual(px, px_legacy, places=8)

	def test_axis_position_crop_bin_wackle(self):
		crop = dict(crop_left=17, crop_right=4, crop_top=8, crop_bot=8, bin_x=4, bin_y=4)
		frame = CT.FrameGeometry2D(self.det, **crop)
		offset, B = -12.5, 0.03
		ref = 0.5 * (self.s_stage_x.max() + self.s_stage_x.min())
		P2 = PM.projection_matrices2D(self.scan, **crop, rotation_center_offset_pix=offset)
		P1 = PM.projection_matrices1D(self.scan, **crop, rotation_center_offset_pix=offset, rotation_center_slope=B)
		# with a moved detector origin the matrices are the same
		moved = self.det.set_origin(10.0, -4.0)
		frame_moved = moved.get_frame_geometry(17, 4, 8, 8, 4, 4)
		np.testing.assert_allclose(PM.projection_matrices2D_frame(frame_moved, self.scan.imageDataset(0), rotation_center_offset_pix=offset),
								   P2, rtol=0, atol=1e-9)
		for i in range(len(self.s_rot)):
			shift = (self.s_stage_x[i] - ref) / frame.pixel_sizex
			# 2D: axis (x = y = 0) at (sizex-1)/2 + (offset + (crop_right-crop_left)/2)/bin_x - shift
			expected = 0.5 * (frame.sizex - 1) + (offset + 0.5 * (4 - 17)) / 4 - shift
			self.assertAlmostEqual(P2[i, 0, 3], expected, places=9)
			# 1D: in frame row r the detector row is crop_top + 4 r + 1.5, the axis in detector pixels
			# offset + B * row from the detector center, mapped to frame pixels
			for r in (0.0, 3.0, frame.sizey - 1.0):
				row = 8 + 4 * r + 1.5
				axis_det = 0.5 * (self.det.sizex - 1) + offset + B * row - shift * 4
				axis_frame = (axis_det - 17 - 1.5) / 4
				self.assertAlmostEqual(P1[i, 0, 2] * r + P1[i, 0, 3], axis_frame, places=9)

	def test_2D_skew(self):
		# fitted 2x4 matrices of createProjectionMatrices.py (uncropped frame) and the axis condition of PM with crop
		A, B = 3.5, 0.02
		frame = CT.FrameGeometry2D(self.det, crop_right=1, bin_x=2, bin_y=2)
		P = PM.projection_matrices2D(self.scan, crop_right=1, bin_x=2, bin_y=2, rotation_center_offset_pix=A,
									 rotation_center_slope=B, material_ct_convention=True)
		L = legacy_matrices(self.s_rot, self.s_stage_x, self.det, 2, 2, frame.sizex, frame.sizey, 0.0, False, True,
							fit_a_pix=A, fit_b_pix=B)
		np.testing.assert_allclose(P, L, rtol=0, atol=1e-9)
		crop = dict(crop_left=17, crop_right=4, crop_top=8, crop_bot=8, bin_x=4, bin_y=4)
		frame = CT.FrameGeometry2D(self.det, **crop)
		ref = 0.5 * (self.s_stage_x.max() + self.s_stage_x.min())
		for material in (True, False):
			P = PM.projection_matrices2D(self.scan, **crop, rotation_center_offset_pix=A, rotation_center_slope=B,
										 detector_center_offsetvy=0.01, material_ct_convention=material)
			for i in range(len(self.s_rot)):
				shift = (self.s_stage_x[i] - ref) / frame.pixel_sizex
				for z in (-0.05, 0.0, 0.03):
					px = P[i, 0, 2] * z + P[i, 0, 3]
					r = P[i, 1, 2] * z + P[i, 1, 3]
					expected = 0.5 * (frame.sizex - 1) + PM.rotation_axis_offset_pix(frame, A, B, r) - shift
					self.assertAlmostEqual(px, expected, places=9)

	def test_read_rotation_center_file(self):
		import os
		import tempfile
		with tempfile.TemporaryDirectory() as d:
			path = os.path.join(d, "rc.log")
			with open(path, "w") as f:
				f.write("rotation_center_offset=0.013\nrotation_center_offset_pix=4.0\nother=1\n"
						"rotation_center_offset_interpfit_a=0.0065\nrotation_center_offset_interpfit_b=0.000013\n")
			offset, a, b = PM.read_rotation_center_file(path, 0.00325)
			self.assertEqual(offset, 4.0)
			self.assertAlmostEqual(a, 2.0)
			self.assertAlmostEqual(b, 0.004)
			with open(path, "w") as f:
				f.write("rotation_center_offset_pix_interp=1.5\nrotation_center_offset_pix_interpfit_a=1.0\n"
						"rotation_center_offset_pix_interpfit_b=0.01\n")
			self.assertEqual(PM.read_rotation_center_file(path, 0.00325), (1.5, 1.0, 0.01))

	def test_1D_without_slope_is_first_row_of_2D(self):
		P2 = PM.projection_matrices2D(self.scan, crop_left=3, crop_right=6, bin_x=2, rotation_center_offset_pix=1.5, petra_compatibility_transform=True)
		P1 = PM.projection_matrices1D(self.scan, crop_left=3, crop_right=6, bin_x=2, rotation_center_offset_pix=1.5, petra_compatibility_transform=True)
		np.testing.assert_array_equal(P1[:, 0, :], P2[:, 0, :])

	def test_1D_rows_only_with_slope(self):
		P = PM.projection_matrices1D(self.scan, crop_left=2, crop_right=3, bin_x=2, rotation_center_offset_pix=4.0)
		Q = PM.projection_matrices1D(self.scan, crop_left=2, crop_right=3, bin_x=2, crop_top=8, crop_bot=6, bin_y=5,
									 rotation_center_offset_pix=4.0)
		np.testing.assert_array_equal(P, Q)
		# no crop and binning in y: the frame row is the detector row
		P = PM.projection_matrices1D(self.scan, crop_left=2, crop_right=3, bin_x=2, rotation_center_offset_pix=4.0,
									 rotation_center_slope=0.1, stage_x_reference=None)
		np.testing.assert_allclose(P[:, 0, 2], 0.05)
		np.testing.assert_allclose(P[:, 0, 3], 0.5 * (2044 / 2 - 1) + (4.0 + 0.5) / 2)

	def test_default_frame_and_stage_reference(self):
		scan = ListScan(self.s_rot, np.full(len(self.s_rot), 5.0), self.det)
		P = PM.projection_matrices2D(scan)
		np.testing.assert_allclose(P[:, 0, 3], 0.5 * (self.det.sizex - 1))
		np.testing.assert_allclose(P[:, 1, 3], 0.5 * (self.det.sizey - 1))
		s = self.s_stage_x
		self.assertAlmostEqual(PM.stage_x_offsets(self.scan.df, "mean").mean(), 0.0, places=12)
		self.assertEqual(PM.stage_x_offsets(self.scan.df, "first")[0], 0.0)
		np.testing.assert_allclose(PM.stage_x_offsets(self.scan.df, 1.0), s - 1.0)
		np.testing.assert_array_equal(PM.stage_x_offsets(self.scan.df, None), 0.0)

	def test_subset_dataset(self):
		df = self.scan.imageDataset(0).iloc[::4]
		P = PM.projection_matrices2D_frame(self.det.get_frame_geometry(), df, stage_x_reference=None)
		self.assertEqual(P.shape[0], len(df))
		np.testing.assert_allclose(P[:, 0, 0], np.cos(np.deg2rad(df["s_rot"].to_numpy())) / self.det.pixel_sizex)

	def test_errors(self):
		with self.assertRaises(ValueError):
			PM.stage_x_offsets(self.scan.df, "median")
		bad = self.scan.df.assign(s_stage_x=np.where(np.arange(len(self.scan.df)) == 2, np.nan, self.s_stage_x))
		frame = self.det.get_frame_geometry()
		with self.assertRaises(ValueError):
			PM.projection_matrices2D_frame(frame, bad)
		PM.projection_matrices2D_frame(frame, bad, stage_x_reference=None)
		with self.assertRaises(ValueError):
			PM.projection_matrices1D_frame(frame, self.scan.df.iloc[:0])
		with self.assertRaises(ValueError):
			PM.projection_matrices2D(self.scan, bin_x=2)  # 2049 columns, crop_right=1 needed


if __name__ == "__main__":
	unittest.main()
