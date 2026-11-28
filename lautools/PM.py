#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Projection matrices of the KCT parallel ray geometry for a recorded scan.

There are two levels of functions:

- projection_matrices2D, projection_matrices1D take a CT.ScanDescription
  (PETRA.ScanDescriptionMicrotomography or NANOCT.ScanDescriptionNanotomography) and the parameters
  of CT.FrameGeometry2D, crop_left, crop_right, crop_top, crop_bot, bin_x and bin_y. They use all
  projections of the scan, scan.imageDataset(0), on the frame
  scan.detectorGeometry().get_frame_geometry(...), and call
- projection_matrices2D_frame, projection_matrices1D_frame, which take the frame (a
  CT.FrameGeometry2D of any detector, e.g. of detectorGeometry(overrideMagnification) of a PETRA
  scan) and the table of the projections (columns s_rot, s_stage_x, e.g. a subset of
  imageDataset(0)) explicitly.

The matrices themselves are created by pykct.PM, this module only translates the scan and the frame
into its parameters: 2x4 matrices of the 3D parallel beam tools kct-pb-* by
pykct.PM.projection_matrix2D (pykct.PM.projection_matrix2D_skew with a skewed rotation axis), 1x4
matrices of the 2D parallel beam tools kct-pb2d-* by pykct.PM.projection_matrix1D.

Every projection (image_key 0 of the scan, in acquisition order) gets its own matrix:

- the angle theta is s_rot (degrees), optionally transformed by the PETRA compatibility
  transform theta = -s_rot - 90 deg;
- the stage position s_stage_x (mm) shifts the detector center along the detector x direction by
  s_stage_x - reference, i.e. detector_center_offsetvx of pykct.PM. A scan moving the stage
  between the projections against ring artifacts ("wackle scan") thus gets a different position of
  the image of the rotation axis for every projection, the projections themselves are not shifted.

The rotation center offset is given in detector pixels relative to the geometric center of the
detector, positive towards higher x, which is the convention of the rotation_center_offset_pix*
values of the rotation center detection and of createProjectionMatrices.py --rotation-center-file.
Detector pixels are the pixels of the recorded frames (camera ROI, unbinned), frame pixels those of
the cropped and binned frame. A skewed rotation axis has in the detector row y (row 0 at the top of
the recorded frame) the offset rotation_center_offset_pix + rotation_center_slope * y. The frame
center is CT.FrameGeometry2D.geometric_center_offset() off the detector center, so the rotation axis
is imaged in the frame row r, the detector row y = crop_top + (bin_y - 1) / 2 + r * bin_y, at the
frame pixel

	(frame.sizex - 1) / 2 + (offset(y) * detector.pixel_sizex - frame.geometric_center_offset()[0]) / frame.pixel_sizex
	                      - (s_stage_x - reference) / frame.pixel_sizex,

see rotation_axis_offset_pix. This holds for both formats, with the 2x4 matrices r is the detector
row PY of the image of a point.

A remainder of the binning that is cut off is a crop, e.g. crop_right = detector.sizex % bin_x,
which reproduces the binning center adjustment of createProjectionMatrices.py.

Theory and conventions, see
https://kulvait.github.io/KCT_doc/posts/working-with-kct-cbct-5-parallel-beam-geometry.html
https://kulvait.github.io/KCT_doc/posts/working-with-kct-cbct-6-parallel-beam-geometry-2d.html
https://kulvait.github.io/KCT_doc/posts/tomographic-notes-1-geometric-conventions.html

@author: Vojtech Kulvait
@date: 2026
"""
import logging
import numbers
import re

import numpy as np


log = logging.getLogger(__name__)

# References of s_stage_x, see stage_x_offsets
STAGE_X_REFERENCES = ("midrange", "mean", "first")


# ================================================================================================
# API
# ================================================================================================


def projection_matrices2D(scan, crop_left=0, crop_right=0, crop_top=0, crop_bot=0, bin_x=1, bin_y=1,
						  rotation_center_offset_pix=0.0, rotation_center_slope=0.0, detector_center_offsetvy=0.0,
						  stage_x_reference="midrange", petra_compatibility_transform=False,
						  material_ct_convention=True):
	"""
	2x4 projection matrices [[a, px0], [b, py0]] of all projections of a scan for kct-pb-*.

	The frame is scan.detectorGeometry().get_frame_geometry(crop_left, crop_right, crop_top,
	crop_bot, bin_x, bin_y), the projections scan.imageDataset(0), the other parameters are those of
	projection_matrices2D_frame.

	Returns
	-------
	numpy.ndarray
		Array of shape (projection count, 2, 4) and dtype float64.
	"""
	frame = scan.detectorGeometry().get_frame_geometry(crop_left, crop_right, crop_top, crop_bot, bin_x, bin_y)
	return projection_matrices2D_frame(frame, scan.imageDataset(0), rotation_center_offset_pix=rotation_center_offset_pix,
									   rotation_center_slope=rotation_center_slope,
									   detector_center_offsetvy=detector_center_offsetvy,
									   stage_x_reference=stage_x_reference,
									   petra_compatibility_transform=petra_compatibility_transform,
									   material_ct_convention=material_ct_convention)


def projection_matrices1D(scan, crop_left=0, crop_right=0, crop_top=0, crop_bot=0, bin_x=1, bin_y=1,
						  rotation_center_offset_pix=0.0, rotation_center_slope=0.0, stage_x_reference="midrange",
						  petra_compatibility_transform=False, material_ct_convention=True):
	"""
	1x4 projection matrices [[a0, a1, a2, a3]] of all projections of a scan for kct-pb2d-*.

	The frame is scan.detectorGeometry().get_frame_geometry(crop_left, crop_right, crop_top,
	crop_bot, bin_x, bin_y), the projections scan.imageDataset(0), the other parameters are those of
	projection_matrices1D_frame.

	Returns
	-------
	numpy.ndarray
		Array of shape (projection count, 1, 4) and dtype float64.
	"""
	frame = scan.detectorGeometry().get_frame_geometry(crop_left, crop_right, crop_top, crop_bot, bin_x, bin_y)
	return projection_matrices1D_frame(frame, scan.imageDataset(0), rotation_center_offset_pix=rotation_center_offset_pix,
									   rotation_center_slope=rotation_center_slope,
									   stage_x_reference=stage_x_reference,
									   petra_compatibility_transform=petra_compatibility_transform,
									   material_ct_convention=material_ct_convention)


def projection_matrices2D_frame(frame, projections, rotation_center_offset_pix=0.0, rotation_center_slope=0.0,
								detector_center_offsetvy=0.0, stage_x_reference="midrange",
								petra_compatibility_transform=False, material_ct_convention=True):
	"""
	2x4 projection matrices [[a, px0], [b, py0]] of the projections on a frame for kct-pb-*.

	Parameters
	----------
	frame : CT.FrameGeometry2D
		The frame of the projections, detector after crop and binning.
	projections : pandas.DataFrame
		The projections, one matrix per row: s_rot (degrees) and, unless stage_x_reference is
		None, s_stage_x (mm), e.g. scan.imageDataset(0) or a subset of its rows.
	rotation_center_offset_pix : float
		Offset of the image of the rotation axis in the detector row 0 from the geometric center of
		the detector in detector pixels, positive towards higher x, see the module documentation;
		without slope in every row.
	rotation_center_slope : float
		Shift of the image of the rotation axis in detector pixels per detector row. With a nonzero
		slope the matrices have the skewed y axis of pykct.PM.projection_matrix2D_skew, every
		detector row images a plane of constant z.
	detector_center_offsetvy : float
		Offset of the detector center along the detector y direction in mm, see
		pykct.PM.projection_matrix2D; with 0 the plane z = 0 is imaged at the center row
		(frame.sizey - 1) / 2 of the frame.
	stage_x_reference : str, float or None
		Reference of s_stage_x, see stage_x_offsets; None ignores s_stage_x, e.g. when the
		projections were already shifted back.
	petra_compatibility_transform : bool
		Use theta = -s_rot - 90 deg, the convention of the PETRA reconstruction scripts
		https://github.com/moosmann/matlab.
	material_ct_convention : bool
		See pykct.PM.projection_matrix2D.

	Returns
	-------
	numpy.ndarray
		Array of shape (len(projections), 2, 4) and dtype float64.
	"""
	from pykct import PM as pykctPM
	theta = projection_angles(projections, petra_compatibility_transform)
	offsetvx = stage_x_offsets(projections, stage_x_reference, frame)
	# Offset at the center row (frame.sizey - 1) / 2, to which projection_matrix2D_skew refers
	offset = rotation_axis_offset_pix(frame, rotation_center_offset_pix, rotation_center_slope, 0.5 * (frame.sizey - 1))
	slope = rotation_center_slope * frame.bin_y / frame.bin_x
	matrices = []
	for i in range(len(theta)):
		kwargs = dict(detector_center_offsetvx=float(offsetvx[i]), detector_center_offsetvy=float(detector_center_offsetvy),
					  rotation_center_offset_pix=offset, material_ct_convention=material_ct_convention)
		if slope == 0.0:
			m = pykctPM.projection_matrix2D(float(theta[i]), frame.sizex, frame.sizey, frame.pixel_sizex, frame.pixel_sizey, **kwargs)
		else:
			m = pykctPM.projection_matrix2D_skew(float(theta[i]), frame.sizex, frame.sizey, frame.pixel_sizex, frame.pixel_sizey,
												 rotation_center_slope=slope, **kwargs)
		matrices.append(m)
	return np.array(matrices, dtype=np.float64).reshape((-1, 2, 4))


def projection_matrices1D_frame(frame, projections, rotation_center_offset_pix=0.0, rotation_center_slope=0.0,
								stage_x_reference="midrange", petra_compatibility_transform=False,
								material_ct_convention=True):
	"""
	1x4 projection matrices [[a0, a1, a2, a3]] of the projections on a frame for kct-pb2d-*.

	A skewed rotation axis is encoded by the row protocol of pykct.PM.projection_matrix1D,
	PX = a0 x + a1 y + a2 r + a3, where r is the row index of the frame, so that kct-pb2d-*
	processing these projections use --z-level2d-origin 0. Without slope, crop_top, crop_bot and
	bin_y of the frame do not change the matrices.

	Parameters
	----------
	frame : CT.FrameGeometry2D
		The frame of the projections, detector after crop and binning.
	projections, rotation_center_offset_pix, stage_x_reference, petra_compatibility_transform, material_ct_convention
		See projection_matrices2D_frame.
	rotation_center_slope : float
		Shift of the image of the rotation axis in detector pixels per detector row, e.g.
		rotation_center_offset_pix_interpfit_b of the rotation center detection, whose
		rotation_center_offset_pix_interpfit_a is then rotation_center_offset_pix.

	Returns
	-------
	numpy.ndarray
		Array of shape (len(projections), 1, 4) and dtype float64.
	"""
	from pykct import PM as pykctPM
	theta = projection_angles(projections, petra_compatibility_transform)
	offsetvx = stage_x_offsets(projections, stage_x_reference, frame)
	offset = rotation_axis_offset_pix(frame, rotation_center_offset_pix, rotation_center_slope, 0.0)
	slope = rotation_center_slope * frame.bin_y / frame.bin_x
	matrices = [
		pykctPM.projection_matrix1D(float(theta[i]), frame.sizex, frame.pixel_sizex,
									detector_center_offsetvx=float(offsetvx[i]),
									rotation_center_offset_pix=offset,
									rotation_center_slope=slope,
									material_ct_convention=material_ct_convention) for i in range(len(theta))
	]
	return np.array(matrices, dtype=np.float64).reshape((-1, 1, 4))


def rotation_axis_offset_pix(frame, rotation_center_offset_pix, rotation_center_slope=0.0, frame_row=0.0):
	"""
	Offset of the image of the rotation axis in frame pixels from the frame center
	(frame.sizex - 1) / 2 in the frame row frame_row, without the stage shift.

	The offset rotation_center_offset_pix + rotation_center_slope * y is in detector pixels from the
	detector center, y = (frame.originy - detector.originy) / detector.pixel_sizey + frame_row *
	bin_y is the detector row of the frame row; frame.geometric_center_offset() moves it to the
	frame center.
	"""
	detector = frame.detector
	y = (frame.originy - detector.originy) / detector.pixel_sizey + frame_row * frame.bin_y
	offset = rotation_center_offset_pix + rotation_center_slope * y
	return (offset * detector.pixel_sizex - frame.geometric_center_offset()[0]) / frame.pixel_sizex


def read_rotation_center_file(path, detector_pixel_size):
	"""
	Rotation center of a detectRotationCenter*.py log: its rotation_center_offset*=value lines.

	Values are relative to the center of the unbinned camera frame (ROI), *_pix* values in detector
	(unbinned camera) pixels, the others in mm, converted by detector_pixel_size. Returns
	(offset_pix, fit_a_pix, fit_b_pix) in detector pixels, the fit offset(y) = fit_a_pix +
	fit_b_pix * y in the detector row y; offset_pix is rotation_center_offset_pix_interp, else
	rotation_center_offset_pix, else rotation_center_offset, as createProjectionMatrices.py. Missing
	values are None.
	"""
	keys = {}
	with open(path, "r") as f:
		for line in f:
			m = re.match(r"^(rotation_center_offset\w*)=\s*(\S+)", line.strip())
			if m is not None:
				try:
					keys[m.group(1)] = float(m.group(2))
				except ValueError:
					pass
	offset = None
	for key, scale in (("rotation_center_offset_pix_interp", 1.0), ("rotation_center_offset_pix", 1.0),
					   ("rotation_center_offset", 1.0 / detector_pixel_size)):
		if key in keys:
			offset = keys[key] * scale
			break
	fit_a, fit_b = keys.get("rotation_center_offset_pix_interpfit_a"), keys.get("rotation_center_offset_pix_interpfit_b")
	if (fit_a is None or fit_b is None) and "rotation_center_offset_interpfit_a" in keys and "rotation_center_offset_interpfit_b" in keys:
		fit_a = keys["rotation_center_offset_interpfit_a"] / detector_pixel_size
		fit_b = keys["rotation_center_offset_interpfit_b"] / detector_pixel_size
	return offset, fit_a, fit_b


def projection_angles(dataset, petra_compatibility_transform=False):
	"""The angles theta in radians of the projections: s_rot (degrees), with
	petra_compatibility_transform -s_rot - 90 deg; ValueError if there are no projections, no s_rot
	or some are not finite."""
	if len(dataset) == 0:
		raise ValueError("No projections.")
	if "s_rot" not in dataset.columns:
		raise ValueError("No s_rot of the projections.")
	theta = np.deg2rad(dataset["s_rot"].to_numpy(dtype=np.float64))
	if not np.all(np.isfinite(theta)):
		raise ValueError("%d of %d projections have no finite s_rot." % (np.sum(~np.isfinite(theta)), len(theta)))
	if petra_compatibility_transform:
		theta = -theta - 0.5 * np.pi
	return theta


def stage_x_offsets(dataset, reference="midrange", frame=None):
	"""
	The shifts s_stage_x - reference (mm) of the projections, detector_center_offsetvx of pykct.PM.

	reference: "midrange" (max + min) / 2 of the projections, as pixel_shift of
	PETRA.imageDataset and createProjectionMatrices.py; "mean"; "first", the first projection; a
	number in mm; None gives zeros (s_stage_x ignored). ValueError if s_stage_x is missing or not
	finite. With a frame whose detector has the pixel size 1.0 (not known, e.g. nanotomography
	without pixelSize_mm) nonzero shifts are logged as a warning, as the mm are then taken as pixels.
	"""
	if reference is None:
		return np.zeros(len(dataset), dtype=np.float64)
	if "s_stage_x" not in dataset.columns:
		raise ValueError("No s_stage_x of the projections, use reference None to ignore it.")
	s = dataset["s_stage_x"].to_numpy(dtype=np.float64)
	if not np.all(np.isfinite(s)):
		raise ValueError("%d of %d projections have no finite s_stage_x, use reference None to ignore it." % (np.sum(~np.isfinite(s)), len(s)))
	if isinstance(reference, numbers.Real) and not isinstance(reference, bool):
		ref = float(reference)
	elif reference == "midrange":
		ref = 0.5 * (s.max() + s.min())
	elif reference == "mean":
		ref = float(s.mean())
	elif reference == "first":
		ref = float(s[0])
	else:
		raise ValueError("stage_x_reference %r is none of %s, a number or None." % (reference, STAGE_X_REFERENCES))
	offsets = s - ref
	if frame is not None and frame.detector.pixel_sizex == 1.0 and np.any(offsets != 0.0):
		log.warning("The s_stage_x of the projections vary by %g mm but the pixel size is 1.0, the shifts are taken as pixels." % (s.max() - s.min()))
	return offsets
