#!/usr/bin/env python
"""
Create projection matrices of a recorded scan in the parallel ray geometry.

Port of createProjectionMatrices.py onto lautools.PM. The scan is a PETRA microtomography scan
(HDF5 file or zarr store) or a P05 nanotomography scan (LogScan.log with --img-dir). The
subcommand selects the format:

    PB3D  2x4 matrices of the 3D parallel beam tools kct-pb-*, lautools.PM.projection_matrices2D_frame;
    PB2D  1x4 matrices of the 2D parallel beam tools kct-pb2d-*, lautools.PM.projection_matrices1D_frame,
          they replace --flat-matrices of createProjectionMatrices.py.

Every projection gets its own matrix: the angle s_rot and the stage position s_stage_x, so that the
moving stage of a wackle scan is taken into account. The options keep the meaning of
createProjectionMatrices.py: the rotation center offsets are relative to the center of the
unbinned camera frame (ROI), --rotation-center-offset-pix in binned pixels, the values of
--rotation-center-file in unbinned pixels. With --rotation-center-file-fitted-matrix the linear fit
of the rotation center gives a skewed rotation axis: PB2D encodes it by the row protocol of the
kct-pb2d-* tools (use --z-level2d-origin 0 with the projections of the frame, also with
--slab-from), PB3D by the skewed y axis of the detector. --detector-sizex and --detector-sizey are
replaced by --crop-left, --crop-right, --crop-top and --crop-bot in unbinned pixels, a remainder of
the binning is cropped at the right and bottom as createProjectionMatrices.py did.

Not ported: the DEN tick file input and --fix-corrupted-h5.

Usage:
    createProjectionMatrices PB2D scan.h5 pm.den --bin-x 2 --bin-y 2 --rotation-center-file rotationaxis.log \\
        --rotation-center-file-fitted-matrix --petra-compatibility-transform
    createProjectionMatrices PB3D scan.h5 pm.den --bin-x 2 --bin-y 2 --crop-top 100 --crop-bot 100

@author: Vojtech Kulvait
@license: GNU GPL v3
@date: 2026
"""

import argparse
import dataclasses
import json
import logging
import os
import sys

import numpy as np

log = logging.getLogger("createProjectionMatrices")


def common_arguments():
	p = argparse.ArgumentParser(add_help=False)
	p.add_argument("inputGeometryInfo", help="PETRA scan: HDF5 file or zarr store; nanotomography scan: LogScan.log, provide --img-dir.")
	p.add_argument("outputProjectionMatrices", help="DEN file to store the projection matrices.")
	src = p.add_argument_group("scan")
	src.add_argument("--img-dir", default=None, help="Nanotomography: directory of the TIFF files of the scan, needed for the detector size.")
	src.add_argument("--pixel-size-mm", type=float, default=None, help="Nanotomography: effective pixel size in mm, the log does not record it.")
	src.add_argument("--override-magnification", type=float, default=None, help="PETRA: magnification to use instead of the recorded one.")
	src.add_argument("--pixel-sizex", type=float, default=None, help="Unbinned pixel size x to use instead of that of the scan.")
	src.add_argument("--pixel-sizey", type=float, default=None, help="Unbinned pixel size y to use instead of that of the scan.")
	src.add_argument("--input-file", default=None, help="DEN file with the projections, its dimensions are checked against the frame and the projection count.")
	frame = p.add_argument_group("frame", "Crop of the recorded frame in unbinned pixels, then binning. A remainder of the binning is cropped at the right and bottom.")
	for side in ("left", "right", "top", "bot"):
		frame.add_argument(f"--crop-{side}", type=int, default=0, help=f"Unbinned pixels cropped at the {side}.")
	frame.add_argument("--bin-x", type=int, default=1, help="X dimension of the binning box.")
	frame.add_argument("--bin-y", type=int, default=1, help="Y dimension of the binning box.")
	rc = p.add_argument_group("rotation center", "Offset of the image of the rotation axis from the center of the unbinned camera frame (ROI), positive towards higher x, defaults to 0.0.")
	g = rc.add_mutually_exclusive_group()
	g.add_argument("--rotation-center-offset", type=float, default=None, help="Offset in physical units of the pixel size.")
	g.add_argument("--rotation-center-offset-pix", type=float, default=None, help="Offset in pixels after binning, i.e. in units of bin_x camera pixels.")
	g.add_argument("--rotation-center-file", default=None, help="Log file of detectRotationCenter*.py with rotation_center_offset*=value lines, *_pix* values in unbinned camera pixels. Priority: linear fit evaluated at --rotation-center-file-fit-y, rotation_center_offset_pix_interp, rotation_center_offset_pix, rotation_center_offset.")
	rc.add_argument("--rotation-center-file-fit-y", type=float, default=None, help="Binned camera row (bin_y camera rows per row, row 0 at the top of the camera) at which the linear fit of --rotation-center-file is evaluated.")
	rc.add_argument("--rotation-center-file-fitted-matrix", action="store_true", help="Skewed rotation axis given by the linear fit of --rotation-center-file.")
	rc.add_argument("--detector-center-offsetvx", type=float, default=0., help="Additional offset in physical units added to the rotation center offset (sign of createProjectionMatrices.py, opposite to the same option of createCameraMatricesForCircularScanTrajectoryPB).")
	geo = p.add_argument_group("geometry")
	geo.add_argument("--material-ct-convention", action="store_true", help="The z axis direction and PY direction will coincide, that is usually not the case in medical CT praxis. See also https://kulvait.github.io/KCT_doc/posts/tomographic-notes-1-geometric-conventions.html.")
	geo.add_argument("--petra-compatibility-transform", action="store_true", help="Use theta = -s_rot - 90 deg, the convention of the PETRA reconstruction scripts https://github.com/moosmann/matlab.")
	geo.add_argument("--ignore-exact-angles", action="store_true", help="Replace the angles by a uniform discretization between the first and the last angle, which have to be nondecreasing.")
	stage = geo.add_mutually_exclusive_group()
	stage.add_argument("--ignore-pixel-shifts", action="store_true", help="Ignore s_stage_x, e.g. when the projections were shifted back.")
	stage.add_argument("--stage-x-reference", default="midrange", help="Reference of s_stage_x: midrange, mean, first or a value in mm, see lautools.PM.stage_x_offsets.")
	p.add_argument("--force", action="store_true", help="Overwrite existing output files.")
	p.add_argument("--write-params-file", action="store_true", help="Write the parameters of the call to OUTPUTPROJECTIONMATRICES.params.")
	p.add_argument("--verbose", action="store_true")
	return p


def stage_reference(value):
	if value in ("midrange", "mean", "first"):
		return value
	try:
		return float(value)
	except ValueError:
		raise ValueError("--stage-x-reference %r is none of midrange, mean, first or a number." % value)


def check_output(path, force):
	if os.path.exists(path):
		if not os.path.isfile(path):
			sys.exit("Path %s exists and is not a file, remove it to proceed." % path)
		if not force:
			sys.exit("File %s exists, add --force to proceed." % path)


def scan_description(ARG, parser):
	from lautools import NANOCT, PETRA
	if ARG.inputGeometryInfo.endswith(".log"):
		if ARG.override_magnification is not None:
			parser.error("--override-magnification applies to PETRA scans only, use --pixel-size-mm")
		scan = NANOCT.ScanDescriptionNanotomography(ARG.inputGeometryInfo, imgDir=ARG.img_dir, pixelSize_mm=ARG.pixel_size_mm)
		detector = scan.detectorGeometry()
	else:
		if ARG.img_dir is not None or ARG.pixel_size_mm is not None:
			parser.error("--img-dir and --pixel-size-mm apply to nanotomography scans (LogScan.log) only")
		scan = PETRA.ScanDescriptionFactory.create(ARG.inputGeometryInfo)
		detector = scan.detectorGeometry(ARG.override_magnification)
	if ARG.pixel_sizex is not None:
		detector = dataclasses.replace(detector, pixel_sizex=ARG.pixel_sizex)
	if ARG.pixel_sizey is not None:
		detector = dataclasses.replace(detector, pixel_sizey=ARG.pixel_sizey)
	return scan, detector


def frame_geometry(detector, ARG):
	"""The frame of the crop and binning, a remainder of the binning cropped at the right and bottom."""
	crop_right = ARG.crop_right + (detector.sizex - ARG.crop_left - ARG.crop_right) % ARG.bin_x
	crop_bot = ARG.crop_bot + (detector.sizey - ARG.crop_top - ARG.crop_bot) % ARG.bin_y
	if crop_right != ARG.crop_right or crop_bot != ARG.crop_bot:
		log.info("Binning remainder cropped: crop_right=%d crop_bot=%d", crop_right, crop_bot)
	return detector.get_frame_geometry(ARG.crop_left, crop_right, ARG.crop_top, crop_bot, ARG.bin_x, ARG.bin_y)


def rotation_center(detector, ARG, parser):
	"""(offset, slope) of the image of the rotation axis in unbinned pixels from the detector center
	and unbinned pixels per unbinned row, see lautools.PM."""
	from lautools import PM
	slope = 0.0
	if ARG.rotation_center_offset is not None:
		offset = ARG.rotation_center_offset / detector.pixel_sizex
	elif ARG.rotation_center_offset_pix is not None:
		offset = ARG.rotation_center_offset_pix * ARG.bin_x
	elif ARG.rotation_center_file is not None:
		offset, fit_a, fit_b = PM.read_rotation_center_file(ARG.rotation_center_file, detector.pixel_sizex)
		has_fit = fit_a is not None and fit_b is not None
		if ARG.rotation_center_file_fitted_matrix:
			if not has_fit:
				parser.error("No linear fit of the rotation center in %s for --rotation-center-file-fitted-matrix" % ARG.rotation_center_file)
			offset, slope = fit_a, fit_b
		elif has_fit and ARG.rotation_center_file_fit_y is not None:
			y = ARG.rotation_center_file_fit_y * ARG.bin_y + 0.5 * (ARG.bin_y - 1)
			offset = fit_a + fit_b * y
			log.info("Fit evaluated at the unbinned row %f: %f + %f y = %f pixels", y, fit_a, fit_b, offset)
		if offset is None:
			parser.error("No rotation_center_offset* value found in %s" % ARG.rotation_center_file)
	else:
		offset = 0.0
	offset += ARG.detector_center_offsetvx / detector.pixel_sizex
	return offset, slope


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(), formatter_class=argparse.RawDescriptionHelpFormatter)
	sub = parser.add_subparsers(dest="format", required=True, metavar="{PB3D,PB2D}")
	common = common_arguments()
	p3 = sub.add_parser("PB3D", aliases=["pb3d"], parents=[common], help="2x4 matrices of the 3D parallel beam tools kct-pb-*.")
	p3.add_argument("--detector-center-offsetvy", type=float, default=0., help="Offset in physical units, the plane z=0 is imaged detector_center_offsetvy/pixel_sizey binned rows below the center row (M-1)/2 in the material CT convention, as createProjectionMatrices.py.")
	p2 = sub.add_parser("PB2D", aliases=["pb2d"], parents=[common], help="1x4 matrices of the 2D parallel beam tools kct-pb2d-*.")
	ARG = parser.parse_args(argv)
	pb3d = ARG.format.upper() == "PB3D"
	active = p3 if pb3d else p2
	logging.basicConfig(level=logging.INFO if ARG.verbose else logging.WARNING, format="%(message)s")
	if ARG.rotation_center_file is None and (ARG.rotation_center_file_fitted_matrix or ARG.rotation_center_file_fit_y is not None):
		active.error("--rotation-center-file-fitted-matrix and --rotation-center-file-fit-y need --rotation-center-file")
	if ARG.rotation_center_file_fitted_matrix and ARG.rotation_center_file_fit_y is not None:
		active.error("--rotation-center-file-fit-y does not apply with --rotation-center-file-fitted-matrix, the rows are given by --crop-top and --bin-y")
	try:
		stage_x_reference = None if ARG.ignore_pixel_shifts else stage_reference(ARG.stage_x_reference)
	except ValueError as e:
		active.error(str(e))

	check_output(ARG.outputProjectionMatrices, ARG.force)
	paramsFile = "%s.params" % ARG.outputProjectionMatrices
	if ARG.write_params_file:
		check_output(paramsFile, ARG.force)

	from denpy import DEN
	from lautools import PM
	scan, detector = scan_description(ARG, active)
	frame = frame_geometry(detector, ARG)
	offset, slope = rotation_center(detector, ARG, active)
	projections = scan.imageDataset(0)
	if ARG.ignore_exact_angles:
		s_rot = projections["s_rot"].to_numpy(dtype=np.float64)
		if np.any(np.diff(s_rot) < 0):
			raise ValueError("The angles s_rot are not a nondecreasing sequence, can not use --ignore-exact-angles.")
		projections = projections.assign(s_rot=np.linspace(s_rot[0], s_rot[-1], len(s_rot)))
	log.info("Detector %d x %d pixel size %g x %g, frame %d x %d pixel size %g x %g, %d projections",
			 detector.sizex, detector.sizey, detector.pixel_sizex, detector.pixel_sizey,
			 frame.sizex, frame.sizey, frame.pixel_sizex, frame.pixel_sizey, len(projections))
	log.info("Rotation center offset %f unbinned pixels in the row 0, slope %f pixels per row", offset, slope)

	if ARG.input_file is not None:
		dimspec = DEN.readHeader(ARG.input_file)["dimspec"]
		if len(dimspec) != 3:
			raise ValueError("%s has %d dimensions, not 3." % (ARG.input_file, len(dimspec)))
		if dimspec[2] != len(projections):
			raise ValueError("%s has %d projections but the scan %d." % (ARG.input_file, dimspec[2], len(projections)))
		if dimspec[0] != frame.sizex:
			raise ValueError("%s has dimx=%d but the frame has %d columns, check the crop and binning." % (ARG.input_file, dimspec[0], frame.sizex))
		if dimspec[1] != frame.sizey:
			if pb3d or slope != 0.0:
				raise ValueError("%s has dimy=%d but the frame has %d rows, check the crop and binning." % (ARG.input_file, dimspec[1], frame.sizey))
			log.warning("%s has dimy=%d but the frame has %d rows.", ARG.input_file, dimspec[1], frame.sizey)

	common = dict(rotation_center_offset_pix=offset, rotation_center_slope=slope, stage_x_reference=stage_x_reference,
				  petra_compatibility_transform=ARG.petra_compatibility_transform,
				  material_ct_convention=ARG.material_ct_convention)
	if pb3d:
		# createProjectionMatrices.py images z=0 at (M-1)/2 + offsetvy/PY, pykct.PM at (M-1)/2 - offsetvy/PY
		matrices = PM.projection_matrices2D_frame(frame, projections, detector_center_offsetvy=-ARG.detector_center_offsetvy, **common)
	else:
		matrices = PM.projection_matrices1D_frame(frame, projections, **common)
	DEN.storeNdarrayAsDEN(ARG.outputProjectionMatrices, matrices, force=ARG.force)
	if ARG.write_params_file:
		with open(paramsFile, "w") as f:
			json.dump(vars(ARG), f, indent=2, sort_keys=True)


if __name__ == "__main__":
	main()
