#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common interface of CT scans, independent of the source of their metadata.

ScanDescription is implemented by PETRA.ScanDescriptionMicrotomography (NeXus HDF5 files and zarr
stores) and NANOCT.ScanDescriptionNanotomography (LogScan.log and TIFF files). Consumers that work
on any scan use only this module: the frame table (scanDataset, imageDataset), the detector
(detectorGeometry) and the beam current (beamCurrentDataset). The values are given as recorded;
how stage positions are referenced or transformed for a geometry is left to the consumer (see
docs/CT_DESIGN.md).

Layout: API (constants, validateScanDataset), classes (DetectorGeometry2D, DetectorGeometry1D,
FrameGeometry2D, FrameGeometry1D, ScanDescription).

Pixel indices refer to pixel centers, the first pixel has index 0, so the geometric center of n
pixels is at index (n - 1) / 2. Pixel sizes are effective (camera pixel / optical magnification) in
mm, or 1.0 when the source does not know them. The detector and frame geometries use world
coordinates in the unit of the pixel size: the center of the pixel i of a detector is at
originx + i * pixel_sizex, pixel indices are obtained by dividing by the pixel size.

@author: Vojtech Kulvait
@date: 2026
"""
import abc
import dataclasses
from typing import Tuple

import numpy as np


# ================================================================================================
# API
# ================================================================================================


# Columns every scanDataset has: index the UNIX time in ms (strictly increasing), time, image_key
# (0 projection, 1 flat field, 2 dark field), image_file, s_rot (degrees), s_stage_x (mm). Other
# columns are those of the source; current (mA) with includeCurrent.
SCAN_DATASET_COLUMNS = ("time", "image_key", "image_file", "s_rot", "s_stage_x")
IMAGE_KEYS = (0, 1, 2)


def validateScanDataset(df, name="scanDataset"):
	"""Raise ValueError listing every violation of the scanDataset contract (SCAN_DATASET_COLUMNS,
	index, image_key values); returns df."""
	problems = []
	missing = [c for c in SCAN_DATASET_COLUMNS if c not in df.columns]
	if missing:
		problems.append(f"missing columns {missing}")
	index = df.index.to_numpy()
	if not np.issubdtype(index.dtype, np.integer):
		problems.append(f"index is {index.dtype}, not the UNIX time in ms as integers")
	elif len(index) > 1 and not np.all(np.diff(index.astype(np.int64)) > 0):
		problems.append("index is not strictly increasing")
	if "image_key" in df.columns:
		unknown = sorted(set(df["image_key"].tolist()) - set(IMAGE_KEYS))
		if unknown:
			problems.append(f"image_key values {unknown} other than {list(IMAGE_KEYS)}")
	if problems:
		raise ValueError(f"{name}: " + "; ".join(problems))
	return df


# ================================================================================================
# Classes
# ================================================================================================


@dataclasses.dataclass(frozen=True)
class DetectorGeometry2D:
	"""The recorded frame: sizex x sizey pixels (the camera ROI) and the effective pixel size, the
	camera pixel size divided by the optical magnification (mm, or 1.0 when not known).

	Coordinates are world coordinates in the unit of the pixel size: the center of the pixel (i, j)
	is at (originx + i * pixel_sizex, originy + j * pixel_sizey), so originx, originy are the
	coordinates of the center of the zero-th pixel. Pixel indices are (x - originx) / pixel_sizex.
	"""
	sizex: int
	sizey: int
	pixel_sizex: float = 1.0
	pixel_sizey: float = 1.0
	originx: float = 0.0
	originy: float = 0.0

	def __post_init__(self):
		if not (self.sizex > 0 and self.sizey > 0):
			raise ValueError(f"DetectorGeometry2D size {self.sizex} x {self.sizey} is not positive")
		if not (self.pixel_sizex > 0 and self.pixel_sizey > 0):
			raise ValueError(f"DetectorGeometry2D pixel size {self.pixel_sizex} x {self.pixel_sizey} is not positive")

	def set_origin(self, originx: float, originy: float):
		"""Return a new DetectorGeometry2D with the given origin."""
		return dataclasses.replace(self, originx=float(originx), originy=float(originy))

	def geometric_centerx(self):
		"""The x coordinate of the center of the detector, originx + pixel_sizex * (sizex - 1) / 2."""
		return self.originx + self.pixel_sizex * (self.sizex - 1) / 2.0

	def geometric_centery(self):
		"""The y coordinate of the center of the detector, originy + pixel_sizey * (sizey - 1) / 2."""
		return self.originy + self.pixel_sizey * (self.sizey - 1) / 2.0

	def get_frame_geometry(self, crop_left=0, crop_right=0, crop_top=0, crop_bot=0, bin_x=1, bin_y=1):
		"""Return a FrameGeometry2D with the given cropping and binning."""
		return FrameGeometry2D(self, crop_left, crop_right, crop_top, crop_bot, bin_x, bin_y)


@dataclasses.dataclass(frozen=True)
class DetectorGeometry1D:
	"""The recorded frame line: sizex pixels (the camera ROI) and the effective pixel size, the
	camera pixel size divided by the optical magnification (mm, or 1.0 when not known). The center
	of the pixel i is at originx + i * pixel_sizex, see DetectorGeometry2D."""
	sizex: int
	pixel_sizex: float = 1.0
	originx: float = 0.0

	def __post_init__(self):
		if not self.sizex > 0:
			raise ValueError(f"DetectorGeometry1D size {self.sizex} is not positive")
		if not self.pixel_sizex > 0:
			raise ValueError(f"DetectorGeometry1D pixel size {self.pixel_sizex} is not positive")

	def set_origin(self, originx: float):
		"""Return a new DetectorGeometry1D with the given origin."""
		return dataclasses.replace(self, originx=float(originx))

	def geometric_centerx(self):
		"""The x coordinate of the center of the detector, originx + pixel_sizex * (sizex - 1) / 2."""
		return self.originx + self.pixel_sizex * (self.sizex - 1) / 2.0

	def get_frame_geometry(self, crop_left=0, crop_right=0, bin_x=1):
		"""Return a FrameGeometry1D with the given cropping and binning."""
		return FrameGeometry1D(self, crop_left, crop_right, bin_x)


@dataclasses.dataclass(frozen=True)
class FrameGeometry2D:
	"""A frame made from a DetectorGeometry2D by cropping and binning.

	crop_left, crop_right, crop_top and crop_bot are in detector pixels; bin_x x bin_y detector
	pixels make one frame pixel, the cropped size has to be divisible by the binning. sizex, sizey
	are in frame pixels, pixel_sizex, pixel_sizey in the unit of the detector pixel size.

	Detector coordinates are the world coordinates of the detector (with its origin). Frame
	coordinates are world coordinates relative to the frame origin (originx, originy), the detector
	coordinates of the center of the frame pixel 0, i.e. of the center of its bin, detector pixel
	crop + (bin - 1) / 2. Frame pixel indices are frame coordinates / frame pixel size.
	"""
	detector: DetectorGeometry2D
	crop_left: int = 0
	crop_right: int = 0
	crop_top: int = 0
	crop_bot: int = 0
	bin_x: int = 1
	bin_y: int = 1

	def __post_init__(self):
		if self.bin_x <= 0 or self.bin_y <= 0:
			raise ValueError(f"Binning {self.bin_x} x {self.bin_y} is not positive")
		if min(self.crop_left, self.crop_right, self.crop_top, self.crop_bot) < 0:
			raise ValueError("Crop values cannot be negative")
		if self.crop_left + self.crop_right >= self.detector.sizex:
			raise ValueError("Horizontal crop exceeds detector dimensions")
		if self.crop_top + self.crop_bot >= self.detector.sizey:
			raise ValueError("Vertical crop exceeds detector dimensions")
		if (self.detector.sizex - self.crop_left - self.crop_right) % self.bin_x:
			raise ValueError("Horizontal frame size is not divisible by binning")
		if (self.detector.sizey - self.crop_top - self.crop_bot) % self.bin_y:
			raise ValueError("Vertical frame size is not divisible by binning")

	@property
	def sizex(self):
		return (self.detector.sizex - self.crop_left - self.crop_right) // self.bin_x

	@property
	def sizey(self):
		return (self.detector.sizey - self.crop_top - self.crop_bot) // self.bin_y

	@property
	def pixel_sizex(self):
		return self.detector.pixel_sizex * self.bin_x

	@property
	def pixel_sizey(self):
		return self.detector.pixel_sizey * self.bin_y

	@property
	def originx(self):
		"""Detector x coordinate of the center of the frame pixel 0."""
		return self.detector.originx + self.detector.pixel_sizex * (self.crop_left + 0.5 * (self.bin_x - 1))

	@property
	def originy(self):
		"""Detector y coordinate of the center of the frame pixel 0."""
		return self.detector.originy + self.detector.pixel_sizey * (self.crop_top + 0.5 * (self.bin_y - 1))

	def detector_coordinate_to_frame_coordinate(self, detector_x: float, detector_y: float) -> Tuple[float, float]:
		"""Detector coordinates to frame coordinates, relative to the frame origin."""
		return detector_x - self.originx, detector_y - self.originy

	def frame_coordinate_to_detector_coordinate(self, frame_x: float, frame_y: float) -> Tuple[float, float]:
		"""Frame coordinates to detector coordinates, the inverse of
		detector_coordinate_to_frame_coordinate."""
		return self.originx + frame_x, self.originy + frame_y

	@property
	def geometric_center(self) -> Tuple[float, float]:
		"""The geometric center of the frame relative to its own origin, pixel_size * (size - 1) / 2."""
		return self.pixel_sizex * (self.sizex - 1) / 2.0, self.pixel_sizey * (self.sizey - 1) / 2.0

	def detector_coordinate_of_geometric_center(self) -> Tuple[float, float]:
		"""The geometric center of the frame in detector coordinates."""
		center_x, center_y = self.geometric_center
		return self.frame_coordinate_to_detector_coordinate(center_x, center_y)

	def geometric_center_offset(self) -> Tuple[float, float]:
		"""The geometric center of the frame relative to the geometric center of the detector,
		detector pixel size * (crop_left - crop_right) / 2 and the same with crop_top, crop_bot."""
		center_x, center_y = self.detector_coordinate_of_geometric_center()
		return center_x - self.detector.geometric_centerx(), center_y - self.detector.geometric_centery()


@dataclasses.dataclass(frozen=True)
class FrameGeometry1D:
	"""A frame line made from a DetectorGeometry1D by cropping and binning, see FrameGeometry2D."""
	detector: DetectorGeometry1D
	crop_left: int = 0
	crop_right: int = 0
	bin_x: int = 1

	def __post_init__(self):
		if self.bin_x <= 0:
			raise ValueError(f"Binning {self.bin_x} is not positive")
		if min(self.crop_left, self.crop_right) < 0:
			raise ValueError("Crop values cannot be negative")
		if self.crop_left + self.crop_right >= self.detector.sizex:
			raise ValueError("Horizontal crop exceeds detector dimensions")
		if (self.detector.sizex - self.crop_left - self.crop_right) % self.bin_x:
			raise ValueError("Horizontal frame size is not divisible by binning")

	@property
	def sizex(self):
		return (self.detector.sizex - self.crop_left - self.crop_right) // self.bin_x

	@property
	def pixel_sizex(self):
		return self.detector.pixel_sizex * self.bin_x

	@property
	def originx(self):
		"""Detector coordinate of the center of the frame pixel 0."""
		return self.detector.originx + self.detector.pixel_sizex * (self.crop_left + 0.5 * (self.bin_x - 1))

	def detector_coordinate_to_frame_coordinate(self, detector_x: float) -> float:
		"""Detector coordinate to frame coordinate, relative to the frame origin."""
		return detector_x - self.originx

	def frame_coordinate_to_detector_coordinate(self, frame_x: float) -> float:
		"""Frame coordinate to detector coordinate, the inverse of
		detector_coordinate_to_frame_coordinate."""
		return self.originx + frame_x

	@property
	def geometric_center(self) -> float:
		"""The geometric center of the frame relative to its own origin, pixel_sizex * (sizex - 1) / 2."""
		return self.pixel_sizex * (self.sizex - 1) / 2.0

	def detector_coordinate_of_geometric_center(self) -> float:
		"""The geometric center of the frame in detector coordinates."""
		return self.frame_coordinate_to_detector_coordinate(self.geometric_center)

	def geometric_center_offset(self) -> float:
		"""The geometric center of the frame relative to the geometric center of the detector,
		detector pixel size * (crop_left - crop_right) / 2."""
		return self.detector_coordinate_of_geometric_center() - self.detector.geometric_centerx()


class ScanDescription(abc.ABC):
	"""A CT scan as recorded. Subclasses read a source and implement scanDataset,
	detectorGeometry and beamCurrentDataset; imageDataset is common."""

	@property
	def name(self):
		"""The source, for messages."""
		return "<scan description>"

	@abc.abstractmethod
	def scanDataset(self, includeCurrent=False):
		"""DataFrame of all frames in acquisition order, see SCAN_DATASET_COLUMNS; with
		includeCurrent the column current (mA) at the frame times."""

	@abc.abstractmethod
	def detectorGeometry(self):
		"""DetectorGeometry2D of the recorded frames."""

	@abc.abstractmethod
	def beamCurrentDataset(self):
		"""DataFrame of the beam current readings: index the UNIX time in ms, columns time and
		current (mA)."""

	def imageDataset(self, image_key=0, includeCurrent=False):
		"""The frames of one image_key of scanDataset, with frame_ind 0..n-1 in acquisition order."""
		df = self.scanDataset(includeCurrent)
		df = df.loc[df["image_key"] == image_key]
		return df.assign(frame_ind=np.arange(len(df)))
