#!/usr/bin/env python
"""
PREPROCESS.py – Core preprocessing utilities for tomography datasets

Author: Vojtech Kulvait
Year: 2026
License: GNU GPL v3
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import median_filter, convolve
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


#	Parameters
#	----------
#	mask : np.ndarray (bool)
#		Binary mask where True indicates detected outliers.
#	max_size : int
#		Maximum allowed size (in pixels) of connected components to keep.
#
#	Returns
#	-------
#	np.ndarray (bool)
#		Mask with only small connected components retained.
def delete_large_components(mask: np.ndarray, max_size: int) -> np.ndarray:
	"""Remove connected components larger than `max_size` pixels."""
	if mask.dtype != bool:
		mask = mask.astype(bool)
	labels, num = ndi.label(mask)
	if num == 0:
		return mask  # already boolean
	# Compute size of each labeled component (labels start at 1)
	sizes = ndi.sum(mask, labels, index=np.arange(1, num + 1)) 
	# Identify labels to keep (small enough components)
	keep_labels = np.where(sizes <= max_size)[0] + 1 
	# Build filtered mask
	filtered_mask = np.isin(labels, keep_labels)
	return filtered_mask


#    Parameters
#    ----------
#    frame : np.ndarray
#        2D image array.
#    iterations : int
#        Number of iterations to apply the filter.
#    filter_size : int
#        Size of the median or Zinger kernel.
#    correct_threshold_abs_sigma : float
#        Number of standard deviations for absolute thresholding.
#    correct_threshold_abs : float | None
#        Absolute value threshold (overrides sigma if set).
#    correct_threshold_rel_sigma : float | None
#        Number of standard deviations for relative thresholding.
#    correct_threshold_rel : float | None
#        Relative threshold (overrides sigma if set).
#    zinger_algorithm : bool
#        Use Zinger-style directional filter instead of median.
#    filter_large_components : bool
#        Remove large connected components from mask after filtering.
#    large_component_minpixcount : int
#        Maximum pixel count of components to keep.
#    epsilon : float
#        Small constant to avoid division by zero in relative threshold.
#
#    Returns
#    -------
#    cleaned_frame : np.ndarray
#        Filtered frame with hot pixels corrected.
#    mask : np.ndarray (bool)
#        Boolean mask of corrected pixels.
#    corrected_pixel_count : int
#        Total number of corrected pixels in the frame.
def remove_hot_pixels(frame, iterations, filter_size, correct_threshold_abs_sigma=3.0, correct_threshold_abs=None, correct_threshold_rel_sigma=None, correct_threshold_rel=None, correct_threshold_local=None, correct_threshold_local_sigma=None, zinger_algorithm=False, filter_large_components=False, large_component_minpixcount=10, epsilon=1e-6):
	"""Detect and correct hot pixels in a 2D image frame using median or Zinger filtering."""
	assert any([
	correct_threshold_abs_sigma is not None,
	correct_threshold_abs is not None,
	correct_threshold_rel_sigma is not None,
	correct_threshold_rel is not None,
	correct_threshold_local is not None,
	correct_threshold_local_sigma is not None,
]), "At least one threshold must be set"
#	log.info("Starting hot pixel removal")
#	log.info("iterations=%d, filter_size=%d, zinger_algorithm=%s"%(iterations, filter_size, zinger_algorithm))
#	if filter_large_components:
#		log.info("filter_large_components=%s, large_component_minpixcount=%d"%(filter_large_components, large_component_minpixcount))
#	log.info("correct_threshold_abs_sigma=%s, correct_threshold_abs=%s, correct_threshold_rel_sigma=%s, correct_threshold_rel=%s, epsilon=%.2e"%(str(correct_threshold_abs_sigma), str(correct_threshold_abs), str(correct_threshold_rel_sigma), str(correct_threshold_rel), epsilon))
	xi = frame.astype(np.float32)
	if zinger_algorithm:
		# Zinger algorithm described in https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-12-17849
		if filter_size % 2 == 0:
			log.warning("Zinger algorithm expects odd filter size, but got %d. Effective filter size will be %d."%(filter_size, filter_size+1))
			filter_size += 1
		kernel = np.zeros((filter_size, filter_size), dtype=np.float32)
		size = filter_size // 2
		offsets = [
			(-size, -size), (-size, 0), (-size, size),
			(0, -size),					(0, size),
			(size, -size),	(size, 0), (size, size),
		]
		for di, dj in offsets:
			kernel[size + di, size + dj] = 1.0
		kernel /= kernel.sum()
	mask_corrupted_pixel = np.zeros_like(xi, dtype=bool)
	for _ in range(iterations):
		if zinger_algorithm:
			frame_filtered = convolve(xi, kernel, mode="reflect")
		else:
			frame_filtered = median_filter(xi, size=filter_size, mode="reflect")
		frame_dif = xi - frame_filtered
		frame_dif_rel = frame_dif / (np.abs(frame_filtered) + epsilon)# Increase detection sensitivity for low-intensity pixels
		#Local filtering of the difference to reduce influence of large-scale variations
		#frame_dif_med = median_filter(np.abs(frame_dif), size=filter_size, mode="reflect")
		#frame_dif_rel = frame_dif / (frame_dif_med + epsilon)
		
		flt = np.zeros_like(xi, dtype=bool)
		if correct_threshold_abs_sigma is not None:
			frame_dif_std = np.std(np.abs(frame_dif))
			#frame_dif_std = np.median(np.abs(frame_dif)) * 1.4826 # Robust estimate of std from median absolute deviation
			flt |= np.abs(frame_dif) > correct_threshold_abs_sigma * frame_dif_std
		if correct_threshold_abs is not None:
			flt |= np.abs(frame_dif) > correct_threshold_abs
		if correct_threshold_rel_sigma is not None:
			#frame_dif_rel_std = np.std(np.abs(frame_dif_rel)) ... use MAD due to unstable division by potentially small values
			frame_dif_rel_mad = np.median(np.abs(frame_dif_rel)) * 1.4826 # Robust estimate of std from median absolute deviation
			flt |= np.abs(frame_dif_rel) > correct_threshold_rel_sigma * frame_dif_rel_mad
		if correct_threshold_rel is not None:
			flt |= np.abs(frame_dif_rel) > correct_threshold_rel
		if correct_threshold_local_sigma is not None or correct_threshold_local is not None:
			frame_dif_med = median_filter(np.abs(frame_dif), size=2*filter_size, mode="reflect")
			frame_dif_local = frame_dif / (frame_dif_med + epsilon) # Local relative difference to adapt to local variations in the image
			if correct_threshold_local_sigma is not None:
				#frame_dif_local_std = np.std(np.abs(frame_dif_local)) ... use MAD due to unstable division by potentially small values
				frame_dif_local_mad = np.median(np.abs(frame_dif_local)) * 1.4826
				flt |= np.abs(frame_dif_local) > correct_threshold_local_sigma * frame_dif_local_mad
			if correct_threshold_local is not None:
				flt |= np.abs(frame_dif_local) > correct_threshold_local
		xi[flt] = frame_filtered[flt]
		mask_corrupted_pixel |= flt
	if filter_large_components:
		mask_corrupted_pixel = delete_large_components(mask_corrupted_pixel, large_component_minpixcount)
		xi[mask_corrupted_pixel == False] = frame[mask_corrupted_pixel == False]
	corrected_pixel_count = int(mask_corrupted_pixel.sum())
	return xi, mask_corrupted_pixel, corrected_pixel_count

