#!/usr/bin/env python
"""
PREPROCESS.py – Core preprocessing utilities for tomography datasets
2023-2026

Author: Vojtech Kulvait
License: GNU GPL v3
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import median_filter, convolve
import warnings


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
def remove_hot_pixels(frame, iterations, filter_size, correct_threshold_abs_sigma=3.0, correct_threshold_abs=None, correct_threshold_rel_sigma=None, correct_threshold_rel=None, zinger_algorithm=False, filter_large_components=False, large_component_minpixcount=10, epsilon=1e-6):
	"""Detect and correct hot pixels in a 2D image frame using median or Zinger filtering."""
	assert any([
	correct_threshold_abs_sigma is not None,
	correct_threshold_abs is not None,
	correct_threshold_rel_sigma is not None,
	correct_threshold_rel is not None
]), "At least one threshold must be set"
	xi = frame.astype(np.float32)
	if zinger_algorithm:
		# Zinger algorithm described in https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-12-17849
		if filter_size % 2 == 0:
			warnings.warn("Zinger algorithm expects odd filter size, but got %d. Effective filter size will be %d."%(filter_size, filter_size+1))
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
		frame_filtered_sign = np.sign(frame_filtered)
		frame_filtered_sign[frame_filtered_sign == 0] = 1
		frame_dif_rel = frame_dif / (frame_filtered + epsilon * frame_filtered_sign)
		flt = np.zeros_like(xi, dtype=bool)
		if correct_threshold_abs_sigma is not None:
			frame_dif_std = np.std(frame_dif)
			flt |= np.abs(frame_dif) > correct_threshold_abs_sigma * frame_dif_std
		if correct_threshold_abs is not None:
			flt |= np.abs(frame_dif) > correct_threshold_abs
		if correct_threshold_rel_sigma is not None:
			frame_dif_rel_std = np.std(frame_dif_rel)
			flt |= np.abs(frame_dif_rel) > correct_threshold_rel_sigma * frame_dif_rel_std
		if correct_threshold_rel is not None:
			flt |= np.abs(frame_dif_rel) > correct_threshold_rel
		xi[flt] = frame_filtered[flt]
		mask_corrupted_pixel |= flt
	if filter_large_components:
		mask_corrupted_pixel = delete_large_components(mask_corrupted_pixel, large_component_minpixcount)
		xi[mask_corrupted_pixel == False] = frame[mask_corrupted_pixel == False]
	corrected_pixel_count = int(mask_corrupted_pixel.sum())
	return xi, mask_corrupted_pixel, corrected_pixel_count

