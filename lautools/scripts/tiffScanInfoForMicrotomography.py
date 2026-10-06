#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-frame statistics for microtomography acquisitions using raw TIFF images and H5 metadata.

This module processes synchrotron tomography scan data from TIFF image sequences and associated H5 metadata files. 
It provides functionality for extracting frame statistics, performing time offset corrections, and
generating visualizations.

@author: Vojtěch Kulvait
@year: 2022-2026
@license: GPL3
"""

# Standard Library imports
import argparse
import multiprocessing
import os
import sys
import time
from bisect import bisect, bisect_left
from multiprocessing.dummy import Pool
from timeit import default_timer as timer

# Third-party imports
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from scipy.ndimage import convolve1d
from scipy.signal import correlate, find_peaks
from skimage.restoration import denoise_tv_chambolle

# Project-specific imports
from denpy import DEN, PETRA, UTILS


def build_parser():
	"""
	Build and return the argument parser for tiffImagesInfo.
	
	Returns:
		argparse.ArgumentParser: Configured parser with all arguments.
	"""
	parser = argparse.ArgumentParser(
		description="Process TIFF image sequences and extract scan statistics."
	)
	parser.add_argument(
		'h5file',
		type=str,
		help="H5 file to read data for exposure correction and current correction."
	)
	parser.add_argument(
		'outputInfoDen',
		type=str,
		help="Output DEN file for scan information."
	)
	parser.add_argument(
		"--type",
		default="IMG",
		const="IMG",
		nargs="?",
		choices=["IMG", "REF", "DAR"],
		help="Process category: IMG (actual images), REF (flat fields), DAR (dark fields). [default: IMG]"
	)
	parser.add_argument(
		"--input-den",
		default=None,
		type=str,
		help="Use this DEN file as source data for processing step evaluation."
	)
	parser.add_argument(
		"--raw-dir",
		default=None,
		type=str,
		help="Raw directory to find files; defaults to parent directory of h5file."
	)
	parser.add_argument(
		"--params-file",
		default="params",
		type=str,
		help="Parameters file path. [default: params in current directory]"
	)
	parser.add_argument(
		"--target-current-value",
		type=float,
		default=None,
		help="Target beam current (mA) for correction; None uses first measurement. [default: None]"
	)
	parser.add_argument(
		"--lower-quantile",
		type=float,
		default=0.9,
		help="Compute statistics for lower quantile fraction of data. [default: 0.9]"
	)
	parser.add_argument(
		"--read-info",
		action="store_true",
		help="Read info instead of recomputing it."
	)
	parser.add_argument(
		"--estimate-time-offset",
		action="store_true",
		help="Estimate time offset and store in params file."
	)
	parser.add_argument(
		"--estimate-insertion-offset",
		action="store_true",
		help="Estimate time offset from insertion events and store in params file."
	)
	parser.add_argument(
		"--correct-time-offset",
		action="store_true",
		help="Apply time_offset_sec from params file (cross-correlation method)."
	)
	parser.add_argument(
		"--correct-insertion-offset",
		action="store_true",
		help="Apply time_insertion_offset_sec from params file (insertion peak method)."
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Enable verbose output."
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="Force overwrite existing output files."
	)
	parser.add_argument(
		"--savefig",
		type=str,
		default=None,
		help="Save figures to this PDF file."
	)
	parser.add_argument(
		"-j",
		type=int,
		default=-1,
		help="Number of threads: 0=no threading, -1=auto-detect. [default: -1]"
	)
	return parser




#If input-den exists use it and use data read from relevant TIFF instead
def getFrame(ARG, df, inputDir, index):
	if inputDir is None:
		return DEN.getFrame(ARG.input_den, index, dtype=np.float32)
	else:
		fileStr = df["image_file"].iloc[index]
		if isinstance(fileStr, bytes):
			fileStr = fileStr.decode("utf-8")
		tiffFile = os.path.join(inputDir, fileStr.lstrip("/"))
		if not os.path.exists(tiffFile):
			raise IOError("File %s does not exist, inputDir=%s." %
						  (tiffFile, inputDir))
		return np.array(Image.open(tiffFile), dtype=np.float32)


#Creating info object is relatively time consuming procedure
#This function takes existing info object and adjusts it to the new time offset
def adjustInfoCurrent(ARG, info, timeOffsetSec):
	scanData = PETRA.scanDataset(ARG.h5file, includeCurrent=True, timeOffsetSec=timeOffsetSec)
	if ARG.type == "REF":
		image_key = 1
	elif ARG.type == "DAR":
		image_key = 2
	else:  # IMG
		image_key = 0
	df = scanData.loc[scanData["image_key"] == image_key]
	df = df.assign(frame_ind=np.arange(len(df)))
	imageCount = len(df)
	zerotime = df["time"].iloc[0]
	info_adjusted = np.zeros([10, imageCount], dtype=np.float64)
	# info[0,:] time in s from the first frame
	# info[1,:] angle in degrees
	# info[2,:] current in mA
	# info[3,:] mean of frame
	# info[4,:] median of frame
	# info[5,:] mean of frame after current correction
	# info[6,:] median of frame after current correction
	# info[7,:] mean of ARG.lower_quantile of the info indexed frame
	# info[8,:] median of ARG.lower_quantile of the frame
	# info[9,:] ARG.lower_quantile
	if ARG.target_current_value is None:
		target_current_value = np.float32(df["current"].iloc[0])
	else:
		target_current_value = ARG.target_current_value
	for i in range(imageCount):
		frameCurrent = np.float32(df["current"].iloc[i]) #Adjusted current value relative to timeOffsetSec
		currentCorrectionFactor = target_current_value / frameCurrent
		info_adjusted[0, i] = info[0, i] #Here we use same time as in original info as offset is applied to current data
		info_adjusted[1, i] = info[1, i] #Angle is not changed
		info_adjusted[2, i] = frameCurrent
		info_adjusted[3, i] = info[3, i] #Mean is not changed
		info_adjusted[4, i] = info[4, i] #Median is not changed
		info_adjusted[5, i] = info_adjusted[3, i] * currentCorrectionFactor #Mean is corrected
		info_adjusted[6, i] = info_adjusted[4, i] * currentCorrectionFactor #Median is corrected
		info_adjusted[7, i] = info[7, i] #Mean of lower quantile is not changed
		info_adjusted[8, i] = info[8, i] #Median of lower quantile is not changed
		info_adjusted[9, i] = info[9, i] #Lower quantile is not changed
	return info_adjusted


#For multithreading
def processFrame(i, ARG, zerotime, df, inputDir, dimx, dimy):
	img = getFrame(ARG, df, inputDir, i)
	if img.shape[0] != dimy or img.shape[1] != dimx:
		raise IOError(
			"Frame %d shape (%d, %d) does not agree with expected (%d, %d)"
			% (i, img.shape[0], img.shape[0], dimx, dimy))
	#Time, angle, beam current, raw mean, raw median, corrected mean, corrected median
	img = img.flatten()
	frameCurrent = np.float32(df["current"].iloc[i])
	mean = np.mean(img)
	median = np.median(img)
	inf = np.zeros([10], dtype=np.float64)
	inf[0] = np.float64(
		(df["time"].iloc[i] - zerotime).to_pytimedelta().total_seconds())
	inf[1] = np.float64(df["s_rot"].iloc[i])
	inf[2] = frameCurrent
	inf[3] = mean
	inf[4] = median
	if ARG.target_current_value is None:
		target_current_value = np.float32(df["current"].iloc[0])
	else:
		target_current_value = ARG.target_current_value
	factor = target_current_value / frameCurrent
	inf[5] = mean * factor
	inf[6] = median * factor
	k = int(ARG.lower_quantile * len(img))
	indices = np.argpartition(img, k)
	flat = img[indices[:k]]
	inf[7] = np.mean(flat)
	inf[8] = np.median(flat)
	inf[9] = ARG.lower_quantile
	return inf


#To write dataframe to den
def createInfoObject(ARG, timeOffsetSec=None):
	scanData = PETRA.scanDataset(ARG.h5file, includeCurrent=True, timeOffsetSec=timeOffsetSec)
	if ARG.type == "REF":
		image_key = 1
	elif ARG.type == "DAR":
		image_key = 2
	else:  # IMG
		image_key = 0
	df = scanData.loc[scanData["image_key"] == image_key]
	df = df.assign(frame_ind=np.arange(len(df)))
	imageCount = len(df)
	inputDir = None
	dimx = 0
	dimy = 0
	if ARG.input_den is not None:
		if not os.path.exists(ARG.input_den):
			raise IOError("File %s does not exist." % (ARG.input_den))
		else:
			inf = DEN.readHeader(ARG.input_den)
			if len(inf["dimspec"]) != 3:
				raise ValueError("File %s has %d dimensions, shall be 3" %
								 (ARG.input_den, len(inf["dimspec"])))
			if inf["dimspec"][2] != imageCount:
				raise ValueError(
					"File %s contains %d images but the df contains %d descriptions"
					% (ARG.input_den, inf["dimspec"][2], imageCount))
			dimx = inf["dimspec"][0]
			dimy = inf["dimspec"][1]
			img = getFrame(ARG, df, inputDir, 0)
	else:
		if ARG.raw_dir is not None:
			inputDir = ARG.raw_dir
		else:
			inputDir = os.path.dirname(os.path.realpath(ARG.h5file))
		img = getFrame(ARG, df, inputDir, 0)
		dimy = img.shape[0]
		dimx = img.shape[1]
	zerotime = df["time"].iloc[0]
	info = np.zeros([10, imageCount], dtype=np.float64)
	# info[0,:] time in s from the first frame
	# info[1,:] angle in degrees
	# info[2,:] current in mA
	# info[3,:] mean of frame
	# info[4,:] median of frame
	# info[5,:] mean of frame after current correction
	# info[6,:] median of frame after current correction
	# info[7,:] mean of ARG.lower_quantile of the info indexed frame
	# info[8,:] median of ARG.lower_quantile of the frame
	# info[9,:] ARG.lower_quantile
	if ARG.j == 0:
		for i in range(imageCount):
			info[:, i] = processFrame(i, ARG, zerotime, df, inputDir, dimx, dimy)
	else:
		if ARG.j < 0:
			ARG.j = multiprocessing.cpu_count()
			print("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()" %
				  (ARG.j))
		else:
			print("Starting threadpool of %d threads, optimal value multiprocessing.cpu_count()=%d" %
				  (ARG.j, multiprocessing.cpu_count()))
		tp = Pool(processes=ARG.j)
		tp_start = timer()
		results = []
		for i in range(imageCount):
			res = tp.apply_async(processFrame,
						   args=(i, ARG, zerotime, df, inputDir, dimx, dimy))
			results.append(res)
		while True:
			time.sleep(5)
			try:
				readyCount = sum([result.ready() for result in results])
				print("After %d seconds, %d/%d tasks are ready, %0.2f %% of processed frames" %
					  (timer() - tp_start, readyCount, imageCount,
					   100 * readyCount / imageCount))
				if readyCount == imageCount:
					break
			except Exception:
				continue
		# exit loop if all tasks returned success
		successful = [result.successful() for result in results]
		if all(successful):
			print("All tasks finished successfully.")
		else:
			print("Some tasks failed.")
		tp.close()
		tp.join()
		for i in range(imageCount):
			info[:, i] = results[i].get()
	return info

def calculateTimeOffset(ARG, info, pixel_offset):
	"""
	Calculate time offset based on pixel offset between frames and beam current.
	
	Parameters:
	- ARG: Argument object containing parameters.
	- info: Numpy array with frame information.
	- pixel_offset: Pixel offset to calculate time difference.
	
	Returns:
	- Time offset in seconds.
	"""
	if pixel_offset < 0:
		i1 = np.arange(-pixel_offset, info.shape[1])
		i2 = i1 + pixel_offset
	else:
		i1 = np.arange(0, info.shape[1] - pixel_offset)
		i2 = i1 + pixel_offset

	time_diffs = info[0, i2] - info[0, i1]
	time_offset = np.mean(time_diffs)

	if getattr(ARG, "verbose", False):
		print(f"Pixel Offset: {pixel_offset}, Estimated Time Offset: {time_offset:.3f} seconds")

	return time_offset


#Estimate time offset between frame time information and beam current information
def estimateTimeOffset(ARG, info, pdf=None):
	# Step 1: Normalize signals
	current_signal = info[2, :] - np.mean(info[2, :])
	mean_signal = info[3, :] - np.mean(info[3, :])

	# Step 2: Cross-correlation
	corr = correlate(mean_signal, current_signal, mode='full')
	offsets = np.arange(-len(current_signal) + 1, len(current_signal))

	# Step 3: Gaussian weighting
	sigma = 0.1*len(offsets)
	gaussian = np.exp(-0.5 * (offsets / sigma) ** 2)
	weighted_corr = corr * gaussian

	# Step 4: Find max correlation offset
	index_offset = offsets[np.argmax(weighted_corr)]

	# Step 5: Visualization
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(offsets, weighted_corr, label='Weighted Cross-Correlation')
	ax.axvline(x=index_offset, color='r', linestyle='--',
			   label=f'Max Corr Lag = {index_offset}')
	ax.set_title('Cross-Correlation between Mean and Beam Current')
	ax.set_xlabel('Frame Shift (Δi)')
	ax.set_ylabel('Correlation')
	ax.legend()
	ax.grid(True)
	fig.tight_layout()

	# Step 6: Output plot
	if pdf is not None:
		pdf.savefig(fig)
		plt.close(fig)
	else:
		plt.show()

	# Step 7: Time offset calculation
	time_offset = calculateTimeOffset(ARG, info, index_offset)

	# Step 8: Optional verbose printing
	if getattr(ARG, "verbose", False):
		print(f"index_offset_est: {index_offset}")
		print(f"time_offset_est: {time_offset:.3f}")

	# Step 9: Return structured result
	return {
		"index_offset_est": index_offset,
		"time_offset_est": time_offset
	}

def find_local_minima_by_angle(x_angle, y_signal, candidate_indices, window_degrees=1.0, quantile_threshold=0.05):
	"""
	Filters indices where y_signal is a local minimum within ±window_degrees in x_angle.
	Handles angle wrapping (0–360°).
	
	Args:
		x_angle (np.ndarray): Angles in degrees.
		y_signal (np.ndarray): Signal values.
		candidate_indices (list/array): Indices to check.
		window_degrees (float): Half-width of angular window.

	Returns:
		np.ndarray: Filtered indices.
	"""
	filtered_indices = []
	x_angle_wrapped = x_angle % 360
	
	if max(x_angle_wrapped) - min(x_angle_wrapped) < 1:
		x_angle_wrapped = x_angle_wrapped * 360 / (max(x_angle_wrapped) - min(x_angle_wrapped))
	pixel_size = np.median(np.diff(np.sort(x_angle_wrapped)))  # more robust than mean
	window_pixels = int(np.round(window_degrees / pixel_size))+1
	n = len(x_angle_wrapped)
	for idx in candidate_indices:
		start = max(0, idx - window_pixels)
		end = min(n, idx + window_pixels + 1)

		neighborhood = y_signal[start:end]
		if len(neighborhood) < 5:
			continue  # not enough data to judge

		# Calculate threshold value at the given quantile
		q_value = np.quantile(neighborhood, quantile_threshold)

		if y_signal[idx] <= q_value:
			filtered_indices.append(idx)

	return np.array(filtered_indices)

def smooth_intensity(x_angle, y_intensity, sigma_degrees):
	# Compute the pixel size in terms of angle (mean of differences between consecutive x_angle)
	if x_angle is not None:
		if np.abs(np.max(x_angle) - np.min(x_angle)) < 1:
			pixel_size = 1
		else:
			pixel_size = np.median(np.diff(x_angle))
	else:
		pixel_size = 1.0
	# Convert sigma from degrees to pixels
	sigma = sigma_degrees / pixel_size
	print("Using sigma_degrees=%0.2f sigma_pixels=%0.2f"%(sigma_degrees, sigma))
	# Create Gaussian kernel based on sigma
	# Define the range of the kernel (e.g., 3 sigma)
	kernel_size = int(6 * sigma) + 1  # 6 sigma to cover most of the distribution
	if kernel_size % 2 == 0:
		kernel_size += 1  # Make sure the kernel size is odd
	
	# Create the x values for the kernel (for angles)
	x_kernel = np.linspace(-3 * sigma, 3 * sigma, kernel_size)
	
	# Compute the Gaussian function
	gaussian_kernel = np.exp(-0.5 * (x_kernel / sigma) ** 2)
	gaussian_kernel /= np.sum(gaussian_kernel)# Normalize to ensure the sum is 1
	
	# Apply the convolution to smooth the y_intensity
	smoothed_y_intensity = convolve1d(y_intensity, gaussian_kernel, mode='reflect')
	
	return smoothed_y_intensity


def predict_insertion_events(signal, time_axis=None, dt=1, angle_data=None,
							  angle_filter_func=None, angle_threshold=1.0, amplitude_threshold=0.05):
	"""
	Predict insertion event positions based on zero-crossings of first derivative.
	
	Parameters:
	- signal (np.ndarray): 1D array of denoised signal values (e.g., intensity).
	- time_axis (np.ndarray or None): 1D array of time values; if None, assume np.arange(len(signal)).
	- dt (float): Time step size, default is 1.
	- angle_data (np.ndarray or None): Optional data (e.g., angle) for filtering using angle_filter_func.
	- angle_filter_func (callable or None): Function to filter zero-crossings, should return indices.
	- angle_threshold (float): Threshold for filtering based on angle, passed to angle_filter_func.
	- amplitude_threshold (float): Minimum peak amplitude difference, passed to angle_filter_func.

	Returns:
	- zero_crossings_idx (np.ndarray): Indices of zero-crossings (filtered if function is provided).
	- time_distances (np.ndarray): Time differences between consecutive zero-crossings.
	- pixel_distances (np.ndarray): Index differences between zero-crossings.
	- first_deriv (np.ndarray): First derivative of the signal.
	- second_deriv (np.ndarray): Second derivative of the signal.
	"""
	signal = np.asarray(signal)
	if time_axis is None:
		time_axis = np.arange(len(signal))
	# First derivative
	first_deriv = np.zeros_like(signal)
	first_deriv[1:-1] = (signal[2:] - signal[:-2]) / (2 * dt)
	first_deriv[0] = signal[1] - signal[0]
	first_deriv[-1] = signal[-1] - signal[-2]
	# Detect zero-crossings, reporting the *start* of the transition
	signs = np.sign(first_deriv)
	zero_crossings_idx = []
	
	i = 0
	while i < len(signs) - 1:
		current_sign = signs[i]
		next_sign = signs[i + 1]
		
		# Direct sign change (e.g., -1 to +1 or +1 to -1)
		if current_sign != 0 and next_sign != 0 and current_sign != next_sign:
			zero_crossings_idx.append(i)
		
		# Transition through a zero plateau (e.g., -1, 0, 0, ..., +1)
		elif current_sign != 0 and next_sign == 0:
			j = i + 1
			while j < len(signs) and signs[j] == 0:
				j += 1
			if j < len(signs) and signs[j] != current_sign:
				zero_crossings_idx.append(i)
		i += 1
	# Second derivative
	second_deriv = np.zeros_like(signal)
	second_deriv[1:-1] = (signal[2:] - 2 * signal[1:-1] + signal[:-2]) / (dt ** 2)
	second_deriv[0] = second_deriv[1]
	second_deriv[-1] = second_deriv[-2]
	# Filter zero-crossings if filtering function is provided
	if angle_filter_func is not None and angle_data is not None:
		zero_crossings_idx = angle_filter_func(
			angle_data, signal, zero_crossings_idx,
			angle_threshold, amplitude_threshold
		)
	# Time and pixel distances between consecutive events
	time_distances = np.diff(time_axis[zero_crossings_idx])
	pixel_distances = np.diff(zero_crossings_idx)
	time_distances = np.append(time_distances, np.nan)
	pixel_distances = np.append(pixel_distances, np.nan)
	return zero_crossings_idx, time_distances, pixel_distances, first_deriv, second_deriv

def predict_insertion_peaks(signal, time_axis=None, dt=1, angle_data=None,
							  angle_filter_func=None, angle_threshold=1.0, amplitude_threshold=0.05,
							  K=5):
	"""
	Predict insertion peak positions based on a heuristic approach for peak selection.
	
	Parameters:
	- signal (np.ndarray): 1D array of denoised signal values (e.g., intensity).
	- time_axis (np.ndarray or None): 1D array of time values; if None, assume np.arange(len(signal)).
	- dt (float): Time step size, default is 1.
	- angle_data (np.ndarray or None): Optional data (e.g., angle) for filtering using angle_filter_func.
	- angle_filter_func (callable or None): Function to filter peaks, should return indices.
	- angle_threshold (float): Threshold for filtering based on angle, passed to angle_filter_func.
	- amplitude_threshold (float): Minimum peak amplitude difference, passed to angle_filter_func.
	- K (int): Number of highest peaks to detect (after applying the heuristic approach).
	
	Returns:
	- insertion_peak_idx (np.ndarray): Indices of the K highest peaks in the first derivative.
	- insertion_time_distances (np.ndarray): Time differences between consecutive peaks.
	- insertion_pixel_distances (np.ndarray): Index differences between consecutive peaks.
	- signal_first_deriv (np.ndarray): First derivative of the signal.
	- signal_second_deriv (np.ndarray): Second derivative of the signal.
	"""
	signal = np.asarray(signal)
	if time_axis is None:
		time_axis = np.arange(len(signal))
	
	# First derivative (same as before)
	signal_first_deriv = np.zeros_like(signal)
	signal_first_deriv[1:-1] = (signal[2:] - signal[:-2]) / (2 * dt)
	signal_first_deriv[0] = signal[1] - signal[0]
	signal_first_deriv[0] = signal[-1] - signal[-2]
	signal_first_deriv -= smooth_intensity(angle_data, signal_first_deriv, 5)
	# Detect peaks in the first derivative
	peaks, _ = find_peaks(signal_first_deriv)  # Finds local maxima (peaks)
	if len(peaks) < K:
		raise ValueError("Not enough peaks found in the signal. Found %d peaks, but K=%d." % (len(peaks), K))
	
	# Get the amplitudes of the peaks
	peak_amplitudes = signal_first_deriv[peaks]
	
	# Select the top K peaks based on amplitude
	top_K_indices = np.argsort(peak_amplitudes)[::-1][:K]
	top_K_amplitudes = peak_amplitudes[top_K_indices]
	
	# Compute the median and 
	median_amplitude = np.median(top_K_amplitudes)

	# Select the top 2K peaks based on amplitude
	top_2K_indices = np.argsort(peak_amplitudes)[::-1][:2*K]
	top_2K_peaks = peaks[top_2K_indices]
	top_2K_amplitudes = peak_amplitudes[top_2K_indices]

	#Compute standard deviation of K peaks closest to the median
	
	# Calculate the amplitude differences from the median
	amplitude_diffs = np.abs(top_2K_amplitudes - median_amplitude)

	# Select the K closest peaks to the median (based on amplitude difference)
	closest_K_indices = np.argsort(amplitude_diffs)[:K]
	closest_K_amplitudes = top_2K_amplitudes[closest_K_indices]

	# Compute the standard deviation of the closest K peaks
	std_amplitude = np.std(closest_K_amplitudes)
	
	# Find peaks within 3 sigma of the median
	sigma_range_indices = np.where(np.abs(top_2K_amplitudes - median_amplitude) <= 3*std_amplitude)[0]
	# Print summary of peaks information
	if len(sigma_range_indices) > 0:
		amplitude_range = f"{top_2K_amplitudes[sigma_range_indices].min()} to {top_2K_amplitudes[sigma_range_indices].max()}"
	else:
		amplitude_range = "No peaks within 3σ"
	summary = (
		f"{len(sigma_range_indices)} peaks within 3σ of median (median: {median_amplitude}, "
		f"std: {std_amplitude}, range: {amplitude_range}); "
		f"{len(top_2K_amplitudes) - len(sigma_range_indices)} peaks outside this range."
	)
	print(summary)

	# If the number of peaks within 2 sigma is less than K, select the largest remaining peaks
	if len(sigma_range_indices) < K:
		# Calculate the distances from the median for all peaks
		remaining_peaks = np.setdiff1d(np.arange(2*K), sigma_range_indices)
		remaining_peaks_amplitude_diffs = np.abs(top_2K_amplitudes[remaining_peaks] - median_amplitude)

		# Select the closest remaining peaks based on their distance to the median amplitude
		remaining_peaks_sorted = remaining_peaks[np.argsort(remaining_peaks_amplitude_diffs)]
		additional_peaks_needed = K - len(sigma_range_indices)

		# Add the closest remaining peaks
		insertion_peak_idx = np.concatenate([top_2K_peaks[sigma_range_indices], top_2K_peaks[remaining_peaks_sorted[:additional_peaks_needed]]])
	else:
		# If we have enough peaks within the 2 sigma range, just use those
		insertion_peak_idx = top_2K_peaks[sigma_range_indices]

	# Second derivative (same as before)
	signal_second_deriv = np.zeros_like(signal)
	signal_second_deriv[1:-1] = (signal[2:] - 2 * signal[1:-1] + signal[:-2]) / (dt ** 2)
	signal_second_deriv[0] = signal_second_deriv[1]
	signal_second_deriv[-1] = signal_second_deriv[-2]
	
	# Sort the peak indices to ensure correct calculation of time and pixel distances
	insertion_peak_idx = np.sort(insertion_peak_idx)
	
	# Filter peaks if filtering function is provided
	if angle_filter_func is not None and angle_data is not None:
		insertion_peak_idx = angle_filter_func(
			angle_data, signal, insertion_peak_idx,
			angle_threshold, amplitude_threshold
		)
		
	print("insertion_peak_idx=%s"%(insertion_peak_idx))
	# Get the time and pixel distances
	insertion_time_distances = np.diff(time_axis[insertion_peak_idx])  # Time differences between consecutive peaks
	insertion_pixel_distances = np.diff(insertion_peak_idx)  # Index differences between consecutive peaks

	# Append NaN for consistency with the previous function's output
	insertion_time_distances = np.append(insertion_time_distances, np.nan)
	insertion_pixel_distances = np.append(insertion_pixel_distances, np.nan)
	
	# Return the selected peaks and their properties
	return insertion_peak_idx, insertion_time_distances, insertion_pixel_distances, signal_first_deriv, signal_second_deriv

def unwrap_make_monotone(info_angle, wrap_increase=360, threshold=None):
	info_angle = np.asarray(info_angle, dtype=float)
	if threshold is None:
		threshold = 0.5 * wrap_increase  # default: half wrap
	corrected = [info_angle[0]]
	unwrap_factor = 0
	for i in range(1, len(info_angle)):
		prev_raw = info_angle[i - 1]
		curr_raw = info_angle[i]
		delta = curr_raw - prev_raw
		# If the jump looks like a wrap (e.g., 350 → 10), increase unwrap factor
		if delta < -threshold:
			unwrap_factor += wrap_increase
		# Unwrap the current angle
		unwrapped = curr_raw + unwrap_factor
		# If still less than the last corrected (small decrease or noise), repeat
		if unwrapped < corrected[-1]:
			unwrapped = corrected[-1]
		corrected.append(unwrapped)
	return np.array(corrected)

def estimate_robust_shift(true_indices, predicted_indices, current_pixel_gaps, max_multiples=3, penalty_weight=0.8):
	"""
	Estimate the best integer shift to align predicted_indices to true_indices,
	searching within ± max_multiples * median_gap, and using current_pixel_gaps as distance measure.

	Parameters:
	- true_indices: array-like, indices of true events (e.g., current insertion indices)
	- predicted_indices: array-like, predicted event indices to shift (e.g., insertion_indices)
	- current_pixel_gaps: array-like, pixel gaps between true events (used as natural scale)
	- max_multiples: int, number of median gaps to set max shift search range (default 3)
	- penalty_weight: float, weight for penalizing large shifts (default 0.8)

	Returns:
	- best_shift: int, estimated best shift in samples to align predicted to true
	"""
	true_indices = np.array(true_indices)
	predicted_indices = np.array(predicted_indices)
	
	if len(true_indices) > 2:
		median_gap = np.median(current_pixel_gaps[np.isfinite(current_pixel_gaps)])  # ignore nan
	elif len(true_indices) == 0 or len(predicted_indices) == 0:
		raise ValueError("Both true_indices and predicted_indices must contain at least one element.")
	else: # len(predicted_indices) >= 1 & len(true_indices) == 1:
		median_gap = np.max(np.abs(predicted_indices - true_indices[0]))
		max_multiples = 1
	max_shift = int(max_multiples * median_gap)
	max_index = np.max([np.max(true_indices), np.max(predicted_indices)]) + max_shift #To have solid upper bound
	best_cost = np.inf
	best_shift = 0
	for shift in range(-max_shift, max_shift + 1):
		shifted_pred = predicted_indices + shift
		# Keep only valid indices inside bounds
		shifted_pred = shifted_pred[(shifted_pred >= 0) & (shifted_pred < max_index)]
		if len(shifted_pred) == 0:
			continue
		# For each true event find closest shifted predicted event and record distance in pixels
		distances = []
		for t_idx in true_indices:
			if len(shifted_pred) == 0:
				break
			closest = shifted_pred[np.argmin(np.abs(shifted_pred - t_idx))]
			distances.append(abs(closest - t_idx))
		if not distances:
			continue
		mean_dist = np.mean(distances)
		cost = mean_dist + penalty_weight * abs(shift)
		if cost < best_cost:
			best_cost = cost
			best_shift = shift
	return -best_shift

def representedInsertionEvents(info_time, info_current, insertion_events=None, tolerance_sec=1.0, verbose=True):
	"""
	Detects insertion events from sampled data, and optionally filters by correspondence with a reference list.

	Parameters:
		info_time (np.ndarray): Time array (seconds).
		info_current (np.ndarray): Beam current values.
		insertion_events (list of dict, optional): Reference events for validation.
		tolerance_sec (float): Time window to consider a match (±).
		verbose (bool): Whether to print debug messages.

	Returns:
		List of validated insertion events with indices and timing info.
	"""
	events = []
	min_indices, _ = find_peaks(-info_current, distance=5)
	max_indices, _ = find_peaks(info_current, distance=5)
	i = j = 0
	while i < len(min_indices) and j < len(max_indices):
		min_idx = min_indices[i]
		max_idx = max_indices[j]
		if max_idx > min_idx:
			segment = info_current[min_idx:max_idx + 1]
			if np.all(np.diff(segment) >= 0):  # monotonic
				dt = info_time[max_idx] - info_time[min_idx]
				dI = info_current[max_idx] - info_current[min_idx]
				if 1.0 < dt < 10.0 and dI > 0.5:
					mid_time = 0.5 * (info_time[min_idx] + info_time[max_idx])
					event = {
						"start_index": min_idx,
						"end_index": max_idx,
						"mid_index": (min_idx + max_idx) // 2,
						"start_time": info_time[min_idx],
						"end_time": info_time[max_idx],
						"mid_time": mid_time,
						"current_increase": dI,
						"duration_sec": dt
					}
					events.append(event)
			i += 1
		else:
			j += 1
	total_detected = len(events)
	if insertion_events is not None:
		reference_times = np.array([ev["mid_time"] for ev in insertion_events])
		# Forward filtering (keep only detected events with a match)
		filtered_events = []
		dropped_events = []
		for ev in events:
			if np.any(np.abs(reference_times - ev["mid_time"]) <= tolerance_sec):
				filtered_events.append(ev)
			else:
				dropped_events.append(ev)
				if verbose:
					print(f"Dropping detected event at t={ev['mid_time']:.3f}s "
						  f"(duration={ev['duration_sec']:.2f}s, dI={ev['current_increase']:.3f}) "
						  f"– no matching reference event within ±{tolerance_sec}s")
		# Backward check (reference events not matched by any detected)
		detected_mid_times = np.array([ev["mid_time"] for ev in events])
		unmatched_references = []
		for ev in insertion_events:
			if not np.any(np.abs(detected_mid_times - ev["mid_time"]) <= tolerance_sec):
				unmatched_references.append(ev)
				if verbose:
					print(f"Reference event at t={ev['mid_time']:.3f}s "
						  f"(duration={ev['duration_sec']:.2f}s, dI={ev['current_increase']:.3f}) "
						  f"– not detected in sampled data within ±{tolerance_sec}s")
		if verbose:
			print(f"\nDetected {total_detected} events.")
			print(f"Retained {len(filtered_events)} events after reference matching.")
			print(f"Dropped {len(dropped_events)} unmatched detected events.")
			print(f"{len(unmatched_references)} reference events were not matched by detection.\n")
		return filtered_events
	if verbose:
		print(f"Detected {total_detected} events (no reference filtering applied).\n")
	return events

def estimate_optimal_time_offset(insertion_events, predicted_peak_times,
								 initial_guess=0.0, max_multiples=3, penalty_weight=0.1,
								 time_resolution=0.01):
	"""
	Estimate the optimal continuous time offset to align predicted_peak_times to insertion_events mid_times.
	
	Parameters:
	- insertion_events: list of dicts, each must have 'mid_time' key
	- predicted_peak_times: array-like, predicted event times to offset
	- initial_guess: float, initial guess for the time offset (center of search)
	- max_multiples: int, multiples of median gap for max offset search range
	- penalty_weight: float, weight for penalizing large offsets
	- time_resolution: float, resolution (in seconds) of the offset search
	
	Returns:
	- optimal_time_offset: float, estimated best time offset to align predicted to true times
	"""
	true_times = np.array([event['mid_time'] for event in insertion_events])
	predicted_times = np.array(predicted_peak_times)
	time_gaps = np.diff(true_times)
	median_gap = np.median(time_gaps[time_gaps > 0])  # filter any zeros or negatives
	max_offset = max_multiples * median_gap
	# Create offsets starting from initial_guess and expanding by search_increment
	offsets = [initial_guess]  # Start with initial guess
	search_increment = time_resolution
	# Expand outwards in both directions, adding/subtracting time_resolution
	while search_increment < max_offset:
		offsets.append(initial_guess + search_increment)
		offsets.insert(0, initial_guess - search_increment)  # Insert in reverse order
		search_increment += time_resolution
	print(f"Initial guess: {initial_guess}, median_gap={median_gap:.2f}s, max_offset={max_offset:.2f}s, "
		  f"will search {len(offsets)} offsets with resolution {time_resolution}s")
	best_cost = np.inf
	optimal_time_offset = 0
	for offset in offsets:
		#true_times relative offset
		shifted_pred = predicted_times - offset
		distances = []
		for t in true_times:
			if len(shifted_pred) == 0:
				break
			closest = shifted_pred[np.argmin(np.abs(shifted_pred - t))]
			distances.append(abs(closest - t))
		if not distances:
			continue
		mean_dist = np.mean(distances)
		cost = mean_dist + penalty_weight * abs(offset - initial_guess)
		if cost < best_cost:
			best_cost = cost
			optimal_time_offset = offset
	return optimal_time_offset	# No rounding to preserve the exact result

#Estimate time offset based on peaks minima
def estimateInsertionEventsTimeOffset(ARG, info, pdf=None):
	# Step 1: Prepare dataset and extract insertion events
	scanData = PETRA.scanDataset(ARG.h5file, includeCurrent=False, timeOffsetSec=0.0)
	if ARG.type == "REF":
		image_key = 1
	elif ARG.type == "DAR":
		image_key = 2
	else:  # IMG
		image_key = 0
	df = scanData.loc[scanData["image_key"] == image_key]
	df = df.assign(frame_ind=np.arange(len(df)))
	zerotime = df["time"].iloc[0]
	endtime = df["time"].iloc[-1]
	totaltime = (endtime - zerotime).to_pytimedelta().total_seconds()
	insertion_events = PETRA.findInsertionEvents(ARG.h5file, timeOffsetSec=0.0, zeroTime=zerotime)
	insertion_events_valid = [
		event for event in insertion_events
		if 0 <= event["mid_time"] <= totaltime # Ensure mid_time is within valid range
	]  
	#Now filter insertion events represented in the dataset
	beamCurrentData = PETRA.beamCurrentDataset(ARG.h5file, timeOffsetSec=0.0)
	beamCurrentTimes = np.array([(x - zerotime).to_pytimedelta().total_seconds()
		for x in beamCurrentData["time"]])
	beamCurrentValues =  beamCurrentData["current"].values
	# Instead of using boolean indexing, find numeric slice bounds
	startind = np.searchsorted(beamCurrentTimes, 0.0, side='left') # first index >= 0.0
	endind = np.searchsorted(beamCurrentTimes, totaltime, side='right')  # first index > totaltime
	# Slice beamCurrentTimes and beamCurrentValues using numeric bounds
	beamCurrentTimes_valid = beamCurrentTimes[startind:endind]
	beamCurrentValues_valid = beamCurrentValues[startind:endind]
	#Reindex insertion events to match the valid beam current times
	insertion_events_valid = adjust_insertion_events_to_slice(insertion_events, startind, endind)
	#This is just formality as the first ehtry usually is already 0.0
	info_time = info[0, :]-info[0,0]
	info_angle = info[1,:]
	info_angle_monotone = unwrap_make_monotone(info_angle)
	info_current = info[2, :]
	insertion_events_represented = representedInsertionEvents(info_time, info_current, insertion_events_valid)
	intertion_event_count = len(insertion_events_represented)

	# Step 2: Perform smoothing of the mean signal by total variation denoising which in 1D resembles molifier
	mean_signal = info[3, :]# Mean signal intensity
	mean_signal_tv = denoise_tv_chambolle(mean_signal, weight=100)
	mean_signal_tv_smooth = smooth_intensity(info_angle_monotone, mean_signal_tv, 0.5)

	#Step 3: Plot overview of the data
	fig, ax = plt.subplots(figsize=(12, 6))
	fig.suptitle("Overview of Mean Signal, TV Denoised Signal and its smooth variant", fontsize=16)
	ax.scatter(info_time, mean_signal, label='Raw Mean Signal', color='blue', s=10, marker='o')
	ax.scatter(info_time, mean_signal_tv, label='TV Denoised Signal', color='red', s=10, marker='x')
	ax.scatter(info_time, mean_signal_tv_smooth, label='Gaussian Smoothed TV Signal', color='green', s=10, marker='^')
	ax.set_xlabel('Time (s)')
	ax.set_ylabel('Signal Intensity')
	ax.set_title('Mean Signal Overview')
	ax.legend()
	ax.grid(True)
	fig.tight_layout()
	if pdf is not None:
		pdf.savefig(fig)
		plt.close(fig)
	else:
		plt.show()
	
	# Step 5: Predict insertion events
	predicted_insertion_idx, predicted_time_diffs, predicted_pixel_diffs, first_deriv, second_deriv = predict_insertion_peaks(mean_signal_tv_smooth, time_axis=info_time, dt=1.0, angle_data=info_angle_monotone, angle_filter_func=None, angle_threshold=1.0, amplitude_threshold=0.05, K=intertion_event_count)

	# Step 6: Estimate number of insertion events and plot data
	predicted_peak_times = info_time[predicted_insertion_idx]
	optimal_time_offset = estimate_optimal_time_offset(insertion_events, predicted_peak_times, time_resolution=1.0)
	optimal_time_offset = estimate_optimal_time_offset(insertion_events, predicted_peak_times, initial_guess=optimal_time_offset, max_multiples=0.1, time_resolution=0.001, penalty_weight=0.0)

	#To plot it let's do quick current corrected dataset
	info_adjusted = adjustInfoCurrent(ARG, info, timeOffsetSec=optimal_time_offset)
	current_signal_adjusted = info_adjusted[2, :]
	if ARG.target_current_value is None:
		target_current_value = np.float32(info_adjusted[2, 0])
	else:
		target_current_value = ARG.target_current_value
	current_signal_cor = target_current_value / current_signal_adjusted
	# Now figure out the position of insertion events in shifted dataset
	scanData_shifted = PETRA.scanDataset(ARG.h5file, includeCurrent=False, timeOffsetSec=optimal_time_offset)
	df_shifted = scanData_shifted.loc[scanData_shifted["image_key"] == image_key]
	df_shifted = df_shifted.assign(frame_ind=np.arange(len(df_shifted)))
	zerotime_shifted = df_shifted["time"].iloc[0]
	endtime_shifted = df_shifted["time"].iloc[-1]
	insertion_events_shifted = PETRA.findInsertionEvents(ARG.h5file, timeOffsetSec=optimal_time_offset, zeroTime=zerotime_shifted)
	insertion_events_shifted_valid = [
		event for event in insertion_events_shifted
		if 0 <= event["mid_time"] <= totaltime# Ensure mid_time is within valid range
	]
	beamCurrentData_shifted = PETRA.beamCurrentDataset(ARG.h5file, timeOffsetSec=optimal_time_offset)
	beamCurrentTimes_shifted = np.array([(x - zerotime_shifted).to_pytimedelta().total_seconds()
		for x in beamCurrentData_shifted["time"]])
	beamCurrentValues_shifted = beamCurrentData_shifted["current"].values
	startind_shifted = np.searchsorted(beamCurrentTimes_shifted, 0.0, side='left') # first index >= 0.0
	endind_shifted = np.searchsorted(beamCurrentTimes_shifted, totaltime, side='right')  # first index > totaltime
	beamCurrentTimes_valid_shifted = beamCurrentTimes_shifted[startind_shifted:endind_shifted]
	beamCurrentValues_valid_shifted = beamCurrentValues_shifted[startind_shifted:endind_shifted]
	# Reindex insertion events to match the valid beam current times
	insertion_events_shifted_valid = adjust_insertion_events_to_slice(insertion_events_shifted, startind_shifted, endind_shifted)
	info_time_shifted = info_adjusted[0, :] - info_adjusted[0, 0]
	info_angle_shifted = info_adjusted[1, :]
	info_angle_monotone_shifted = unwrap_make_monotone(info_angle_shifted)
	info_current_shifted = info_adjusted[2, :]
	insertion_events_represented_shifted = representedInsertionEvents(info_time_shifted, info_current_shifted, insertion_events_shifted_valid)
	insertion_events_represented_shifted_midpoints = np.array([ev["mid_time"] for ev in insertion_events_represented_shifted])
	fig, ax1 = plt.subplots(figsize=(20, 6))
	fig.suptitle("Time Offset Estimation based on insertion peaks of %0.2fs"%(optimal_time_offset), fontsize=16)
	# Plot TV denoised intensity (reference) on primary y-axis with current corrected signal
	l1, = ax1.plot(info_time, mean_signal_tv, label='TV Denoised Intensity', color='darkred')
	l2, = ax1.plot(info_time, mean_signal_tv * current_signal_cor, label='Corrected TV Intensity', color='green')
	# Predicted insertion events
	for t in insertion_events_represented_shifted_midpoints:
		ax1.axvline(t, color='blue', linestyle='--', alpha=0.7, label='Insertion Events With Proper Shift')
	ax1.scatter(info_time[predicted_insertion_idx], mean_signal_tv[predicted_insertion_idx], color='blue', s=20, label='Predicted Insertion Events', marker='o')
	ax1.set_xlabel('Time (s)')
	ax1.set_ylabel('Intensity', color='darkgreen')
	ax1.tick_params(axis='y', labelcolor='darkgreen')
	# Annotate reference zero crossings
	for i, idx in enumerate(predicted_insertion_idx):
		if np.isnan(predicted_time_diffs[i]):
			continue
		txt = f"Δt={predicted_time_diffs[i]:.3f}s\nΔpix={predicted_pixel_diffs[i]}"
		ax1.annotate(txt,
				  (info_time[idx], mean_signal_tv[idx]),
				  textcoords="offset points",
				  xytext=(10, 10),
				  ha='left',
				  fontsize=8,
				  color='darkred',
				  bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
	# Create secondary y-axis for current_signal
	ax2 = ax1.twinx()
	ax2.plot(info_time, current_signal_adjusted, label=f'Current Signal (shifted by {optimal_time_offset:.3f}s)', color='purple', alpha=0.7)
	# Vertical lines for shifted insertion points in current signal
	ax2.set_ylabel('Current', color='purple')
	ax2.tick_params(axis='y', labelcolor='purple')
	# Annotate shifted zero crossings
	for i, ev in enumerate(insertion_events_represented_shifted):
		if i < len(insertion_events_represented_shifted) - 1:
			continue
		ev_time = ev["mid_time"]
		ev_current = current_signal_adjusted[ev["mid_index"]]
		curent_time_gap = ev["mid_time"] - insertion_events_represented_shifted[i - 1]["mid_time"]
		curent_pixel_gap = ev["mid_index"] - insertion_events_represented_shifted[i - 1]["mid_index"]
		txt = f"Δt={curent_time_gap:.3f}s\nΔpix={curent_pixel_gap}"
		ax2.annotate(txt,
				 (ev_time, ev_current),
				 textcoords="offset points",
				 xytext=(10, 10),
				 ha='left',
				 fontsize=8,
				 color='darkgreen',
				 bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.5))
	# Build unified legend
	handles1, labels1 = ax1.get_legend_handles_labels()
	handles2, labels2 = ax2.get_legend_handles_labels()
	by_label = dict(zip(labels1 + labels2, handles1 + handles2))  # Avoid duplicates
	# Move the legend inside the plot area to avoid overlapping with the title/plot
	fig.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
	fig.tight_layout()
	# Step 6: Output plot
	if pdf is not None:
		pdf.savefig(fig)
		plt.close(fig)
	else:
		plt.show()
	# Step 7: Legacy offsets
	offsets_legacy = estimateTimeOffset(ARG, info, pdf=pdf)
	index_offset_legacy = offsets_legacy["index_offset_est"]
	time_offset_legacy = offsets_legacy["time_offset_est"]

	# Step 8: Optional verbose printing
	if getattr(ARG, "verbose", False):
		print(f"index_offset_est: {index_offset_legacy}")
		print(f"time_offset_est: {time_offset_legacy:.3f}")
		print(f"time_insertion_offset_est: {optimal_time_offset:.3f}")
		print(f"insertion_event_count: {len(insertion_events_represented_shifted)}")

	# Step 9: Return structured result
	return {
		"index_offset_est": index_offset_legacy,
		"time_offset_est": time_offset_legacy,
		"time_insertion_offset_est": optimal_time_offset,
		"insertion_event_count": len(insertion_events_represented_shifted)
	}


def getCurrentData(ARG, zerotime, endtime, timeOffsetSec):
	# Apply negative time offset to current to leave frame timestamps unchanged
	beamCurrentData = PETRA.beamCurrentDataset(ARG.h5file, timeOffsetSec)
	startind = bisect(list(beamCurrentData["time"]), zerotime)
	endind = bisect_left(list(beamCurrentData["time"]), endtime)
	startind = max(0, startind - 1)
	endind = min(endind + 1, len(beamCurrentData))
	beamCurrentData = beamCurrentData.iloc[startind:endind]
	return beamCurrentData, startind, endind

def adjust_insertion_events_to_slice(insertion_events, startind, endind):
	"""
	Filters and adjusts insertion events to match a restricted beamCurrentData slice.

	Parameters:
		insertion_events (list): Original events from full data.
		startind (int): Start index of the restricted slice.
		endind (int): End index (exclusive) of the restricted slice.

	Returns:
		list: Filtered and adjusted insertion events.
	"""
	adjusted_events = []
	for evt in insertion_events:
		if startind <= evt["start_index"] < endind and startind <= evt["end_index"] < endind:
			new_evt = evt.copy()
			new_evt["start_index"] -= startind
			new_evt["end_index"] -= startind
			adjusted_events.append(new_evt)
	return adjusted_events

def	plotInfoOverview(info, beamCurrentData, zerotime, insertion_events=None, mainLabel=None, pdf=None):
	"""
	Plot overview of the scan information including beam current, angles, means, medians, and corrected values.
	Parameters:
		info (np.ndarray): Info object containing scan statistics.
		beamCurrentData (pd.DataFrame): DataFrame with beam current measurements.
		zerotime (pd.Timestamp): Timestamp of the first frame.
		insertion_events (list): List of insertion events with start and end indices.
		mainLabel (str): Main title for the plot.
		pdf (PdfPages): Optional PdfPages object to save the figure.
	"""
	figure, axis = plt.subplots(3, 4)
	if mainLabel is not None:
		figure.suptitle(mainLabel, fontsize=16)
	beamCurrentTimes = [(x - zerotime).to_pytimedelta().total_seconds()
						for x in beamCurrentData["time"]]
	beamCurrentValues = beamCurrentData["current"].values
	#Plot current vs. time, mean intensity vs. time
	# Scatter beam current measurements, smaller dots
	axis[0, 0].plot(beamCurrentTimes, beamCurrentValues, color="#DDCC77", linewidth=10, label="Measured Beam Current", zorder=1)
	axis[0, 0].scatter(info[0, :], info[2, :], color="#332288", s=5, label="Current Acquisition Events", zorder=2, alpha=0.7)
	# Add insertion events with distinct markers and colors
	if insertion_events is not None:
		for ev in insertion_events:
			axis[0, 0].scatter(beamCurrentTimes[ev["start_index"]], beamCurrentValues[ev["start_index"]],
				color='green', s=100, marker='D', label='Insertion Start (Min)', edgecolor='black', zorder=3)
			axis[0, 0].scatter(beamCurrentTimes[ev["end_index"]], beamCurrentValues[ev["end_index"]],
				color='red', s=100, marker='D', label='Insertion End (Max)', edgecolor='black', zorder=3)
	# Set labels and title
	axis[0, 0].set_title("Current vs time")
	axis[0, 0].set_xlabel("Time [s]")
	axis[0, 0].set_ylabel("Beam current [mA]")
	handles, labels = axis[0, 0].get_legend_handles_labels()
	by_label = dict(zip(labels, handles))
	axis[0, 0].legend(by_label.values(), by_label.keys(), loc="center right")
	
	# --- Beam Current vs Time with Mean Intensity ---
	axis[0, 1].scatter(beamCurrentTimes, beamCurrentData["current"], color="red", s=2, zorder=2)
	axis[0, 1].scatter(info[0, :], info[2, :], color="green", s=2, zorder=3)
	axis[0, 1].set_xlim([0.0, np.max(beamCurrentTimes)])  # Ensure time axis covers full range
	axis[0, 1].set_title("Beam Current Measurements vs Time")
	axis[0, 1].set_xlabel("Time [s]")
	axis[0, 1].set_ylabel("Beam Current [mA]")
	# Adding intensity on secondary y-axis
	ax2 = axis[0, 1].twinx()
	ax2.set_ylabel("Mean Intensity")
	ax2.plot(info[0, :], info[3, :], color="blue", zorder=1)
	
	axis[0, 2].plot(info[0, :], info[1, :])
	axis[0, 2].set_title("Angle vs time")
	axis[0, 3].scatter(info[6, :], info[5, :])
	axis[0, 3].set_title("Corrected mean vs corrected median")
	
	# --- Uncorrected Mean vs Time ---
	axis[1, 0].scatter(info[0, :], info[3, :], color="blue", s=2)
	axis[1, 0].set_title("Uncorrected Mean vs Time")
	axis[1, 0].set_xlabel("Time [s]")
	axis[1, 0].set_ylabel("Uncorrected Mean")
	# --- Uncorrected Median vs Time ---
	axis[1, 1].scatter(info[0, :], info[4, :], color="green", s=1)
	axis[1, 1].set_title("Uncorrected Median vs Time")
	axis[1, 1].set_xlabel("Time [s]")
	axis[1, 1].set_ylabel("Uncorrected Median")
	# --- Corrected Mean vs Time ---
	axis[1, 2].scatter(info[0, :], info[5, :], color="red", s=1)
	axis[1, 2].set_title("Corrected Mean vs Time")
	axis[1, 2].set_xlabel("Time [s]")
	axis[1, 2].set_ylabel("Corrected Mean")
	axis[1, 3].plot(info[0, :], info[6, :])
	axis[1, 3].set_title("Corrected median vs time")
	
	axis[2, 0].scatter(info[0, :], info[7, :])
	axis[2, 0].set_title("Uncorrected mean of bottom 90 % vs time")
	axis[2, 1].scatter(info[0, :], info[8, :])
	axis[2, 1].set_title("Uncorrected median of bottom 90% vs time")
	axis[2, 2].scatter(info[2, :], info[5, :])
	axis[2, 2].set_title("Corrected mean vs current")
	axis[2, 3].scatter(info[2, :], info[6, :])
	axis[2, 3].set_title("Corrected median vs current")
	if pdf is not None:
		fig = plt.gcf()
		fig.set_size_inches(3 * 11.75, 3 * 8.25)
		pdf.savefig(fig, bbox_inches='tight')
		plt.close(fig)
	else:
		plt.show()

def main(argv=None):
	parser = build_parser()
	ARG = parser.parse_args(argv)
	print("START tiffScanInfoForMicrotomography %s" % " ".join(sys.argv[1:]))
	if ARG.raw_dir is not None:
		inputDir = ARG.raw_dir
	else:
		inputDir = os.path.dirname(os.path.realpath(ARG.h5file))
	
	if ARG.params_file is not None:
		params = UTILS.readParamsFile(ARG.params_file)
	else:
		params = {}
	if params.get("time_offset_sec") is not None and ARG.correct_time_offset:
		timeOffsetSec = float(params["time_offset_sec"])
		print("Using time offset from params file: %fs" % (timeOffsetSec))
	if params.get("time_insertion_offset_sec") is not None and ARG.correct_insertion_offset:
		timeOffsetSec = float(params["time_insertion_offset_sec"])
		print("Using time offset from params file: %fs" % (timeOffsetSec))
	else:
		timeOffsetSec = 0.0
	
	if ARG.type == "REF":
		image_key = 1
	elif ARG.type == "DAR":
		image_key = 2
	else:  # IMG
		image_key = 0
	scanData = PETRA.scanDataset(ARG.h5file, includeCurrent=True, timeOffsetSec=timeOffsetSec)
	df = scanData.loc[scanData["image_key"] == image_key]
	df = df.assign(frame_ind=np.arange(len(df)))
	zerotime = df["time"].iloc[0]
	endtime = df["time"].iloc[-1]
	totaltime = (endtime - zerotime).to_pytimedelta().total_seconds()
	
	if ARG.verbose:
		print(f"Processing scan {os.path.basename(inputDir)} with {len(df)} frames, total duration {totaltime:.3f}s")
	
	if ARG.read_info and os.path.exists(ARG.outputInfoDen):
		info = DEN.getNumpyArray(ARG.outputInfoDen)
		#If info contains further frames, e.g. including corrected frames, take only the first one now
		if len(info.shape) == 3:
			if ARG.correct_time_offset or ARG.correct_insertion_offset:
				info = info[1, :, :].copy()
			else:
				info = info[0, :, :].copy()
	else:
		info = createInfoObject(ARG, timeOffsetSec)
	
	if ARG.savefig is not None:
		matplotlib.use('Agg')
		pdf = PdfPages(ARG.savefig)
	else:
		pdf = None
	#First plot the data from info frame
	beamcurrentData, startind, endind = getCurrentData(ARG, zerotime, endtime, timeOffsetSec)
	insertion_events = PETRA.findInsertionEvents(ARG.h5file, timeOffsetSec, zerotime)
	insertion_events_valid = adjust_insertion_events_to_slice(insertion_events, startind, endind)
	plotInfoOverview(info, beamcurrentData, zerotime, insertion_events_valid, mainLabel="Characteristisc of the scan for timeSecOffeset=%f" %
						(timeOffsetSec), pdf=pdf)
	
	if ARG.estimate_time_offset:
		offset_params = estimateTimeOffset(ARG, info, pdf=pdf)
		offset_correction = offset_params["time_offset_est"]
		timeOffsetSec = timeOffsetSec + offset_correction
		info_corrected = adjustInfoCurrent(ARG, info, timeOffsetSec)
		beamcurrentData, startind, endind = getCurrentData(ARG, zerotime, endtime, timeOffsetSec)
		insertion_events = PETRA.findInsertionEvents(ARG.h5file, timeOffsetSec, zerotime)
		insertion_events_valid = adjust_insertion_events_to_slice(insertion_events, startind, endind)
		plotInfoOverview(info_corrected, beamcurrentData, zerotime, insertion_events_valid,
						mainLabel="Corrected characteristisc of the scan for timeSecOffeset=%f" %
						(timeOffsetSec), pdf=pdf)
		info = np.stack([info, info_corrected], axis=0)  # Shape will be (2, 10, n)
		params["time_offset_sec"] = timeOffsetSec
		DEN.storeNdarrayAsDEN(ARG.outputInfoDen, info, force=ARG.force)
		UTILS.writeParamsFile(params, ARG.params_file)
	elif ARG.estimate_insertion_offset:
		offset_params = estimateInsertionEventsTimeOffset(ARG, info, pdf=pdf)
		offset_correction = offset_params["time_insertion_offset_est"]
		timeOffsetSec_legacy = timeOffsetSec + offset_params["time_offset_est"]
		timeOffsetSec = timeOffsetSec + offset_correction
		info_corrected = adjustInfoCurrent(ARG, info, timeOffsetSec)
		beamcurrentData, startind, endind = getCurrentData(ARG, zerotime, endtime, timeOffsetSec)
		insertion_events = PETRA.findInsertionEvents(ARG.h5file, timeOffsetSec, zerotime)
		insertion_events_valid = adjust_insertion_events_to_slice(insertion_events, startind, endind)
		plotInfoOverview(info_corrected, beamcurrentData, zerotime, insertion_events_valid,
						mainLabel="Corrected characteristisc of the scan for timeSecOffeset=%f estimated by peaks" %
						(timeOffsetSec), pdf=pdf)
		info_corrected_legacy = adjustInfoCurrent(ARG, info, timeOffsetSec_legacy)
		beamCurrentData_legacy, startind, endind = getCurrentData(ARG, zerotime, endtime, timeOffsetSec_legacy)
		insertion_events = PETRA.findInsertionEvents(ARG.h5file, timeOffsetSec, zerotime)
		insertion_events_valid = adjust_insertion_events_to_slice(insertion_events, startind, endind)
		plotInfoOverview(info_corrected_legacy, beamCurrentData_legacy, zerotime, insertion_events_valid,
						mainLabel="Corrected characteristisc of the scan for timeSecOffeset=%f legacy estimate" %
						(timeOffsetSec_legacy), pdf=pdf)
		info = np.stack([info, info_corrected], axis=0)
		params["time_insertion_offset_sec"] = timeOffsetSec
		params["time_offset_sec"] = timeOffsetSec_legacy
		params["insertion_event_count"] = offset_params["insertion_event_count"]
		DEN.storeNdarrayAsDEN(ARG.outputInfoDen, info, force=ARG.force)
		UTILS.writeParamsFile(params, ARG.params_file)
	else:
		DEN.storeNdarrayAsDEN(ARG.outputInfoDen, info, force=ARG.force)
	
	if ARG.savefig is not None:
		pdf.close()
	print("END tiffScanInfoForMicrotomography h5=%s" % (ARG.h5file))
