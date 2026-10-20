#!/usr/bin/env python
"""
Create a DEN file containing angle, shift and current coordinates for selected imaging data.

Output layout:
    2D array, float32 with zdim=number of frames in imaging sequence
    (3, N) when --include-current is specified, last column is 0

    0 ... s_rot ... tick_array[0, :] rotation angles in degrees
    1 ... pixel_shift ...tick_array[1, :] s_stage_x values for wackel scans, or zero if unavailable
    2 ... current_mA ... tick_array[2, :] ring current, or zero if unavailable , if --include-current is not specified value undefined

    DEN dimensions are xdim=N and ydim=2 or 3. This is not a 3D array.
    Each row contains one quantity across all selected imaging frames.

Frame selection:
    Only records with image_key == 0 are included.
    --from is a zero-based position within those imaging records.
    --frame-count selects that many records; omitted means all remaining.

Angle normalization:
    Original angles are preserved by default.
    --normalize-angles subtracts the minimum angle in the selected range.
    The minimum becomes zero, not necessarily the first angle.

Created: 05/2026

@author: Vojtech Kulvait
@license: GNU GPL v3
"""

import argparse
from lautools import NANOCT
from denpy import PETRA
from denpy import DEN
import numpy as np

def main():
	parser = argparse.ArgumentParser(description="Process log files and generate DEN files with tick information.")
	subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation")
	for mode, input_name, help_text in (
		("micro", "h5_file", "Process microCT HDF5 metadata"), # micro subcommand
		("nano", "log_file", "Process nanoCT log metadata"), # nano subcommand
	):
		subparser = subparsers.add_parser(mode, help=help_text)
		subparser.add_argument(input_name, type=str, help=f"Path to input {mode} log file")
		subparser.add_argument("tck_den_file", type=str, help="Output DEN file with tick information (columns: s_rot, pixel_shift, current_mA)")
		subparser.add_argument("--from", dest="from_idx", type=int, default=0, help="Start index for processing (default: 0)")
		subparser.add_argument("--frame-count", dest="frame_count", type=int, default=None, help="Number of frames to process (default: all)")
		normalization = subparser.add_mutually_exclusive_group()
		normalization.add_argument("--normalize-angles", action="store_true", help="Subtract the first imaging angle of the full sequence, before cropping (default: preserve original angles)")
		normalization.add_argument("--normalize-cropped-angles", action="store_true", help="Subtract the first imaging angle of the selected range, making the cropped sequence start at zero")
		subparser.add_argument("--include-current", action="store_true", help="Include ring current in output (default: False)")
	ARG = parser.parse_args()
	if ARG.mode == "micro":
		df = PETRA.scanDataset(ARG.h5_file, includeCurrent=ARG.include_current)
	elif ARG.mode == "nano":
		df = NANOCT.scanDataset(ARG.log_file)
	df_img = df[df["image_key"]==0]
	df_img_len = len(df_img)
	# Validate user-specified frame range
	start_idx = ARG.from_idx
	count_frames = ARG.frame_count if ARG.frame_count is not None else (df_img_len - start_idx)
	end_idx = start_idx + count_frames
	if count_frames <= 0:
		raise ValueError("Frame count must be a positive integer.")
	if start_idx < 0 or start_idx >= df_img_len or end_idx > df_img_len:
		raise ValueError(f"Frame range [{start_idx}, {end_idx}) is out of bounds for available frames (0 to {df_img_len}).")
	# create 3D array with zdim=number of frames in imaging sequence, xdim=2
	# 0 ... s_rot ... angle of rotation in degrees
	# 1 ... pixel_shift ... for wackel scans, 0 otherwise
	# 2 ... current_mA ... ring current, 0 N/A
	tick_array = np.zeros((3, count_frames), dtype=np.float32)
	for idx in range(count_frames):
		tick_array[0, idx] = df_img["s_rot"].iat[start_idx + idx] if "s_rot" in df_img.columns else 0
		tick_array[1, idx] = df_img["s_stage_x"].iat[start_idx + idx] if "s_stage_x" in df_img.columns else 0
		tick_array[2, idx] = df_img["current"].iat[start_idx + idx] if "current" in df_img.columns else 0
	# Normalize angles if requested to start from zero
	if ARG.normalize_angles and count_frames > 0:
		min_s_rot = df_img["s_rot"].iat[0] if "s_rot" in df_img.columns else 0.0
	elif ARG.normalize_cropped_angles and count_frames > 0:
		min_s_rot = df_img["s_rot"].iat[start_idx] if "s_rot" in df_img.columns else 0.0
	else:
		min_s_rot = 0.0
		tick_array[0, :] -= min_s_rot
	# create DEN file
	DEN.storeNdarrayAsDEN(ARG.tck_den_file, tick_array, force=True)

if __name__ == "__main__":
	main()
