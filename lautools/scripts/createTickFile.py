#!/usr/bin/env python
"""
This script creates simple den file with array providing tick coordinates of imaging data.
3D array with zdim=number of frames in imaging sequence
xdim=2
0 ... s_rot ... angle of rotation in degrees
1 ... pixel_shift ... for wackel scans, 0 otherwise
2 ... current_mA ... ring current, 0 N/A

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
	# micro subcommand
	parser_micro = subparsers.add_parser("micro", help="Process microCT log files")
	parser_micro.add_argument("h5_file", type=str, help="Path to input HDF5 log file")
	parser_micro.add_argument("den_file", type=str, help="Output DEN file path")
	# nano subcommand
	parser_nano = subparsers.add_parser("nano", help="Process nanoCT log files")
	parser_nano.add_argument("log_file", type=str, help="Path to input log file, e.g. LogScan.log")
	parser_nano.add_argument("den_file", type=str, help="Output DEN file path")
	args = parser.parse_args()
	if args.mode == "micro":
		df = PETRA.scanDataset(args.h5_file, includeCurrent=True)
	elif args.mode == "nano":
		df = NANOCT.scanDataset(args.log_file)
	df_img = df[df["image_key"]==0]
	# create 3D array with zdim=number of frames in imaging sequence, xdim=2
	# 0 ... s_rot ... angle of rotation in degrees
	# 1 ... pixel_shift ... for wackel scans, 0 otherwise
	# 2 ... current_mA ... ring current, 0 N/A
	tick_array = np.zeros((3, len(df_img)), dtype=np.float32)
	min_s_rot = df_img["s_rot"].min() if "s_rot" in df_img.columns else 0.0
	for idx in range(len(df_img)):
		tick_array[0, idx] = df_img["s_rot"].iat[idx] - min_s_rot if "s_rot" in df_img.columns else 0
		tick_array[1, idx] = df_img["s_stage_x"].iat[idx] if "s_stage_x" in df_img.columns else 0
		tick_array[2, idx] = df_img["current"].iat[idx] if "current" in df_img.columns else 0
	# create DEN file
	DEN.storeNdarrayAsDEN(args.den_file, tick_array, force=True)

