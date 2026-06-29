import os
import numpy as np
from algotom.rec.reconstruction import dfi_reconstruction
from skimage import filters, feature, measure
from skimage.filters import meijering, sato
from pathlib import Path
import json
import logging

# Create a logger specific to this module
log = logging.getLogger(__name__)
log.setLevel(logging.INFO) # Set the logging level to INFO
# Create a console handler and set its level to INFO
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# Add the handler to the logger
log.addHandler(ch)

class JsonCache:
	def __init__(self, path=None):
		self.entries = {}
		if path is not None:
			self.path = path
		if os.path.exists(path):
			self.load(path)

	def _make_key(
		self,
		sinogram_ind,
		cor,
		reconstruction="dfi",
		apply_log=False,
	):
		return (
			int(sinogram_ind),
			round(float(cor), 6),
			str(reconstruction),
			bool(apply_log),
		)

	def load(self, path):
		path = Path(path)
		with open(path, "r") as f:
			raw = json.load(f)
		self.entries = {}
		for entry in raw:
			key = self._make_key(
				sinogram_ind=entry["sinogram_ind"],
				cor=entry["cor"],
				reconstruction=entry["reconstruction"],
				apply_log=entry["apply_log"],
			)
			self.entries[key] = entry

	def save(self, path = None):
		if path is None:
			path = self.path
		if path is None:
			log.warning("No path specified for saving cache.")
			return
		path = Path(path)
		with open(path, "w") as f:
			json.dump(
				list(self.entries.values()),
				f,
				indent=2,
			)

	def get_entry(
		self,
		sinogram_ind,
		cor,
		reconstruction="dfi",
		apply_log=False,
		create=False,
	):
		key = self._make_key(
			sinogram_ind,
			cor,
			reconstruction,
			apply_log,
		)
		entry = self.entries.get(key)
		if entry is None and create:
			entry = {
				"sinogram_ind": int(sinogram_ind),
				"cor": float(cor),
				"reconstruction": reconstruction,
				"apply_log": apply_log,
				"scores": {},
			}
			self.entries[key] = entry
		return entry

	def get_json(self):
		return list(self.entries.values())

	def get_score(
		self,
		method,
		sinogram_ind,
		cor,
		reconstruction="dfi",
		apply_log=False,
	):
		entry = self.get_entry(
			sinogram_ind,
			cor,
			reconstruction,
			apply_log,
		)
		if entry is None:
			return None
		return entry["scores"].get(method)

	def set_score(
		self,
		method,
		value,
		sinogram_ind,
		cor,
		reconstruction="dfi",
		apply_log=False,
	):
		entry = self.get_entry(
			sinogram_ind,
			cor,
			reconstruction,
			apply_log,
			create=True,
		)
		entry["scores"][method] = value

	def get_scores(
		self,
		sinogram_ind,
		cor,
		reconstruction="dfi",
		apply_log=False,
	):
		entry = self.get_entry(
			sinogram_ind,
			cor,
			reconstruction,
			apply_log,
			create=True,
		)
		return entry["scores"]


def compute_scores(sinogram, sinogram_ind, angles, cor_points, metrics=None, apply_log=False, cache=None, verbose=True):
	all_metrics = {
		"blur_effect_3",
		"blur_effect_11",
		"blur_effect_30",
		"canny_pix_count",
		"shanon_entropy",
		"laplacian_var",
		"ridge_meijering",
		"ridge_sato",
	}
	metrics = set(metrics or all_metrics)
	results = []
	if len(sinogram.shape) == 3:
		f = sinogram[sinogram_ind]
	else:
		f = sinogram
	center = f.shape[1] / 2 - 0.5
	offsets = cor_points - center
	for cor in cor_points:
		scores_cache = cache.get_scores(sinogram_ind=sinogram_ind, cor=cor, reconstruction="dfi", apply_log=apply_log) if cache is not None else {}
		score_entry = {m: scores_cache.get(m) for m in metrics}
		missing = [m for m, v in score_entry.items() if v is None]
		if missing:
			x = dfi_reconstruction(f, cor, angles=angles, apply_log=apply_log)
			h, w = x.shape
			crop_h, crop_w = int(h * 0.2), int(w * 0.2)
			x0, y0 = (w - crop_w) // 2, (h - crop_h) // 2
			x = x[y0:y0 + crop_h, x0:x0 + crop_w]
			q01, q99 = np.quantile(x, [0.01, 0.99])
			im = np.clip((x - q01) / (q99 - q01 + 1e-8), 0, 1)
			im = (im * 255).astype(np.uint8)
			if "blur_effect_3" in missing:
				score_entry["blur_effect_3"] = measure.blur_effect(
					im, h_size=3)
			if "blur_effect_11" in missing:
				score_entry["blur_effect_11"] = measure.blur_effect(
					im, h_size=11)
			if "blur_effect_30" in missing:
				score_entry["blur_effect_30"] = measure.blur_effect(
					im, h_size=30)
			if "canny_pix_count" in missing:
				canny = feature.canny(im)
				score_entry["canny_pix_count"] = int(np.sum(canny))
			if "shanon_entropy" in missing:
				score_entry["shanon_entropy"] = measure.shannon_entropy(im)
			if "laplacian_var" in missing:
				laplacian = filters.laplace(im)
				score_entry["laplacian_var"] = np.sqrt(np.sum(laplacian**2))
			if "ridge_meijering" in missing:
				ridge = meijering(im, black_ridges=False, sigmas=(1, 2, 3))
				score_entry["ridge_meijering"] = np.sqrt(np.sum(ridge**2))
			if "ridge_sato" in missing:
				ridge = sato(im, black_ridges=False, sigmas=(1, 2, 3))
				score_entry["ridge_sato"] = np.sqrt(np.sum(ridge**2))
			scores_cache.update(score_entry)
		results.append(score_entry)
		if verbose:
			msg = f"COR={cor:10.3f} offset={cor-center:10.3f}"
			for k, v in score_entry.items():
				msg += f" {k}={v:.6f}" if isinstance(v, float) else f" {k}={v}"
			print(msg)
	return results

def estimate_cor(sinogram, sinogram_ind, angles, start_offset=0, test_radius=None, refinement_steps=3, pointCount=5, metric=None, apply_log=False, cache=None, verbose=True):
	if len(sinogram.shape) == 3:
		f = sinogram[sinogram_ind]
	else:
		f = sinogram
	center = f.shape[1] / 2 - 0.5
	log.info(f"Estimating COR for sinogram index {sinogram_ind} with size {f.shape[1]} and center at {center:.4f}, start_offset={start_offset:.4f}, test_radius={test_radius}, refinement_steps={refinement_steps}, pointCount={pointCount}, metric={metric}, apply_log={apply_log}")
	if test_radius is None:
		window = (0, f.shape[1] - 1)
	else:
		window = (center + start_offset - test_radius, center + start_offset + test_radius)
		window = (max(window[0], 0), min(window[1], f.shape[1] - 1))
	offsets = np.linspace(window[0], window[1], pointCount)
	if metric is None:
		metric = "ridge_sato"
	if cache is None:
		cache = JsonCache()
	for step in range(refinement_steps):
		eps = (offsets[1] - offsets[0]) / 10
		entries = compute_scores(f, sinogram_ind, angles, offsets, metrics=[metric], apply_log=apply_log, cache=cache, verbose=verbose)
		scores = np.array([e[metric] for e in entries])
		idx = np.argsort(scores)
		best_idx = idx[0]
		best_cor = offsets[best_idx]
		#Indices to the left and right of the best index
		left_idx = idx[idx < best_idx]
		right_idx = idx[idx > best_idx]
		rightOffset = None
		leftOffset = None
		if len(right_idx) > 0:
			right_idx = right_idx[np.argsort(scores[right_idx])]
			right = right_idx[0]
		else:
			right = best_idx
		if right != best_idx + 1:
			rightOffset = offsets[right] + eps
			if verbose:
				print(f"step {step}: adjusting right offset to {rightOffset:.4f} because right idx is {right} and best idx is {best_idx}")
		else:
			rightOffset = offsets[right]
		if len(left_idx) > 0:
			left_idx = left_idx[np.argsort(scores[left_idx])]
			left = left_idx[0]
			leftOffset = offsets[left]
		else:
			left = best_idx
		if left != best_idx - 1:
			leftOffset = offsets[left] - eps
			if verbose:
				print(f"step {step}: adjusting left offset to {leftOffset:.4f} because left idx is {left} and best idx is {best_idx}")
		else:
			leftOffset = offsets[left]
		log.info(f"step {step}: best_cor={best_cor:.4f} best_offset={best_cor - center:.4f} search window=({offsets[0] - center:.4f}, {offsets[-1] - center:.4f}) next search window=({leftOffset - center:.4f}, {rightOffset - center:.4f})")
		leftOffset = max(leftOffset, window[0])
		rightOffset = min(rightOffset, window[1])
		offsets = np.linspace(leftOffset, rightOffset, pointCount)
		if cache is not None:
			cache.save()
	return best_cor

def estimate_cor(sinogram, sinogram_ind, angles, start_offset=0, test_radius=None, refinement_steps=3, pointCount=5, metric=None, apply_log=False, cache=None, verbose=True):
	if len(sinogram.shape) == 3:
		f = sinogram[sinogram_ind]
	else:
		f = sinogram
	center = f.shape[1] / 2 - 0.5
	log.info(f"Estimating COR for sinogram index {sinogram_ind} with size {f.shape[1]} and center at {center:.4f}, start_offset={start_offset:.4f}, test_radius={test_radius}, refinement_steps={refinement_steps}, pointCount={pointCount}, metric={metric}, apply_log={apply_log}")
	if test_radius is None:
		window = (0, f.shape[1] - 1)
	else:
		window = (center + start_offset - test_radius, center + start_offset + test_radius)
		window = (max(window[0], 0), min(window[1], f.shape[1] - 1))
	offsets = np.linspace(window[0], window[1], pointCount)
	if metric is None:
		metric = "ridge_sato"
	if cache is None:
		cache = JsonCache()
	for step in range(refinement_steps):
		eps = (offsets[1] - offsets[0]) / 10
		entries = compute_scores(f, sinogram_ind, angles, offsets, metrics=[metric], apply_log=apply_log, cache=cache, verbose=verbose)
		scores = np.array([e[metric] for e in entries])
		best_cor = offsets[np.argmin(scores)]
		coef = np.polyfit(offsets, scores, 2)
		a, b, c = coef
		if a > 0:
			best_offset = -b / (2 * a)
		else:
			best_offset = best_cor
			log.warning(f"Quadratic fit is not convex (a={a:.6f}), using best offset {best_offset:.4f} without adjustment")
		# where is optimum in current interval?
		searchSize = (offsets[-1] - offsets[0])
		t = (best_offset - offsets[0]) / searchSize
		center_conf = 1.0 - 2.0 * abs(t - 0.5)
		center_conf = np.clip(center_conf, 0.0, 1.0)
		shrinkedRadius = searchSize * (0.49 - 0.25 * center_conf)
		best_offset = np.clip(best_offset, offsets[0], offsets[-1])
		leftOffset = best_offset - shrinkedRadius
		rightOffset = best_offset + shrinkedRadius
		leftOffset = max(leftOffset, window[0])
		rightOffset = min(rightOffset, window[1])
		log.info(f"step {step}: best_offset={best_offset - center:.4f} best_cor={best_offset} search window=({offsets[0] - center:.4f}, {offsets[-1] - center:.4f}) next search window=({leftOffset - center:.4f}, {rightOffset - center:.4f})")
		offsets = np.linspace(leftOffset, rightOffset, pointCount)
		if cache is not None:
			cache.save()
	return best_cor
