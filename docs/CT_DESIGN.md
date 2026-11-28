# lautools.CT: scan description and geometry transforms (design)

Status: 2026-11-27. Implemented: section 3 (`CT.ScanDescription`, `CT.DetectorGeometry2D/1D`,
`CT.FrameGeometry/1D`, `CT.validateScanDataset`, the PETRA and NANOCT subclasses) and
`integerShifts` of the PETRA `scanDataset`. Design only: the transforms (5, 6).

## 1. Purpose

A common interface for CT scans from different sources (PETRA microtomography: NeXus HDF5 and
zarr; P05 nanotomography: LogScan.log and TIFF files), so that consumers such as a projection
matrix builder (port of `syscripts/createProjectionMatrices.py`) work on any of them without
parsing the source themselves.

Two parts:

- `ScanDescription`: the scan as recorded. The frame table keeps the columns as derived from the
  source; nothing in it is corrected or re-referenced.
- Transforms: chains of small functions that derive new columns from the recorded ones, e.g. stage
  positions relative to a reference, without jitter, in pixels, after crop and binning. They never
  change a recorded column. The consumer finally reads `angle`, `pix_offset_vx`, `pix_offset_vy`
  and the frame size `N x M`.

## 2. Modules

```
lautools/CT.py       ScanDescription (abstract), DetectorGeometry2D/1D, FrameGeometry/1D, GeometryDataset,
                     AngleTransform, StageTransformX, StageTransformY and the built-in transforms,
                     validateScanDataset. No HDF5 or NANOCT code.
lautools/PETRA.py    ScanDescriptionMicrotomography(CT.ScanDescription): today's class (HDF5, zarr,
                     writers, findInsertionEvents, ...); PETRA.ScanDescription stays as its name.
lautools/NANOCT.py   ScanDescriptionNanotomography(CT.ScanDescription): LogScan.log + TIFF directory.
```

The module functions of PETRA and NANOCT (`PETRA.scanDataset`, `PETRA.imageDataset(includePixelShift=True)`,
`NANOCT.scanDataset`, ...) keep their current behaviour.

## 3. ScanDescription

Implemented by each source:

| method | returns |
|---|---|
| `scanDataset(includeCurrent=False, fixKnownTimestampIssues=True)` | all frames, columns as derived from the source |
| `detectorGeometry()` | `DetectorGeometry2D` |
| `beamCurrentDataset()` | beam current readings (nano: from the log rows) |

Implemented once in the base:

| method | returns |
|---|---|
| `imageDataset(image_key=0, includeCurrent=False)` | the rows of one key with `frame_ind` (0..n-1 in acquisition order) |
| `transformDataset(transforms, image_key=0)` | `GeometryDataset`: `imageDataset` with the columns the transforms add, see 6 |

Column contract of `scanDataset` (checked by `CT.validateScanDataset` in the tests of each source):

| column | unit | required |
|---|---|---|
| index | UNIX time in ms, strictly increasing (timestamps corrected, see PETRA) | yes |
| `time` | datetime64 | yes |
| `image_key` | 0 projection, 1 flat, 2 dark | yes |
| `image_file` | file name | yes |
| `s_rot` | degrees | yes |
| `s_stage_x` | mm | yes (nano: 0) |
| `s_stage_z` | mm | if logged |
| `current` | mA | with `includeCurrent` |
| others | as in the source | free |

Flat fields are taken with the sample moved out along x (e.g. 26.87 mm against 0.25 mm for the
projections in example01), so stage transforms are fitted on the projections only.

`DetectorGeometry2D` is minimal: `sizex` (N), `sizey` (M) of the recorded frame (ROI) and
`pixel_sizex`, `pixel_sizey` (effective, camera pixel / optical magnification, mm; 1.0 when the
source does not know it, nanotomography without a given pixel size). No unit is stored.
Coordinates are world coordinates in the unit of the pixel size: pixel i is centred at
`originx + i * pixel_sizex` (`originx`, `originy`, default 0, changed by `set_origin`), pixel
indices are obtained by dividing by the pixel size. `geometric_centerx()`, `geometric_centery()`
give `origin + pixel_size * (N-1)/2`.
`DetectorGeometry1D` is the same for one line. PETRA takes the values from the camera group (as
`getExperimentInfo`; `detectorGeometry(overrideMagnification=None)`), NANOCT from the first TIFF
and the `pixelSize_mm` option of its constructor.

`FrameGeometry2D(detector, crop_left, crop_right, crop_top, crop_bot, bin_x, bin_y)` (and
`FrameGeometry1D`) is a cropped and binned frame of a detector: `sizex`, `sizey` in frame pixels
(the cropped size has to be divisible by the binning), `pixel_sizex`, `pixel_sizey`. Its origin
`originx`, `originy` is the detector coordinate of the centre of frame pixel 0, the centre of its
bin (detector pixel crop + (bin-1)/2); frame coordinates are relative to it. `geometric_center`
(relative to the frame origin), `detector_coordinate_of_geometric_center()` and
`geometric_center_offset()`, the frame centre minus the detector centre, i.e.
`pixel_size * (crop_left - crop_right) / 2`, are world coordinates as well. `PM` builds the frame
from the same parameters to place the rotation axis in the cropped and binned projections.

PETRA `scanDataset(integerShifts=True, overrideMagnification=None)` puts `s_stage_x` (mm, float)
on the grid pos0 + k * pixel_sizex: pos0 is the circular mean of the projection positions modulo
the pixel size, refined to zero mean projection jitter, so the zero of `s_stage_x` is kept and the
frames differ by whole pixels. Added columns: `s_stage_x_orig` (as recorded) and
`s_stage_x_jitter` = orig - grid. Frames without a position stay NaN. A warning is logged when a
projection is more than 0.2 um off the grid (a step scan with steps that are not whole pixels).
On the examples: fly scans one grid point, jitter up to 51 nm; example10 199 grid points, 0.5 nm.

## 4. Conventions

- Detector and frame pixels: 0-based indices at pixel centres; the geometric centre of a frame
  of N pixels is at index (N-1)/2.
- `s_stage_x` increases along the detector x axis (increasing column index); `s_stage_z` along
  the detector y axis. A stage value is read as a displacement of the detector as a whole.
- Every stage column produced by a transform is the displacement of the geometric centre of the
  current frame from the reference ray, in the column's unit, positive along the detector axis.
  The reference ray is where the values are zero; a reference transform chooses it.
- A consumer finds the reference ray of projection i at frame index `(N-1)/2 - pix_offset_vx[i]`
  (and `(M-1)/2 - pix_offset_vy[i]`). This is the sign of today's `createProjectionMatrices`
  (`px0 -= pixel_shift`). Rotation centre offsets of the detectRotationCenter tools are relative to
  the detector centre and are added by the consumer.
- The reference must lie within the acquired positions, otherwise the reconstruction is centred
  where no data was taken: `GeometryDataset.check()` (and every reference transform) warns when the
  values of a stage column do not include zero.

## 5. Column names

A column name is `prefix` or `prefix_suffix`. The prefix fixes the quantity and its unit:

| prefix | quantity | unit |
|---|---|---|
| `s_rot` | rotation angle | degrees |
| `rad_rot` | rotation angle | radians |
| `s_stage_x`, `s_stage_z` | stage position or displacement | mm |
| `pix_offset_vx`, `pix_offset_vy` | displacement of the frame centre | pixels of that column's frame |

Rules (conventions, checked when a transform is applied):

- No column is ever overwritten; in particular the recorded columns (`s_rot`, `s_stage_x`, ...)
  stay as they are. A transform reads `prefix + from_suffix` and writes `prefix + to_suffix`
  (e.g. `from_suffix=""`, `to_suffix="invert"`: `s_rot` -> `s_rot_invert`); chains are formed by
  suffixes.
- A transform that keeps the unit keeps the prefix. A transform that changes the unit writes to
  the prefix of the new unit (mm -> pixels: `s_stage_x` -> `pix_offset_vx`; degrees -> radians:
  `s_rot` -> `rad_rot`), so a prefix never holds two units.
- The consumer reads the columns without suffix of the final prefixes: `rad_rot` (or `s_rot`),
  `pix_offset_vx`, `pix_offset_vy`. The last transform of a chain writes to `to_suffix=""`.

## 6. Transforms

```
AngleTransform      angle values -> angle values
StageTransformX     (N, pixel size, x values) -> (N', pixel size', x values')
StageTransformY     (M, pixel size, y values) -> (M', pixel size', y values')
```

Common interface: `from_suffix`, `to_suffix`, `describe()` (a dict for the history), and
`apply(dataset) -> dataset`. A stage transform knows the size and the pixel size of its input
column and gives those of its output column; they are stored per column in the dataset (6.3), so
`N x M` of the final `pix_offset_vx`, `pix_offset_vy` is known. Transforms that need the whole
column (reference, grid) are fitted when they are applied, on the projections of the dataset; a
transform may also be constructed with `(det, img)` to fit beforehand and inspect the result.

### 6.1 Built-in angle transforms

| transform | values' |
|---|---|
| `AngleFunction(f)` | f(a) for any float -> float function |
| `AngleNormalize()` | a - a[0] |
| `AngleLinspace()` | linspace(a[0], a[-1], n), requires non-decreasing a (`--ignore-exact-angles`) |
| `AngleToRadians()` | `s_rot` -> `rad_rot`, a * pi / 180 |
| `AnglePetraCompatibility()` | -a - pi/2 on `rad_rot` (`--petra-compatibility-transform`) |

### 6.2 Built-in stage transforms (X; Y the same on `s_stage_z` / `pix_offset_vy`)

Reference and jitter, in mm (`s_stage_x` prefix), size and pixel size unchanged:

| transform | values' |
|---|---|
| `StageFunction(f)` | f(x) for any float -> float function |
| `StageZero()` | 0 (e.g. vertical stage not used) |
| `StageReference(mode)` | x - r, r = midrange (default; the convention of denpy/lautools `imageDataset`), min, max, mean, first, or a number (warned if outside the acquired range) |
| `StageGrid(mode)` | the pixel grid of the projections, see below; x' = pos0 + k p - r with r taken on the grid |
| `StageP05Reco()` | the convention of `p05_reco.m` (matlab repository): 0 if std(x) <= 1 um, else round((x - min x)/p) minus its mean, in mm |

`StageGrid`, with the effective pixel size p of the input column: `k = round((x - x[0])/p)`,
`jitter = x - (x[0] + k p)`, `pos0 = x[0] + mean(jitter)`, `k = round((x - pos0)/p)` again,
`jitter = x - (pos0 + k p)`. The reference r is `pos0 + (min k + max k)/2 p` for midrange,
`pos0 + min k p` for min, and so on; with mean it is not on the grid and the values are not
multiples of p. The jitter is written to `s_stage_x_<to_suffix>_jitter`. Reported once: when k is
not constant (step scan) the number of positions, the k range, the maximum jitter and the
difference of the reference to the one of `p05_reco.m`; a warning when the maximum jitter exceeds
0.2 um. On the example files: fly scans give k constant (all values 0, jitter up to 51 nm);
example10_step_scan gives k in -158..40, jitter <= 1 nm, and midrange differs from the p05_reco.m
mean by 0.823 px.

To pixels and frame operations (`pix_offset_vx` prefix):

| transform | N' | pixel size' | values' |
|---|---|---|---|
| `StageToPixels()` | N | p | `s_stage_x...` -> `pix_offset_vx...`, x / p |
| `StageCrop(left, right)` | N - left - right | p | x + (left - right)/2 |
| `StageBin(b)` (remainder dropped at the right) | N // b | p b | (x - (N mod b)/2) / b |

Worked examples: N = 5, `StageBin(2)` on x = 0 gives x' = -0.25, N' = 2: the reference ray is at
0.5 - (-0.25) = 0.75, the detector centre in binned indices (today's `binning_center_adjustment`
of 0.5 unbinned pixels). N = 5120, `StageCrop(left=2)` on x = 0 gives N' = 5118, x' = 1: the ray is
at 2558.5 - 1 = 2557.5, the old centre 2559.5 minus the two cropped columns. Crop and binning in
any order and number; each applies its rule to the frame it receives.

### 6.3 GeometryDataset

A DataFrame (the projections of `imageDataset` with the added columns) with `attrs`:

- `columns`: per column added by a transform: unit, size (N or M), pixel size and unit for stage
  columns, the source column and `describe()` of the transform;
- `history`: the transforms in the order applied.

`check()` warns about stage columns whose values do not include zero; `frameSize()` returns
`(N, M)` of `pix_offset_vx`, `pix_offset_vy`.

### 6.4 Example

```python
from lautools import PETRA, CT
scan = PETRA.ScanDescriptionFactory.create("scan.h5")
ds = scan.transformDataset([
	CT.AngleToRadians(to_suffix=""),                                   # s_rot -> rad_rot
	CT.StageGrid("midrange", to_suffix="ref"),                         # s_stage_x -> s_stage_x_ref
	CT.StageToPixels(from_suffix="ref", to_suffix="det"),              # -> pix_offset_vx_det
	CT.StageCrop(left=2, from_suffix="det", to_suffix="crop"),         # -> pix_offset_vx_crop
	CT.StageBin(2, from_suffix="crop", to_suffix=""),                  # -> pix_offset_vx
	CT.StageZero(axis="y", to_suffix="ref"),                           # s_stage_z -> s_stage_z_ref = 0
	CT.StageToPixels(axis="y", from_suffix="ref", to_suffix="det"),
	CT.StageBin(2, axis="y", from_suffix="det", to_suffix=""),         # -> pix_offset_vy
])
N, M = ds.frameSize()
```

Comparing with the MATLAB reconstruction is the same chain with `CT.StageP05Reco()` in place of
`CT.StageGrid(...)`.

## 7. Tests

1. Transforms alone, synthetic: each built-in on hand-made columns; crop and binning chains
   against the formulas (odd and even N, asymmetric crops, crop before and after binning); the
   naming rules (no overwrite, unit change needs a new prefix); `check()` warnings.
2. `StageGrid` and `StageP05Reco`: synthetic jitter and step patterns; then example01-09 (all
   shifts 0) and example10_step_scan (integers, 199 positions, the 0.823 px difference).
3. PETRA: `ScanDescriptionMicrotomography` passes `validateScanDataset` on example01-10;
   `detectorGeometry()` agrees with `getExperimentInfo` (pix_size, ROI); all existing PETRA tests
   unchanged.
4. Cross-check with `createProjectionMatrices.py`: for a few bin/crop cases and example05,
   example10, `(N-1)/2 - pix_offset_vx` and N equal its `px0` (without rotation centre offset) and
   detector size.
5. NANOCT: `ScanDescriptionNanotomography` on a synthetic log and TIFF directory; pixel size unit
   "px" without `pixelSize_mm`.

## 8. Implementation order

1. `CT.py`: `ScanDescription` base, `DetectorGeometry2D/1D`, `FrameGeometry/1D`, `GeometryDataset`, the transforms; tests 1-2
   on synthetic data.
2. PETRA: `ScanDescriptionMicrotomography`, `detectorGeometry()`; tests 2-4.
3. NANOCT: `ScanDescriptionNanotomography`; test 5.
4. API manual: a CT section.

## 9. Open questions

1. Crop and binning only on `pix_offset_v*` columns (as above), or also on mm columns?
2. Names of the radians prefix (`rad_rot`) and of the jitter column (`..._jitter`).
3. `StageGrid` with mode mean: exact mean (values not on the grid, as p05_reco.m) or rounded to
   the grid?
4. Should `transformDataset` also accept flats (image_key 1) for consumers that need them, with
   the reference fitted on the projections?
