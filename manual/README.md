# lautools documentation

- `lautools_api_manual.pdf` – the lautools Python API (scan metadata `PETRA`, `TOMOGRAPHY`,
  `NANOCT`) and the lautools command line tools (`tiffScanDataToZarrForMicrotomography`, ...).
  The LaTeX source `lautools_api_manual.tex` is maintained here; build it with
  [tectonic](https://tectonic-typesetting.github.io/): `tectonic lautools_api_manual.tex`.
- `laupy_lautools_manual.pdf` – the user manual for lautools-browser and the laupy CLI tools.
  This PDF is a copy. Its LaTeX source is maintained only in the laupy repository
  (`manual/laupy_lautools_manual.tex`, next to the laupy API reference), so the sources
  cannot diverge. After changing the manual there, rebuild it with `tectonic` and copy
  the PDF here.

The data formats and the TIFF stack conversion of denpy are described in the denpy manual
(`manual/denpy_manual.pdf` of denpy).
