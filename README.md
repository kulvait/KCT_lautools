# lautools

**lautools** is a Python package for preprocessing, filtering, and managing tomographic datasets. It provides reusable Python modules and command-line utilities for cleaning, correcting, and analyzing raw tomography data. As the Python codebase grows, **lautools** is intended to be a stable, maintainable foundation for tomographic data workflows.

It is designed to integrate seamlessly with:

- **laupy** – for orchestration, job submission, and SLURM workflows  
- **denpy** – for handling DEN and Zarr dataset formats  

By separating preprocessing and I/O from orchestration, **lautools** provides a reliable, modular codebase for researchers and engineers working on tomographic data processing.

## Motivation

The primary motivation behind **lautools** is to provide a **stable, reusable Python toolkit** for tomographic data preprocessing. Many beamline experiments produce large datasets that require cleaning, filtering, or hot pixel removal before reconstruction. Traditional approaches often rely on fragile scripts or fragmented software.

**lautools** aims to:

- Provide **clean, reusable Python modules** for preprocessing and analysis  
- Offer **command-line utilities** for common tasks, e.g., hot pixel removal (`removeHotPixels`)  
- Ensure **compatibility and stability** for integration with orchestration pipelines (`laupy`)  
- Support **DEN and Zarr formats** for flexible storage and I/O  
- Facilitate **experimentation** with new algorithms while maintaining stable core functions  


## Installation

SSH clone:

```bash
git clone git@github.com:kulvait/KCT_lautools.git
```

To install the package, execute the following command

```bash
pip install git+https://github.com/kulvait/KCT_lautools.git
```

For an editable local install from the git directory, use the following command

```bash
git clone https://github.com/kulvait/KCT_lautools.git
cd KCT_lautools
pip install --user --upgrade -e .
```


### Upgrading the Package
To update the package, use

```bash
pip install --upgrade git+https://github.com/kulvait/KCT_lautools.git
```

For a local upgrade from the git directory:

```bash
pip install --user --upgrade .
```

For a local development editable upgrade from the git directory:

```bash
pip install --user --upgrade --editable .
``` 

## Command-Line Tools

The **lautools** package installs several command-line utilities via `console_scripts`.  
These tools become available to the user automatically after installing the package.

### Provided tools

- **`removeHotPixels`**  
  A utility for detecting and correcting hot pixels in raw tomographic datasets.  
  It leverages configurable filtering strategies, including median and Zinger-style filters,  
  and supports both absolute and relative thresholding.  

  **Key features:**
  - Works with Zarr and DEN file formats, leveraging `denpy` for efficient I/O.  
  - Outputs both cleaned data and a binary mask of corrected pixels.  
  - Can be used as a standalone CLI tool or imported programmatically:
    ```python
    from lautools.preprocessing import remove_hot_pixels
    cleaned, mask, count = remove_hot_pixels(frame, iterations=2, filter_size=5)
    ```
  - Configurable via command-line arguments for thresholding, iterations, and filter size.

This approach allows users to integrate hot pixel correction into pipelines or use it interactively for preprocessing tomographic data.


## Acknowledgements

The development of this package was supported by [Hi ACTS Use Case Initiatives 2026](https://www.hi-acts.de/en/use-case-initiatives) within the project ***Advanced reconstruction pipeline for tomography experiments at PETRA III***.

## Licensing

Unless otherwise specified in the source files, this project is licensed under the GNU General Public License v3.0.

Copyright (C) 2025-2026 Vojtěch Kulvait

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
