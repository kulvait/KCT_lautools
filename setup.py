import subprocess
from pathlib import Path
from git import Repo, InvalidGitRepositoryError, NoSuchPathError

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()

        try:
            repo = Repo(Path(__file__).resolve().parent, search_parent_directories=True)
            commit = repo.head.commit.hexsha
        except (InvalidGitRepositoryError, NoSuchPathError):
            commit = "unknown"

        target = Path(self.build_lib) / "lautools" / "build_info.py"
        target.write_text(f'GIT_COMMIT = "{commit}"\n')

try:
    exec(open("lautools/version.py").read())
except FileNotFoundError:
    __version__ = "0.0.0"

pkg_requires = [
    "denpy[zarr]>=3.0.0",   # DEN, denpy.TIFF, ZARZS; the extra brings zarr, tifffile, imagecodecs, numcodecs
    "laupy",                # pipelines, submitslurm (and redis) of lautools-browser
    "pyside6",              # lautools-browser
    "numpy",
    "scipy",
    "pandas",
    "h5py",                 # PETRA: scan HDF5 files
    "zarr",
    "imagecodecs",          # zarr image codecs (htj2k, jpeg2k, ...); a recent version: extra codecs
    "scikit-image",
    "matplotlib",           # tiffScanInfoFor* reports
    "pillow",               # TIFF reading of the nanotomography scripts
    "platformdirs",
    "GitPython",
]

extras = {
    # lautools.COR (rotation centre with algotom, used by syscripts detectRotationCenter*)
    "cor": [
        "algotom",
    ],
    # a recent imagecodecs: htj2k, jpeg2k, jpegxl, avif, zfp (conda-forge and PyPI for Python 3.12+)
    "codecs": [
        "imagecodecs>=2026.11.11",
    ],
    # kept for existing installs (lautools[gui]): PySide6 is a requirement now
    "gui": [],
    "full": [
        "algotom",
        "imagecodecs>=2026.11.11",
    ],
}

setup(
    name="lautools",
    url="https://github.com/kulvait/KCT_lautools",
    author="Vojtěch Kulvait",
    author_email="vojtech.kulvait@hereon.de",
    packages=find_packages(),
    package_data={"lautools": ["schema.sql"]},
    install_requires=pkg_requires,
    extras_require=extras,
    entry_points={
        "console_scripts": [
            "lautools-browser = lautools.browser.main:main",
            "binDenFile = lautools.scripts.binDenFile:main",
            "selectAngleSequenceForFBP = lautools.scripts.selectAngleSequenceForFBP:main",
            "removeHotPixels = lautools.scripts.removeHotPixels:main",
            "createTickFile = lautools.scripts.createTickFile:main",
            "createWorkingDirectoryForMicrotomography = lautools.scripts.createWorkingDirectoryForMicrotomography:main",
            "createWorkingDirectoryForNanotomography = lautools.scripts.createWorkingDirectoryForNanotomography:main",
            "tiffScanInfoForNanotomography = lautools.scripts.tiffScanInfoForNanotomography:main",
            "tiffScanInfoForMicrotomography = lautools.scripts.tiffScanInfoForMicrotomography:main",
            "tiffScanDataToZarrForNanotomography = lautools.scripts.tiffScanDataToZarrForNanotomography:main",
            "tiffScanDataToZarrForMicrotomography = lautools.scripts.tiffScanDataToZarrForMicrotomography:main",
            "currentCorrect = lautools.scripts.currentCorrect:main",
        ]
    },
    version=__version__,
    cmdclass={"build_py": BuildPy},
    license="GPL3",
    description="Python package for tomographic data preprocessing and analysis",
)
