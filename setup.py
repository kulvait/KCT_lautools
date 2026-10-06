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
    "denpy",
    "laupy",
    "numpy",
    "zarr",
    "scipy",
    "pandas",
    "imagecodecs>=2026.8.16",
    "scikit-image",
    "termcolor",
    "platformdirs",
    "GitPython",
]

extras = {
    "gui": [
        "pyside6",
    ],
    "gpu": [
        "redis",
        "pycuda",
        "pyopencl",
    ],
    "full": [
        "pyside6",
        "redis",
        "pycuda",
        "pyopencl",
    ],
}

setup(
    name="lautools",
    url="https://github.com/kulvait/KCT_lautools",
    author="Vojtěch Kulvait",
    author_email="vojtech.kulvait@hereon.de",
    packages=find_packages(),
    install_requires=pkg_requires,
    extras_require=extras,
    entry_points={
        "console_scripts": [
            "lautools-browser = lautools.browser.main:main",
            "binDenFile = lautools.scripts.binDenFile:main",
            "GPU = lautools.scripts.GPU:main",
            "removeHotPixels = lautools.scripts.removeHotPixels:main",
            "createTickFile = lautools.scripts.createTickFile:main",
            "createWorkingDirectoryForMicrotomography = lautools.scripts.createWorkingDirectoryForMicrotomography:main",
            "createWorkingDirectoryForNanotomography = lautools.scripts.createWorkingDirectoryForNanotomography:main",
            "tiffScanInfoForNanotomography = lautools.scripts.tiffScanInfoForNanotomography:main",
            "tiffScanInfoForMicrotomography = lautools.scripts.tiffScanInfoForMicrotomography:main",
            "tiffScanDataToZarrForNanotomography = lautools.scripts.tiffScanDataToZarrForNanotomography:main",
            "tiffScanDataToZarrForMicrotomography = lautools.scripts.tiffScanDataToZarrForMicrotomography:main",
        ]
    },
    version=__version__,
    cmdclass={"build_py": BuildPy},
    license="GPL3",
    description="Python package for tomographic data preprocessing and analysis",
)
