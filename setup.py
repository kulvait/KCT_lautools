from setuptools import setup, find_packages

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
    "pytz",
    "imagecodecs>=2026.6.6",
    "scikit-image",
]

extras = {
    "gui": [
        "wxPython",
        "shellx",
        "termcolor",
    ],
    "gpu": [
        "redis",
        "pycuda",
        "pyopencl",
        "termcolor",
    ],
    "full": [
        "wxPython",
        "shellx",
        "termcolor",
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
            "removeHotPixels = lautools.scripts.removeHotPixels:main",
            "createTickFile = lautools.scripts.createTickFile:main",
            "GPU = lautools.scripts.GPU:main",
            "lautools-browser = lautools.browser:main",
        ]
    },
    version=__version__,
    license="GPL3",
    description="Python package for tomographic data preprocessing and analysis",
)
