from .version import __version__
from .build_info import GIT_COMMIT
from .preprocess import remove_hot_pixels

def about():
    return f"lautools {__version__} (git: {GIT_COMMIT})"
