from .version import __version__
from .build_info import GIT_COMMIT
from .preprocess import remove_hot_pixels

def about():
    if GIT_COMMIT is None:
        return f"lautools {__version__}"
    else:
        return f"lautools {__version__} (git: {GIT_COMMIT})"
