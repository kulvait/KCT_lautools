from .version import __version__

def about():
    from .build_info import GIT_COMMIT
    if GIT_COMMIT is None:
        return f"lautools {__version__}"
    else:
        return f"lautools {__version__} (git: {GIT_COMMIT})"

def remove_hot_pixels(*args, **kwargs):
    # Lazy import to avoid importing numpy/scipy during GUI startup.
    from .preprocess import remove_hot_pixels as _remove_hot_pixels
    return _remove_hot_pixels(*args, **kwargs)
