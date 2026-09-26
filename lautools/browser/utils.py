# Create a new file: lautools/browser/ui_utils.py

"""Shared UI utilities for opening files and terminals."""

import subprocess
import shlex
import logging
from pathlib import Path
from typing import Callable, List, Optional

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

if not log.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s : %(message)s",
        datefmt="%d.%m.%Y %H:%M:%S",
    )
    ch.setFormatter(formatter)
    log.addHandler(ch)

log.propagate = False

def open_files_mousepad(files: List[str], on_error: Optional[Callable[[str], None]] = None) -> bool:
    """
    Open files with mousepad editor.

    Parameters
    ----------
    files : List[str]
        List of file paths to open.
    on_error : Optional[Callable[[str], None]]
        Optional callback to handle errors (e.g., update status label).
    
    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    if files is None or len(files) == 0:
        return False
    if not subprocess.Popen(["mousepad"] + [str(f) for f in files]):
        error_msg = f"Failed to open files in mousepad: {files}"
        log.error(error_msg)
        if on_error:
            on_error(error_msg)
        return False
    return True


def open_files_vim(files: List[str], working_dir: Optional[Path] = None, 
                   on_error: Optional[Callable[[str], None]] = None) -> bool:
    """
    Open files with vim in a terminal.
    
    Parameters
    ----------
    files : List[str]
        List of file paths to open.
    working_dir : Optional[Path]
        Working directory for the terminal. If single file, uses its parent.
    on_error : Optional[Callable[[str], None]]
        Optional callback to handle errors.
    
    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    if files is None or len(files) == 0:
        return False
    directory = working_dir if working_dir else Path.cwd()
    if len(files) == 1:
        directory = Path(files[0]).parent
    try:
        cmd = f"vim {' '.join(map(shlex.quote, files))}"
        subprocess.Popen([
            "xfce4-terminal", 
            "--working-directory", str(directory), 
            "--command", cmd
        ])
        return True
    except Exception as e:
        error_msg = f"Failed to open files in vim: {e}"
        log.error(error_msg)
        if on_error:
            on_error(error_msg)
        return False


def open_terminal(directory: Path, on_error: Optional[Callable[[str], None]] = None) -> bool:
    """
    Open a terminal in the specified directory.
    
    Parameters
    ----------
    directory : Path
        Directory to open terminal in.
    on_error : Optional[Callable[[str], None]]
        Optional callback to handle errors.
    
    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    if directory is None or not directory.exists() or not directory.is_dir():
        error_msg = f"Cannot open terminal: invalid directory {directory}"
        log.warning(error_msg)
        if on_error:
            on_error(error_msg)
        return False
    
    try:
        subprocess.Popen([
            "xfce4-terminal", 
            "--working-directory", str(directory)
        ])
        return True
    except Exception as e:
        error_msg = f"Failed to open terminal in {directory}: {e}"
        log.error(error_msg)
        if on_error:
            on_error(error_msg)
        return False

def open_hdf5view(file: str, on_error: Optional[Callable[[str], None]] = None) -> bool:
    """
    Open files with mousepad editor.

    Parameters
    ----------
    file : str
	    File path to open.
    on_error : Optional[Callable[[str], None]]
        Optional callback to handle errors (e.g., update status label).
    
    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    if file is None or not Path(file).exists():
        return False
    file = str(Path(file).resolve())
    if not subprocess.Popen(["hdf5view", "-f", str(file)]):
        error_msg = f"Failed to open file in hdf5view: {file}"
        log.error(error_msg)
        if on_error:
            on_error(error_msg)
        return False
    return True
