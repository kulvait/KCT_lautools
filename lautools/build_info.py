from pathlib import Path
import subprocess

from .version import __version__


def _git_commit():
    try:
        package_dir = Path(__file__).resolve().parent

        return subprocess.check_output(
            ["git", "-C", str(package_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    except (subprocess.CalledProcessError, FileNotFoundError):
        return None






GIT_COMMIT = _git_commit()
