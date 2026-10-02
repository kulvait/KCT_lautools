from pathlib import Path
from git import Repo, InvalidGitRepositoryError, NoSuchPathError
from .version import __version__


def _git_commit():
    try:
        package_dir = Path(__file__).resolve().parent
        repo = Repo(package_dir, search_parent_directories=True)
        return repo.head.commit.hexsha
    except (InvalidGitRepositoryError, NoSuchPathError):
        return None

GIT_COMMIT = _git_commit()
