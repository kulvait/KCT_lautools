from __future__ import annotations

import os
import shutil
from pathlib import Path


FALLBACK_BIN = Path("/data/hereon/wp/group/laupy/bin")


def _normalize_directory(value) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser()
    try:
        path = path.resolve()
    except OSError:
        path = path.absolute()
    return str(path)


def _deduplicate(paths: list[str]) -> list[str]:
    result = []
    seen = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def location_directories(entries) -> list[str]:
    paths = []
    for entry in entries:
        directory = _normalize_directory(entry.location.disk_location)
        if directory:
            paths.append(directory)
    return _deduplicate(paths)


def effective_path(
    global_entries,
    project_entries,
    inherited_path: str | None = None,
) -> str:
    inherited = inherited_path if inherited_path is not None else os.environ.get("PATH", os.defpath)
    parts = [
        *location_directories(project_entries),
        *location_directories(global_entries),
        *inherited.split(os.pathsep),
    ]
    return os.pathsep.join(_deduplicate(parts))


def command_exists(command: str, path_value: str) -> bool:
    return shutil.which(command, path=path_value) is not None
