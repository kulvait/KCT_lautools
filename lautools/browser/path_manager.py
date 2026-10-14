from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from lautools.browser.settings_store import LautoolsLocation
from lautools.db import LautoolsProjectPath, LautoolsPath


@dataclass
class ManagedPathEntry:
    location: LautoolsLocation
    is_global: bool = False
    is_project: bool = False


class PathManager:
    def __init__(self, db):
        self.db = db

    def _normalize_directory(self, directory: str | Path) -> Path:
        path = Path(directory).expanduser()
        if not path.is_absolute():
            raise ValueError(f"PATH directory must be absolute: {path}")
        try:
            return path.resolve()
        except OSError:
            return path.absolute()

    def _validate_directory(self, directory: str | Path) -> Path:
        path = self._normalize_directory(directory)
        text = str(path)
        if os.pathsep in text or "\n" in text or "\r" in text:
            raise ValueError(f"Invalid PATH directory: {text}")
        if path.exists() and not path.is_dir():
            raise ValueError(f"Not a directory: {text}")
        return path

    def _location_by_directory(self, directory: Path):
        target = str(directory)
        for location in self.db.list_locations():
            if location.disk_location is None:
                continue
            try:
                existing = self._normalize_directory(location.disk_location)
            except ValueError:
                continue
            if str(existing) == target:
                return location
        return None

    def ensure_location(self, directory: str | Path):
        path = self._validate_directory(directory)
        existing = self._location_by_directory(path)
        if existing is not None:
            return existing
        return self.db.add_location(
            LautoolsLocation(
                id=None,
                name=path.name or str(path),
                disk_location=path,
            )
        )

    def list_entries(self, project_id: int | None = None) -> list[ManagedPathEntry]:
        locations = {
            location.id: location
            for location in self.db.list_locations()
            if location.id is not None and location.disk_location is not None
        }

        global_members = {
            entry.location_id: entry
            for entry in self.db.list_path_entries()
        }

        project_members = {}
        if project_id is not None:
            project_members = {
                entry.location_id: entry
                for entry in self.db.list_project_path_entries(project_id)
            }

        result = []
        for location_id, location in sorted(
            locations.items(),
            key=lambda item: str(item[1].disk_location).casefold(),
        ):
            result.append(
                ManagedPathEntry(
                    location=location,
                    is_global=location_id in global_members,
                    is_project=location_id in project_members,
                )
            )
        return result

    def save_memberships(
        self,
        entries: list[ManagedPathEntry],
        project_id: int | None = None,
    ) -> None:
        global_entries = []
        project_entries = []

        for global_position, entry in enumerate(
            [entry for entry in entries if entry.is_global]
        ):
            if entry.location.id is None:
                raise ValueError("Global PATH entry location is unsaved.")
            global_entries.append(
                LautoolsPath(
                    location_id=entry.location.id,
                    position=global_position,
                )
            )

        for project_position, entry in enumerate(
            [entry for entry in entries if entry.is_project]
        ):
            if entry.location.id is None:
                raise ValueError("Project PATH entry location is unsaved.")
            project_entries.append(
                LautoolsProjectPath(
                    project_id=project_id,
                    location_id=entry.location.id,
                    position=project_position,
                )
            )

        with self.db.transaction():
            self.db.replace_path_entries(global_entries)
            if project_id is not None:
                self.db.replace_project_path_entries(project_id, project_entries)

    def add_directory(
        self,
        entries: list[ManagedPathEntry],
        directory: str | Path,
        project_id: int | None = None,
    ) -> list[ManagedPathEntry]:
        location = self.ensure_location(directory)
        for entry in entries:
            if entry.location.id == location.id:
                return entries

        entries = list(entries)
        entries.append(
            ManagedPathEntry(
                location=location,
                is_global=project_id is None,
                is_project=project_id is not None,
            )
        )
        return entries

    def effective_directories(self, project_id: int | None = None) -> list[str]:
        global_map = {
            entry.location_id: entry.position
            for entry in self.db.list_path_entries()
        }

        project_map = {}
        if project_id is not None:
            project_map = {
                entry.location_id: entry.position
                for entry in self.db.list_project_path_entries(project_id)
            }

        locations = {
            location.id: location
            for location in self.db.list_locations()
            if location.id is not None and location.disk_location is not None
        }

        ordered = []

        for location_id, position in sorted(
            project_map.items(),
            key=lambda item: (item[1], item[0]),
        ):
            location = locations.get(location_id)
            if location is None:
                continue
            ordered.append(str(self._normalize_directory(location.disk_location)))

        for location_id, position in sorted(
            global_map.items(),
            key=lambda item: (item[1], item[0]),
        ):
            location = locations.get(location_id)
            if location is None:
                continue
            ordered.append(str(self._normalize_directory(location.disk_location)))

        return list(dict.fromkeys(ordered))

    def effective_path(
        self,
        project_id: int | None = None,
        inherited_path: str | None = None,
    ) -> str:
        inherited = (
            inherited_path
            if inherited_path is not None
            else os.environ.get("PATH", os.defpath)
        )
        parts = [
            *self.effective_directories(project_id),
            *inherited.split(os.pathsep),
        ]
        return os.pathsep.join(dict.fromkeys(parts))

    def apply_to_session(self, session, project_id: int | None = None) -> str:
        path_value = self.effective_path(project_id)
        session.set_path(path_value)
        return path_value
