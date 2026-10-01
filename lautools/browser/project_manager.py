from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Callable

from lautools.beamtime_scanner import (BeamtimeCandidate, inspect_candidate, parse_metadata_text)
from lautools.db import (
    Beamtime,
    BeamtimeStorage,
    LaupyDB,
    LaupyProject,
    LaupyProjectWorkspace,
)

ProgressCallback = Callable[[str], None]


def _now() -> datetime:
    return datetime.now().replace(microsecond=0)

def default_beamtime_label(root: Path, beamline: str | None) -> str:
    """beamline_year_beamtimeID from the metadata and the GPFS path."""
    parts = root.parts
    beamline = beamline or (parts[-4] if len(parts) >= 4 else "")
    year = parts[-3] if len(parts) >= 3 else ""
    return "_".join(part for part in (beamline, year, root.name) if part)

@dataclass
class BeamtimeDetail:
    beamtime: Beamtime
    storage: BeamtimeStorage | None


@dataclass
class ProjectInfo:
    project: LaupyProject
    workspaces: list[LaupyProjectWorkspace] = field(default_factory=list)
    beamtimes: list[BeamtimeDetail] = field(default_factory=list)

    @property
    def id(self) -> int | None:
        return self.project.id

    @property
    def name(self) -> str:
        return self.project.name

    @property
    def path(self) -> Path:
        return self.project.path

    @property
    def description(self) -> str | None:
        return self.project.description

    @property
    def project_size_bytes(self) -> int | None:
        return self.project.project_size_bytes

    @property
    def last_inspected(self) -> str | None:
        value = self.project.last_inspected
        return value.isoformat(timespec="seconds") if value else None


class BeamtimeManager:
    """Filesystem operations only; persistence belongs to ProjectManager."""

    def find_beamtime_root(self, path: Path) -> Path | None:
        parts = path.resolve().parts
        for i, part in enumerate(parts):
            if (
                part == "gpfs"
                and i + 4 < len(parts)
                and parts[i + 3] == "data"
                and parts[i + 4].isdigit()
            ):
                return Path(*parts[: i + 5])
        return None

    def load_metadata(self, root: Path) -> tuple[dict[str, Any] | None, str | None]:
        """Accept both the usual filename and archive-stub metadata.json."""
        for name in (f"beamtime-metadata-{root.name}.json", "metadata.json", f"beamtime-metadata-{root.name}.txt"):
            metadata_file = root / name
            try:
                text = metadata_file.read_text(encoding="utf-8")
            except FileNotFoundError:
                continue
            try:
                data, json_text = parse_metadata_text(text)
            except ValueError as exc:
                raise ValueError(f"{exc}: {metadata_file}") from exc
            metadata_id = data.get("beamtimeId")
            if metadata_id is not None and str(metadata_id) != root.name:
                raise ValueError(
                    f"Beamtime ID {metadata_id!r} does not match "
                    f"directory {root.name!r}: {metadata_file}"
                )
            # Store only the JSON object so metadata_json in the database is always valid JSON.
            return data, json_text
        return None, None

    def inspect_storage(
        self,
        root: Path,
        previous: BeamtimeStorage,
        candidate: BeamtimeCandidate | None = None,
    ) -> BeamtimeStorage:
        candidate = candidate or inspect_candidate(root)
        if candidate is None or not candidate.accepted:
            raise ValueError(f"Not a recognizable beamtime directory: {root}")

        now = _now()
        if candidate.archived:
            previous.on_tape = True

        if candidate.is_stub:
            # Only metadata + README remain: a reference, not data on GPFS.
            # Keep sizes and raw sub-directory samples as last measured.
            previous.on_gpfs = False
            if candidate.readme_mtime is not None:
                previous.last_on_gpfs = candidate.readme_mtime
            previous.raw_exists = False
            previous.processed_exists = False
            previous.scratch_cc_exists = False
            previous.scratch_cc_writable = False
            previous.shared_exists = (root / "shared").is_dir()
            previous.last_inspected = now
            return previous

        raw = root / "raw"
        if candidate.raw_exists:
            try:
                subdirs = sorted(
                    entry.name for entry in raw.iterdir() if entry.is_dir()
                )
            except OSError:
                pass  # Keep previously recorded values.
            else:
                previous.raw_subdir_count = len(subdirs)
                previous.raw_subdir_samples = subdirs
        else:
            previous.raw_subdir_count = 0
            previous.raw_subdir_samples = []

        previous.on_gpfs = True
        previous.last_on_gpfs = now
        previous.raw_exists = candidate.raw_exists
        previous.processed_exists = candidate.processed_exists
        previous.scratch_cc_exists = candidate.scratch_cc_exists
        previous.shared_exists = (root / "shared").is_dir()
        previous.scratch_cc_writable = (
            candidate.scratch_cc_writable if candidate.scratch_cc_exists else False
        )
        previous.last_inspected = now
        return previous

    def dir_size_bytes(
        self,
        path: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> int | None:
        if not path.is_dir():
            return None
        if progress_callback is not None:
            progress_callback(f"Counting size of {path}")

        total = 0
        try:
            for item in path.rglob("*"):
                try:
                    if item.is_file():
                        total += item.stat().st_size
                except OSError:
                    # An incomplete scan is not a valid zero/small size.
                    return None
        except OSError:
            return None
        return total

class ProjectManager:
    def __init__(self, db: LaupyDB):
        self.db = db
        self.beamtime_manager = BeamtimeManager()

    def _project(self, project: LaupyProject | int) -> LaupyProject:
        project_id = project if isinstance(project, int) else project.id
        if project_id is None:
            raise ValueError("Project has not been saved")
        result = self.db.get_project(project_id)
        if result is None:
            raise ValueError(f"Project {project_id} does not exist")
        return result

    def get_project_info(self, project: LaupyProject | int) -> ProjectInfo:
        project = self._project(project)
        workspaces = self.db.list_workspaces_for_project(project.id)
        linked = self.db.list_beamtimes_for_project(project.id)
        return ProjectInfo(
            project=project,
            workspaces=workspaces,
            beamtimes=[
                BeamtimeDetail(
                    beamtime=beamtime,
                    storage=self.db.get_beamtime_storage(beamtime.id),
                )
                for beamtime in linked
            ],
        )

    def register_project(
        self,
        path: Path,
        name: str | None = None,
        description: str | None = None,
    ) -> LaupyProject:
        path = Path(path).resolve()
        existing = self.db.get_project_by_path(path)
        if existing is not None:
            return existing
        project = self.db.add_project(
            LaupyProject(
                id=None,
                name=name or path.name,
                path=path,
                description=description,
            )
        )
        self.db.connection.commit()
        return project

    def register_workspace(
        self,
        project: LaupyProject | int,
        path: Path,
        description: str | None = None,
    ) -> LaupyProjectWorkspace:
        project = self._project(project)
        path = Path(path).resolve()
        if path.parent != project.path.resolve() or not path.is_dir():
            raise ValueError(
                "Workspace must be an existing immediate child directory "
                "of the project"
            )
        existing = next(
            (
                workspace
                for workspace in self.db.list_workspaces_for_project(project.id)
                if workspace.path.resolve() == path
            ),
            None,
        )
        if existing is not None:
            return existing
        workspace = self.db.add_workspace(
            LaupyProjectWorkspace(
                id=None,
                project_id=project.id,
                name=path.name,
                path=path,
                description=description,
            )
        )
        self.db.connection.commit()
        return workspace

    def list_workspaces(
        self, project: LaupyProject | int,
    ) -> list[LaupyProjectWorkspace]:
        project = self._project(project)
        return self.db.list_workspaces_for_project(project.id)



    def _beamtime_from_metadata(
        self,
        root: Path,
        data: dict[str, Any] | None,
        metadata_text: str | None,
        existing: Beamtime | None,
        readme_text: str | None = None,
    ) -> Beamtime:
        data = data or {}

        def person(prefix: str) -> dict[str, str | None]:
            value = data.get(prefix) or {}
            if not isinstance(value, dict):
                value = {}
            return {
                f"{prefix}_username": value.get("username"),
                f"{prefix}_lastname": value.get("lastname"),
                f"{prefix}_institute": value.get("institute"),
                f"{prefix}_email": value.get("email"),
                f"{prefix}_user_id": value.get("userId"),
            }

        users = data.get("users") or {}
        if not isinstance(users, dict):
            users = {}

        core_path = data.get("corePath")
        beamline = data.get("beamline") or (existing.beamline if existing else None)
        return Beamtime(
            id=existing.id if existing else None,
            beamtime_id=root.name,
            beamline=beamline,
            label=(existing.label if existing is not None and existing.label else default_beamtime_label(root, beamline)),
            beamline_alias=data.get("beamlineAlias"),
            beamline_setup=data.get("beamtimeSetup") or data.get("beamlineSetup"),
            facility=data.get("facility"),
            proposal_id=data.get("proposalId"),
            proposal_type=data.get("proposalType"),
            event_start=data.get("eventStart"),
            event_end=data.get("eventEnd"),
            generated=data.get("generated"),
            # Prefer the actually scanned location. Metadata's corePath
            # may refer to a different spelling or an obsolete location.
            core_path=root,
            contact=data.get("contact"),
            retention_period=(
                str(data["retentionPeriod"])
                if data.get("retentionPeriod") is not None else None
            ),
            title=data.get("title"),
            description=(
                readme_text if readme_text is not None
                else existing.description if existing else None
            ),
            unix_id=(
                str(data["unixId"]) if data.get("unixId") is not None else None
            ),
            users_door_db=users.get("doorDb"),
            users_special=users.get("special"),
            users_unknown=users.get("unknown"),
            metadata_json=metadata_text,
            created_at=existing.created_at if existing else None,
            **person("applicant"),
            **person("leader"),
            **person("pi"),
        )

    def scan_beamtime(
        self,
        root: Path,
        progress_callback: ProgressCallback | None = None,
        candidate: BeamtimeCandidate | None = None,
    ) -> BeamtimeDetail:
        """Update the database from one positively identified directory."""
        root = Path(os.path.abspath(root))
        if not root.name.isdigit():
            raise ValueError(f"Expected numeric beamtime directory: {root}")
        candidate = candidate or inspect_candidate(root)
        if candidate is None or not candidate.accepted:
            raise ValueError(f"Not a recognizable beamtime directory: {root}")

        if progress_callback is not None:
            progress_callback(f"Reading beamtime metadata from {root}")
        existing = self.db.get_beamtime_by_key(root.name)
        try:
            data, text = self.beamtime_manager.load_metadata(root)
        except (OSError, ValueError):
            # Keep previously stored metadata if the file is unreadable now.
            data, text = None, None

        readme = candidate.readme_text
        if data is None and existing is not None:
            beamtime = existing
            if readme is not None or beamtime.core_path != root:
                if readme is not None:
                    beamtime.description = readme
                beamtime.core_path = root
                beamtime = self.db.add_beamtime(beamtime)
        else:
            beamtime = self.db.add_beamtime(
                self._beamtime_from_metadata(
                    root, data, text, existing, readme_text=readme,
                )
            )

        storage = self.db.get_beamtime_storage(beamtime.id)
        storage = self.beamtime_manager.inspect_storage(
            root, storage or BeamtimeStorage(beamtime_id=beamtime.id), candidate
        )
        self.db.upsert_beamtime_storage(storage)
        self.db.connection.commit()
        return BeamtimeDetail(beamtime=beamtime, storage=storage)

    def mark_beamtime_offloaded(self, beamtime_id: int) -> None:
        """An absent root changes only its GPFS status, never cached sizes."""
        self.db.connection.execute(
            """
            UPDATE beamtime_storage
            SET on_gpfs = 0, last_inspected = ?
            WHERE beamtime_id = ? AND on_gpfs = 1
            """,
            (_now().isoformat(timespec="seconds"), beamtime_id),
        )
        self.db.connection.commit()

    def link_beamtime(
        self,
        project: LaupyProject | int,
        beamtime: Beamtime | int,
    ) -> None:
        project = self._project(project)
        beamtime_id = beamtime if isinstance(beamtime, int) else beamtime.id
        if beamtime_id is None or self.db.get_beamtime(beamtime_id) is None:
            raise ValueError(f"Beamtime {beamtime_id} does not exist")
        self.db.link_beamtime_project(beamtime_id, project.id)

    def update_beamtime_user_fields(
        self,
        beamtime: Beamtime | int,
        label: str | None,
        description: str | None,
    ) -> Beamtime:
        """Update fields owned by the user rather than metadata refresh."""
        beamtime_id = (
            beamtime
            if isinstance(beamtime, int)
            else beamtime.id
        )
        if beamtime_id is None:
            raise ValueError("Beamtime has not been saved")

        now = _now().isoformat(timespec="seconds")
        self.db.connection.execute(
            """
            UPDATE beamtime
            SET label = ?,
                description = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                label.strip() if label else None,
                description.strip() if description else None,
                now,
                beamtime_id,
            ),
        )
        self.db.connection.commit()

        updated = self.db.get_beamtime(beamtime_id)
        if updated is None:
            raise ValueError(
                f"Beamtime {beamtime_id} no longer exists"
            )
        return updated

    def sync_beamtime_projects_from_disk(
        self,
        beamtime: Beamtime | int,
    ) -> list[LaupyProject]:
        """Register immediate scratch_cc/kct_* folders as Laupy projects.

        Existing links are preserved. Database projects whose folders have
        disappeared remain stored, but are not returned by this scan.
        """
        beamtime_id = (
            beamtime
            if isinstance(beamtime, int)
            else beamtime.id
        )
        stored = self.db.get_beamtime(beamtime_id)

        if stored is None:
            raise ValueError(
                f"Beamtime {beamtime_id} does not exist"
            )

        if stored.core_path is None:
            raise ValueError(
                f"Beamtime {stored.beamtime_id} has no known core path"
            )

        scratch = stored.core_path / "scratch_cc"

        if not scratch.is_dir():
            return [
                project
                for project in self.db.list_projects_for_beamtime(
                    stored.id
                )
                if project.path.is_dir()
            ]

        try:
            paths = sorted(
                (
                    entry
                    for entry in scratch.iterdir()
                    if (
                        entry.is_dir()
                        and entry.name.casefold().startswith("kct_")
                    )
                ),
                key=lambda path: path.name.casefold(),
            )
        except OSError:
            paths = []

        for path in paths:
            try:
                project = self.register_project(
                    path,
                    name=path.name,
                )
                self.link_beamtime(project, stored)
            except (OSError, ValueError, sqlite3.Error):
                continue

        projects = self.db.list_projects_for_beamtime(
            stored.id
        )

        return sorted(
            (
                project
                for project in projects
                if project.path.is_dir()
            ),
            key=lambda project: project.name.casefold(),
        )

    def refresh_project_metadata(
        self,
        project: LaupyProject | int,
        progress_callback: ProgressCallback | None = None,
    ) -> ProjectInfo:
        project = self._project(project)
        root = self.beamtime_manager.find_beamtime_root(project.path)
        if root is not None:
            candidate = inspect_candidate(root)
            if candidate is not None and candidate.accepted:
                detail = self.scan_beamtime(root, progress_callback, candidate)
                self.db.link_beamtime_project(detail.beamtime.id, project.id)
        return self.get_project_info(project)

    def refresh_project_sizes(
        self,
        project: LaupyProject | int,
        progress_callback: ProgressCallback | None = None,
    ) -> ProjectInfo:
        project = self._project(project)
        size = self.beamtime_manager.dir_size_bytes(project.path, progress_callback)
        if size is not None:
            now = _now().isoformat(timespec="seconds")
            self.db.connection.execute(
                """
                UPDATE laupy_project
                SET project_size_bytes = ?,
                    project_size_bytes_timestamp = ?,
                    last_inspected = ?
                WHERE id = ?
                """,
                (size, now, now, project.id),
            )
            self.db.connection.commit()
        return self.get_project_info(project)

    def refresh_workspace_size(
        self,
        workspace: LaupyProjectWorkspace | int,
        progress_callback: ProgressCallback | None = None,
    ) -> LaupyProjectWorkspace:
        workspace_id = workspace if isinstance(workspace, int) else workspace.id
        stored = self.db.get_workspace(workspace_id)
        if stored is None:
            raise ValueError(f"Workspace {workspace_id} does not exist")
        size = self.beamtime_manager.dir_size_bytes(stored.path, progress_callback)
        if size is not None:
            now = _now().isoformat(timespec="seconds")
            self.db.connection.execute(
                """
                UPDATE laupy_project_workspace
                SET workspace_size_bytes = ?,
                    workspace_size_bytes_timestamp = ?,
                    last_inspected = ?
                WHERE id = ?
                """,
                (size, now, now, stored.id),
            )
            self.db.connection.commit()
        return self.db.get_workspace(stored.id)

    def refresh_beamtime_sizes(
        self,
        beamtime: Beamtime | int,
        progress_callback: ProgressCallback | None = None,
    ) -> BeamtimeStorage:
        beamtime_id = beamtime if isinstance(beamtime, int) else beamtime.id
        stored = self.db.get_beamtime(beamtime_id)
        if stored is None or stored.core_path is None:
            raise ValueError(f"Beamtime {beamtime_id} has no known storage path")
        storage = self.db.get_beamtime_storage(stored.id)
        storage = storage or BeamtimeStorage(beamtime_id=stored.id)

        if storage.on_gpfs is False:
            return storage

        for subdir, size_field, timestamp_field in (
            ("raw", "raw_size_bytes", "raw_size_bytes_timestamp"),
            ("processed", "processed_size_bytes",
             "processed_size_bytes_timestamp"),
            ("scratch_cc", "scratch_cc_size_bytes",
             "scratch_cc_size_bytes_timestamp"),
        ):
            area = stored.core_path / subdir
            if not area.is_dir():
                continue  # Archive stub: preserve historical size.
            size = self.beamtime_manager.dir_size_bytes(area, progress_callback)
            if size is None:
                continue
            setattr(storage, size_field, size)
            setattr(storage, timestamp_field, _now())

        storage.last_inspected = _now()
        self.db.upsert_beamtime_storage(storage)
        self.db.connection.commit()
        return storage

    def refresh_project_cache(
        self,
        project: LaupyProject | int,
        progress_callback: ProgressCallback | None = None,
    ) -> ProjectInfo:
        self.refresh_project_metadata(project, progress_callback)
        return self.refresh_project_sizes(project, progress_callback)

    def refresh_project_sizes_threadsafe(
        self,
        project: LaupyProject | int,
        progress_callback: ProgressCallback | None = None,
    ) -> ProjectInfo:
        project_id = project if isinstance(project, int) else project.id
        thread_db = LaupyDB(self.db.db_path)
        try:
            return ProjectManager(thread_db).refresh_project_sizes(
                project_id, progress_callback
            )
        finally:
            thread_db.close()

    def refresh_project_cache_threadsafe(
        self,
        project: LaupyProject | int,
        progress_callback: ProgressCallback | None = None,
    ) -> ProjectInfo:
        project_id = project if isinstance(project, int) else project.id
        thread_db = LaupyDB(self.db.db_path)
        try:
            return ProjectManager(thread_db).refresh_project_cache(
                project_id, progress_callback
            )
        finally:
            thread_db.close()

    def record_project_open(
        self,
        project: LaupyProject | int,
        workspace: LaupyProjectWorkspace | int | None = None,
    ) -> None:
        project = self._project(project)
        workspace_id = (
            workspace if isinstance(workspace, int)
            else workspace.id if workspace is not None else None
        )
        self.db.add_history(
            project_id=project.id, workspace_id=workspace_id, action="open"
        )

    def record_project_close(self) -> None:
        self.db.add_history(
            project_id=None, workspace_id=None, action="close"
        )

    def current_project(self) -> LaupyProject | None:
        events = self.db.list_history(limit=1)
        if not events or events[0].project_id is None:
            return None
        return self.db.get_project(events[0].project_id)

    def sync_workspaces_from_disk(
        self, project: LaupyProject | int,
    ) -> list[LaupyProjectWorkspace]:
        project = self._project(project)
        try:
            paths = sorted(
                (
                    entry for entry in project.path.iterdir()
                    if entry.is_dir() and entry.name.startswith("wd")
                ),
                key=lambda path: path.name.lower(),
            )
        except OSError:
            return []

        workspaces = []
        for path in paths:
            try:
                workspaces.append(self.register_workspace(project, path))
            except (OSError, ValueError, sqlite3.Error):
                continue
        return workspaces

    def listed_beamtime_ids(self) -> set[int]:
        rows = self.db.connection.execute(
            "SELECT beamtime_id FROM lautools_app_listed_beamtime"
        ).fetchall()
        return {row[0] for row in rows}

    def set_beamtime_listed(self, beamtime_id: int, listed: bool) -> None:
        if listed:
            self.db.add_listed_beamtime(beamtime_id)
        else:
            self.db.connection.execute(
                "DELETE FROM lautools_app_listed_beamtime WHERE beamtime_id = ?",
                (beamtime_id,),
            )
            self.db.connection.commit()

    def set_beamtime_label(self, beamtime_id: int, label: str | None) -> None:
        self.db.set_beamtime_label(beamtime_id, label)
