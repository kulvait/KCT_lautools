from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Callable

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


@dataclass
class BeamtimeDetail:
    """A beamtime linked to a project, with its separately stored state."""

    beamtime: Beamtime
    storage: BeamtimeStorage | None


@dataclass
class ProjectInfo:
    """Browser-facing view assembled from database rows."""

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
        """Recognize .../gpfs/<a>/<b>/data/<numeric-id>/... paths.

        An independent project outside that layout has no inferred beamtime.
        It can still be linked explicitly with ProjectManager.link_beamtime().
        """
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
        metadata_file = root / f"beamtime-metadata-{root.name}.json"

        try:
            text = metadata_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None, None
        except OSError:
            raise

        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object in {metadata_file}")

        metadata_id = data.get("beamtimeId")
        if metadata_id is not None and str(metadata_id) != root.name:
            raise ValueError(
                f"Beamtime ID {metadata_id!r} does not match directory "
                f"{root.name!r}: {metadata_file}"
            )

        return data, text

    def inspect_storage(
        self,
        root: Path,
        previous: BeamtimeStorage,
    ) -> BeamtimeStorage:
        raw = root / "raw"
        processed = root / "processed"
        scratch = root / "scratch_cc"
        shared = root / "shared"

        raw_subdirs: list[str] = []
        if raw.is_dir():
            try:
                raw_subdirs = sorted(
                    entry.name for entry in raw.iterdir() if entry.is_dir()
                )
            except OSError:
                # Do not turn an unreadable directory into a reported zero.
                previous.raw_subdir_count = None
                previous.raw_subdir_samples = None
            else:
                previous.raw_subdir_count = len(raw_subdirs)
                previous.raw_subdir_samples = raw_subdirs[:12]
        else:
            previous.raw_subdir_count = 0
            previous.raw_subdir_samples = []

        now = _now()
        previous.on_gpfs = root.is_dir()
        if previous.on_gpfs:
            previous.last_on_gpfs = now
        # A GPFS inspection tells us nothing about tape. Preserve on_tape.
        previous.raw_exists = raw.is_dir()
        previous.processed_exists = processed.is_dir()
        previous.scratch_cc_exists = scratch.is_dir()
        previous.shared_exists = shared.is_dir()
        previous.scratch_cc_writable = (
            os.access(scratch, os.W_OK) if scratch.is_dir() else False
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
                    continue
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
        """Read cached data without scanning the filesystem."""
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
        """Register a project even when it has no beamtime or workspace."""
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
        """Register one immediate child WD of a project.

        This records an existing directory; it does not create one.
        """
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
        self,
        project: LaupyProject | int,
    ) -> list[LaupyProjectWorkspace]:
        project = self._project(project)
        return self.db.list_workspaces_for_project(project.id)

    def _beamtime_from_metadata(
        self,
        root: Path,
        data: dict[str, Any] | None,
        metadata_text: str | None,
        existing: Beamtime | None,
    ) -> Beamtime:
        """Preserve user description when refreshing external metadata."""
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
        return Beamtime(
            id=existing.id if existing else None,
            beamtime_id=root.name,
            beamline=data.get("beamline"),
            beamline_alias=data.get("beamlineAlias"),
            beamline_setup=(
                data.get("beamtimeSetup") or data.get("beamlineSetup")
            ),
            facility=data.get("facility"),
            proposal_id=data.get("proposalId"),
            proposal_type=data.get("proposalType"),
            event_start=data.get("eventStart"),
            event_end=data.get("eventEnd"),
            generated=data.get("generated"),
            core_path=Path(core_path) if core_path else root,
            contact=data.get("contact"),
            retention_period=(
                str(data["retentionPeriod"])
                if data.get("retentionPeriod") is not None
                else None
            ),
            title=data.get("title"),
            description=existing.description if existing else None,
            unix_id=(
                str(data["unixId"])
                if data.get("unixId") is not None
                else None
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
    ) -> BeamtimeDetail:
        """Scan a beamtime independently of any Laupy project."""
        root = Path(root).resolve()
        if not root.name.isdigit():
            raise ValueError(f"Expected numeric beamtime directory: {root}")
        if not root.is_dir():
            raise ValueError(f"Beamtime directory does not exist: {root}")

        if progress_callback is not None:
            progress_callback(f"Reading beamtime metadata from {root}")

        data, text = self.beamtime_manager.load_metadata(root)
        existing = self.db.get_beamtime_by_key(root.name)

        if data is None and existing is not None:
            # Missing metadata must not erase an earlier successful scan.
            beamtime = existing
        else:
            beamtime = self.db.add_beamtime(
                self._beamtime_from_metadata(root, data, text, existing)
            )

        storage = self.db.get_beamtime_storage(beamtime.id)
        storage = self.beamtime_manager.inspect_storage(
            root,
            storage or BeamtimeStorage(beamtime_id=beamtime.id),
        )
        self.db.upsert_beamtime_storage(storage)
        self.db.connection.commit()

        return BeamtimeDetail(beamtime=beamtime, storage=storage)

    def link_beamtime(
        self,
        project: LaupyProject | int,
        beamtime: Beamtime | int,
    ) -> None:
        """Explicitly associate an independent project with a beamtime."""
        project = self._project(project)
        beamtime_id = beamtime if isinstance(beamtime, int) else beamtime.id
        if beamtime_id is None or self.db.get_beamtime(beamtime_id) is None:
            raise ValueError(f"Beamtime {beamtime_id} does not exist")
        self.db.link_beamtime_project(beamtime_id, project.id)

    def refresh_project_metadata(
        self,
        project: LaupyProject | int,
        progress_callback: ProgressCallback | None = None,
    ) -> ProjectInfo:
        """Discover a GPFS beamtime when possible; never remove manual links."""
        project = self._project(project)
        root = self.beamtime_manager.find_beamtime_root(project.path)

        if root is not None and root.is_dir():
            detail = self.scan_beamtime(root, progress_callback)
            self.db.link_beamtime_project(detail.beamtime.id, project.id)

        return self.get_project_info(project)

    def refresh_project_sizes(
        self,
        project: LaupyProject | int,
        progress_callback: ProgressCallback | None = None,
    ) -> ProjectInfo:
        """Count the project only, not its linked beamtimes' storage."""
        project = self._project(project)
        size = self.beamtime_manager.dir_size_bytes(
            project.path, progress_callback
        )
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
        workspace_id = (
            workspace if isinstance(workspace, int) else workspace.id
        )
        stored = self.db.get_workspace(workspace_id)
        if stored is None:
            raise ValueError(f"Workspace {workspace_id} does not exist")

        size = self.beamtime_manager.dir_size_bytes(
            stored.path, progress_callback
        )
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
        """Explicit operation: may be expensive on a large beamtime."""
        beamtime_id = beamtime if isinstance(beamtime, int) else beamtime.id
        stored = self.db.get_beamtime(beamtime_id)
        if stored is None or stored.core_path is None:
            raise ValueError(
                f"Beamtime {beamtime_id} has no known storage path"
            )

        root = stored.core_path
        storage = self.db.get_beamtime_storage(stored.id)
        storage = storage or BeamtimeStorage(beamtime_id=stored.id)

        for subdir, size_field, timestamp_field in (
            ("raw", "raw_size_bytes", "raw_size_bytes_timestamp"),
            ("processed", "processed_size_bytes",
             "processed_size_bytes_timestamp"),
            ("scratch_cc", "scratch_cc_size_bytes",
             "scratch_cc_size_bytes_timestamp"),
        ):
            size = self.beamtime_manager.dir_size_bytes(
                root / subdir, progress_callback
            )
            setattr(storage, size_field, size)
            setattr(storage, timestamp_field, _now() if size is not None else None)

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
            else workspace.id if workspace is not None
            else None
        )
        # The schema's composite FK rejects a workspace from another project.
        self.db.add_history(
            project_id=project.id,
            workspace_id=workspace_id,
            action="open",
        )

    def record_project_close(self) -> None:
        self.db.add_history(
            project_id=None,
            workspace_id=None,
            action="close",
        )

    def current_project(self) -> LaupyProject | None:
        """Unlike 'last non-null project', a close event clears this result."""
        events = self.db.list_history(limit=1)
        if not events or events[0].project_id is None:
            return None
        return self.db.get_project(events[0].project_id)

    def sync_workspaces_from_disk(
        self,
        project: LaupyProject | int,
    ) -> list[LaupyProjectWorkspace]:
        """wd* directories currently on disk, registered for history/sizes.

        Database rows whose directory has disappeared are not returned.
        """
        project = self._project(project)
        try:
            paths = sorted(
                (
                    entry
                    for entry in project.path.iterdir()
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
