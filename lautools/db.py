from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator


def _now() -> datetime:
    return datetime.now().replace(microsecond=0)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") if value is not None else None


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _json_loads(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value)


@dataclass
class Beamtime:
    id: int | None
    beamtime_id: str
    label: str | None = None
    beamline: str | None = None
    beamline_alias: str | None = None
    beamline_setup: str | None = None
    facility: str | None = None
    proposal_id: str | None = None
    proposal_type: str | None = None
    event_start: str | None = None
    event_end: str | None = None
    generated: str | None = None
    core_path: Path | None = None
    applicant_username: str | None = None
    applicant_lastname: str | None = None
    applicant_institute: str | None = None
    applicant_email: str | None = None
    applicant_user_id: str | None = None
    contact: str | None = None
    leader_username: str | None = None
    leader_lastname: str | None = None
    leader_institute: str | None = None
    leader_email: str | None = None
    leader_user_id: str | None = None
    pi_username: str | None = None
    pi_lastname: str | None = None
    pi_institute: str | None = None
    pi_email: str | None = None
    pi_user_id: str | None = None
    retention_period: str | None = None
    title: str | None = None
    description: str | None = None
    unix_id: str | None = None
    users_door_db: Any = None
    users_special: Any = None
    users_unknown: Any = None
    metadata_json: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class BeamtimeStorage:
    beamtime_id: int
    on_gpfs: bool | None = None
    on_tape: bool | None = None
    last_on_gpfs: datetime | None = None
    raw_exists: bool | None = None
    raw_subdir_count: int | None = None
    raw_subdir_samples: list[str] | None = None
    raw_size_bytes: int | None = None
    raw_size_bytes_timestamp: datetime | None = None
    processed_exists: bool | None = None
    processed_size_bytes: int | None = None
    processed_size_bytes_timestamp: datetime | None = None
    scratch_cc_exists: bool | None = None
    scratch_cc_writable: bool | None = None
    scratch_cc_size_bytes: int | None = None
    scratch_cc_size_bytes_timestamp: datetime | None = None
    shared_exists: bool | None = None
    last_inspected: datetime | None = None


@dataclass
class LaupyProject:
    id: int | None
    name: str
    path: Path
    description: str | None = None
    created_at: datetime | None = None
    project_size_bytes: int | None = None
    project_size_bytes_timestamp: datetime | None = None
    last_inspected: datetime | None = None


@dataclass
class LaupyProjectWorkspace:
    id: int | None
    project_id: int
    name: str
    path: Path
    description: str | None = None
    created_at: datetime | None = None
    workspace_size_bytes: int | None = None
    workspace_size_bytes_timestamp: datetime | None = None
    last_inspected: datetime | None = None


@dataclass
class BeamtimeProjectLink:
    beamtime_id: int
    project_id: int
    created_at: datetime | None = None


@dataclass
class ListedBeamtime:
    beamtime_id: int
    listed_at: datetime | None = None
    last_access: datetime | None = None
    pinned: bool = False


@dataclass
class ListedProject:
    project_id: int
    listed_at: datetime | None = None
    last_access: datetime | None = None
    pinned: bool = False


@dataclass
class AppHistory:
    id: int | None
    opened_at: datetime | None = None
    project_id: int | None = None
    workspace_id: int | None = None
    action: str | None = None
 
@dataclass
class LautoolsLocation:
    id: int | None
    name: str
    disk_location: Path | None = None
    upstream: str | None = None
    git_managed: bool | None = None
    git_root: Path | None = None
    git_subdir: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class LaupyRecipeCollection:
    location_id: int
    use_as_cookbook: bool = False
    use_as_workbench: bool = False


@dataclass
class LaupyRecipeInstance:
    id: int | None
    collection_location_id: int
    name: str
    relative_path: Path
    source_instance_id: int | None = None
    created_at: datetime | None = None
    cloned_at: datetime | None = None
    last_inspected: datetime | None = None


@dataclass
class LautoolsPath:
    location_id: int
    position: int = 0


@dataclass
class LautoolsConfig:
    default_cookbook_location_id: int | None = None
    default_workbench_location_id: int | None = None
    updated_at: datetime | None = None


class LaupyDB:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")

        self._init_schema()

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        try:
            yield
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def _init_schema(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        schema = schema_path.read_text(encoding="utf-8")
        self.connection.executescript(schema)
        self.connection.commit()

    def _row_to_beamtime(self, row) -> Beamtime:
        return Beamtime(
            id=row["id"],
            beamtime_id=row["beamtime_id"],
            label=row["label"],
            beamline=row["beamline"],
            beamline_alias=row["beamline_alias"],
            beamline_setup=row["beamline_setup"],
            facility=row["facility"],
            proposal_id=row["proposal_id"],
            proposal_type=row["proposal_type"],
            event_start=row["event_start"],
            event_end=row["event_end"],
            generated=row["generated"],
            core_path=Path(row["core_path"]) if row["core_path"] else None,
            applicant_username=row["applicant_username"],
            applicant_lastname=row["applicant_lastname"],
            applicant_institute=row["applicant_institute"],
            applicant_email=row["applicant_email"],
            applicant_user_id=row["applicant_user_id"],
            contact=row["contact"],
            leader_username=row["leader_username"],
            leader_lastname=row["leader_lastname"],
            leader_institute=row["leader_institute"],
            leader_email=row["leader_email"],
            leader_user_id=row["leader_user_id"],
            pi_username=row["pi_username"],
            pi_lastname=row["pi_lastname"],
            pi_institute=row["pi_institute"],
            pi_email=row["pi_email"],
            pi_user_id=row["pi_user_id"],
            retention_period=row["retention_period"],
            title=row["title"],
            description=row["description"],
            unix_id=row["unix_id"],
            users_door_db=_json_loads(row["users_door_db"]),
            users_special=_json_loads(row["users_special"]),
            users_unknown=_json_loads(row["users_unknown"]),
            metadata_json=row["metadata_json"],
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )

    def _row_to_storage(self, row) -> BeamtimeStorage:
        return BeamtimeStorage(
            beamtime_id=row["beamtime_id"],
            on_gpfs=bool(row["on_gpfs"]) if row["on_gpfs"] is not None else None,
            on_tape=bool(row["on_tape"]) if row["on_tape"] is not None else None,
            last_on_gpfs=_parse_dt(row["last_on_gpfs"]),
            raw_exists=bool(row["raw_exists"]) if row["raw_exists"] is not None else None,
            raw_subdir_count=row["raw_subdir_count"],
            raw_subdir_samples=_json_loads(row["raw_subdir_samples"]),
            raw_size_bytes=row["raw_size_bytes"],
            raw_size_bytes_timestamp=_parse_dt(row["raw_size_bytes_timestamp"]),
            processed_exists=bool(row["processed_exists"]) if row["processed_exists"] is not None else None,
            processed_size_bytes=row["processed_size_bytes"],
            processed_size_bytes_timestamp=_parse_dt(row["processed_size_bytes_timestamp"]),
            scratch_cc_exists=bool(row["scratch_cc_exists"]) if row["scratch_cc_exists"] is not None else None,
            scratch_cc_writable=bool(row["scratch_cc_writable"]) if row["scratch_cc_writable"] is not None else None,
            scratch_cc_size_bytes=row["scratch_cc_size_bytes"],
            scratch_cc_size_bytes_timestamp=_parse_dt(row["scratch_cc_size_bytes_timestamp"]),
            shared_exists=bool(row["shared_exists"]) if row["shared_exists"] is not None else None,
            last_inspected=_parse_dt(row["last_inspected"]),
        )

    def _row_to_project(self, row) -> LaupyProject:
        return LaupyProject(
            id=row["id"],
            name=row["name"],
            path=Path(row["path"]),
            description=row["description"],
            created_at=_parse_dt(row["created_at"]),
            project_size_bytes=row["project_size_bytes"],
            project_size_bytes_timestamp=_parse_dt(row["project_size_bytes_timestamp"]),
            last_inspected=_parse_dt(row["last_inspected"]),
        )

    def _row_to_workspace(self, row) -> LaupyProjectWorkspace:
        return LaupyProjectWorkspace(
            id=row["id"],
            project_id=row["project_id"],
            name=row["name"],
            path=Path(row["path"]),
            description=row["description"],
            created_at=_parse_dt(row["created_at"]),
            workspace_size_bytes=row["workspace_size_bytes"],
            workspace_size_bytes_timestamp=_parse_dt(row["workspace_size_bytes_timestamp"]),
            last_inspected=_parse_dt(row["last_inspected"]),
        )

    def _row_to_link(self, row) -> BeamtimeProjectLink:
        return BeamtimeProjectLink(
            beamtime_id=row["beamtime_id"],
            project_id=row["project_id"],
            created_at=_parse_dt(row["created_at"]),
        )

    def _row_to_listed_beamtime(self, row) -> ListedBeamtime:
        return ListedBeamtime(
            beamtime_id=row["beamtime_id"],
            listed_at=_parse_dt(row["listed_at"]),
            last_access=_parse_dt(row["last_access"]),
            pinned=bool(row["pinned"]),
        )

    def _row_to_listed_project(self, row) -> ListedProject:
        return ListedProject(
            project_id=row["project_id"],
            listed_at=_parse_dt(row["listed_at"]),
            last_access=_parse_dt(row["last_access"]),
            pinned=bool(row["pinned"]),
        )

    def _row_to_history(self, row) -> AppHistory:
        return AppHistory(
            id=row["id"],
            opened_at=_parse_dt(row["opened_at"]),
            project_id=row["project_id"],
            workspace_id=row["workspace_id"],
            action=row["action"],
        )

    def _row_to_location(self, row) -> LautoolsLocation:
        return LautoolsLocation(
            id=row["id"],
            name=row["name"],
            disk_location=Path(row["disk_location"]) if row["disk_location"] else None,
            upstream=row["upstream"],
            git_managed=(
                bool(row["git_managed"])
                if row["git_managed"] is not None else None
            ),
            git_root=Path(row["git_root"]) if row["git_root"] else None,
            git_subdir=row["git_subdir"],
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )

    def _row_to_recipe_collection(self, row) -> LaupyRecipeCollection:
        return LaupyRecipeCollection(
            location_id=row["location_id"],
            use_as_cookbook=bool(row["use_as_cookbook"]),
            use_as_workbench=bool(row["use_as_workbench"]),
        )

    def _row_to_recipe_instance(self, row) -> LaupyRecipeInstance:
        return LaupyRecipeInstance(
            id=row["id"],
            collection_location_id=row["collection_location_id"],
            name=row["name"],
            relative_path=Path(row["relative_path"]),
            source_instance_id=row["source_instance_id"],
            created_at=_parse_dt(row["created_at"]),
            cloned_at=_parse_dt(row["cloned_at"]),
            last_inspected=_parse_dt(row["last_inspected"]),
        )

    def _row_to_path_entry(self, row) -> LautoolsPath:
        return LautoolsPath(
            location_id=row["location_id"],
            position=row["position"],
        )

    def add_beamtime(self, beamtime: Beamtime) -> Beamtime:
        now = _iso(_now())
        params = {
            "beamtime_id": beamtime.beamtime_id,
            "label": beamtime.label,
            "beamline": beamtime.beamline,
            "beamline_alias": beamtime.beamline_alias,
            "beamline_setup": beamtime.beamline_setup,
            "facility": beamtime.facility,
            "proposal_id": beamtime.proposal_id,
            "proposal_type": beamtime.proposal_type,
            "event_start": beamtime.event_start,
            "event_end": beamtime.event_end,
            "generated": beamtime.generated,
            "core_path": str(beamtime.core_path) if beamtime.core_path else None,
            "applicant_username": beamtime.applicant_username,
            "applicant_lastname": beamtime.applicant_lastname,
            "applicant_institute": beamtime.applicant_institute,
            "applicant_email": beamtime.applicant_email,
            "applicant_user_id": beamtime.applicant_user_id,
            "contact": beamtime.contact,
            "leader_username": beamtime.leader_username,
            "leader_lastname": beamtime.leader_lastname,
            "leader_institute": beamtime.leader_institute,
            "leader_email": beamtime.leader_email,
            "leader_user_id": beamtime.leader_user_id,
            "pi_username": beamtime.pi_username,
            "pi_lastname": beamtime.pi_lastname,
            "pi_institute": beamtime.pi_institute,
            "pi_email": beamtime.pi_email,
            "pi_user_id": beamtime.pi_user_id,
            "retention_period": beamtime.retention_period,
            "title": beamtime.title,
            "description": beamtime.description,
            "unix_id": beamtime.unix_id,
            "users_door_db": _json_dumps(beamtime.users_door_db),
            "users_special": _json_dumps(beamtime.users_special),
            "users_unknown": _json_dumps(beamtime.users_unknown),
            "metadata_json": beamtime.metadata_json,
            "created_at": _iso(beamtime.created_at or _now()),
            "updated_at": now,
        }
        self.connection.execute(
            """
            INSERT INTO beamtime (
                beamtime_id, label, beamline, beamline_alias, beamline_setup, facility,
                proposal_id, proposal_type, event_start, event_end, generated,
                core_path, applicant_username, applicant_lastname,
                applicant_institute, applicant_email, applicant_user_id,
                contact, leader_username, leader_lastname, leader_institute,
                leader_email, leader_user_id, pi_username, pi_lastname,
                pi_institute, pi_email, pi_user_id, retention_period, title,
                description, unix_id, users_door_db, users_special,
                users_unknown, metadata_json, created_at, updated_at
            ) VALUES (
                :beamtime_id, :label, :beamline, :beamline_alias, :beamline_setup, :facility,
                :proposal_id, :proposal_type, :event_start, :event_end, :generated,
                :core_path, :applicant_username, :applicant_lastname,
                :applicant_institute, :applicant_email, :applicant_user_id,
                :contact, :leader_username, :leader_lastname, :leader_institute,
                :leader_email, :leader_user_id, :pi_username, :pi_lastname,
                :pi_institute, :pi_email, :pi_user_id, :retention_period, :title,
                :description, :unix_id, :users_door_db, :users_special,
                :users_unknown, :metadata_json, :created_at, :updated_at
            )
            ON CONFLICT(beamtime_id) DO UPDATE SET
                -- A rescan must not discard a label chosen by the user
                label=COALESCE(excluded.label, beamtime.label),
                beamline=excluded.beamline,
                beamline_alias=excluded.beamline_alias,
                beamline_setup=excluded.beamline_setup,
                facility=excluded.facility,
                proposal_id=excluded.proposal_id,
                proposal_type=excluded.proposal_type,
                event_start=excluded.event_start,
                event_end=excluded.event_end,
                generated=excluded.generated,
                core_path=excluded.core_path,
                applicant_username=excluded.applicant_username,
                applicant_lastname=excluded.applicant_lastname,
                applicant_institute=excluded.applicant_institute,
                applicant_email=excluded.applicant_email,
                applicant_user_id=excluded.applicant_user_id,
                contact=excluded.contact,
                leader_username=excluded.leader_username,
                leader_lastname=excluded.leader_lastname,
                leader_institute=excluded.leader_institute,
                leader_email=excluded.leader_email,
                leader_user_id=excluded.leader_user_id,
                pi_username=excluded.pi_username,
                pi_lastname=excluded.pi_lastname,
                pi_institute=excluded.pi_institute,
                pi_email=excluded.pi_email,
                pi_user_id=excluded.pi_user_id,
                retention_period=excluded.retention_period,
                title=excluded.title,
                description=excluded.description,
                unix_id=excluded.unix_id,
                users_door_db=excluded.users_door_db,
                users_special=excluded.users_special,
                users_unknown=excluded.users_unknown,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            params,
        )
        row = self.connection.execute(
            "SELECT * FROM beamtime WHERE beamtime_id = ?",
            (beamtime.beamtime_id,),
        ).fetchone()
        return self._row_to_beamtime(row)

    def get_beamtime(self, beamtime_id: int) -> Beamtime | None:
        row = self.connection.execute(
            "SELECT * FROM beamtime WHERE id = ?",
            (beamtime_id,),
        ).fetchone()
        return self._row_to_beamtime(row) if row else None

    def get_beamtime_by_key(self, beamtime_key: str) -> Beamtime | None:
        row = self.connection.execute(
            "SELECT * FROM beamtime WHERE beamtime_id = ?",
            (beamtime_key,),
        ).fetchone()
        return self._row_to_beamtime(row) if row else None

    def list_beamtimes(self) -> list[Beamtime]:
        rows = self.connection.execute(
            "SELECT * FROM beamtime ORDER BY beamtime_id"
        ).fetchall()
        return [self._row_to_beamtime(r) for r in rows]

    def upsert_beamtime_storage(
        self,
        storage: BeamtimeStorage,
    ) -> BeamtimeStorage:
        params = {
            "beamtime_id": storage.beamtime_id,
            "on_gpfs": int(storage.on_gpfs) if storage.on_gpfs is not None else None,
            "on_tape": int(storage.on_tape) if storage.on_tape is not None else None,
            "last_on_gpfs": _iso(storage.last_on_gpfs),
            "raw_exists": int(storage.raw_exists) if storage.raw_exists is not None else None,
            "raw_subdir_count": storage.raw_subdir_count,
            "raw_subdir_samples": _json_dumps(storage.raw_subdir_samples),
            "raw_size_bytes": storage.raw_size_bytes,
            "raw_size_bytes_timestamp": _iso(storage.raw_size_bytes_timestamp),
            "processed_exists": int(storage.processed_exists) if storage.processed_exists is not None else None,
            "processed_size_bytes": storage.processed_size_bytes,
            "processed_size_bytes_timestamp": _iso(storage.processed_size_bytes_timestamp),
            "scratch_cc_exists": int(storage.scratch_cc_exists) if storage.scratch_cc_exists is not None else None,
            "scratch_cc_writable": int(storage.scratch_cc_writable) if storage.scratch_cc_writable is not None else None,
            "scratch_cc_size_bytes": storage.scratch_cc_size_bytes,
            "scratch_cc_size_bytes_timestamp": _iso(storage.scratch_cc_size_bytes_timestamp),
            "shared_exists": int(storage.shared_exists) if storage.shared_exists is not None else None,
            "last_inspected": _iso(storage.last_inspected),
        }
        self.connection.execute(
            """
            INSERT INTO beamtime_storage (
                beamtime_id, on_gpfs, on_tape, last_on_gpfs,
                raw_exists, raw_subdir_count, raw_subdir_samples,
                raw_size_bytes, raw_size_bytes_timestamp,
                processed_exists, processed_size_bytes,
                processed_size_bytes_timestamp,
                scratch_cc_exists, scratch_cc_writable,
                scratch_cc_size_bytes, scratch_cc_size_bytes_timestamp,
                shared_exists, last_inspected
            ) VALUES (
                :beamtime_id, :on_gpfs, :on_tape, :last_on_gpfs,
                :raw_exists, :raw_subdir_count, :raw_subdir_samples,
                :raw_size_bytes, :raw_size_bytes_timestamp,
                :processed_exists, :processed_size_bytes,
                :processed_size_bytes_timestamp,
                :scratch_cc_exists, :scratch_cc_writable,
                :scratch_cc_size_bytes, :scratch_cc_size_bytes_timestamp,
                :shared_exists, :last_inspected
            )
            ON CONFLICT(beamtime_id) DO UPDATE SET
                on_gpfs=excluded.on_gpfs,
                on_tape=excluded.on_tape,
                last_on_gpfs=excluded.last_on_gpfs,
                raw_exists=excluded.raw_exists,
                raw_subdir_count=excluded.raw_subdir_count,
                raw_subdir_samples=excluded.raw_subdir_samples,
                raw_size_bytes=excluded.raw_size_bytes,
                raw_size_bytes_timestamp=excluded.raw_size_bytes_timestamp,
                processed_exists=excluded.processed_exists,
                processed_size_bytes=excluded.processed_size_bytes,
                processed_size_bytes_timestamp=excluded.processed_size_bytes_timestamp,
                scratch_cc_exists=excluded.scratch_cc_exists,
                scratch_cc_writable=excluded.scratch_cc_writable,
                scratch_cc_size_bytes=excluded.scratch_cc_size_bytes,
                scratch_cc_size_bytes_timestamp=excluded.scratch_cc_size_bytes_timestamp,
                shared_exists=excluded.shared_exists,
                last_inspected=excluded.last_inspected
            """,
            params,
        )
        row = self.connection.execute(
            "SELECT * FROM beamtime_storage WHERE beamtime_id = ?",
            (storage.beamtime_id,),
        ).fetchone()
        return self._row_to_storage(row)

    def get_beamtime_storage(self, beamtime_id: int) -> BeamtimeStorage | None:
        row = self.connection.execute(
            "SELECT * FROM beamtime_storage WHERE beamtime_id = ?",
            (beamtime_id,),
        ).fetchone()
        return self._row_to_storage(row) if row else None

    def add_project(self, project: LaupyProject) -> LaupyProject:
        params = {
            "name": project.name,
            "path": str(project.path),
            "description": project.description,
            "created_at": _iso(project.created_at or _now()),
            "project_size_bytes": project.project_size_bytes,
            "project_size_bytes_timestamp": _iso(project.project_size_bytes_timestamp),
            "last_inspected": _iso(project.last_inspected),
        }
        self.connection.execute(
            """
            INSERT INTO laupy_project (
                name, path, description, created_at,
                project_size_bytes, project_size_bytes_timestamp, last_inspected
            ) VALUES (
                :name, :path, :description, :created_at,
                :project_size_bytes, :project_size_bytes_timestamp, :last_inspected
            )
            ON CONFLICT(path) DO UPDATE SET
                name=excluded.name,
                description=excluded.description,
                project_size_bytes=excluded.project_size_bytes,
                project_size_bytes_timestamp=excluded.project_size_bytes_timestamp,
                last_inspected=excluded.last_inspected
            """,
            params,
        )
        row = self.connection.execute(
            "SELECT * FROM laupy_project WHERE path = ?",
            (str(project.path),),
        ).fetchone()
        return self._row_to_project(row)

    def get_project(self, project_id: int) -> LaupyProject | None:
        row = self.connection.execute(
            "SELECT * FROM laupy_project WHERE id = ?",
            (project_id,),
        ).fetchone()
        return self._row_to_project(row) if row else None

    def get_project_by_path(self, path: Path) -> LaupyProject | None:
        row = self.connection.execute(
            "SELECT * FROM laupy_project WHERE path = ?",
            (str(Path(path).resolve()),),
        ).fetchone()
        return self._row_to_project(row) if row else None

    def list_projects(self) -> list[LaupyProject]:
        rows = self.connection.execute(
            "SELECT * FROM laupy_project ORDER BY name"
        ).fetchall()
        return [self._row_to_project(r) for r in rows]

    def add_workspace(self, workspace: LaupyProjectWorkspace) -> LaupyProjectWorkspace:
        params = {
            "project_id": workspace.project_id,
            "name": workspace.name,
            "path": str(workspace.path),
            "description": workspace.description,
            "created_at": _iso(workspace.created_at or _now()),
            "workspace_size_bytes": workspace.workspace_size_bytes,
            "workspace_size_bytes_timestamp": _iso(workspace.workspace_size_bytes_timestamp),
            "last_inspected": _iso(workspace.last_inspected),
        }
        self.connection.execute(
            """
            INSERT INTO laupy_project_workspace (
                project_id, name, path, description, created_at,
                workspace_size_bytes, workspace_size_bytes_timestamp, last_inspected
            ) VALUES (
                :project_id, :name, :path, :description, :created_at,
                :workspace_size_bytes, :workspace_size_bytes_timestamp, :last_inspected
            )
            ON CONFLICT(path) DO UPDATE SET
                project_id=excluded.project_id,
                name=excluded.name,
                description=excluded.description,
                workspace_size_bytes=excluded.workspace_size_bytes,
                workspace_size_bytes_timestamp=excluded.workspace_size_bytes_timestamp,
                last_inspected=excluded.last_inspected
            """,
            params,
        )
        row = self.connection.execute(
            "SELECT * FROM laupy_project_workspace WHERE path = ?",
            (str(workspace.path),),
        ).fetchone()
        return self._row_to_workspace(row)

    def get_workspace(self, workspace_id: int) -> LaupyProjectWorkspace | None:
        row = self.connection.execute(
            "SELECT * FROM laupy_project_workspace WHERE id = ?",
            (workspace_id,),
        ).fetchone()
        return self._row_to_workspace(row) if row else None

    def list_workspaces_for_project(
        self,
        project_id: int,
    ) -> list[LaupyProjectWorkspace]:
        rows = self.connection.execute(
            """
            SELECT * FROM laupy_project_workspace
            WHERE project_id = ?
            ORDER BY name
            """,
            (project_id,),
        ).fetchall()
        return [self._row_to_workspace(r) for r in rows]

    def link_beamtime_project(self, beamtime_id: int, project_id: int) -> None:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO beamtime_project_link (
                beamtime_id, project_id, created_at
            ) VALUES (?, ?, ?)
            """,
            (beamtime_id, project_id, _iso(_now())),
        )
        self.connection.commit()

    def list_projects_for_beamtime(self, beamtime_id: int) -> list[LaupyProject]:
        rows = self.connection.execute(
            """
            SELECT p.*
            FROM laupy_project p
            JOIN beamtime_project_link l ON l.project_id = p.id
            WHERE l.beamtime_id = ?
            ORDER BY p.name
            """,
            (beamtime_id,),
        ).fetchall()
        return [self._row_to_project(r) for r in rows]

    def list_beamtimes_for_project(self, project_id: int) -> list[Beamtime]:
        rows = self.connection.execute(
            """
            SELECT b.*
            FROM beamtime b
            JOIN beamtime_project_link l ON l.beamtime_id = b.id
            WHERE l.project_id = ?
            ORDER BY b.beamtime_id
            """,
            (project_id,),
        ).fetchall()
        return [self._row_to_beamtime(r) for r in rows]

    def add_listed_beamtime(self, beamtime_id: int, pinned: bool = False) -> None:
        self.connection.execute(
            """
            INSERT INTO lautools_app_listed_beamtime (
                beamtime_id, listed_at, pinned
            ) VALUES (?, ?, ?)
            ON CONFLICT(beamtime_id) DO UPDATE SET
                listed_at=excluded.listed_at,
                pinned=excluded.pinned
            """,
            (beamtime_id, _iso(_now()), int(pinned)),
        )
        self.connection.commit()

    def set_beamtime_label(self, beamtime_id: int, label: str | None) -> None:
        """Store a user label; None or blank restores the default."""
        self.connection.execute(
            "UPDATE beamtime SET label = ?, updated_at = ? WHERE id = ?",
            (label or None, _iso(_now()), beamtime_id),
        )
        self.connection.commit()

    def add_listed_project(self, project_id: int, pinned: bool = False) -> None:
        self.connection.execute(
            """
            INSERT INTO lautools_app_listed_project (
                project_id, listed_at, pinned
            ) VALUES (?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                listed_at=excluded.listed_at,
                pinned=excluded.pinned
            """,
            (project_id, _iso(_now()), int(pinned)),
        )
        self.connection.commit()

    def add_history(
        self,
        project_id: int | None = None,
        workspace_id: int | None = None,
        action: str | None = None,
        opened_at: datetime | None = None,
    ) -> AppHistory:
        self.connection.execute(
            """
            INSERT INTO lautools_app_history (
                opened_at, project_id, workspace_id, action
            ) VALUES (?, ?, ?, ?)
            """,
            (_iso(opened_at or _now()), project_id, workspace_id, action),
        )
        self.connection.commit()
        row = self.connection.execute(
            "SELECT * FROM lautools_app_history ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return self._row_to_history(row)

    def list_history(self, limit: int = 100) -> list[AppHistory]:
        rows = self.connection.execute(
            """
            SELECT * FROM lautools_app_history
            ORDER BY opened_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._row_to_history(r) for r in rows]

    def last_project(self) -> LaupyProject | None:
        row = self.connection.execute(
            """
            SELECT p.*
            FROM lautools_app_history h
            JOIN laupy_project p ON p.id = h.project_id
            ORDER BY h.opened_at DESC, h.id DESC
            LIMIT 1
            """
        ).fetchone()
        return self._row_to_project(row) if row else None

    def last_workspace(self, project_id: int) -> LaupyProjectWorkspace | None:
        row = self.connection.execute(
            """
            SELECT w.*
            FROM lautools_app_history h
            JOIN laupy_project_workspace w ON w.id = h.workspace_id
            WHERE h.project_id = ?
            ORDER BY h.opened_at DESC, h.id DESC
            LIMIT 1
            """,
            (project_id,),
        ).fetchone()
        return self._row_to_workspace(row) if row else None

    def list_listed_projects(self) -> list[LaupyProject]:
        rows = self.connection.execute(
            """
            SELECT p.*
            FROM laupy_project p
            JOIN lautools_app_listed_project l ON l.project_id = p.id
            ORDER BY l.pinned DESC, l.last_access IS NULL, l.last_access DESC,
                     l.listed_at DESC
            """
        ).fetchall()
        return [self._row_to_project(r) for r in rows]

    def list_listed_beamtimes(self) -> list[Beamtime]:
        rows = self.connection.execute(
            """
            SELECT b.*
            FROM beamtime b
            JOIN lautools_app_listed_beamtime l ON l.beamtime_id = b.id
            ORDER BY l.pinned DESC, l.last_access IS NULL, l.last_access DESC,
                     l.listed_at DESC
            """
        ).fetchall()
        return [self._row_to_beamtime(r) for r in rows]


    # ------------------------------------------------------------------
    # Locations / collections / recipe instances
    # ------------------------------------------------------------------

    def list_locations(self) -> list[LautoolsLocation]:
        rows = self.connection.execute(
            "SELECT * FROM lautools_location ORDER BY name COLLATE NOCASE, id"
        ).fetchall()
        return [self._row_to_location(r) for r in rows]

    def get_location(self, location_id: int) -> LautoolsLocation | None:
        row = self.connection.execute(
            "SELECT * FROM lautools_location WHERE id = ?",
            (location_id,),
        ).fetchone()
        return self._row_to_location(row) if row else None

    def add_location(self, location: LautoolsLocation) -> LautoolsLocation:
        now = _iso(_now())
        self.connection.execute(
            """
            INSERT INTO lautools_location (
                name, disk_location, upstream, git_managed,
                git_root, git_subdir, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                location.name,
                str(location.disk_location) if location.disk_location else None,
                location.upstream,
                int(location.git_managed) if location.git_managed is not None else None,
                str(location.git_root) if location.git_root else None,
                location.git_subdir,
                _iso(location.created_at or _now()),
                now,
            ),
        )
        row = self.connection.execute(
            "SELECT * FROM lautools_location ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return self._row_to_location(row)

    def update_location(self, location: LautoolsLocation) -> LautoolsLocation:
        if location.id is None:
            raise ValueError("Location has not been saved")
        self.connection.execute(
            """
            UPDATE lautools_location
            SET name = ?, disk_location = ?, upstream = ?, git_managed = ?,
                git_root = ?, git_subdir = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                location.name,
                str(location.disk_location) if location.disk_location else None,
                location.upstream,
                int(location.git_managed) if location.git_managed is not None else None,
                str(location.git_root) if location.git_root else None,
                location.git_subdir,
                _iso(_now()),
                location.id,
            ),
        )
        row = self.connection.execute(
            "SELECT * FROM lautools_location WHERE id = ?",
            (location.id,),
        ).fetchone()
        return self._row_to_location(row)

    def remove_location(self, location_id: int) -> None:
        self.connection.execute(
            "DELETE FROM lautools_location WHERE id = ?",
            (location_id,),
        )

    def list_recipe_collections(self) -> list[LaupyRecipeCollection]:
        rows = self.connection.execute(
            """
            SELECT * FROM laupy_recipe_collections
            ORDER BY location_id
            """
        ).fetchall()
        return [self._row_to_recipe_collection(r) for r in rows]

    def get_recipe_collection(
        self,
        location_id: int,
    ) -> LaupyRecipeCollection | None:
        row = self.connection.execute(
            """
            SELECT * FROM laupy_recipe_collections
            WHERE location_id = ?
            """,
            (location_id,),
        ).fetchone()
        return self._row_to_recipe_collection(row) if row else None

    def upsert_recipe_collection(
        self,
        collection: LaupyRecipeCollection,
    ) -> LaupyRecipeCollection:
        self.connection.execute(
            """
            INSERT INTO laupy_recipe_collections (
                location_id, use_as_cookbook, use_as_workbench
            ) VALUES (?, ?, ?)
            ON CONFLICT(location_id) DO UPDATE SET
                use_as_cookbook = excluded.use_as_cookbook,
                use_as_workbench = excluded.use_as_workbench
            """,
            (
                collection.location_id,
                int(collection.use_as_cookbook),
                int(collection.use_as_workbench),
            ),
        )
        row = self.connection.execute(
            """
            SELECT * FROM laupy_recipe_collections
            WHERE location_id = ?
            """,
            (collection.location_id,),
        ).fetchone()
        return self._row_to_recipe_collection(row)

    def remove_recipe_collection(self, location_id: int) -> None:
        self.connection.execute(
            "DELETE FROM laupy_recipe_collections WHERE location_id = ?",
            (location_id,),
        )

    def list_recipe_instances_for_collection(
        self,
        collection_location_id: int,
    ) -> list[LaupyRecipeInstance]:
        rows = self.connection.execute(
            """
            SELECT * FROM laupy_recipe_instances
            WHERE collection_location_id = ?
            ORDER BY name COLLATE NOCASE, id
            """,
            (collection_location_id,),
        ).fetchall()
        return [self._row_to_recipe_instance(r) for r in rows]

    def get_recipe_instance(
        self,
        recipe_instance_id: int,
    ) -> LaupyRecipeInstance | None:
        row = self.connection.execute(
            "SELECT * FROM laupy_recipe_instances WHERE id = ?",
            (recipe_instance_id,),
        ).fetchone()
        return self._row_to_recipe_instance(row) if row else None

    def add_recipe_instance(
        self,
        recipe_instance: LaupyRecipeInstance,
    ) -> LaupyRecipeInstance:
        self.connection.execute(
            """
            INSERT INTO laupy_recipe_instances (
                collection_location_id, name, relative_path,
                source_instance_id, created_at, cloned_at, last_inspected
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                recipe_instance.collection_location_id,
                recipe_instance.name,
                str(recipe_instance.relative_path),
                recipe_instance.source_instance_id,
                _iso(recipe_instance.created_at or _now()),
                _iso(recipe_instance.cloned_at),
                _iso(recipe_instance.last_inspected),
            ),
        )
        row = self.connection.execute(
            "SELECT * FROM laupy_recipe_instances ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return self._row_to_recipe_instance(row)

    def update_recipe_instance(
        self,
        recipe_instance: LaupyRecipeInstance,
    ) -> LaupyRecipeInstance:
        if recipe_instance.id is None:
            raise ValueError("Recipe instance has not been saved")
        self.connection.execute(
            """
            UPDATE laupy_recipe_instances
            SET collection_location_id = ?, name = ?, relative_path = ?,
                source_instance_id = ?, cloned_at = ?, last_inspected = ?
            WHERE id = ?
            """,
            (
                recipe_instance.collection_location_id,
                recipe_instance.name,
                str(recipe_instance.relative_path),
                recipe_instance.source_instance_id,
                _iso(recipe_instance.cloned_at),
                _iso(recipe_instance.last_inspected),
                recipe_instance.id,
            ),
        )
        row = self.connection.execute(
            "SELECT * FROM laupy_recipe_instances WHERE id = ?",
            (recipe_instance.id,),
        ).fetchone()
        return self._row_to_recipe_instance(row)

    def get_config(self) -> LautoolsConfig:
        row = self.connection.execute(
            "SELECT * FROM lautools_config WHERE id = 1"
        ).fetchone()
        if row is None:
            return LautoolsConfig()
        return LautoolsConfig(
            default_cookbook_location_id=row["default_cookbook_location_id"],
            default_workbench_location_id=row["default_workbench_location_id"],
            updated_at=_parse_dt(row["updated_at"])
        )

    def set_config(self, config: LautoolsConfig) -> None:
        self.connection.execute(
            """
            INSERT INTO lautools_config (
                id, default_cookbook_location_id,
                default_workbench_location_id, updated_at
            ) VALUES (1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                default_cookbook_location_id =
                    excluded.default_cookbook_location_id,
                default_workbench_location_id =
                    excluded.default_workbench_location_id,
                updated_at = excluded.updated_at
            """,
            (
                config.default_cookbook_location_id,
                config.default_workbench_location_id,
                _iso(_now()),
            ),
        )
