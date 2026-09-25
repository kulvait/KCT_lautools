from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sqlite3


@dataclass
class Location:
    id: int
    name: str
    path: Path
    last_access: datetime | None
    description: str | None
    last_selected: bool


@dataclass
class ProjectCache:
    location_id: int
    beamtime_root: Path | None = None
    beamtime_id: str | None = None
    beamline: str | None = None
    beamline_alias: str | None = None
    beamline_setup: str | None = None
    facility: str | None = None
    proposal_id: str | None = None
    proposal_type: str | None = None
    event_start: str | None = None
    event_end: str | None = None
    generated: str | None = None
    applicant_username: str | None = None
    applicant_lastname: str | None = None
    applicant_institute: str | None = None
    applicant_email: str | None = None
    applicant_user_id: str | None = None
    raw_exists: bool | None = None
    processed_exists: bool | None = None
    scratch_cc_exists: bool | None = None
    shared_exists: bool | None = None
    raw_subdir_count: int | None = None
    raw_subdir_samples: list[str] | None = None
    project_size_bytes: int | None = None
    raw_size_bytes: int | None = None
    processed_size_bytes: int | None = None
    scratch_cc_size_bytes: int | None = None
    last_inspected: str | None = None


class LaupyDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS locations (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                path TEXT NOT NULL UNIQUE,
                last_access TEXT,
                description TEXT,
                last_selected INTEGER NOT NULL DEFAULT 0,
                last_working_directory TEXT
            )
        """)

        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS project_cache (
                location_id INTEGER PRIMARY KEY,
                beamtime_root TEXT,
                beamtime_id TEXT,
                beamline TEXT,
                beamline_alias TEXT,
                beamline_setup TEXT,
                facility TEXT,
                proposal_id TEXT,
                proposal_type TEXT,
                event_start TEXT,
                event_end TEXT,
                generated TEXT,
                applicant_username TEXT,
                applicant_lastname TEXT,
                applicant_institute TEXT,
                applicant_email TEXT,
                applicant_user_id TEXT,
                raw_exists INTEGER,
                processed_exists INTEGER,
                scratch_cc_exists INTEGER,
                shared_exists INTEGER,
                raw_subdir_count INTEGER,
                raw_subdir_samples TEXT,
                project_size_bytes INTEGER,
                raw_size_bytes INTEGER,
                processed_size_bytes INTEGER,
                scratch_cc_size_bytes INTEGER,
                last_inspected TEXT,
                FOREIGN KEY(location_id) REFERENCES locations(id)
                    ON DELETE CASCADE
            )
        """)

        # Handle databases created with older schemas.
        columns = {
            row[1]
            for row in self.connection.execute(
                "PRAGMA table_info(locations)"
            )
        }

        if "last_access" not in columns:
            self.connection.execute(
                "ALTER TABLE locations ADD COLUMN last_access TEXT"
            )

        if "description" not in columns:
            self.connection.execute(
                "ALTER TABLE locations ADD COLUMN description TEXT"
            )

        if "last_selected" not in columns:
            self.connection.execute(
                "ALTER TABLE locations "
                "ADD COLUMN last_selected INTEGER NOT NULL DEFAULT 0"
            )

        if "last_working_directory" not in columns:
            self.connection.execute(
                "ALTER TABLE locations "
                "ADD COLUMN last_working_directory TEXT"
            )

        self.connection.commit()

    def _row_to_location(self, row) -> Location:
        return Location(
            id=row[0],
            name=row[1],
            path=Path(row[2]),
            last_access=(
                datetime.fromisoformat(row[3])
                if row[3] is not None
                else None
            ),
            description=row[4],
            last_selected=bool(row[5]),
        )

    def _row_to_project_cache(self, row) -> ProjectCache:
        return ProjectCache(
            location_id=row[0],
            beamtime_root=Path(row[1]) if row[1] else None,
            beamtime_id=row[2],
            beamline=row[3],
            beamline_alias=row[4],
            beamline_setup=row[5],
            facility=row[6],
            proposal_id=row[7],
            proposal_type=row[8],
            event_start=row[9],
            event_end=row[10],
            generated=row[11],
            applicant_username=row[12],
            applicant_lastname=row[13],
            applicant_institute=row[14],
            applicant_email=row[15],
            applicant_user_id=row[16],
            raw_exists=bool(row[17]) if row[17] is not None else None,
            processed_exists=bool(row[18]) if row[18] is not None else None,
            scratch_cc_exists=bool(row[19]) if row[19] is not None else None,
            shared_exists=bool(row[20]) if row[20] is not None else None,
            raw_subdir_count=row[21],
            raw_subdir_samples=(
                json.loads(row[22]) if row[22] else None
            ),
            project_size_bytes=row[23],
            raw_size_bytes=row[24],
            processed_size_bytes=row[25],
            scratch_cc_size_bytes=row[26],
            last_inspected=row[27],
        )

    def list_locations(self) -> list[Location]:
        rows = self.connection.execute("""
            SELECT id, name, path, last_access, description, last_selected
            FROM locations
            ORDER BY name
        """).fetchall()

        return [self._row_to_location(row) for row in rows]

    def list_recent_locations(self) -> list[Location]:
        rows = self.connection.execute("""
            SELECT id, name, path, last_access, description, last_selected
            FROM locations
            ORDER BY
                last_access IS NULL,
                last_access DESC,
                name
        """).fetchall()

        return [self._row_to_location(row) for row in rows]

    def add_location(
        self,
        path: Path,
        name: str | None = None,
    ) -> Location | None:
        path = path.resolve()

        if name is None:
            name = path.name

        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO locations
                (name, path)
            VALUES (?, ?)
            """,
            (name, str(path)),
        )

        self.connection.commit()

        if cursor.rowcount == 0:
            return None

        return self.get_location_by_path(path)

    def get_location(self, location_id: int) -> Location:
        row = self.connection.execute("""
            SELECT id, name, path, last_access, description, last_selected
            FROM locations
            WHERE id = ?
        """, (location_id,)).fetchone()

        if row is None:
            raise ValueError(
                f"Location {location_id} does not exist"
            )

        return self._row_to_location(row)

    def get_location_by_path(self, path: Path) -> Location | None:
        row = self.connection.execute("""
            SELECT id, name, path, last_access, description, last_selected
            FROM locations
            WHERE path = ?
        """, (str(path.resolve()),)).fetchone()
        if row is None:
            return None
        return self._row_to_location(row)

    def get_last_selected_location(self) -> Location | None:
        row = self.connection.execute("""
            SELECT id, name, path, last_access, description, last_selected
            FROM locations
            WHERE last_selected = 1
            ORDER BY id DESC
            LIMIT 1
        """).fetchone()
        if row is None:
            return None
        return self._row_to_location(row)

    def get_working_directory(self, location_id: int) -> str | None:
        row = self.connection.execute(
            "SELECT last_working_directory FROM locations WHERE id = ?",
            (location_id,),
        ).fetchone()
        return row[0] if row else None

    def set_working_directory(self, location_id: int, wd_name: str | None):
        self.connection.execute(
            "UPDATE locations SET last_working_directory = ? WHERE id = ?",
            (wd_name, location_id),
        )
        self.connection.commit()

    def update_last_access(self, location_id: int):
        now = datetime.now().isoformat(timespec="seconds")

        self.connection.execute(
            """
            UPDATE locations
            SET last_access = ?
            WHERE id = ?
            """,
            (now, location_id),
        )

        self.connection.commit()

    def rename_location(self, location_id: int, name: str):
        name = name.strip()

        if not name:
            raise ValueError("Location name cannot be empty")

        self.connection.execute(
            """
            UPDATE locations
            SET name = ?
            WHERE id = ?
            """,
            (name, location_id),
        )

        self.connection.commit()

    def update_description(
        self,
        location_id: int,
        description: str | None,
    ):
        if description is not None:
            description = description.strip()

        if description == "":
            description = None

        self.connection.execute(
            """
            UPDATE locations
            SET description = ?
            WHERE id = ?
            """,
            (description, location_id),
        )

        self.connection.commit()

    def set_last_selected(self, location_id: int):
        self.connection.execute(
            "UPDATE locations SET last_selected = 0"
        )
        self.connection.execute(
            """
            UPDATE locations
            SET last_selected = 1
            WHERE id = ?
            """,
            (location_id,),
        )

        self.connection.commit()

    def remove_location(self, location_id: int):
        self.connection.execute(
            "DELETE FROM project_cache WHERE location_id = ?",
            (location_id,),
        )
        self.connection.execute(
            "DELETE FROM locations WHERE id = ?",
            (location_id,),
        )

        self.connection.commit()

    def get_project_cache(self, location_id: int) -> ProjectCache | None:
        row = self.connection.execute("""
            SELECT
                location_id,
                beamtime_root,
                beamtime_id,
                beamline,
                beamline_alias,
                beamline_setup,
                facility,
                proposal_id,
                proposal_type,
                event_start,
                event_end,
                generated,
                applicant_username,
                applicant_lastname,
                applicant_institute,
                applicant_email,
                applicant_user_id,
                raw_exists,
                processed_exists,
                scratch_cc_exists,
                shared_exists,
                raw_subdir_count,
                raw_subdir_samples,
                project_size_bytes,
                raw_size_bytes,
                processed_size_bytes,
                scratch_cc_size_bytes,
                last_inspected
            FROM project_cache
            WHERE location_id = ?
        """, (location_id,)).fetchone()

        if row is None:
            return None

        return self._row_to_project_cache(row)

    def upsert_project_cache(
        self,
        location_id: int,
        cache: ProjectCache,
    ):
        raw_subdir_samples_json = None
        if cache.raw_subdir_samples is not None:
            raw_subdir_samples_json = json.dumps(
                cache.raw_subdir_samples
            )

        self.connection.execute(
            """
            INSERT INTO project_cache (
                location_id,
                beamtime_root,
                beamtime_id,
                beamline,
                beamline_alias,
                beamline_setup,
                facility,
                proposal_id,
                proposal_type,
                event_start,
                event_end,
                generated,
                applicant_username,
                applicant_lastname,
                applicant_institute,
                applicant_email,
                applicant_user_id,
                raw_exists,
                processed_exists,
                scratch_cc_exists,
                shared_exists,
                raw_subdir_count,
                raw_subdir_samples,
                project_size_bytes,
                raw_size_bytes,
                processed_size_bytes,
                scratch_cc_size_bytes,
                last_inspected
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(location_id) DO UPDATE SET
                beamtime_root = excluded.beamtime_root,
                beamtime_id = excluded.beamtime_id,
                beamline = excluded.beamline,
                beamline_alias = excluded.beamline_alias,
                beamline_setup = excluded.beamline_setup,
                facility = excluded.facility,
                proposal_id = excluded.proposal_id,
                proposal_type = excluded.proposal_type,
                event_start = excluded.event_start,
                event_end = excluded.event_end,
                generated = excluded.generated,
                applicant_username = excluded.applicant_username,
                applicant_lastname = excluded.applicant_lastname,
                applicant_institute = excluded.applicant_institute,
                applicant_email = excluded.applicant_email,
                applicant_user_id = excluded.applicant_user_id,
                raw_exists = excluded.raw_exists,
                processed_exists = excluded.processed_exists,
                scratch_cc_exists = excluded.scratch_cc_exists,
                shared_exists = excluded.shared_exists,
                raw_subdir_count = excluded.raw_subdir_count,
                raw_subdir_samples = excluded.raw_subdir_samples,
                project_size_bytes = excluded.project_size_bytes,
                raw_size_bytes = excluded.raw_size_bytes,
                processed_size_bytes = excluded.processed_size_bytes,
                scratch_cc_size_bytes = excluded.scratch_cc_size_bytes,
                last_inspected = excluded.last_inspected
            """,
            (
                location_id,
                str(cache.beamtime_root) if cache.beamtime_root else None,
                cache.beamtime_id,
                cache.beamline,
                cache.beamline_alias,
                cache.beamline_setup,
                cache.facility,
                cache.proposal_id,
                cache.proposal_type,
                cache.event_start,
                cache.event_end,
                cache.generated,
                cache.applicant_username,
                cache.applicant_lastname,
                cache.applicant_institute,
                cache.applicant_email,
                cache.applicant_user_id,
                (
                    int(cache.raw_exists)
                    if cache.raw_exists is not None
                    else None
                ),
                (
                    int(cache.processed_exists)
                    if cache.processed_exists is not None
                    else None
                ),
                (
                    int(cache.scratch_cc_exists)
                    if cache.scratch_cc_exists is not None
                    else None
                ),
                (
                    int(cache.shared_exists)
                    if cache.shared_exists is not None
                    else None
                ),
                cache.raw_subdir_count,
                raw_subdir_samples_json,
                cache.project_size_bytes,
                cache.raw_size_bytes,
                cache.processed_size_bytes,
                cache.scratch_cc_size_bytes,
                cache.last_inspected,
            ),
        )

        self.connection.commit()

    def clear_project_cache(self, location_id: int):
        self.connection.execute(
            "DELETE FROM project_cache WHERE location_id = ?",
            (location_id,),
        )
        self.connection.commit()
