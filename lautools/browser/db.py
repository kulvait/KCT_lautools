from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sqlite3


@dataclass
class Location:
    id: int
    name: str
    path: Path
    last_access: datetime | None


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
                last_access TEXT
            )
        """)

        # Handle databases created with the old schema.
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

        self.connection.commit()

    def list_locations(self) -> list[Location]:
        rows = self.connection.execute("""
            SELECT id, name, path, last_access
            FROM locations
            ORDER BY name
        """).fetchall()

        return [
            Location(
                id=row[0],
                name=row[1],
                path=Path(row[2]),
                last_access=(
                    datetime.fromisoformat(row[3])
                    if row[3] is not None
                    else None
                ),
            )
            for row in rows
        ]

    def list_recent_locations(self) -> list[Location]:
        rows = self.connection.execute("""
            SELECT id, name, path, last_access
            FROM locations
            ORDER BY
                last_access IS NULL,
                last_access DESC
        """).fetchall()

        return [
            Location(
                id=row[0],
                name=row[1],
                path=Path(row[2]),
                last_access=(
                    datetime.fromisoformat(row[3])
                    if row[3] is not None
                    else None
                ),
            )
            for row in rows
        ]

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
            SELECT id, name, path, last_access
            FROM locations
            WHERE id = ?
        """, (location_id,)).fetchone()

        if row is None:
            raise ValueError(
                f"Location {location_id} does not exist"
            )

        return Location(
            id=row[0],
            name=row[1],
            path=Path(row[2]),
            last_access=(
                datetime.fromisoformat(row[3])
                if row[3] is not None
                else None
            ),
        )

    def get_location_by_path(self, path: Path) -> Location | None:
        row = self.connection.execute("""
            SELECT id, name, path, last_access
            FROM locations
            WHERE path = ?
        """, (str(path.resolve()),)).fetchone()

        if row is None:
            return None

        return Location(
            id=row[0],
            name=row[1],
            path=Path(row[2]),
            last_access=(
                datetime.fromisoformat(row[3])
                if row[3] is not None
                else None
            ),
        )

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

    def remove_location(self, location_id: int):
        self.connection.execute(
            "DELETE FROM locations WHERE id = ?",
            (location_id,),
        )

        self.connection.commit()
