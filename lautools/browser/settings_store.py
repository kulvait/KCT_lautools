from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class LautoolsLocation:
    # Negative IDs identify unsaved dialog entries.
    id: int
    name: str
    disk_location: str | None = None
    upstream: str | None = None
    git_managed: bool | None = None
    git_root: str | None = None
    git_subdir: str | None = None


@dataclass
class FolderEntry:
    location: LautoolsLocation
    use_as_cookbook: bool = False
    use_as_workbench: bool = False
    is_in_path: bool = False
    path_position: int = 0


@dataclass
class LautoolsConfig:
    default_cookbook_location_id: int | None = None
    default_workbench_location_id: int | None = None




def load_settings(db) -> tuple[list[FolderEntry], LautoolsConfig]:
    rows = db.connection.execute(
        """
        SELECT l.*,
               COALESCE(p.use_as_cookbook, 0) AS use_as_cookbook,
               COALESCE(p.use_as_workbench, 0) AS use_as_workbench,
               (x.location_id IS NOT NULL) AS is_in_path,
               COALESCE(x.position, 0) AS path_position
        FROM lautools_location l
        LEFT JOIN laupy_recipe_collections p ON p.location_id = l.id
        LEFT JOIN lautools_paths x ON x.location_id = l.id
        ORDER BY l.name COLLATE NOCASE, l.id
        """
    ).fetchall()

    entries = [
        FolderEntry(
            location=LautoolsLocation(
                id=row["id"],
                name=row["name"],
                disk_location=row["disk_location"],
                upstream=row["upstream"],
                git_managed=(
                    bool(row["git_managed"])
                    if row["git_managed"] is not None else None
                ),
                git_root=row["git_root"],
                git_subdir=row["git_subdir"],
            ),
            use_as_cookbook=bool(row["use_as_cookbook"]),
            use_as_workbench=bool(row["use_as_workbench"]),
            is_in_path=bool(row["is_in_path"]),
            path_position=row["path_position"],
        )
        for row in rows
    ]

    row = db.connection.execute(
        "SELECT * FROM lautools_config WHERE id = 1"
    ).fetchone()
    config = (
        LautoolsConfig(
            row["default_cookbook_location_id"],
            row["default_workbench_location_id"],
        )
        if row else LautoolsConfig()
    )
    return entries, config


def save_settings(
    db,
    entries: list[FolderEntry],
    config: LautoolsConfig,
    removed_ids: set[int],
) -> None:
    """Save locations, memberships and defaults in one transaction."""
    by_id = {entry.location.id: entry for entry in entries}
    if len(by_id) != len(entries):
        raise ValueError("Duplicate location IDs.")

    for selected_id, role in (
        (config.default_cookbook_location_id, "use_as_cookbook"),
        (config.default_workbench_location_id, "use_as_workbench"),
    ):
        if selected_id is not None:
            entry = by_id.get(selected_id)
            if entry is None or not getattr(entry, role):
                raise ValueError("The selected default is not eligible.")

    for entry in entries:
        location = entry.location
        if not location.name.strip():
            raise ValueError("Each location needs a name.")
        if not location.disk_location and not location.upstream:
            raise ValueError("Each location needs a directory or upstream.")
        if entry.is_in_path and not location.disk_location:
            raise ValueError("PATH entries need a local directory.")

    now = datetime.now().isoformat(timespec="seconds")
    actual_ids: dict[int, int] = {}

    with db.transaction():
        # Temporarily clear defaults while memberships are changed.
        db.connection.execute(
            """
            UPDATE lautools_config
            SET default_cookbook_location_id = NULL,
                default_workbench_location_id = NULL
            WHERE id = 1
            """
        )

        for location_id in removed_ids:
            db.connection.execute(
                "DELETE FROM lautools_location WHERE id = ?",
                (location_id,),
            )

        for entry in entries:
            location = entry.location
            values = (
                location.name.strip(),
                location.disk_location,
                location.upstream,
                location.git_managed,
                location.git_root,
                location.git_subdir,
                now,
            )

            if location.id < 0:
                cursor = db.connection.execute(
                    """
                    INSERT INTO lautools_location (
                        name, disk_location, upstream, git_managed,
                        git_root, git_subdir, updated_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (*values, now),
                )
                location_id = cursor.lastrowid
            else:
                location_id = location.id
                cursor = db.connection.execute(
                    """
                    UPDATE lautools_location
                    SET name = ?, disk_location = ?, upstream = ?,
                        git_managed = ?, git_root = ?, git_subdir = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (*values, location_id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("A location was removed externally.")

            actual_ids[location.id] = location_id

            if entry.use_as_cookbook or entry.use_as_workbench:
                db.connection.execute(
                     """
                    INSERT INTO laupy_recipe_collections (
                        location_id, use_as_cookbook, use_as_workbench
                    ) VALUES (?, ?, ?)
                    ON CONFLICT(location_id) DO UPDATE SET
                        use_as_cookbook = excluded.use_as_cookbook,
                        use_as_workbench = excluded.use_as_workbench
                    """,
                    (
                        location_id,
                        entry.use_as_cookbook,
                        entry.use_as_workbench,
                    ),
                )
            else:
                db.connection.execute(
                    "DELETE FROM laupy_recipe_collections WHERE location_id = ?",
                    (location_id,),
                )

            if entry.is_in_path:
                db.connection.execute(
                    """
                    INSERT INTO lautools_paths (location_id, position)
                    VALUES (?, ?)
                    ON CONFLICT(location_id) DO UPDATE SET
                        position = excluded.position
                    """,
                    (location_id, entry.path_position),
                )
            else:
                db.connection.execute(
                    "DELETE FROM lautools_paths WHERE location_id = ?",
                    (location_id,),
                )

        db.connection.execute(
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
                actual_ids.get(config.default_cookbook_location_id),
                actual_ids.get(config.default_workbench_location_id),
                now,
            ),
        )
