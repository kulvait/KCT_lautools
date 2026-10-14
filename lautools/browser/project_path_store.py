from __future__ import annotations

from dataclasses import dataclass

from lautools.browser.settings_store import LautoolsLocation


@dataclass
class ProjectPathEntry:
    location: LautoolsLocation
    is_global_path: bool = False
    is_project_path: bool = False
    global_path_position: int = 0
    project_path_position: int = 0


def load_project_path_settings(db, project_id: int) -> list[ProjectPathEntry]:
    rows = db.connection.execute(
        """
        SELECT l.*,
               (gp.location_id IS NOT NULL) AS is_global_path,
               COALESCE(gp.position, 0) AS global_path_position,
               (pp.location_id IS NOT NULL) AS is_project_path,
               COALESCE(pp.position, 0) AS project_path_position
        FROM lautools_location l
        LEFT JOIN lautools_paths gp ON gp.location_id = l.id
        LEFT JOIN lautools_project_paths pp
            ON pp.location_id = l.id AND pp.project_id = ?
        ORDER BY l.name COLLATE NOCASE, l.id
        """,
        (project_id,),
    ).fetchall()

    return [
        ProjectPathEntry(
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
            is_global_path=bool(row["is_global_path"]),
            is_project_path=bool(row["is_project_path"]),
            global_path_position=row["global_path_position"],
            project_path_position=row["project_path_position"],
        )
        for row in rows
    ]


def save_project_path_settings(db, project_id: int, entries: list[ProjectPathEntry]) -> None:
    with db.transaction():
        db.connection.execute(
            "DELETE FROM lautools_project_paths WHERE project_id = ?",
            (project_id,),
        )
        for entry in entries:
            if entry.is_project_path:
                db.connection.execute(
                    """
                    INSERT INTO lautools_project_paths (
                        project_id, location_id, position
                    ) VALUES (?, ?, ?)
                    """,
                    (project_id, entry.location.id, entry.project_path_position),
                )
