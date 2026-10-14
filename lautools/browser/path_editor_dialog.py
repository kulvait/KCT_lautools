from datetime import datetime
from pathlib import Path
import os
import sqlite3

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QFileDialog,
    QLabel, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout,
)


class PathEditorDialog(QDialog):
    def __init__(self, db, project_id=None, parent=None):
        super().__init__(parent)
        self.db = db
        self.project_id = project_id

        self.setWindowTitle("PATH — Global and project")
        self.resize(800, 450)
        layout = QVBoxLayout(self)
        note = QLabel(
            "Global: applies to every project.\n"
            "This project: adds the directory only to the selected project.\n"
            "Both may be checked. Uncheck both to remove PATH membership.\n"
            "Global changes affect all projects; inherited system PATH is unchanged."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            ["Directory", "Global", "This project"]
        )
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        rows = db.connection.execute(
            """
            SELECT l.id, l.disk_location,
                   (g.location_id IS NOT NULL) AS global_member,
                   (p.location_id IS NOT NULL) AS local_member
            FROM lautools_location l
            LEFT JOIN lautools_paths g ON g.location_id = l.id
            LEFT JOIN lautools_project_paths p
                ON p.location_id = l.id AND p.project_id = ?
            WHERE NULLIF(TRIM(l.disk_location), '') IS NOT NULL
            ORDER BY l.disk_location, l.id
            """,
            (project_id,),
        ).fetchall()
        for row in rows:
            self._add_row(
                row["id"], row["disk_location"],
                bool(row["global_member"]), bool(row["local_member"]),
            )

        add = QPushButton("Add directory…")
        add.clicked.connect(self._browse)
        layout.addWidget(add)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.table.resizeColumnsToContents()

    def _add_row(self, location_id, directory, global_member, local_member):
        row = self.table.rowCount()
        self.table.insertRow(row)

        item = QTableWidgetItem(directory)
        item.setData(Qt.UserRole, location_id)
        self.table.setItem(row, 0, item)

        for column, checked in ((1, global_member), (2, local_member)):
            item = QTableWidgetItem()
            flags = Qt.ItemIsSelectable
            if column == 1 or self.project_id is not None:
                flags |= Qt.ItemIsEnabled | Qt.ItemIsUserCheckable
            item.setFlags(flags)
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.table.setItem(row, column, item)

    def _browse(self):
        directory = QFileDialog.getExistingDirectory(self, "Add PATH directory")
        if not directory:
            return
        directory = str(Path(directory).resolve())

        for row in range(self.table.rowCount()):
            existing = self.table.item(row, 0).text()
            if Path(existing).expanduser().resolve() == Path(directory):
                self.table.selectRow(row)
                return

        self._add_row(
            None, directory,
            global_member=self.project_id is None,
            local_member=self.project_id is not None,
        )

    def _save(self):
        try:
            with self.db.transaction():
                for row in range(self.table.rowCount()):
                    item = self.table.item(row, 0)
                    location_id = item.data(Qt.UserRole)
                    directory = item.text()
                    global_member = (
                        self.table.item(row, 1).checkState() == Qt.Checked
                    )
                    local_member = (
                        self.project_id is not None
                        and self.table.item(row, 2).checkState() == Qt.Checked
                    )

                    if global_member or local_member:
                        path = Path(directory).expanduser()
                        if (
                            not path.is_absolute()
                            or os.pathsep in directory
                            or "\n" in directory
                            or "\r" in directory
                        ):
                            raise ValueError(f"Invalid PATH directory: {directory}")
                        if path.exists() and not path.is_dir():
                            raise ValueError(f"Not a directory: {directory}")

                    if location_id is None:
                        if not (global_member or local_member):
                            continue
                        now = datetime.now().isoformat(timespec="seconds")
                        cursor = self.db.connection.execute(
                            """
                            INSERT INTO lautools_location
                                (name, disk_location, created_at, updated_at)
                            VALUES (?, ?, ?, ?)
                            """,
                            (Path(directory).name or directory, directory, now, now),
                        )
                        location_id = cursor.lastrowid

                    # Modify memberships only; never delete the location itself.
                    for table, enabled, project_id in (
                        ("lautools_paths", global_member, None),
                        ("lautools_project_paths", local_member, self.project_id),
                    ):
                        if table == "lautools_project_paths" and project_id is None:
                            continue

                        scope = "" if project_id is None else "project_id = ? AND "
                        args = () if project_id is None else (project_id,)

                        if enabled:
                            # Append new memberships; retain existing positions.
                            columns = "location_id, position"
                            values = (location_id,)
                            where = ""
                            if project_id is not None:
                                columns = "project_id, " + columns
                                values = (project_id, location_id)
                                where = " WHERE project_id = ?"
                            placeholders = ", ".join("?" for _ in values)
                            self.db.connection.execute(
                                f"""
                                INSERT INTO {table} ({columns})
                                SELECT {placeholders}, COALESCE(MAX(position), -1) + 1
                                FROM {table}{where}
                                ON CONFLICT DO NOTHING
                                """.replace(
                                    f"FROM {table}\n",
                                    f"FROM {table} WHERE 1\n",
                                ),
                                (*values, *args),
                            )
                        else:
                            self.db.connection.execute(
                                f"DELETE FROM {table} WHERE {scope}location_id = ?",
                                (*args, location_id),
                            )
        except (ValueError, OSError, sqlite3.Error) as exc:
            QMessageBox.warning(self, "Cannot save PATH", str(exc))
            return

        self.accept()
