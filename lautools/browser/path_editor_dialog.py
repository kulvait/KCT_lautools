from pathlib import Path
import os
import sqlite3

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QFileDialog,
    QLabel, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout,
)
from lautools.browser.path_manager import PathManager

class PathEditorDialog(QDialog):
    def __init__(self, db, project_id=None, parent=None):
        super().__init__(parent)
        self.db = db
        self.project_id = project_id
        self.path_manager = PathManager(db)

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
        self.entries = self.path_manager.list_entries(project_id)
        for entry in self.entries:
             self._add_row(
                entry.location.id,
                str(entry.location.disk_location),
                entry.is_global,
                entry.is_project,
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
        self.entries = self.path_manager.add_directory(
            self.entries,
            directory,
            self.project_id,
        )
        self._add_row(
            self.entries[-1].location.id,
            str(self.entries[-1].location.disk_location),
            self.entries[-1].is_global,
            self.entries[-1].is_project,
        )

    def _save(self):
        try:
            for row, entry in enumerate(self.entries):
                entry.is_global = (
                    self.table.item(row, 1).checkState() == Qt.Checked
                )
                entry.is_project = (
                    self.project_id is not None
                    and self.table.item(row, 2).checkState() == Qt.Checked
                )
            self.path_manager.save_memberships(self.entries, self.project_id)
        except (ValueError, OSError, sqlite3.Error) as exc:
            QMessageBox.warning(self, "Cannot save PATH", str(exc))
            return

        self.accept()
