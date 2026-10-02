from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os
import sys

from git import Repo
from git.exc import GitError, InvalidGitRepositoryError
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from lautools.browser.settings_store import (
    FolderEntry, LautoolsConfig, LautoolsLocation,
    load_settings, save_settings,
)


DEFAULT_UPSTREAM = (
    "https://github.com/kulvait/KCT_laupy_pipelines.git"
)


def inspect_git(location: LautoolsLocation) -> tuple[LautoolsLocation, str]:
    """Return a snapshot without modifying Git configuration or upstream."""
    updated = replace(
        location, git_managed=None, git_root=None, git_subdir=None
    )
    if not location.disk_location:
        return updated, "No local directory."

    path = Path(location.disk_location).expanduser()
    try:
        if not path.is_dir():
            return updated, "Directory unavailable."
        repo = Repo(path, search_parent_directories=True)
    except InvalidGitRepositoryError:
        return replace(updated, git_managed=False), "Not Git-managed."
    except (GitError, OSError, ValueError) as exc:
        return updated, f"Inspection failed: {exc}"

    try:
        if repo.bare or not repo.working_tree_dir:
            return updated, "No usable Git working tree."

        root = Path(repo.working_tree_dir).resolve()
        subdir = path.resolve().relative_to(root).as_posix()
        updated = replace(
            updated,
            git_managed=True,
            git_root=str(root),
            git_subdir=subdir,
        )
        origin = next(
            (remote for remote in repo.remotes if remote.name == "origin"),
            None,
        )
        urls = list(origin.urls) if origin is not None else []
        description = (
            f"Repository: {root}\n"
            f"Within repository: {subdir}\n"
            f"Detected origin: {', '.join(urls) or '(none)'}"
        )
        if location.upstream and urls and location.upstream not in urls:
            description += "\nConfigured upstream differs from origin."
        return updated, description
    except (GitError, OSError, ValueError) as exc:
        return updated, f"Inspection failed: {exc}"
    finally:
        repo.close()


class LocationEditor(QDialog):
    def __init__(self, location: LautoolsLocation, parent=None):
        super().__init__(parent)
        self.location = replace(location)
        self.setWindowTitle("Edit location")
        self.resize(650, 250)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(location.name)
        self.path_edit = QLineEdit(location.disk_location or "")
        self.upstream_edit = QLineEdit(location.upstream or "")
        self.upstream_edit.setPlaceholderText("Optional Git repository URL")

        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(self.path_edit)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        path_layout.addWidget(browse)

        form.addRow("Name:", self.name_edit)
        form.addRow("Directory:", path_row)
        form.addRow("Configured upstream:", self.upstream_edit)
        layout.addLayout(form)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.PlainText)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)

        inspect_button = QPushButton("Inspect Git")
        inspect_button.clicked.connect(self._inspect)
        layout.addWidget(inspect_button)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._inspect()

    def _browse(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Choose directory",
            self.path_edit.text() or str(Path.home()),
        )
        if directory:
            self.path_edit.setText(directory)
            self._inspect()

    def _read_fields(self):
        text = self.path_edit.text().strip()
        self.location = replace(
            self.location,
            name=self.name_edit.text().strip(),
            disk_location=(
                os.path.abspath(os.path.expanduser(text)) if text else None
            ),
            upstream=self.upstream_edit.text().strip() or None,
        )

    def _inspect(self):
        self._read_fields()
        self.location, description = inspect_git(self.location)
        self.status.setText(description)

    def _accept(self):
        self._inspect()
        if not self.location.name:
            QMessageBox.warning(self, "Name required", "Enter a name.")
            return
        if not self.location.disk_location and not self.location.upstream:
            QMessageBox.warning(
                self, "Location required",
                "Enter a directory or an upstream URL.",
            )
            return
        self.accept()


class DefaultSetupThread(QThread):
    """Filesystem operations only; never accesses the SQLite connection."""
    completed = Signal()
    failed = Signal(str)

    def __init__(self, root: Path, parent=None):
        super().__init__(parent)
        self.root = root

    def run(self):
        recipe = self.root / "laupy-pipelines"
        workbench = self.root / "workbench"
        try:
            # lexists also detects dangling symlinks.
            if os.path.lexists(recipe) or os.path.lexists(workbench):
                raise ValueError(
                    "A default destination already exists. "
                    "Register it manually; nothing will be overwritten."
                )
            repo = Repo.clone_from(DEFAULT_UPSTREAM, recipe)
            repo.close()
            repo = Repo.init(workbench)
            repo.close()
        except Exception as exc:
            self.failed.emit(
                f"{exc}\n\n"
                "Any created directories have been left on disk. "
                "No database defaults were saved."
            )
        else:
            self.completed.emit()


class SettingsDialog(QDialog):
    NAME, DIRECTORY, UPSTREAM, GIT, RECIPE, WORKBENCH, PATH = range(7)

    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.db = db
        self.entries, self.config = load_settings(db)
        self.original_ids = {
            entry.location.id for entry in self.entries
        }
        self.next_id = -1
        self._setup_thread = None
        self._busy = False

        self.setWindowTitle("Settings — Lautools")
        self.resize(1100, 600)
        outer = QVBoxLayout(self)

        # Disable only the editable content during default setup.
        self.content = QWidget()
        layout = QVBoxLayout(self.content)
        outer.addWidget(self.content)

        environment = QGroupBox("Running Python environment")
        form = QFormLayout(environment)
        for title, value in (
            ("Environment root:", sys.prefix),
            ("Base environment:", sys.base_prefix),
            ("Interpreter:", sys.executable),
            ("Database:", str(db.db_path)),
        ):
            label = QLabel(value)
            label.setTextFormat(Qt.PlainText)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            form.addRow(title, label)
        layout.addWidget(environment)

        defaults = QGroupBox("Default project folders")
        form = QFormLayout(defaults)
        self.recipe_combo = QComboBox()
        self.workbench_combo = QComboBox()
        form.addRow("Recipes:", self.recipe_combo)
        form.addRow("Workbench:", self.workbench_combo)
        self.recipe_combo.currentIndexChanged.connect(self._defaults_changed)
        self.workbench_combo.currentIndexChanged.connect(
            self._defaults_changed
        )
        layout.addWidget(defaults)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Name", "Directory", "Upstream", "Git",
            "Recipe", "Workbench", "In PATH",
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self._role_changed)
        layout.addWidget(self.table, 1)

        actions = QHBoxLayout()
        for title, callback in (
            ("Add...", self._add),
            ("Edit...", self._edit),
            ("Remove", self._remove),
            ("Inspect Git", self._inspect),
            ("Set up defaults...", self._setup_defaults),
        ):
            button = QPushButton(title)
            button.clicked.connect(callback)
            actions.addWidget(button)
        layout.addLayout(actions)

        note = QLabel(
            "Changes are saved only with Save. Removing a registration "
            "does not delete its directory. 'In PATH' records membership; "
            "it does not change the application's PATH yet."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        self.buttons.accepted.connect(self._save)
        self.buttons.rejected.connect(self.reject)
        outer.addWidget(self.buttons)
        self._render()

    def _render(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.entries))
        for row, entry in enumerate(self.entries):
            location = entry.location
            git_text = (
                "Unknown" if location.git_managed is None
                else "Yes" if location.git_managed else "No"
            )
            for column, text in enumerate((
                location.name,
                location.disk_location or "",
                location.upstream or "",
                git_text,
            )):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                if column == self.GIT:
                    item.setToolTip(
                        f"Root: {location.git_root or '(unknown)'}\n"
                        f"Subdirectory: {location.git_subdir or '(unknown)'}"
                    )
                self.table.setItem(row, column, item)

            for column, checked in (
                (self.RECIPE, entry.use_as_recipe),
                (self.WORKBENCH, entry.use_as_workbench),
                (self.PATH, entry.is_in_path),
            ):
                item = QTableWidgetItem()
                item.setFlags(
                    Qt.ItemIsEnabled | Qt.ItemIsSelectable
                    | Qt.ItemIsUserCheckable
                )
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                self.table.setItem(row, column, item)
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()
        self._refresh_defaults()

    def _refresh_defaults(self):
        for combo, role, attribute in (
            (self.recipe_combo, "use_as_recipe",
             "default_recipe_location_id"),
            (self.workbench_combo, "use_as_workbench",
             "default_workbench_location_id"),
        ):
            selected_id = getattr(self.config, attribute)
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(Not configured)", None)
            for entry in self.entries:
                if getattr(entry, role):
                    location = entry.location
                    combo.addItem(
                        f"{location.name} — "
                        f"{location.disk_location or '(remote only)'}",
                        location.id,
                    )
            index = combo.findData(selected_id)
            combo.setCurrentIndex(index if index >= 0 else 0)
            setattr(self.config, attribute, combo.currentData())
            combo.blockSignals(False)

    def _defaults_changed(self, *_):
        self.config.default_recipe_location_id = (
            self.recipe_combo.currentData()
        )
        self.config.default_workbench_location_id = (
            self.workbench_combo.currentData()
        )

    def _role_changed(self, item):
        entry = self.entries[item.row()]
        checked = item.checkState() == Qt.Checked
        if item.column() == self.RECIPE:
            entry.use_as_recipe = checked
        elif item.column() == self.WORKBENCH:
            entry.use_as_workbench = checked
        elif item.column() == self.PATH:
            entry.is_in_path = checked
        self._refresh_defaults()

    def _new_location(self, name="", directory=None, upstream=None):
        location = LautoolsLocation(
            self.next_id, name, directory, upstream
        )
        self.next_id -= 1
        return location

    def _selected_entry(self):
        row = self.table.currentRow()
        return self.entries[row] if row >= 0 else None

    def _add(self):
        editor = LocationEditor(self._new_location(), self)
        if editor.exec() == QDialog.Accepted:
            self.entries.append(FolderEntry(editor.location))
            self._render()

    def _edit(self):
        entry = self._selected_entry()
        if entry is None:
            return
        editor = LocationEditor(entry.location, self)
        if editor.exec() == QDialog.Accepted:
            entry.location = editor.location
            self._render()

    def _remove(self):
        entry = self._selected_entry()
        if entry is None:
            return
        answer = QMessageBox.question(
            self, "Remove registration?",
            f"Remove '{entry.location.name}' from the database registry?\n"
            "Its files will not be deleted.",
        )
        if answer == QMessageBox.Yes:
            self.entries.remove(entry)
            self._render()

    def _inspect(self):
        entry = self._selected_entry()
        if entry is None:
            return
        entry.location, description = inspect_git(entry.location)
        self._render()
        message = QMessageBox(self)
        message.setWindowTitle("Git inspection")
        message.setTextFormat(Qt.PlainText)
        message.setText(description)
        message.exec()

    def _setup_defaults(self):
        # First-install setup only; never replaces existing defaults.
        if (
            self.config.default_recipe_location_id is not None
            or self.config.default_workbench_location_id is not None
        ):
            QMessageBox.warning(
                self, "Defaults already selected",
                "Default setup is for an unconfigured installation.",
            )
            return

        root = Path(self.db.db_path).expanduser().resolve().parent
        recipe = root / "laupy-pipelines"
        workbench = root / "workbench"

        if os.path.lexists(recipe) or os.path.lexists(workbench):
            QMessageBox.warning(
                self, "Destinations exist",
                "Register the existing directories with Add instead. "
                "Default setup will not overwrite or reuse them.",
            )
            return

        message = QMessageBox(self)
        message.setWindowTitle("Set up default folders?")
        message.setTextFormat(Qt.PlainText)
        message.setText(
            f"Clone:\n{DEFAULT_UPSTREAM}\ninto:\n{recipe}\n\n"
            f"Initialize an empty Git workbench:\n{workbench}\n\n"
            "Directories are created immediately. Database registration "
            "is staged until Save; Cancel will not delete the directories."
        )
        message.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        message.setDefaultButton(QMessageBox.No)
        if message.exec() != QMessageBox.Yes:
            return

        self._busy = True
        self.content.setEnabled(False)
        self.buttons.setEnabled(False)
        self._setup_thread = DefaultSetupThread(root, self)
        self._setup_thread.completed.connect(self._defaults_created)
        self._setup_thread.failed.connect(self._setup_failed)
        self._setup_thread.finished.connect(self._setup_finished)
        self._setup_thread.start()

    def _defaults_created(self):
        root = Path(self.db.db_path).expanduser().resolve().parent
        recipe, _ = inspect_git(self._new_location(
            "Default recipes",
            str(root / "laupy-pipelines"),
            DEFAULT_UPSTREAM,
        ))
        workbench, _ = inspect_git(self._new_location(
            "Private workbench", str(root / "workbench")
        ))
        self.entries.extend([
            FolderEntry(recipe, use_as_recipe=True),
            FolderEntry(workbench, use_as_workbench=True),
        ])
        self.config = LautoolsConfig(recipe.id, workbench.id)
        self._render()

    def _setup_failed(self, description):
        message = QMessageBox(self)
        message.setWindowTitle("Default setup failed")
        message.setTextFormat(Qt.PlainText)
        message.setText(description)
        message.exec()

    def _setup_finished(self):
        self._busy = False
        self.content.setEnabled(True)
        self.buttons.setEnabled(True)

    def _save(self):
        current_ids = {
            entry.location.id
            for entry in self.entries
            if entry.location.id > 0
        }
        try:
            save_settings(
                self.db, self.entries, self.config,
                self.original_ids - current_ids,
            )
        except Exception as exc:
            message = QMessageBox(self)
            message.setWindowTitle("Settings not saved")
            message.setTextFormat(Qt.PlainText)
            message.setText(str(exc))
            message.exec()
            return
        self.accept()

    def reject(self):
        if not self._busy:
            super().reject()

    def closeEvent(self, event):
        if self._busy:
            event.ignore()
        else:
            super().closeEvent(event)
