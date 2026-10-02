from __future__ import annotations

from dataclasses import replace
from itertools import count
import logging
import os
from pathlib import Path
import re

from PySide6.QtCore import QThread, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from lautools.browser.settings_store import load_settings
from lautools.project_creator import (
    INFO_FILENAME,
    RESERVED_BEAMTIME_NAMES,
    create_project_from_recipe,
    validate_recipe_copy,
)


log = logging.getLogger(__name__)


def _slug(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "project"


def _default_project_basename(beamtime, scratch: Path) -> str:
    label = beamtime.label or beamtime.beamtime_id or "project"
    base = f"kct_{_slug(str(label))}"

    if not os.path.lexists(scratch / base):
        return base

    # Minimum three digits, with no artificial upper limit.
    for index in count(1):
        candidate = f"{base}_{index:03d}"
        if not os.path.lexists(scratch / candidate):
            return candidate


def _validate_recipe_entries(recipe: Path) -> None:
    """Preflight checks before copying or modifying INFO."""
    reserved = [
        name
        for name in RESERVED_BEAMTIME_NAMES
        if os.path.lexists(recipe / name)
    ]
    if reserved:
        raise ValueError(
            "The recipe contains reserved top-level names:\n"
            + ", ".join(reserved)
            + "\n\nRemove these entries from the recipe. "
            "The project uses actual beamtime storage for these names."
        )

    info = recipe / INFO_FILENAME
    if os.path.lexists(info):
        if info.is_symlink() or not info.is_file():
            raise ValueError(
                f"Recipe INFO must be a regular file, not a symlink "
                f"or directory:\n{info}"
            )


class _CreationThread(QThread):
    """Filesystem operations only; no database access."""

    def __init__(self, arguments: dict, parent=None):
        super().__init__(parent)
        self.arguments = arguments
        self.result: tuple[Path, Path] | None = None
        self.error: str | None = None

    def run(self):
        try:
            # Repeat preflight in case the recipe changed after confirmation.
            _validate_recipe_entries(self.arguments["recipe"])
            self.result = create_project_from_recipe(**self.arguments)
        except Exception as exc:
            log.exception("Recipe project creation failed")
            self.error = str(exc)


class NewProjectFromRecipeDialog(QDialog):
    """Choose a recipe and destinations; delegate creation to the Python API."""

    def __init__(
        self,
        project_manager,
        beamtime,
        scratch,
        size_service=None,
        parent=None,
    ):
        super().__init__(parent)

        # Snapshot the database model; do not pass a database connection
        # or project_manager to the worker.
        self._beamtime = replace(beamtime)
        self._scratch = Path(scratch).expanduser().resolve()

        # Retained for compatibility with BrowserWindow's constructor call.
        # size_service is not needed for project creation.
        self._recipes_root: Path | None = None
        self._workbench: Path | None = None
        self._project_path: Path | None = None
        self._workbench_copy: Path | None = None
        self._worker: _CreationThread | None = None
        self._busy = False
        self._automatic_copy_name = True

        self.setWindowTitle(
            f"New project from recipe — {beamtime.beamtime_id}"
        )
        self.resize(800, 450)

        outer = QVBoxLayout(self)
        self.content = QWidget()
        layout = QFormLayout(self.content)
        outer.addWidget(self.content)

        self.recipe_combo = QComboBox()
        layout.addRow("Recipe:", self.recipe_combo)

        self.recipes_label = self._path_label()
        self.workbench_label = self._path_label()
        layout.addRow("Recipes directory:", self.recipes_label)
        layout.addRow("Workbench:", self.workbench_label)

        self.project_edit = QLineEdit()
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_project)

        project_row = QWidget()
        project_layout = QHBoxLayout(project_row)
        project_layout.setContentsMargins(0, 0, 0, 0)
        project_layout.addWidget(self.project_edit)
        project_layout.addWidget(browse)
        layout.addRow("Scratch project directory:", project_row)

        self.copy_name_edit = QLineEdit()
        self.copy_name_edit.textEdited.connect(self._copy_name_edited)
        layout.addRow("Workbench copy name:", self.copy_name_edit)

        self.preview = self._path_label()
        layout.addRow("Copy destination:", self.preview)

        explanation = QLabel(
            "The recipe is copied into the workbench. Its top-level "
            "entries are linked into the scratch project.\n\n"
            "Reserved names raw, processed, scratch_cc and shared must "
            "not occur in the recipe; links to existing beamtime storage "
            "directories are created instead.\n\n"
            "INFO is created or prepended with project metadata in the "
            "workbench copy, then linked into the project. Other existing "
            "recipe symlinks remain symlinks; their targets are not copied."
        )
        explanation.setWordWrap(True)
        layout.addRow(explanation)

        self.status = self._path_label()
        outer.addWidget(self.status)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.buttons.button(QDialogButtonBox.Ok).setText("Create project")
        self.buttons.accepted.connect(self._create)
        self.buttons.rejected.connect(self.reject)
        outer.addWidget(self.buttons)

        self.recipe_combo.currentIndexChanged.connect(self._update_name)
        self.project_edit.textChanged.connect(self._update_name)
        self.copy_name_edit.textChanged.connect(self._update_preview)

        try:
            if self._beamtime.core_path is None:
                raise ValueError("The beamtime has no known core path.")

            core = Path(self._beamtime.core_path).expanduser().resolve(
                strict=True
            )
            expected_scratch = (core / "scratch_cc").resolve(strict=True)

            if not self._scratch.is_dir():
                raise ValueError(
                    f"Scratch directory is unavailable:\n{self._scratch}"
                )
            if self._scratch != expected_scratch:
                raise ValueError(
                    "Scratch directory does not match the selected beamtime."
                )

            self._beamtime = replace(self._beamtime, core_path=core)
            basename = _default_project_basename(
                self._beamtime, self._scratch
            )
            self.project_edit.setText(str(self._scratch / basename))

            entries, config = load_settings(project_manager.db)
            by_id = {entry.location.id: entry for entry in entries}

            def default_directory(location_id, role, title):
                entry = by_id.get(location_id)
                if (
                    entry is None
                    or not getattr(entry, role)
                    or not entry.location.disk_location
                ):
                    raise ValueError(
                        f"Configure a default {title} directory in Settings."
                    )

                directory = Path(
                    entry.location.disk_location
                ).expanduser().resolve(strict=True)

                if not directory.is_dir():
                    raise ValueError(f"Not a directory:\n{directory}")
                return directory

            self._recipes_root = default_directory(
                config.default_recipe_location_id,
                "use_as_recipe",
                "recipe",
            )
            self._workbench = default_directory(
                config.default_workbench_location_id,
                "use_as_workbench",
                "workbench",
            )

            self.recipes_label.setText(str(self._recipes_root))
            self.workbench_label.setText(str(self._workbench))

            recipes = sorted(
                (
                    path
                    for path in self._recipes_root.iterdir()
                    if not path.name.startswith(".")
                    and not path.is_symlink()
                    and path.is_dir()
                ),
                key=lambda path: path.name.casefold(),
            )

            for recipe in recipes:
                self.recipe_combo.addItem(recipe.name, recipe)

            if not recipes:
                raise ValueError(
                    "No recipe subfolders found in the default "
                    "recipes directory."
                )

            self._update_name()

        except Exception as exc:
            self.status.setText(str(exc))
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)

    @staticmethod
    def _path_label():
        label = QLabel()
        label.setWordWrap(True)
        label.setTextFormat(Qt.PlainText)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return label

    def selected_project_path(self) -> Path | None:
        return self._project_path

    def selected_workbench_path(self) -> Path | None:
        return self._workbench_copy

    def _browse_project(self):
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select or create an empty project directory inside scratch_cc",
            str(self._scratch),
        )
        if selected:
            self.project_edit.setText(selected)

    def _copy_name_edited(self, _text):
        self._automatic_copy_name = False

    def _update_name(self, *_):
        recipe = self.recipe_combo.currentData()
        if self._automatic_copy_name and recipe is not None:
            project_name = Path(self.project_edit.text().strip()).name
            self.copy_name_edit.setText(
                f"{project_name}_{recipe.name}"
            )
        self._update_preview()

    def _update_preview(self, *_):
        if self._workbench is not None:
            self.preview.setText(
                str(self._workbench / self.copy_name_edit.text())
            )

    def _message(self, title, text):
        message = QMessageBox(self)
        message.setWindowTitle(title)
        message.setTextFormat(Qt.PlainText)
        message.setText(text)
        message.exec()

    def _create(self):
        recipe = self.recipe_combo.currentData()
        project_text = self.project_edit.text().strip()

        if (
            recipe is None
            or not project_text
            or self._recipes_root is None
            or self._workbench is None
        ):
            self._message(
                "Selection required",
                "Choose a recipe and project directory.",
            )
            return

        project = Path(project_text).expanduser()
        if not project.is_absolute():
            project = self._scratch / project

        arguments = {
            "beamtime": self._beamtime,
            "recipes_root": self._recipes_root,
            "recipe": recipe,
            "workbench": self._workbench,
            "project_dir": project,
            "copy_name": self.copy_name_edit.text(),
        }

        try:
            recipe, destination, project = validate_recipe_copy(
                **arguments
            )
            _validate_recipe_entries(recipe)

            # Use actual filesystem state for the confirmation preview.
            core = Path(self._beamtime.core_path)
            storage_preview = "\n".join(
                f"{name} -> {core / name}"
                if (core / name).is_dir()
                else f"{name}: unavailable; no link will be created"
                for name in RESERVED_BEAMTIME_NAMES
            )

            # Pass canonical paths to the worker.
            arguments["recipe"] = recipe
            arguments["project_dir"] = project

        except Exception as exc:
            self._message("Cannot create project", str(exc))
            return

        confirmation = QMessageBox(self)
        confirmation.setWindowTitle("Create project?")
        confirmation.setTextFormat(Qt.PlainText)
        confirmation.setText(
            f"Copy recipe:\n{recipe}\n\n"
            f"Into workbench:\n{destination}\n\n"
            f"Create project:\n{project}\n\n"
            f"Beamtime storage links:\n{storage_preview}\n\n"
            "INFO will receive a project summary in the workbench copy."
        )
        confirmation.setStandardButtons(
            QMessageBox.Yes | QMessageBox.No
        )
        confirmation.setDefaultButton(QMessageBox.No)

        if confirmation.exec() != QMessageBox.Yes:
            return

        self._busy = True
        self.content.setEnabled(False)
        self.buttons.setEnabled(False)
        self.status.setText(
            "Copying recipe, writing INFO and creating project links…"
        )

        self._worker = _CreationThread(arguments, self)
        self._worker.finished.connect(self._creation_finished)
        self._worker.start()

    def _creation_finished(self):
        worker = self._worker
        self._busy = False
        self.content.setEnabled(True)
        self.buttons.setEnabled(True)

        if worker.result is None:
            self.status.setText("Project creation failed.")
            self._message(
                "Project creation failed",
                worker.error or "Unknown error.",
            )
        else:
            self._project_path, self._workbench_copy = worker.result
            super().accept()

        worker.deleteLater()
        self._worker = None

    def reject(self):
        if not self._busy:
            super().reject()

    def closeEvent(self, event):
        if self._busy:
            event.ignore()
        else:
            super().closeEvent(event)
