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

from lautools.collection_manager import CollectionManager
from lautools.project_creator import create_project

RESERVED_BEAMTIME_NAMES = ("raw", "processed", "scratch_cc", "shared")

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s : %(message)s",
        datefmt="%d.%m.%Y %H:%M:%S",
    )
    handler.setFormatter(formatter)
    log.addHandler(handler)
log.propagate = False


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
            clonedRecipeInstance = self.parent().collection_manager.clone_recipe_instance(
                source_instance=self.arguments["recipe_instance"],
                destination_collection=self.parent()._workbench_collection_id,
                new_name=self.arguments["cloned_recipe_name"],
            )
            cloned_recipe_path = self.parent().collection_manager.get_recipe_instance_path(clonedRecipeInstance)
            project = create_project(
                beamtime=self.arguments["beamtime"],
                project_dir=self.arguments["project_dir"],
                recipe_instance_directory=cloned_recipe_path,
            )
            self.result = (project, cloned_recipe_path)
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
        self._cookbook_root: Path | None = None
        self._workbench: Path | None = None
        self._project_path: Path | None = None
        self._workbench_copy: Path | None = None
        self._worker: _CreationThread | None = None
        self._busy = False
        self._automatic_copy_name = True

        self.setWindowTitle(f"New project from recipe for beamtime {beamtime.beamtime_id}")
        self.resize(800, 450)

        outer = QVBoxLayout(self)
        self.content = QWidget()
        layout = QFormLayout(self.content)
        outer.addWidget(self.content)

        self.recipe_combo = QComboBox()
        layout.addRow("Recipe:", self.recipe_combo)

        self.cookbook_label = self._path_label()
        self.workbench_label = self._path_label()
        layout.addRow("Cookbook:", self.cookbook_label)
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

            self.collection_manager = CollectionManager(project_manager.db)
            config = project_manager.db.get_config()
            if config.default_cookbook_location_id is None:
                raise ValueError(
                    "Configure a default cookbook collection in Settings."
                )
            if config.default_workbench_location_id is None:
                raise ValueError(
                    "Configure a default workbench collection in Settings."
                )

            cookbook_location = project_manager.db.get_location(
                config.default_cookbook_location_id
            )
            workbench_location = project_manager.db.get_location(
                config.default_workbench_location_id,
            )
            if cookbook_location is None or cookbook_location.disk_location is None:
                raise ValueError("Default cookbook collection has no disk location.")
            if workbench_location is None or workbench_location.disk_location is None:
                raise ValueError("Default workbench collection has no disk location.")

            self._cookbook_root = cookbook_location.disk_location
            self._workbench = workbench_location.disk_location
            self._cookbook_collection_id = config.default_cookbook_location_id
            self._workbench_collection_id = config.default_workbench_location_id

            self.cookbook_label.setText(str(self._cookbook_root))
            self.workbench_label.setText(str(self._workbench))

            recipe_instances = self.collection_manager.sync_recipe_instances_from_disk(
                self._cookbook_collection_id
            )

            for recipe_instance in recipe_instances:
                self.recipe_combo.addItem(recipe_instance.name, recipe_instance)

            if not recipe_instances:
                raise ValueError(
                    "No recipe instances found in the default cookbook collection."
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
        recipe_instance = self.recipe_combo.currentData()
        project_text = self.project_edit.text().strip()

        if (
            recipe_instance is None
            or not project_text
            or self._cookbook_root is None
            or self._workbench is None
        ):
            log.error("Missing recipe or project directory: %s, %s", recipe_instance, project_text)
            self._message(
                "Selection required",
                "Choose a recipe and project directory.",
            )
            return

        project = Path(project_text).expanduser()
        if not project.is_absolute():
            project = self._scratch / project

        clone_name = self.copy_name_edit.text().strip()

        try:
            source_path = self.collection_manager.get_recipe_instance_path(
                recipe_instance
            )
            destination = self._workbench / clone_name
            if os.path.lexists(destination):
                raise ValueError(
                    f"Workbench recipe instance already exists:\n{destination}"
                )
            # Use actual filesystem state for the confirmation preview.
            core = Path(self._beamtime.core_path)
            storage_preview = "\n".join(
                f"{name} -> {core / name}"
                if (core / name).is_dir()
                else f"{name}: unavailable; no link will be created"
                for name in RESERVED_BEAMTIME_NAMES
            )

        except Exception as exc:
            log.exception("Preflight checks failed with exception {%s}", exc)
            self._message("Cannot create project", str(exc))
            return

        confirmation = QMessageBox(self)
        confirmation.setWindowTitle("Create project?")
        confirmation.setTextFormat(Qt.PlainText)
        confirmation.setText(
            f"Clone recipe instance:\n{source_path}\n\n"
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

        self._worker = _CreationThread(
            {
                "beamtime": self._beamtime,
                "project_dir": project,
                "recipe_instance": recipe_instance,
                "cloned_recipe_name": clone_name,
            },
            self,
        )
        self._worker.finished.connect(self._creation_finished)
        self._worker.start()

    def _creation_finished(self):
        worker = self._worker
        self._busy = False
        self.content.setEnabled(True)
        self.buttons.setEnabled(True)

        if worker.result is None:
            log.error("Project creation failed: %s", worker.error)
            self.status.setText("Project creation failed.")
            self._message("Project creation failed", worker.error or "Unknown error.",)
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
