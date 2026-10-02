from __future__ import annotations

import logging
import os
import re
from pathlib import Path
import shutil

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

def _directory_name(value: str) -> str:
    """Validate a single directory name, not a path."""
    name = value.strip()
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\0" in name
    ):
        raise ValueError("Enter a directory name without path separators.")
    return name


def validate_creation(
    recipes_root: Path,
    recipe: Path,
    workbench: Path,
    scratch: Path,
    project: Path,
    copy_name: str,
) -> tuple[Path, Path, Path]:
    """Validate and return canonical recipe, copy and project paths."""
    recipes_root = recipes_root.resolve(strict=True)
    workbench = workbench.resolve(strict=True)
    scratch = scratch.resolve(strict=True)

    for directory in (recipes_root, workbench, scratch):
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

    if recipe.is_symlink():
        raise ValueError("Recipe folders must not themselves be symlinks.")
    recipe = recipe.resolve(strict=True)
    if recipe.parent != recipes_root or not recipe.is_dir():
        raise ValueError("Select an immediate recipe subfolder.")

    if workbench == scratch or workbench.is_relative_to(scratch):
        raise ValueError(
            "The workbench must be outside this beamtime's scratch area."
        )

    project = Path(os.path.abspath(project.expanduser()))
    if project.is_symlink():
        raise ValueError("The project directory must not be a symlink.")
    project = project.resolve()

    if project == scratch or not project.is_relative_to(scratch):
        raise ValueError(
            "The project directory must be strictly inside scratch_cc."
        )
    if not project.parent.is_dir():
        raise ValueError(
            "The project's parent directory must already exist."
        )
    if os.path.lexists(project):
        if not project.is_dir() or any(project.iterdir()):
            raise ValueError(
                "Select a new directory or an existing empty directory."
            )

    destination = workbench / _directory_name(copy_name)
    if os.path.lexists(destination):
        raise ValueError(
            f"The workbench destination already exists:\n{destination}"
        )

    # Prevent recursive copying or overlapping source/output trees.
    for left, right in (
        (recipe, destination),
        (recipe, project),
        (destination, project),
    ):
        if (
            left == right
            or left.is_relative_to(right)
            or right.is_relative_to(left)
        ):
            raise ValueError(
                "Recipe, workbench copy and project must not overlap."
            )

    return recipe, destination, project


def create_project_from_recipe(
    recipes_root: Path,
    recipe: Path,
    workbench: Path,
    scratch: Path,
    project: Path,
    copy_name: str,
) -> tuple[Path, Path]:
    """Copy the full recipe tree and link its top-level entries.

    No database or Git writes.

    On failure, remove only links created by this operation. Keep any
    workbench copy, complete or partial, for inspection and recovery.
    """
    recipe, destination, project = validate_creation(
        recipes_root, recipe, workbench, scratch, project, copy_name
    )

    created_links: list[tuple[Path, str]] = []
    created_project = False
    reserved_destination = False

    try:
        # Reserve exclusively: never merge into somebody else's directory.
        destination.mkdir()
        reserved_destination = True
        shutil.copytree(
            recipe,
            destination,
            dirs_exist_ok=True,
            symlinks=True,
        )

        if not os.path.lexists(project):
            project.mkdir()
            created_project = True
        elif (
            project.is_symlink()
            or not project.is_dir()
            or any(project.iterdir())
        ):
            raise ValueError(
                "The project directory changed during copying; "
                "it must still be empty."
            )

        for entry in sorted(destination.iterdir(), key=lambda p: p.name):
            link = project / entry.name
            target = str(entry)  # Absolute path to the workbench entry.
            os.symlink(
                target,
                link,
                target_is_directory=entry.is_dir(),
            )
            created_links.append((link, target))

        return project, destination

    except Exception as exc:
        cleanup_errors = []

        # Never recursively delete the project or workbench.
        for link, target in reversed(created_links):
            try:
                if link.is_symlink() and os.readlink(link) == target:
                    link.unlink()
            except OSError as cleanup_exc:
                cleanup_errors.append(str(cleanup_exc))

        if created_project:
            try:
                project.rmdir()  # Only succeeds if empty.
            except OSError as cleanup_exc:
                cleanup_errors.append(str(cleanup_exc))

        details = str(exc)
        if reserved_destination:
            details += (
                "\n\nThe workbench copy was retained, possibly incomplete:"
                f"\n{destination}\n"
                "Inspect it before removing it or retrying with another name."
            )
        if cleanup_errors:
            details += "\n\nCleanup warnings:\n" + "\n".join(cleanup_errors)
        raise RuntimeError(details) from exc


class _CreationThread(QThread):
    """Filesystem operations only; no SQLite access from this thread."""

    def __init__(self, arguments, parent=None):
        super().__init__(parent)
        self.arguments = arguments
        self.result: tuple[Path, Path] | None = None
        self.error: str | None = None

    def run(self):
        try:
            self.result = create_project_from_recipe(*self.arguments)
        except Exception as exc:
            log.exception("Recipe project creation failed")
            self.error = str(exc)


class NewProjectFromRecipeDialog(QDialog):
    def __init__(
        self,
        project_manager,
        beamtime,
        scratch,
        size_service=None,
        parent=None,
    ):
        super().__init__(parent)
        # size_service is accepted for compatibility with BrowserWindow.
        self._scratch = Path(scratch).expanduser().resolve()
        self._recipes_root: Path | None = None
        self._workbench: Path | None = None
        self._project_path: Path | None = None
        self._workbench_copy: Path | None = None
        self._worker: _CreationThread | None = None
        self._beamtime = beamtime
        self._project_basename = self._default_project_basename(beamtime)
        self._busy = False
        self._automatic_copy_name = True

        self.setWindowTitle(
            f"New project from recipe — {beamtime.beamtime_id}"
        )
        self.resize(760, 380)

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

        self.project_edit = QLineEdit(str(self._scratch / self._project_basename))
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
            "The full recipe tree is copied into the workbench. "
            "Each top-level entry is then linked into the scratch project.\n"
            "Existing symlinks within the recipe remain symlinks; their "
            "external targets are not copied."
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
                    raise ValueError(f"Not a directory: {directory}")
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
                    path for path in self._recipes_root.iterdir()
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

    def _default_project_basename(self, beamtime) -> str:
        label = getattr(beamtime, "label", None) or getattr(
            beamtime, "beamtime_id", None
        ) or "project"
        base = f"kct_{_slug(label)}"
        candidate = self._scratch / base
        if not os.path.lexists(candidate):
            return base
        for index in range(1, 1000):
            numbered = f"{base}_{index:03d}"
            candidate = self._scratch / numbered
            if not os.path.lexists(candidate):
                return numbered
        raise ValueError(f"Could not find a free project directory name under {self._scratch}")

    def selected_project_path(self) -> Path | None:
        return self._project_path

    def selected_workbench_path(self) -> Path | None:
        return self._workbench_copy

    def _browse_project(self):
        # Qt's directory chooser can also create a new directory.
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
            self.copy_name_edit.setText(f"{project_name}_{recipe.name}")
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
        if recipe is None or not project_text:
            self._message("Selection required", "Choose a recipe and project.")
            return

        project = Path(project_text).expanduser()
        if not project.is_absolute():
            project = self._scratch / project

        arguments = (
            self._recipes_root,
            recipe,
            self._workbench,
            self._scratch,
            project,
            self.copy_name_edit.text(),
        )
        try:
            _, destination, project = validate_creation(*arguments)
        except Exception as exc:
            self._message("Cannot create project", str(exc))
            return

        confirmation = QMessageBox(self)
        confirmation.setWindowTitle("Create project?")
        confirmation.setTextFormat(Qt.PlainText)
        confirmation.setText(
            f"Copy recipe:\n{recipe}\n\n"
            f"Into workbench:\n{destination}\n\n"
            f"Create top-level symlinks in:\n{project}"
        )
        confirmation.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        confirmation.setDefaultButton(QMessageBox.No)
        if confirmation.exec() != QMessageBox.Yes:
            return

        self._busy = True
        self.content.setEnabled(False)
        self.buttons.setEnabled(False)
        self.status.setText("Copying recipe and creating project links…")
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
