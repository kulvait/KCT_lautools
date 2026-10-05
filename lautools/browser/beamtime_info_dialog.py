from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lautools.browser.project_config_dialog import (
    _format_person,
    format_bytes,
    format_flag,
    format_time,
)
from lautools.browser.size_service_qt import SizeServiceBridge
from lautools.browser.utils import open_terminal
from lautools.size_service import (
    BEAMTIME_AREAS,
    SizeEvent,
    SizeEventKind,
    SizeService,
)


TERMINAL_EVENTS = {
    SizeEventKind.FINISHED,
    SizeEventKind.SKIPPED,
    SizeEventKind.FAILED,
    SizeEventKind.CANCELLED,
}

REFRESH_ORDER = ("processed", "scratch_cc", "raw")
PROBLEM_BRUSH = QBrush(QColor(255, 200, 200))
PROJECT_ID_ROLE = Qt.UserRole


def _resolve(path: Path) -> Path:
    try:
        return Path(path).resolve()
    except OSError:
        return Path(path)


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


class _SizeRow:
    def __init__(self):
        self.exists_label = QLabel("")
        self.size_label = QLabel("Not counted")
        self.updated_label = QLabel("—")
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.button = QPushButton("Count")


class BeamtimeInfoDialog(QDialog):
    """Editable beamtime information plus live storage and project sizes."""

    (
        PROJECT_NAME,
        PROJECT_PATH,
        PROJECT_EXISTS,
        PROJECT_RECIPE,
        PROJECT_RECIPE_PATH,
        PROJECT_RECIPE_EXISTS,
        PROJECT_SIZE,
        PROJECT_UPDATED,
        PROJECT_STATUS,
    ) = range(9)

    def __init__(
        self,
        project_manager,
        beamtime,
        size_service: SizeService | None = None,
        parent=None,
    ):
        super().__init__(parent)

        self.project_manager = project_manager
        self.db = project_manager.db
        self.beamtime = beamtime
        self.size_service = size_service

        self._area_paths: dict[str, Path] = {}
        self._project_rows: dict[Path, tuple[int, int]] = {}
        self._active: dict[Path, str] = {}

        self.setWindowTitle(
            f"Beamtime Info - {beamtime.beamtime_id}"
        )
        self.resize(1040, 920)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.CustomizeWindowHint
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinMaxButtonsHint
            | Qt.WindowCloseButtonHint
        )
        self.setSizeGripEnabled(True)

        outer = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(self._create_beamtime_box())
        layout.addWidget(self._create_storage_box())
        layout.addWidget(self._create_projects_box())
        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        outer.addLayout(self._create_refresh_row())

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._bridge = None
        if self.size_service is not None:
            self._bridge = SizeServiceBridge(
                self.size_service,
                parent=self,
            )
            self._bridge.sizeEvent.connect(self._on_size_event)
        else:
            self.refresh_status_label.setText(
                "Size service unavailable; cached sizes are read-only."
            )

        self._load_beamtime()
        self._load_projects()
        self._reload_sizes()
        self._restore_active_state()
        self._update_buttons()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _create_beamtime_box(self) -> QGroupBox:
        box = QGroupBox("Beamtime")
        layout = QVBoxLayout(box)

        form = QFormLayout()

        self.label_edit = QLineEdit()
        self.label_edit.setToolTip(
            "Name shown in the Beamtime menu."
        )
        form.addRow("Label:", self.label_edit)

        self.description_edit = QPlainTextEdit()
        self.description_edit.setMaximumHeight(90)
        form.addRow("Description:", self.description_edit)

        self.beamtime_id_label = QLabel("")
        self.beamtime_id_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        form.addRow("Beamtime ID:", self.beamtime_id_label)

        self.title_label = QLabel("")
        self.title_label.setWordWrap(True)
        self.title_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Title:", self.title_label)

        self.beamline_label = QLabel("")
        self.beamline_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        form.addRow("Beamline:", self.beamline_label)

        core_path_widget = QWidget()
        core_path_layout = QHBoxLayout(core_path_widget)
        core_path_layout.setContentsMargins(0, 0, 0, 0)

        self.core_path_label = QLabel("")
        self.core_path_label.setWordWrap(True)
        self.core_path_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        self.open_core_path_button = QPushButton("Open terminal")
        self.open_core_path_button.clicked.connect(
            self._open_core_path_terminal
        )

        core_path_layout.addWidget(self.core_path_label, 1)
        core_path_layout.addWidget(self.open_core_path_button)
        form.addRow("Core path:", core_path_widget)

        self.proposal_id_label = QLabel("")
        self.proposal_id_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        form.addRow("Proposal ID:", self.proposal_id_label)

        self.proposal_type_label = QLabel("")
        self.proposal_type_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        form.addRow("Proposal type:", self.proposal_type_label)

        self.facility_label = QLabel("")
        self.facility_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        form.addRow("Facility:", self.facility_label)

        self.setup_label = QLabel("")
        self.setup_label.setWordWrap(True)
        self.setup_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Setup:", self.setup_label)

        self.contact_label = QLabel("")
        self.contact_label.setWordWrap(True)
        self.contact_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Contact:", self.contact_label)

        self.retention_label = QLabel("")
        self.retention_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        form.addRow("Retention period:", self.retention_label)

        self.unix_id_label = QLabel("")
        self.unix_id_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        form.addRow("Unix ID:", self.unix_id_label)

        self.applicant_label = QLabel("")
        self.applicant_label.setWordWrap(True)
        self.applicant_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        form.addRow("Applicant:", self.applicant_label)

        self.leader_label = QLabel("")
        self.leader_label.setWordWrap(True)
        self.leader_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        form.addRow("Leader:", self.leader_label)

        self.pi_label = QLabel("")
        self.pi_label.setWordWrap(True)
        self.pi_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("PI:", self.pi_label)


        self.event_start_label = QLabel("")
        self.event_start_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        form.addRow("Event start:", self.event_start_label)

        self.event_end_label = QLabel("")
        self.event_end_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        form.addRow("Event end:", self.event_end_label)

        layout.addLayout(form)
        
        self.metadata_box = QGroupBox(f"Full JSON metadata")
        metadata_layout = QVBoxLayout(self.metadata_box)

        self.metadata_json_edit = QPlainTextEdit()
        self.metadata_json_edit.setReadOnly(True)
        self.metadata_json_edit.setMinimumHeight(180)
        self.metadata_json_edit.setLineWrapMode(QPlainTextEdit.NoWrap)

        metadata_layout.addWidget(self.metadata_json_edit)
        layout.addWidget(self.metadata_box)

        return box

    def _create_storage_box(self) -> QGroupBox:
        self.storage_box = QGroupBox("Beamtime storage")
        layout = QVBoxLayout(self.storage_box)

        grid = QGridLayout()
        for column, header in enumerate(
            ("Area", "Exists", "Size", "Updated", "Status", "")
        ):
            grid.addWidget(QLabel(f"<b>{header}</b>"), 0, column)

        self.area_rows: dict[str, _SizeRow] = {}

        for index, area in enumerate(BEAMTIME_AREAS, start=1):
            row = _SizeRow()
            row.button.clicked.connect(
                partial(self._request_area, area)
            )

            grid.addWidget(QLabel(area), index, 0)
            grid.addWidget(row.exists_label, index, 1)
            grid.addWidget(row.size_label, index, 2)
            grid.addWidget(row.updated_label, index, 3)
            grid.addWidget(row.status_label, index, 4)
            grid.addWidget(row.button, index, 5)

            self.area_rows[area] = row

        grid.setColumnStretch(4, 1)
        layout.addLayout(grid)

        form = QFormLayout()

        self.shared_label = QLabel("")
        self.gpfs_label = QLabel("")
        self.gpfs_extra_label = QLabel("")
        self.gpfs_extra_label.setWordWrap(True)
        self.scratch_writable_label = QLabel("")
        self.raw_subdir_count_label = QLabel("")
        self.raw_subdirs_edit = QPlainTextEdit()
        self.raw_subdirs_edit.setReadOnly(True)
        self.raw_subdirs_edit.setMaximumHeight(110)
        self.raw_subdirs_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.storage_inspected_label = QLabel("")

        form.addRow("shared exists:", self.shared_label)
        form.addRow("On GPFS:", self.gpfs_label)
        form.addRow("Archive info:", self.gpfs_extra_label)
        form.addRow(
            "scratch_cc writable:",
            self.scratch_writable_label,
        )
        form.addRow(
            "raw subdirectories count:",
            self.raw_subdir_count_label,
        )
        form.addRow("raw subdirectories:", self.raw_subdirs_edit)
        form.addRow("Last inspected:", self.storage_inspected_label)

        layout.addLayout(form)
        return self.storage_box

    def _create_projects_box(self) -> QGroupBox:
        box = QGroupBox("Laupy projects in scratch_cc/kct_*")
        layout = QVBoxLayout(box)

        controls = QHBoxLayout()

        self.scan_projects_button = QPushButton(
            "Scan projects and sizes"
        )
        self.scan_projects_button.setToolTip(
            "Find immediate kct_* directories in scratch_cc, register and "
            "link them to this beamtime, then count scratch_cc once so all "
            "project sizes are refreshed."
        )
        self.scan_projects_button.clicked.connect(
            self._scan_projects_and_sizes
        )

        controls.addWidget(self.scan_projects_button)
        controls.addStretch()
        layout.addLayout(controls)

        # Create the project table with 9 columns for various project attributes.
        self.project_table = QTableWidget(0, 9)
        self.project_table.setHorizontalHeaderLabels([
            "Project",
            "Path",
            "Folder exists",
            "Recipe",
            "Recipe folder",
            "Recipe exists",
            "Size",
            "Updated",
            "Status",
        ])
        self.project_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.project_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.project_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.project_table.verticalHeader().setVisible(False)
        self.project_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.project_table.customContextMenuRequested.connect(
            self._project_context_menu
        )
        header = self.project_table.horizontalHeader()
        for column in range(self.project_table.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.PROJECT_PATH, QHeaderView.Stretch)
        header.setSectionResizeMode(self.PROJECT_RECIPE_PATH, QHeaderView.Stretch)
        self.project_table.setMinimumHeight(220)
        layout.addWidget(self.project_table)

        return box

    def _create_refresh_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.refresh_all_button = QPushButton("Refresh all sizes")
        self.refresh_all_button.clicked.connect(
            self._refresh_all_sizes
        )

        self.force_check = QCheckBox("Force recount")
        self.force_check.setToolTip(
            "Count again even if this path was counted recently."
        )

        self.refresh_status_label = QLabel("")
        self.refresh_status_label.setWordWrap(True)

        row.addWidget(self.refresh_all_button)
        row.addWidget(self.force_check)
        row.addWidget(self.refresh_status_label, 1)

        return row

    # ------------------------------------------------------------------
    # Beamtime data
    # ------------------------------------------------------------------

    def _default_label(self) -> str:
        parts = self.beamtime.core_path.parts if self.beamtime.core_path else ()
        year = parts[-3] if len(parts) >= 3 else ""

        return "_".join(
            part
            for part in (
                self.beamtime.beamline or "",
                year,
                self.beamtime.beamtime_id,
            )
            if part
        )

    def _open_core_path_terminal(self) -> None:
        if self.beamtime.core_path is None:
            self.refresh_status_label.setText(
                "Beamtime has no core path"
            )
            return

        open_terminal(
            self.beamtime.core_path,
            on_error=lambda error: self.refresh_status_label.setText(
                f"Error: {error}"
            ),
        )

    def _load_beamtime(self) -> None:
        refreshed = self.db.get_beamtime(self.beamtime.id)
        if refreshed is not None:
            self.beamtime = refreshed

        self.label_edit.setText(
            self.beamtime.label or self._default_label()
        )
        self.description_edit.setPlainText(
            self.beamtime.description or ""
        )

        self.beamtime_id_label.setText(self.beamtime.beamtime_id or "—")
        self.title_label.setText(self.beamtime.title or "—")
        self.beamline_label.setText(self.beamtime.beamline or "—")
        self.core_path_label.setText(
            str(self.beamtime.core_path)
            if self.beamtime.core_path
            else "—"
        )
        self.proposal_id_label.setText(self.beamtime.proposal_id or "—")
        self.proposal_type_label.setText(self.beamtime.proposal_type or "—")
        self.facility_label.setText(self.beamtime.facility or "—")
        self.setup_label.setText(self.beamtime.beamline_setup or "—")
        self.contact_label.setText(self.beamtime.contact or "—")
        self.retention_label.setText(self.beamtime.retention_period or "—")
        self.unix_id_label.setText(self.beamtime.unix_id or "—")
        self.applicant_label.setText(_format_person(self.beamtime, "applicant"))
        self.leader_label.setText(_format_person(self.beamtime, "leader"))
        self.pi_label.setText(_format_person(self.beamtime, "pi"))
        if self.beamtime.generated is not None:
            self.metadata_box.setTitle(f"Full JSON metadata, generated {self.beamtime.generated}")
        self.event_start_label.setText(self.beamtime.event_start or "—")
        self.event_end_label.setText(self.beamtime.event_end or "—")

        self.metadata_json_edit.setPlainText(
            self.beamtime.metadata_json or ""
        )

        self.open_core_path_button.setEnabled(
            self.beamtime.core_path is not None
            and self.beamtime.core_path.is_dir()
        )

        if self.beamtime.core_path is None:
            self._area_paths = {}
        else:
            core = _resolve(self.beamtime.core_path)
            self._area_paths = {
                area: core / area
                for area in BEAMTIME_AREAS
            }
            self.storage_box.setTitle(
                f"Beamtime storage {core}"
            )

    # ------------------------------------------------------------------
    # Project discovery
    # ------------------------------------------------------------------

    def _beamtime_on_gpfs(self) -> bool | None:
        storage = self.db.get_beamtime_storage(self.beamtime.id)
        return storage.on_gpfs if storage is not None else None

    def _load_projects(self) -> None:
        try:
            projects = self.db.list_projects_for_beamtime(self.beamtime.id)
        except Exception as exc:
            self.refresh_status_label.setText(
                f"Cannot load linked projects: {exc}"
            )
            projects = []

        projects = sorted(projects, key=lambda p: p.name.casefold())
        on_gpfs = self._beamtime_on_gpfs()

        self.project_table.clearContents()
        self.project_table.setRowCount(len(projects))
        self._project_rows.clear()

        for row, project in enumerate(projects):
            path = _resolve(project.path)
            self._project_rows[path] = (project.id, row)

            try:
                health = self.project_manager.get_project_health(
                    project, beamtime_on_gpfs=on_gpfs
                )
            except Exception as exc:
                health = None
                self.refresh_status_label.setText(
                    f"Cannot inspect {project.path}: {exc}"
                )

            if health is None or not health.has_recipe:
                recipe_name, recipe_path, recipe_exists = "—", "—", "—"
            else:
                recipe_name = (
                    health.recipe_instance.name
                    if health.recipe_instance else "?"
                )
                recipe_path = (
                    str(health.recipe_path)
                    if health.recipe_path else "(no disk location)"
                )
                recipe_exists = format_flag(health.recipe_exists)

            values = {
                self.PROJECT_NAME: project.name,
                self.PROJECT_PATH: str(project.path),
                self.PROJECT_EXISTS: format_flag(
                    health.project_exists if health else None
                ),
                self.PROJECT_RECIPE: recipe_name,
                self.PROJECT_RECIPE_PATH: recipe_path,
                self.PROJECT_RECIPE_EXISTS: recipe_exists,
                self.PROJECT_SIZE: "",
                self.PROJECT_UPDATED: "",
                self.PROJECT_STATUS: "",
            }
            tooltip = (
                "\n".join(health.problems())
                if health and health.needs_attention
                else str(project.path)
            )
            for column, text in values.items():
                item = QTableWidgetItem(text)
                item.setToolTip(tooltip)
                if health is not None and health.needs_attention:
                    item.setBackground(PROBLEM_BRUSH)
                self.project_table.setItem(row, column, item)

            self.project_table.item(row, self.PROJECT_NAME).setData(
                PROJECT_ID_ROLE, project.id
            )

        if not projects:
            self.project_table.setRowCount(1)
            self.project_table.setItem(
                0,
                self.PROJECT_NAME,
                QTableWidgetItem(
                    "(No linked kct_* projects; use Scan projects and sizes)"
                ),
            )

    def _scan_projects_and_sizes(self) -> None:
        try:
            projects = (
                self.project_manager.sync_beamtime_projects_from_disk(
                    self.beamtime
                )
            )
        except Exception as exc:
            self.refresh_status_label.setText(
                f"Project scan failed: {exc}"
            )
            return

        self._load_projects()
        self._reload_sizes()

        scratch = self._area_paths.get("scratch_cc")
        if (
            self.size_service is not None
            and scratch is not None
            and scratch.is_dir()
        ):
            self.size_service.request(
                scratch,
                force=self.force_check.isChecked(),
            )
            self.refresh_status_label.setText(
                f"Found {len(projects)} linked project(s); "
                "scratch_cc size scan requested"
            )
        else:
            self.refresh_status_label.setText(
                f"Found {len(projects)} linked project(s); "
                "scratch_cc is not available for size counting"
            )

        self._restore_active_state()
        self._update_buttons()

    # ------------------------------------------------------------------
    # Project context menu
    # ------------------------------------------------------------------

    def _project_id_at_row(self, row: int) -> int | None:
        item = self.project_table.item(row, self.PROJECT_NAME)
        return item.data(PROJECT_ID_ROLE) if item is not None else None

    def _project_context_menu(self, pos) -> None:
        item = self.project_table.itemAt(pos)
        if item is None:
            return
        project_id = self._project_id_at_row(item.row())
        if project_id is None:
            return

        try:
            health = self.project_manager.get_project_health(
                project_id, beamtime_on_gpfs=self._beamtime_on_gpfs()
            )
        except Exception as exc:
            self.refresh_status_label.setText(str(exc))
            return

        busy = self._is_busy(_resolve(health.project.path))

        menu = QMenu(self)
        remove_entry = menu.addAction("Remove database entry")
        remove_folder = menu.addAction("Remove project folder…")
        remove_folder.setEnabled(health.project_exists is True and not busy)
        remove_recipe = menu.addAction("Remove recipe from workbench…")
        remove_recipe.setEnabled(health.has_recipe)

        chosen = menu.exec(self.project_table.viewport().mapToGlobal(pos))
        if chosen is remove_entry:
            self._remove_project_entry(health)
        elif chosen is remove_folder:
            self._remove_project_folder(health)
        elif chosen is remove_recipe:
            self._remove_project_recipe(health)

    def _confirm(self, title: str, text: str) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(title)
        box.setTextFormat(Qt.PlainText)
        box.setText(text)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        return box.exec() == QMessageBox.Yes

    def _after_project_change(self, message: str) -> None:
        self._load_projects()
        self._reload_sizes()
        self._restore_active_state()
        self._update_buttons()
        self.refresh_status_label.setText(message)

    def _remove_project_entry(self, health) -> None:
        project = health.project
        if not self._confirm(
            "Remove database entry?",
            f"Remove project '{project.name}' from the database?\n\n"
            f"{project.path}\n\n"
            "Workspaces, history and the recipe link records are removed "
            "as well. Files on disk are not touched.",
        ):
            return
        try:
            self.project_manager.remove_project_entry(project.id)
        except Exception as exc:
            self.refresh_status_label.setText(f"Removal failed: {exc}")
            return
        self._after_project_change(f"Removed database entry {project.name}")

    def _remove_project_folder(self, health) -> None:
        project = health.project
        if not self._confirm(
            "Remove project folder?",
            f"Permanently delete the folder:\n{project.path}\n\n"
            "Symlinks inside it (raw, processed, recipe links…) are removed, "
            "their targets are not. Real directories inside, such as wd* "
            "workspaces, ARE deleted.\n\n"
            "The database entry is kept.",
        ):
            return
        try:
            removed = self.project_manager.remove_project_folder(project.id)
        except Exception as exc:
            self.refresh_status_label.setText(f"Removal failed: {exc}")
            return
        self._after_project_change(f"Deleted {removed}")

    def _remove_project_recipe(self, health) -> None:
        where = (
            str(health.recipe_path)
            if health.recipe_path else "(unknown location)"
        )
        state = (
            "The folder will be permanently deleted."
            if health.recipe_exists
            else "The folder is missing; only the database record is removed."
        )
        if not self._confirm(
            "Remove recipe from workbench?",
            f"Remove recipe instance of '{health.project.name}':\n"
            f"{where}\n\n{state}\n\n"
            "Project links pointing into it will become dangling.",
        ):
            return
        try:
            removed = self.project_manager.remove_project_recipe(
                health.project.id
            )
        except Exception as exc:
            self.refresh_status_label.setText(f"Removal failed: {exc}")
            return
        self._after_project_change(
            f"Deleted {removed}" if removed else "Removed recipe record"
        )

    # ------------------------------------------------------------------
    # Cached sizes
    # ------------------------------------------------------------------

    def _set_project_cell(
        self,
        row: int,
        column: int,
        text: str,
    ) -> None:
        item = self.project_table.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            self.project_table.setItem(row, column, item)
        item.setText(text)

    def _reload_sizes(self) -> None:
        storage = self.db.get_beamtime_storage(
            self.beamtime.id
        )

        for area, row in self.area_rows.items():
            if storage is None:
                row.exists_label.setText("?")
                row.size_label.setText("Not counted")
                row.updated_label.setText("—")
                continue

            row.exists_label.setText(
                format_flag(getattr(storage, f"{area}_exists"))
            )
            row.size_label.setText(
                format_bytes(
                    getattr(storage, f"{area}_size_bytes")
                )
            )
            row.updated_label.setText(
                format_time(
                    getattr(
                        storage,
                        f"{area}_size_bytes_timestamp",
                    )
                )
            )

        if storage is None:
            self.shared_label.setText("—")
            self.gpfs_label.setText("?")
            self.gpfs_extra_label.setText("—")
            self.scratch_writable_label.setText("?")
            self.raw_subdir_count_label.setText("—")
            self.raw_subdirs_edit.setPlainText("")
            self.storage_inspected_label.setText("—")
        else:
            self.shared_label.setText(
                format_flag(storage.shared_exists)
            )
            self.gpfs_label.setText(
                format_flag(storage.on_gpfs)
            )

            if storage.on_gpfs:
                self.gpfs_extra_label.setText("—")
            else:
                extras = []
                if storage.last_on_gpfs:
                    extras.append(
                        "Last on GPFS: "
                        + format_time(storage.last_on_gpfs)
                    )
                extras.append(
                    "On tape: " + format_flag(storage.on_tape)
                )
                self.gpfs_extra_label.setText("\n".join(extras))

            self.scratch_writable_label.setText(
                format_flag(storage.scratch_cc_writable)
            )

            count = storage.raw_subdir_count
            self.raw_subdir_count_label.setText(
                "?" if count is None else str(count)
            )
            self.raw_subdirs_edit.setPlainText(
                "\n".join(storage.raw_subdir_samples or [])
            )

            self.storage_inspected_label.setText(
                format_time(storage.last_inspected)
            )

        for project_id, row in self._project_rows.values():
            project = self.db.get_project(project_id)
            if project is None:
                continue

            self._set_project_cell(
                row,
                self.PROJECT_SIZE,
                format_bytes(project.project_size_bytes),
            )
            self._set_project_cell(
                row,
                self.PROJECT_UPDATED,
                format_time(
                    project.project_size_bytes_timestamp
                ),
            )

    # ------------------------------------------------------------------
    # Size requests
    # ------------------------------------------------------------------

    def _request(self, path: Path) -> bool:
        if self.size_service is None:
            return False

        if not path.is_dir():
            self.refresh_status_label.setText(
                f"Not accessible: {path}"
            )
            return False

        self.size_service.request(
            path,
            force=self.force_check.isChecked(),
        )
        return True

    def _request_area(self, area: str) -> None:
        path = self._area_paths.get(area)
        if path is not None:
            self._request(path)

    def _refresh_all_sizes(self) -> None:
        paths: list[Path] = []

        for area in REFRESH_ORDER:
            path = self._area_paths.get(area)
            if path is not None and path.is_dir():
                paths.append(path)

        requested = sum(
            1 for path in paths
            if self._request(path)
        )

        self.refresh_status_label.setText(
            f"Requested {requested} size count(s); "
            "project sizes are updated by the scratch_cc scan"
        )
        self._restore_active_state()
        self._update_buttons()

    # ------------------------------------------------------------------
    # Size events
    # ------------------------------------------------------------------

    def _tracked(self) -> dict[Path, Callable[[str], None]]:
        tracked: dict[Path, Callable[[str], None]] = {}

        for area, path in self._area_paths.items():
            tracked[path] = (
                self.area_rows[area].status_label.setText
            )

        for path, (_, row) in self._project_rows.items():
            tracked[path] = partial(
                self._set_project_cell,
                row,
                self.PROJECT_STATUS,
            )

        return tracked

    @staticmethod
    def _status_text(
        event: SizeEvent,
        path: Path,
    ) -> str:
        direct = path == event.path
        via = "" if direct else f" (via {event.path.name})"

        if event.kind == SizeEventKind.QUEUED:
            return f"Queued{via}"

        if event.kind == SizeEventKind.STARTED:
            return f"Counting{via}…"

        if event.kind == SizeEventKind.PROGRESS:
            if direct:
                return (
                    f"Counting {event.files_scanned:,} files; "
                    f"{format_bytes(event.size_bytes)} so far"
                )
            return f"Counting{via}…"

        if event.kind == SizeEventKind.FINISHED:
            if event.errors:
                return f"Updated; {event.errors} entries unreadable"
            return "Updated"

        if event.kind == SizeEventKind.SKIPPED:
            return "Up to date"

        if event.kind == SizeEventKind.FAILED:
            return f"Failed: {event.message or 'unknown error'}"

        if event.kind == SizeEventKind.CANCELLED:
            return "Cancelled"

        return ""

    def _on_size_event(self, event: SizeEvent) -> None:
        root = event.path

        if event.kind == SizeEventKind.QUEUED:
            self._active[root] = "queued"
        elif event.kind in (
            SizeEventKind.STARTED,
            SizeEventKind.PROGRESS,
        ):
            self._active[root] = "running"
        elif event.kind in TERMINAL_EVENTS:
            self._active.pop(root, None)

        tracked = self._tracked()
        affected = [
            path
            for path in tracked
            if event.concerns(path) or _is_within(path, root)
        ]

        for path in affected:
            tracked[path](
                self._status_text(event, path)
            )

        if affected and event.kind in (
            SizeEventKind.FINISHED,
            SizeEventKind.SKIPPED,
        ):
            self._reload_sizes()

        if affected and event.kind == SizeEventKind.FAILED:
            self.refresh_status_label.setText(
                f"Count of {root} failed: {event.message}"
            )
        elif (
            affected
            and not self._active
            and event.kind in TERMINAL_EVENTS
        ):
            self.refresh_status_label.setText(
                "All requested counts finished"
            )

        self._update_buttons()

    def _restore_active_state(self) -> None:
        if self.size_service is None:
            return

        self._active.update(
            self.size_service.active_paths()
        )

        tracked = self._tracked()

        for root, state in self._active.items():
            for path, set_status in tracked.items():
                if not _is_within(path, root):
                    continue

                via = (
                    ""
                    if path == root
                    else f" (via {root.name})"
                )
                set_status(
                    f"Counting{via}…"
                    if state == "running"
                    else f"Queued{via}"
                )

    def _is_busy(self, path: Path) -> bool:
        return any(
            _is_within(path, root)
            for root in self._active
        )

    def _update_buttons(self) -> None:
        available = self.size_service is not None

        for area, row in self.area_rows.items():
            path = self._area_paths.get(area)
            row.button.setEnabled(
                available
                and path is not None
                and path.is_dir()
                and not self._is_busy(path)
            )

        scratch = self._area_paths.get("scratch_cc")
        scratch_available = (
            scratch is not None and scratch.is_dir()
        )

        self.scan_projects_button.setEnabled(
            scratch_available
            and (
                self.size_service is None
                or not self._is_busy(scratch)
            )
        )
        self.refresh_all_button.setEnabled(available)
        self.force_check.setEnabled(available)

    # ------------------------------------------------------------------
    # Save / close
    # ------------------------------------------------------------------

    def accept(self) -> None:
        label = self.label_edit.text().strip()
        description = self.description_edit.toPlainText().strip()

        try:
            self.project_manager.update_beamtime_user_fields(
                self.beamtime.id,
                label=label or self._default_label(),
                description=description or None,
            )
        except Exception as exc:
            self.refresh_status_label.setText(
                f"Could not save beamtime: {exc}"
            )
            return

        super().accept()

    def done(self, result: int) -> None:
        # Size scans continue after the dialog closes.
        if self._bridge is not None:
            self._bridge.detach()
            self._bridge = None

        super().done(result)
