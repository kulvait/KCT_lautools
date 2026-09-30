from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt
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
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lautools.browser.project_config_dialog import (
    format_bytes,
    format_flag,
    format_time,
)
from lautools.browser.size_service_qt import SizeServiceBridge
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
    """Edit one beamtime and manage its Laupy projects and cached sizes."""

    (
        PROJECT_NAME,
        PROJECT_PATH,
        PROJECT_SIZE,
        PROJECT_UPDATED,
        PROJECT_STATUS,
    ) = range(5)

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
            f"Configure Beamtime {beamtime.beamtime_id}"
        )
        self.resize(980, 850)
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
        form = QFormLayout(box)

        self.label_edit = QLineEdit()
        self.label_edit.setToolTip(
            "Name displayed in the Beamtime menu."
        )
        form.addRow("Label:", self.label_edit)

        beamtime_id = QLabel(self.beamtime.beamtime_id)
        beamtime_id.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Beamtime ID:", beamtime_id)

        beamline = QLabel(self.beamtime.beamline or "—")
        form.addRow("Beamline:", beamline)

        title = QLabel(self.beamtime.title or "—")
        title.setWordWrap(True)
        title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Title:", title)

        proposal = QLabel(self.beamtime.proposal_id or "—")
        form.addRow("Proposal:", proposal)

        modality = QLabel(self.beamtime.beamline_setup or "—")
        modality.setWordWrap(True)
        modality.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Modality:", modality)

        path = QLabel(
            str(self.beamtime.core_path)
            if self.beamtime.core_path
            else "—"
        )
        path.setWordWrap(True)
        path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Core path:", path)

        self.description_edit = QPlainTextEdit()
        self.description_edit.setMinimumHeight(110)
        form.addRow("Description:", self.description_edit)

        return box

    def _create_storage_box(self) -> QGroupBox:
        box = QGroupBox("Beamtime storage")
        layout = QVBoxLayout(box)

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

        self.gpfs_label = QLabel("")
        self.tape_label = QLabel("")
        self.last_gpfs_label = QLabel("")
        self.scratch_writable_label = QLabel("")
        self.last_inspected_label = QLabel("")

        form.addRow("On GPFS:", self.gpfs_label)
        form.addRow("On tape:", self.tape_label)
        form.addRow("Last on GPFS:", self.last_gpfs_label)
        form.addRow(
            "scratch_cc writable:",
            self.scratch_writable_label,
        )
        form.addRow("Last inspected:", self.last_inspected_label)

        layout.addLayout(form)
        return box

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
            "project sizes are updated."
        )
        self.scan_projects_button.clicked.connect(
            self._scan_projects_and_sizes
        )

        controls.addWidget(self.scan_projects_button)
        controls.addStretch()
        layout.addLayout(controls)

        self.project_table = QTableWidget(0, 5)
        self.project_table.setHorizontalHeaderLabels([
            "Project",
            "Path",
            "Size",
            "Updated",
            "Status",
        ])
        self.project_table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
        self.project_table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )
        self.project_table.setSelectionMode(
            QAbstractItemView.SingleSelection
        )
        self.project_table.verticalHeader().setVisible(False)

        header = self.project_table.horizontalHeader()
        header.setSectionResizeMode(
            self.PROJECT_NAME,
            QHeaderView.ResizeToContents,
        )
        header.setSectionResizeMode(
            self.PROJECT_PATH,
            QHeaderView.Stretch,
        )
        header.setSectionResizeMode(
            self.PROJECT_SIZE,
            QHeaderView.ResizeToContents,
        )
        header.setSectionResizeMode(
            self.PROJECT_UPDATED,
            QHeaderView.ResizeToContents,
        )
        header.setSectionResizeMode(
            self.PROJECT_STATUS,
            QHeaderView.ResizeToContents,
        )

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

        if self.beamtime.core_path is None:
            self._area_paths = {}
        else:
            core = _resolve(self.beamtime.core_path)
            self._area_paths = {
                area: core / area
                for area in BEAMTIME_AREAS
            }

    # ------------------------------------------------------------------
    # Project discovery
    # ------------------------------------------------------------------

    def _load_projects(self) -> None:
        try:
            projects = self.db.list_projects_for_beamtime(
                self.beamtime.id
            )
        except Exception as exc:
            self.refresh_status_label.setText(
                f"Cannot load linked projects: {exc}"
            )
            projects = []

        projects = sorted(
            projects,
            key=lambda project: project.name.casefold(),
        )

        self.project_table.setRowCount(len(projects))
        self._project_rows.clear()

        for row, project in enumerate(projects):
            path = _resolve(project.path)
            self._project_rows[path] = (project.id, row)

            name_item = QTableWidgetItem(project.name)
            name_item.setToolTip(str(project.path))

            path_item = QTableWidgetItem(str(project.path))
            path_item.setToolTip(str(project.path))

            self.project_table.setItem(
                row,
                self.PROJECT_NAME,
                name_item,
            )
            self.project_table.setItem(
                row,
                self.PROJECT_PATH,
                path_item,
            )

            for column in (
                self.PROJECT_SIZE,
                self.PROJECT_UPDATED,
                self.PROJECT_STATUS,
            ):
                self.project_table.setItem(
                    row,
                    column,
                    QTableWidgetItem(""),
                )

        if not projects:
            self.project_table.setRowCount(1)
            item = QTableWidgetItem(
                "(No linked kct_* projects; use Scan projects and sizes)"
            )
            self.project_table.setItem(
                0,
                self.PROJECT_NAME,
                item,
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
            # One scratch_cc scan updates scratch_cc itself and every
            # registered kct_* project below it.
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
            self.gpfs_label.setText("?")
            self.tape_label.setText("?")
            self.last_gpfs_label.setText("—")
            self.scratch_writable_label.setText("?")
            self.last_inspected_label.setText("—")
        else:
            self.gpfs_label.setText(
                format_flag(storage.on_gpfs)
            )
            self.tape_label.setText(
                format_flag(storage.on_tape)
            )
            self.last_gpfs_label.setText(
                format_time(storage.last_on_gpfs)
            )
            self.scratch_writable_label.setText(
                format_flag(storage.scratch_cc_writable)
            )
            self.last_inspected_label.setText(
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
