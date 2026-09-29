from __future__ import annotations

from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
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

# Request order for "Refresh all": small and useful results first.
REFRESH_ORDER = ("processed", "scratch_cc", "raw")

METADATA_FIELDS = (
    ("Beamtime ID", "beamtime_id"),
    ("Title", "title"),
    ("Beamline", "beamline"),
    ("Beamline alias", "beamline_alias"),
    ("Setup", "beamline_setup"),
    ("Facility", "facility"),
    ("Proposal ID", "proposal_id"),
    ("Proposal type", "proposal_type"),
    ("Event start", "event_start"),
    ("Event end", "event_end"),
    ("Metadata generated", "generated"),
    ("Core path", "core_path"),
    ("Contact", "contact"),
    ("Retention period", "retention_period"),
    ("Unix ID", "unix_id"),
)

PEOPLE_FIELDS = (
    ("Applicant", "applicant"),
    ("Leader", "leader"),
    ("PI", "pi"),
)


def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "Not counted"

    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def format_time(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


def format_flag(value: bool | None) -> str:
    if value is None:
        return "?"
    return "yes" if value else "no"


def _resolve(path: Path) -> Path:
    try:
        return Path(path).resolve()
    except OSError:
        return Path(path)


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _format_person(beamtime, prefix: str) -> str:
    def value(name: str):
        return getattr(beamtime, f"{prefix}_{name}", None)

    lastname = value("lastname")
    username = value("username")
    parts = []
    if lastname or username:
        parts.append(
            f"{lastname or ''} ({username})".strip()
            if username
            else lastname
        )
    for name in ("institute", "email"):
        if value(name):
            parts.append(value(name))
    if value("user_id"):
        parts.append(f"id {value('user_id')}")
    return ", ".join(parts) if parts else "—"


class _SizeRow:
    """Widgets showing the cached size of one directory."""

    def __init__(self):
        self.exists_label = QLabel("")
        self.size_label = QLabel("Not counted")
        self.updated_label = QLabel("—")
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.button = QPushButton("Count")

    def clear(self, status: str = ""):
        self.exists_label.setText("")
        self.size_label.setText("Not counted")
        self.updated_label.setText("—")
        self.status_label.setText(status)


class ProjectConfigDialog(QDialog):
    """Project settings plus live size information.

    Size counting is delegated to SizeService. The dialog only queues
    requests and redraws rows whenever a scan covering them reports back.
    """

    WS_NAME, WS_SIZE, WS_UPDATED, WS_STATUS = range(4)

    def __init__(
        self,
        project_manager,
        project,
        project_info=None,
        size_service: SizeService | None = None,
        parent=None,
    ):
        super().__init__(parent)

        self.project_manager = project_manager
        self.db = project_manager.db
        self.project = project
        self.project_info = project_info
        self.size_service = size_service

        self._project_path = _resolve(project.path)
        self._beamtimes = []
        self._current_beamtime_id: int | None = None
        self._beamtime_descriptions: dict[int, str] = {}
        self._original_descriptions: dict[int, str] = {}
        self._area_paths: dict[str, Path] = {}
        self._workspace_rows: dict[Path, tuple[int, int]] = {}
        # Requested root path -> "queued" | "running".
        self._active: dict[Path, str] = {}

        self.setWindowTitle("Configure Project")
        self.resize(860, 920)

        outer = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(self._create_project_box())
        layout.addWidget(self._create_beamtime_box())
        layout.addWidget(self._create_storage_box())
        layout.addWidget(self._create_workspace_box())
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
            self._bridge = SizeServiceBridge(self.size_service, parent=self)
            self._bridge.sizeEvent.connect(self._on_size_event)
        else:
            self.refresh_status_label.setText(
                "Size service not available; sizes are read-only."
            )

        self._load_beamtimes()
        self._load_workspaces()
        self._reload_sizes()
        self._restore_active_state()
        self._update_buttons()

    # ------------------------------------------------------------------
    # Values read by the browser window after exec()
    # ------------------------------------------------------------------

    def project_name(self) -> str:
        return self.name_edit.text().strip()

    def project_description(self) -> str:
        return self.description_edit.toPlainText().strip()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _create_project_box(self) -> QGroupBox:
        box = QGroupBox("Laupy project")
        form = QFormLayout(box)

        self.name_edit = QLineEdit(self.project.name)
        form.addRow("Name:", self.name_edit)

        path_label = QLabel(str(self.project.path))
        path_label.setWordWrap(True)
        path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Path:", path_label)

        self.description_edit = QPlainTextEdit(self.project.description or "")
        self.description_edit.setMaximumHeight(100)
        form.addRow("Description:", self.description_edit)

        self.project_row = _SizeRow()
        self.project_row.button.clicked.connect(
            lambda: self._request(self._project_path)
        )
        size_widget = QWidget()
        size_layout = QHBoxLayout(size_widget)
        size_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.addWidget(self.project_row.size_label)
        size_layout.addWidget(QLabel("updated"))
        size_layout.addWidget(self.project_row.updated_label)
        size_layout.addWidget(self.project_row.status_label, 1)
        size_layout.addWidget(self.project_row.button)
        form.addRow("Size:", size_widget)

        return box

    def _create_beamtime_box(self) -> QGroupBox:
        box = QGroupBox("Beamtime")
        layout = QVBoxLayout(box)

        selector = QHBoxLayout()
        self.beamtime_combo = QComboBox()
        self.beamtime_combo.currentIndexChanged.connect(
            self._on_beamtime_changed
        )
        self.rescan_metadata_button = QPushButton("Rescan metadata")
        self.rescan_metadata_button.clicked.connect(self._rescan_metadata)
        selector.addWidget(QLabel("Linked beamtime:"))
        selector.addWidget(self.beamtime_combo, 1)
        selector.addWidget(self.rescan_metadata_button)
        layout.addLayout(selector)

        form = QFormLayout()
        self.metadata_labels: dict[str, QLabel] = {}
        for title, attribute in METADATA_FIELDS:
            label = QLabel("")
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.metadata_labels[attribute] = label
            form.addRow(f"{title}:", label)

        for title, prefix in PEOPLE_FIELDS:
            label = QLabel("")
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.metadata_labels[prefix] = label
            form.addRow(f"{title}:", label)

        self.beamtime_description_edit = QPlainTextEdit()
        self.beamtime_description_edit.setMaximumHeight(80)
        form.addRow("Beamtime notes:", self.beamtime_description_edit)

        layout.addLayout(form)
        return box

    def _create_storage_box(self) -> QGroupBox:
        box = QGroupBox("Beamtime storage")
        layout = QVBoxLayout(box)

        grid = QGridLayout()
        for column, header in enumerate(
            ("Area", "Exists", "Size", "Updated", "Status", "")
        ):
            label = QLabel(f"<b>{header}</b>")
            grid.addWidget(label, 0, column)

        self.area_rows: dict[str, _SizeRow] = {}
        for index, area in enumerate(BEAMTIME_AREAS, start=1):
            row = _SizeRow()
            row.button.clicked.connect(partial(self._request_area, area))
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
        self.tape_label = QLabel("")
        self.scratch_writable_label = QLabel("")
        self.raw_subdirs_label = QLabel("")
        self.raw_subdirs_label.setWordWrap(True)
        self.storage_inspected_label = QLabel("")
        form.addRow("shared exists:", self.shared_label)
        form.addRow("On GPFS:", self.gpfs_label)
        form.addRow("On tape:", self.tape_label)
        form.addRow("scratch_cc writable:", self.scratch_writable_label)
        form.addRow("raw subdirectories:", self.raw_subdirs_label)
        form.addRow("Last inspected:", self.storage_inspected_label)
        layout.addLayout(form)

        return box

    def _create_workspace_box(self) -> QGroupBox:
        box = QGroupBox("Workspaces (wd* directories on disk)")
        layout = QVBoxLayout(box)

        self.workspace_table = QTableWidget(0, 4)
        self.workspace_table.setHorizontalHeaderLabels(
            ["Workspace", "Size", "Updated", "Status"]
        )
        self.workspace_table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
        self.workspace_table.setSelectionMode(
            QAbstractItemView.NoSelection
        )
        self.workspace_table.verticalHeader().setVisible(False)
        header = self.workspace_table.horizontalHeader()
        header.setSectionResizeMode(self.WS_NAME, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.WS_SIZE, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(
            self.WS_UPDATED, QHeaderView.ResizeToContents
        )
        header.setSectionResizeMode(self.WS_STATUS, QHeaderView.Stretch)
        self.workspace_table.setMinimumHeight(160)

        layout.addWidget(self.workspace_table)
        return box

    def _create_refresh_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.refresh_all_button = QPushButton("Refresh all sizes")
        self.refresh_all_button.clicked.connect(self._refresh_all)
        self.force_check = QCheckBox("Force recount")
        self.force_check.setToolTip(
            "Count again even if the size was counted less than a minute ago"
        )
        self.refresh_status_label = QLabel("")
        self.refresh_status_label.setWordWrap(True)

        row.addWidget(self.refresh_all_button)
        row.addWidget(self.force_check)
        row.addWidget(self.refresh_status_label, 1)
        return row

    # ------------------------------------------------------------------
    # Beamtime metadata
    # ------------------------------------------------------------------

    def _load_beamtimes(self, keep_id: int | None = None):
        self._beamtimes = self.db.list_beamtimes_for_project(self.project.id)
        for beamtime in self._beamtimes:
            description = beamtime.description or ""
            self._original_descriptions.setdefault(beamtime.id, description)
            self._beamtime_descriptions.setdefault(beamtime.id, description)

        self.beamtime_combo.blockSignals(True)
        self.beamtime_combo.clear()
        for beamtime in self._beamtimes:
            text = beamtime.beamtime_id
            if beamtime.title:
                text += f" – {beamtime.title}"
            self.beamtime_combo.addItem(text, beamtime.id)
        self.beamtime_combo.setEnabled(len(self._beamtimes) > 1)
        self.beamtime_combo.blockSignals(False)

        if not self._beamtimes:
            self.beamtime_combo.addItem("(No linked beamtime)", None)
            self._show_beamtime(None)
            return

        index = 0
        if keep_id is not None:
            found = self.beamtime_combo.findData(keep_id)
            if found >= 0:
                index = found
        self.beamtime_combo.setCurrentIndex(index)
        self._show_beamtime(self.beamtime_combo.itemData(index))

    def _on_beamtime_changed(self, index: int):
        self._show_beamtime(self.beamtime_combo.itemData(index))
        self._reload_sizes()
        self._restore_active_state()
        self._update_buttons()

    def _store_current_description(self):
        if self._current_beamtime_id is not None:
            self._beamtime_descriptions[self._current_beamtime_id] = (
                self.beamtime_description_edit.toPlainText()
            )

    def _show_beamtime(self, beamtime_id: int | None):
        self._store_current_description()
        self._current_beamtime_id = beamtime_id
        beamtime = next(
            (b for b in self._beamtimes if b.id == beamtime_id), None
        )

        if beamtime is None:
            for label in self.metadata_labels.values():
                label.setText("")
            self.metadata_labels["beamtime_id"].setText(
                "No beamtime linked to this project"
            )
            self.beamtime_description_edit.setPlainText("")
            self.beamtime_description_edit.setEnabled(False)
            self._area_paths = {}
            for row in self.area_rows.values():
                row.clear("No beamtime linked")
            return

        for _, attribute in METADATA_FIELDS:
            value = getattr(beamtime, attribute, None)
            self.metadata_labels[attribute].setText(
                str(value) if value not in (None, "") else "—"
            )
        for _, prefix in PEOPLE_FIELDS:
            self.metadata_labels[prefix].setText(
                _format_person(beamtime, prefix)
            )

        self.beamtime_description_edit.setEnabled(True)
        self.beamtime_description_edit.setPlainText(
            self._beamtime_descriptions.get(beamtime.id, "")
        )

        core = _resolve(beamtime.core_path) if beamtime.core_path else None
        self._area_paths = (
            {area: core / area for area in BEAMTIME_AREAS} if core else {}
        )
        for row in self.area_rows.values():
            row.status_label.setText("")

    def _rescan_metadata(self):
        self.refresh_status_label.setText("Rescanning beamtime metadata...")
        try:
            self.project_manager.refresh_project_metadata(self.project)
        except Exception as exc:
            self.refresh_status_label.setText(f"Metadata rescan failed: {exc}")
            return

        self._store_current_description()
        self._load_beamtimes(keep_id=self._current_beamtime_id)
        self._reload_sizes()
        self._restore_active_state()
        self._update_buttons()
        self.refresh_status_label.setText("Beamtime metadata rescanned")

    # ------------------------------------------------------------------
    # Workspaces
    # ------------------------------------------------------------------

    def _load_workspaces(self):
        workspaces = self.project_manager.sync_workspaces_from_disk(
            self.project
        )

        self.workspace_table.setRowCount(len(workspaces))
        self._workspace_rows.clear()

        for row, workspace in enumerate(workspaces):
            path = _resolve(workspace.path)
            self._workspace_rows[path] = (workspace.id, row)
            name_item = QTableWidgetItem(workspace.name)
            name_item.setToolTip(str(workspace.path))
            self.workspace_table.setItem(row, self.WS_NAME, name_item)
            for column in (self.WS_SIZE, self.WS_UPDATED, self.WS_STATUS):
                self.workspace_table.setItem(row, column, QTableWidgetItem(""))

        if not workspaces:
            self.workspace_table.setRowCount(1)
            self.workspace_table.setItem(
                0, self.WS_NAME, QTableWidgetItem("(No wd* directories)")
            )

    def _set_workspace_cell(self, row: int, column: int, text: str):
        item = self.workspace_table.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            self.workspace_table.setItem(row, column, item)
        item.setText(text)

    # ------------------------------------------------------------------
    # Sizes from the database
    # ------------------------------------------------------------------

    def _reload_sizes(self):
        """Redraw cached sizes. Only reads the database, never counts."""
        project = self.db.get_project(self.project.id)
        if project is not None:
            self.project = project
        self.project_row.size_label.setText(
            format_bytes(self.project.project_size_bytes)
        )
        self.project_row.updated_label.setText(
            format_time(self.project.project_size_bytes_timestamp)
        )

        storage = None
        if self._current_beamtime_id is not None:
            storage = self.db.get_beamtime_storage(self._current_beamtime_id)

        for area, row in self.area_rows.items():
            if self._current_beamtime_id is None:
                continue
            if storage is None:
                row.exists_label.setText("?")
                row.size_label.setText("Not counted")
                row.updated_label.setText("—")
                continue
            row.exists_label.setText(
                format_flag(getattr(storage, f"{area}_exists"))
            )
            row.size_label.setText(
                format_bytes(getattr(storage, f"{area}_size_bytes"))
            )
            row.updated_label.setText(
                format_time(getattr(storage, f"{area}_size_bytes_timestamp"))
            )

        if storage is None:
            for label in (
                self.shared_label,
                self.gpfs_label,
                self.tape_label,
                self.scratch_writable_label,
                self.raw_subdirs_label,
                self.storage_inspected_label,
            ):
                label.setText("—")
        else:
            self.shared_label.setText(format_flag(storage.shared_exists))
            self.gpfs_label.setText(
                format_flag(storage.on_gpfs)
                + (
                    f" (last seen {format_time(storage.last_on_gpfs)})"
                    if storage.last_on_gpfs
                    else ""
                )
            )
            self.tape_label.setText(format_flag(storage.on_tape))
            self.scratch_writable_label.setText(
                format_flag(storage.scratch_cc_writable)
            )
            samples = storage.raw_subdir_samples or []
            count = storage.raw_subdir_count
            text = "?" if count is None else str(count)
            if samples:
                text += ": " + ", ".join(samples)
                if count is not None and count > len(samples):
                    text += ", …"
            self.raw_subdirs_label.setText(text)
            self.storage_inspected_label.setText(
                format_time(storage.last_inspected)
            )

        for workspace_id, row in self._workspace_rows.values():
            workspace = self.db.get_workspace(workspace_id)
            if workspace is None:
                continue
            self._set_workspace_cell(
                row, self.WS_SIZE, format_bytes(workspace.workspace_size_bytes)
            )
            self._set_workspace_cell(
                row,
                self.WS_UPDATED,
                format_time(workspace.workspace_size_bytes_timestamp),
            )

    # ------------------------------------------------------------------
    # Requests to the size service
    # ------------------------------------------------------------------

    def _request(self, path: Path) -> bool:
        if self.size_service is None:
            return False
        if not path.is_dir():
            self.refresh_status_label.setText(f"Not accessible: {path}")
            return False
        self.size_service.request(path, force=self.force_check.isChecked())
        return True

    def _request_area(self, area: str):
        path = self._area_paths.get(area)
        if path is not None:
            self._request(path)

    def _refresh_all(self):
        paths: list[Path] = []

        scratch = self._area_paths.get("scratch_cc")
        project_covered = (
            scratch is not None
            and scratch.is_dir()
            and _is_within(self._project_path, scratch)
        )
        if not project_covered:
            paths.append(self._project_path)

        for area in REFRESH_ORDER:
            path = self._area_paths.get(area)
            if path is not None and path.is_dir():
                paths.append(path)

        requested = sum(1 for path in paths if self._request(path))
        self.refresh_status_label.setText(
            f"Requested {requested} size count(s); "
            "results appear as each finishes"
        )

    # ------------------------------------------------------------------
    # Size events
    # ------------------------------------------------------------------

    def _tracked(self) -> dict[Path, Callable[[str], None]]:
        """Every displayed path mapped to a function that sets its status."""
        tracked: dict[Path, Callable[[str], None]] = {
            self._project_path: self.project_row.status_label.setText,
        }
        for area, path in self._area_paths.items():
            tracked[path] = self.area_rows[area].status_label.setText
        for path, (_, row) in self._workspace_rows.items():
            tracked[path] = partial(
                self._set_workspace_cell, row, self.WS_STATUS
            )
        return tracked

    @staticmethod
    def _status_text(event: SizeEvent, path: Path) -> str:
        direct = path == event.path
        via = "" if direct else f" (via {event.path.name})"
        kind = event.kind

        if kind == SizeEventKind.QUEUED:
            return f"Queued{via}"
        if kind == SizeEventKind.STARTED:
            return f"Counting{via}…"
        if kind == SizeEventKind.PROGRESS:
            if direct:
                return (
                    f"Counting… {event.files_scanned:,} files, "
                    f"{format_bytes(event.size_bytes)} so far"
                )
            return f"Counting{via}…"
        if kind == SizeEventKind.FINISHED:
            if event.errors:
                return f"Updated; {event.errors} entries unreadable"
            return "Updated"
        if kind == SizeEventKind.SKIPPED:
            return "Up to date (counted recently)"
        if kind == SizeEventKind.FAILED:
            return f"Failed: {event.message or 'unknown error'}"
        if kind == SizeEventKind.CANCELLED:
            return "Cancelled"
        return ""

    def _on_size_event(self, event: SizeEvent):
        root = event.path

        if event.kind == SizeEventKind.QUEUED:
            self._active[root] = "queued"
        elif event.kind in (SizeEventKind.STARTED, SizeEventKind.PROGRESS):
            self._active[root] = "running"
        elif event.kind in TERMINAL_EVENTS:
            self._active.pop(root, None)

        tracked = self._tracked()
        affected = [path for path in tracked if _is_within(path, root)]

        for path in affected:
            tracked[path](self._status_text(event, path))

        if affected and event.kind in (
            SizeEventKind.FINISHED,
            SizeEventKind.SKIPPED,
        ):
            self._reload_sizes()

        if affected and event.kind == SizeEventKind.FAILED:
            self.refresh_status_label.setText(
                f"Count of {root} failed: {event.message}"
            )
        elif not self._active and affected and event.kind in TERMINAL_EVENTS:
            self.refresh_status_label.setText("All requested counts finished")

        self._update_buttons()

    def _restore_active_state(self):
        """Show scans already queued or running before the dialog opened."""
        if self.size_service is None:
            return
        self._active.update(self.size_service.active_paths())

        tracked = self._tracked()
        for root, state in self._active.items():
            for path, set_status in tracked.items():
                if not _is_within(path, root):
                    continue
                via = "" if path == root else f" (via {root.name})"
                set_status(
                    f"Counting{via}…" if state == "running" else f"Queued{via}"
                )

    def _is_busy(self, path: Path) -> bool:
        return any(_is_within(path, root) for root in self._active)

    def _update_buttons(self):
        available = self.size_service is not None

        self.project_row.button.setEnabled(
            available and not self._is_busy(self._project_path)
        )
        for area, row in self.area_rows.items():
            path = self._area_paths.get(area)
            row.button.setEnabled(
                available
                and path is not None
                and path.is_dir()
                and not self._is_busy(path)
            )
        self.refresh_all_button.setEnabled(available)
        self.force_check.setEnabled(available)

    # ------------------------------------------------------------------
    # Accept / close
    # ------------------------------------------------------------------

    def _save_beamtime_descriptions(self):
        self._store_current_description()
        now = datetime.now().isoformat(timespec="seconds")
        changed = False

        for beamtime_id, text in self._beamtime_descriptions.items():
            text = text.strip()
            if text == self._original_descriptions.get(beamtime_id, "").strip():
                continue
            self.db.connection.execute(
                "UPDATE beamtime SET description = ?, updated_at = ? "
                "WHERE id = ?",
                (text or None, now, beamtime_id),
            )
            changed = True

        if changed:
            self.db.connection.commit()

    def accept(self):
        self._save_beamtime_descriptions()
        super().accept()

    def done(self, result: int):
        # Stop listening; scans keep running and still update the database.
        if self._bridge is not None:
            self._bridge.detach()
            self._bridge = None
        super().done(result)
