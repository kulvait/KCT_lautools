from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from lautools.beamtime_scanner import (
    DEFAULT_BASE,
    BeamtimeCandidate,
    BeamtimeScanner,
    ScanEvent,
    ScanEventKind,
    path_key,
)
from lautools.browser.utils import open_terminal

log = logging.getLogger(__name__)


class _ScanBridge(QObject):
    """Deliver worker-thread events to the GUI thread."""
    scanEvent = Signal(object)


class BeamtimeListDialog(QDialog):
    """Display persisted beamtimes; use a GPFS scan to update their state."""

    COL_SELECT, COL_ID, COL_BEAMLINE, COL_YEAR, COL_GPFS = range(5)
    COL_TAPE, COL_RAW, COL_PROCESSED, COL_SCRATCH, COL_META, COL_PATH = range(5, 11)

    def __init__(self, project_manager, parent=None):
        super().__init__(parent)
        self.project_manager = project_manager
        self.db = project_manager.db
        self._rows: dict[str, int] = {}
        self._known_paths: dict[str, int] = {}
        self._added = self._updated = self._offloaded = self._failed = 0

        self.setWindowTitle("Beamtimes")
        self.resize(1100, 640)
        layout = QVBoxLayout(self)
        layout.addLayout(self._create_options_row())
        layout.addWidget(self._create_table(), 1)
        layout.addLayout(self._create_selection_row())
        layout.addWidget(self._create_progress())
        layout.addWidget(self.status_label)

        buttons = QDialogButtonBox()
        self.save_button = buttons.addButton(
            "Add selected to list", QDialogButtonBox.AcceptRole
        )
        self.save_button.clicked.connect(self._save_selected)
        buttons.addButton(QDialogButtonBox.Close).clicked.connect(self.reject)
        layout.addWidget(buttons)

        self._bridge = _ScanBridge()
        self._bridge.scanEvent.connect(self._on_scan_event)
        self._scanner = BeamtimeScanner(self._bridge.scanEvent.emit)
        self._load_from_db()

    def _create_options_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.base_edit = QLineEdit(str(DEFAULT_BASE))
        self.writable_check = QCheckBox("Scan only writable scratch_cc")
        self.writable_check.setToolTip(
            "Limit discovered rows to directories whose scratch_cc accepts "
            "a file. Uncheck to find read-only areas and archive stubs."
        )
        self.writable_check.setChecked(False)
        self.scan_button = QPushButton("Scan")
        self.scan_button.clicked.connect(self._start_scan)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_scan)
        row.addWidget(QLabel("GPFS base:"))
        row.addWidget(self.base_edit, 1)
        row.addWidget(self.writable_check)
        row.addWidget(self.scan_button)
        row.addWidget(self.cancel_button)
        return row

    def _create_table(self) -> QTableWidget:
        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels([
            "", "Beamtime", "Beamline", "Year", "GPFS", "Tape",
            "raw", "processed", "scratch_cc", "metadata", "Path",
        ])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        header = self.table.horizontalHeader()
        for column in range(self.COL_PATH):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_PATH, QHeaderView.Stretch)
        return self.table

    def _create_selection_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        select_all = QPushButton("Select all")
        select_all.clicked.connect(lambda: self._set_all_checked(True))
        select_none = QPushButton("Select none")
        select_none.clicked.connect(lambda: self._set_all_checked(False))
        select_writable = QPushButton("Select writable")
        select_writable.clicked.connect(self._select_writable)
        row.addWidget(select_all)
        row.addWidget(select_none)
        row.addWidget(select_writable)
        row.addStretch()
        return row

    def _create_progress(self) -> QProgressBar:
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        return self.progress

    @staticmethod
    def _cell(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        return item

    @classmethod
    def _flag(cls, value: bool | None) -> QTableWidgetItem:
        return cls._cell("?" if value is None else "yes" if value else "no")

    def _load_from_db(self) -> None:
        self.table.setRowCount(0)
        self._rows.clear()
        for beamtime in self.db.list_beamtimes():
            self._show_beamtime(beamtime)
        self.status_label.setText(
            f"{len(self._rows)} beamtimes in database. Scan to update."
        )

    def _show_beamtime(self, beamtime) -> None:
        storage = self.db.get_beamtime_storage(beamtime.id)
        key = beamtime.beamtime_id
        row = self._rows.get(key)
        if row is None:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._rows[key] = row
            select = QTableWidgetItem()
            select.setFlags(
                Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
            )
            select.setCheckState(Qt.Unchecked)
            self.table.setItem(row, self.COL_SELECT, select)
        self.table.item(row, self.COL_SELECT).setData(Qt.UserRole, beamtime.id)

        path = beamtime.core_path
        parts = path.parts if path else ()
        self.table.setItem(row, self.COL_ID, QTableWidgetItem(key))
        self.table.setItem(
            row, self.COL_BEAMLINE,
            QTableWidgetItem(
                beamtime.beamline or (parts[-4] if len(parts) >= 4 else "")
            ),
        )
        self.table.setItem(
            row, self.COL_YEAR,
            QTableWidgetItem(parts[-3] if len(parts) >= 3 else ""),
        )
        self.table.setItem(
            row, self.COL_GPFS,
            self._flag(storage.on_gpfs if storage else None),
        )
        self.table.setItem(
            row, self.COL_TAPE,
            self._flag(storage.on_tape if storage else None),
        )
        self.table.setItem(
            row, self.COL_RAW,
            self._flag(storage.raw_exists if storage else None),
        )
        self.table.setItem(
            row, self.COL_PROCESSED,
            self._flag(storage.processed_exists if storage else None),
        )

        scratch = "?"
        if storage is not None and storage.scratch_cc_exists is not None:
            scratch = "no"
            if storage.scratch_cc_exists:
                scratch = (
                    "writable" if storage.scratch_cc_writable else "read-only"
                )
        self.table.setItem(row, self.COL_SCRATCH, self._cell(scratch))
        self.table.setItem(
            row, self.COL_META, self._flag(bool(beamtime.metadata_json))
        )
        path_item = QTableWidgetItem(str(path) if path else "")
        path_item.setToolTip(str(path) if path else "")
        if path is not None:
            path_item.setData(Qt.UserRole, path)
        self.table.setItem(row, self.COL_PATH, path_item)

        if beamtime.description:
            for col in (self.COL_ID, self.COL_TAPE):
                self.table.item(row, col).setToolTip(beamtime.description)

    def _start_scan(self) -> None:
        base = Path(self.base_edit.text().strip() or str(DEFAULT_BASE))
        if self._scanner.running:
            self.status_label.setText("Previous scan is still stopping")
            return
        if not base.is_dir():
            self.status_label.setText(f"Not a directory: {base}")
            return

        self._known_paths = {}
        for beamtime in self.db.list_beamtimes():
            storage = self.db.get_beamtime_storage(beamtime.id)
            if (
                beamtime.core_path is not None
                and storage is not None
                and storage.on_gpfs is True
            ):
                self._known_paths[path_key(beamtime.core_path)] = beamtime.id
        self._added = self._updated = self._offloaded = self._failed = 0
        self.scan_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setVisible(True)
        self.status_label.setText(f"Scanning {base}...")
        self._scanner.start(
            base=base,
            writable_only=self.writable_check.isChecked(),
            known_on_gpfs=[Path(path) for path in self._known_paths],
        )

    def _cancel_scan(self) -> None:
        self._scanner.cancel()
        self.status_label.setText("Cancelling scan...")

    def _on_scan_event(self, event: ScanEvent) -> None:
        if event.kind == ScanEventKind.PROGRESS:
            if event.scanned % 25 == 0:
                self.status_label.setText(
                    f"Checked {event.scanned} directories, "
                    f"found {event.found}"
                )
            return

        if event.kind == ScanEventKind.FOUND and event.candidate is not None:
            self._store_candidate(event.candidate)
            return

        if event.kind == ScanEventKind.OFFLOADED and event.current is not None:
            beamtime_id = self._known_paths.get(path_key(event.current))
            if beamtime_id is not None:
                self.project_manager.mark_beamtime_offloaded(beamtime_id)
                beamtime = self.db.get_beamtime(beamtime_id)
                if beamtime is not None:
                    self._show_beamtime(beamtime)
                self._offloaded += 1
            return

        if event.kind in (
            ScanEventKind.FINISHED,
            ScanEventKind.CANCELLED,
            ScanEventKind.FAILED,
        ):
            self.progress.setVisible(False)
            self.scan_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            summary = (
                f"{self._added} new, {self._updated} updated, "
                f"{self._offloaded} no longer on GPFS, "
                f"{self._failed} failed"
            )
            if event.kind == ScanEventKind.FAILED:
                self.status_label.setText(
                    f"Scan failed: {event.message}; {summary}"
                )
            elif event.kind == ScanEventKind.CANCELLED:
                self.status_label.setText(f"Scan cancelled; {summary}")
            else:
                self.status_label.setText(
                    f"Scan finished ({event.scanned} directories); {summary}"
                )

    def _store_candidate(self, candidate: BeamtimeCandidate) -> None:
        is_new = candidate.beamtime_id not in self._rows
        try:
            detail = self.project_manager.scan_beamtime(
                candidate.path, candidate=candidate
            )
        except Exception as exc:
            self._failed += 1
            log.warning("Cannot scan beamtime %s: %s", candidate.path, exc)
            return

        self._show_beamtime(detail.beamtime)
        if is_new:
            self._added += 1
        else:
            self._updated += 1

    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.COL_SELECT)
            if item is not None:
                item.setCheckState(state)

    def _select_writable(self) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.COL_SELECT)
            scratch = self.table.item(row, self.COL_SCRATCH)
            if item is not None:
                item.setCheckState(
                    Qt.Checked
                    if scratch is not None and scratch.text() == "writable"
                    else Qt.Unchecked
                )

    def _save_selected(self) -> None:
        ids = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.COL_SELECT)
            if item is not None and item.checkState() == Qt.Checked:
                beamtime_id = item.data(Qt.UserRole)
                if beamtime_id is not None:
                    ids.append(beamtime_id)
        if not ids:
            self.status_label.setText("Nothing selected")
            return

        for beamtime_id in ids:
            self.db.add_listed_beamtime(beamtime_id)
        self.status_label.setText(
            f"Added {len(ids)} beamtime(s) to the application list"
        )

    def _show_context_menu(self, position) -> None:
        item = self.table.itemAt(position)
        if item is None:
            return
        path_item = self.table.item(item.row(), self.COL_PATH)
        path = path_item.data(Qt.UserRole) if path_item else None
        if not isinstance(path, Path):
            return

        menu = QMenu(self)
        action = menu.addAction("Open Terminal Here")
        action.triggered.connect(
            lambda: open_terminal(
                path, on_error=self.status_label.setText
            )
        )
        menu.exec(self.table.viewport().mapToGlobal(position))

    def done(self, result: int) -> None:
        self._scanner.cancel()
        self._scanner.wait(timeout=2.0)
        super().done(result)
