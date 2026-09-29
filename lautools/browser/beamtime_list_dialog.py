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
)

log = logging.getLogger(__name__)


class _ScanBridge(QObject):
    """Moves scanner events from the worker thread into the GUI thread."""

    scanEvent = Signal(object)


class BeamtimeListDialog(QDialog):
    """Scan GPFS for accessible beamtimes and save the selected ones.

    Results appear while the scan runs; nothing is written to the database
    until the user presses "Save selected".
    """

    COL_SELECT, COL_ID, COL_BEAMLINE, COL_YEAR = range(4)
    COL_RAW, COL_PROCESSED, COL_SCRATCH, COL_META, COL_PATH = range(4, 9)

    def __init__(self, project_manager, parent=None):
        super().__init__(parent)

        self.project_manager = project_manager
        self.db = project_manager.db
        self._candidates: list[BeamtimeCandidate] = []
        self._known_ids = {
            beamtime.beamtime_id for beamtime in self.db.list_beamtimes()
        }

        self.setWindowTitle("Find beamtimes")
        self.resize(1000, 640)

        layout = QVBoxLayout(self)
        layout.addLayout(self._create_options_row())
        layout.addWidget(self._create_table(), 1)
        layout.addLayout(self._create_selection_row())
        layout.addWidget(self._create_progress())
        layout.addWidget(self.status_label)

        buttons = QDialogButtonBox()
        self.save_button = buttons.addButton(
            "Save selected", QDialogButtonBox.AcceptRole
        )
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save_selected)
        buttons.addButton(QDialogButtonBox.Close).clicked.connect(self.reject)
        layout.addWidget(buttons)

        self._bridge = _ScanBridge()
        self._bridge.scanEvent.connect(self._on_scan_event)
        self._scanner = BeamtimeScanner(self._bridge.scanEvent.emit)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _create_options_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.base_edit = QLineEdit(str(DEFAULT_BASE))
        self.writable_check = QCheckBox("With writable scratch_cc")
        self.writable_check.setToolTip(
            "Keep only beamtimes where a file can actually be created "
            "in scratch_cc"
        )
        self.writable_check.setChecked(True)
        self.hide_known_check = QCheckBox("Hide already saved")
        self.hide_known_check.setChecked(True)

        self.scan_button = QPushButton("Scan")
        self.scan_button.clicked.connect(self._start_scan)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_scan)

        row.addWidget(QLabel("GPFS base:"))
        row.addWidget(self.base_edit, 1)
        row.addWidget(self.writable_check)
        row.addWidget(self.hide_known_check)
        row.addWidget(self.scan_button)
        row.addWidget(self.cancel_button)
        return row

    def _create_table(self) -> QTableWidget:
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels([
            "", "Beamtime", "Beamline", "Year",
            "raw", "processed", "scratch_cc", "metadata", "Path",
        ])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)

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
        # Directory count is unknown up front, so show a busy indicator.
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.status_label = QLabel("Ready to scan")
        self.status_label.setWordWrap(True)
        return self.progress

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _start_scan(self):
        base = Path(self.base_edit.text().strip() or str(DEFAULT_BASE))
        if not base.is_dir():
            self.status_label.setText(f"Not a directory: {base}")
            return

        self.table.setRowCount(0)
        self._candidates.clear()
        self._known_ids = {
            beamtime.beamtime_id for beamtime in self.db.list_beamtimes()
        }

        self.scan_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.progress.setVisible(True)
        self.status_label.setText(f"Scanning {base}...")

        self._scanner.start(
            base=base,
            writable_only=self.writable_check.isChecked(),
        )

    def _cancel_scan(self):
        self._scanner.cancel()
        self.status_label.setText("Cancelling scan...")

    def _on_scan_event(self, event: ScanEvent):
        if event.kind == ScanEventKind.PROGRESS:
            if event.scanned % 25 == 0:
                self.status_label.setText(
                    f"Checked {event.scanned} directories, "
                    f"found {event.found}"
                )
            return

        if event.kind == ScanEventKind.FOUND and event.candidate is not None:
            self._add_candidate(event.candidate)
            return

        if event.kind in (
            ScanEventKind.FINISHED,
            ScanEventKind.CANCELLED,
            ScanEventKind.FAILED,
        ):
            self.progress.setVisible(False)
            self.scan_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.save_button.setEnabled(bool(self._candidates))

            if event.kind == ScanEventKind.FAILED:
                self.status_label.setText(f"Scan failed: {event.message}")
            elif event.kind == ScanEventKind.CANCELLED:
                self.status_label.setText(
                    f"Scan cancelled after {event.scanned} directories; "
                    f"{event.found} beamtimes listed"
                )
            else:
                self.status_label.setText(
                    f"Scan finished: {event.found} accessible beamtimes "
                    f"in {event.scanned} directories"
                )

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------

    @staticmethod
    def _flag_item(value: bool) -> QTableWidgetItem:
        item = QTableWidgetItem("yes" if value else "no")
        item.setTextAlignment(Qt.AlignCenter)
        return item

    def _add_candidate(self, candidate: BeamtimeCandidate):
        known = candidate.beamtime_id in self._known_ids
        if known and self.hide_known_check.isChecked():
            return

        self._candidates.append(candidate)
        row = self.table.rowCount()
        self.table.insertRow(row)

        select_item = QTableWidgetItem()
        select_item.setFlags(
            Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
        )
        # Pre-select usable beamtimes; the user can still change this.
        select_item.setCheckState(
            Qt.Checked if candidate.scratch_cc_writable else Qt.Unchecked
        )
        select_item.setData(Qt.UserRole, candidate)
        self.table.setItem(row, self.COL_SELECT, select_item)

        id_text = candidate.beamtime_id + (" (saved)" if known else "")
        self.table.setItem(row, self.COL_ID, QTableWidgetItem(id_text))
        self.table.setItem(
            row, self.COL_BEAMLINE, QTableWidgetItem(candidate.beamline or "")
        )
        self.table.setItem(
            row, self.COL_YEAR, QTableWidgetItem(candidate.year or "")
        )
        self.table.setItem(
            row, self.COL_RAW, self._flag_item(candidate.raw_exists)
        )
        self.table.setItem(
            row, self.COL_PROCESSED, self._flag_item(candidate.processed_exists)
        )

        scratch_text = "no"
        if candidate.scratch_cc_exists:
            scratch_text = (
                "writable" if candidate.scratch_cc_writable else "read-only"
            )
        scratch_item = QTableWidgetItem(scratch_text)
        scratch_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, self.COL_SCRATCH, scratch_item)

        self.table.setItem(
            row, self.COL_META, self._flag_item(candidate.has_metadata)
        )
        path_item = QTableWidgetItem(str(candidate.path))
        path_item.setToolTip(str(candidate.path))
        self.table.setItem(row, self.COL_PATH, path_item)

    def _set_all_checked(self, checked: bool):
        state = Qt.Checked if checked else Qt.Unchecked
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.COL_SELECT)
            if item is not None:
                item.setCheckState(state)

    def _select_writable(self):
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.COL_SELECT)
            if item is None:
                continue
            candidate = item.data(Qt.UserRole)
            item.setCheckState(
                Qt.Checked
                if candidate is not None and candidate.scratch_cc_writable
                else Qt.Unchecked
            )

    def _checked_candidates(self) -> list[BeamtimeCandidate]:
        selected = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.COL_SELECT)
            if item is not None and item.checkState() == Qt.Checked:
                candidate = item.data(Qt.UserRole)
                if candidate is not None:
                    selected.append(candidate)
        return selected

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _save_selected(self):
        candidates = self._checked_candidates()
        if not candidates:
            self.status_label.setText("Nothing selected")
            return

        saved = 0
        failed = 0

        for candidate in candidates:
            try:
                detail = self.project_manager.scan_beamtime(candidate.path)
            except Exception as exc:
                failed += 1
                log.warning("Cannot save beamtime %s: %s", candidate.path, exc)
                continue

            # The probe result is more reliable than the permission bits
            # recorded during scan_beamtime.
            storage = detail.storage
            if storage is not None:
                storage.scratch_cc_writable = candidate.scratch_cc_writable
                self.db.upsert_beamtime_storage(storage)

            self.db.add_listed_beamtime(detail.beamtime.id)
            saved += 1

        self.db.connection.commit()
        self._known_ids = {
            beamtime.beamtime_id for beamtime in self.db.list_beamtimes()
        }

        message = f"Saved {saved} beamtime(s)"
        if failed:
            message += f"; {failed} could not be read"
        self.status_label.setText(message)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def done(self, result: int):
        self._scanner.cancel()
        self._scanner.wait(timeout=2.0)
        super().done(result)
