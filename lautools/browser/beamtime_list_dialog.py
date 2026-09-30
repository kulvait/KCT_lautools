from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor
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

COLOR_NOT_ON_GPFS = QColor(255, 245, 157)   # yellow
COLOR_WRITABLE = QColor(200, 230, 201)     # green
COLOR_READ_ONLY = QColor(255, 205, 210)    # red


class _ScanBridge(QObject):
    """Deliver worker-thread events to the GUI thread."""

    scanEvent = Signal(object)


class BeamtimeListDialog(QDialog):
    """Database is the ground truth; a scan only updates it.

    The tick in the first column mirrors lautools_app_listed_beamtime and is
    written immediately when toggled.
    """

    (
        COL_SELECT,
        COL_ID,
        COL_BEAMLINE,
        COL_YEAR,
        COL_PI,
        COL_TITLE,
        COL_DESCRIPTION,
        COL_GPFS,
        COL_TAPE,
        COL_RAW,
        COL_PROCESSED,
        COL_SCRATCH,
        COL_META,
        COL_PATH,
    ) = range(14)

    def __init__(self, project_manager, parent=None):
        super().__init__(parent)

        self.project_manager = project_manager
        self.db = project_manager.db
        self._known_paths: dict[str, int] = {}
        self._known_ids: set[str] = set()
        self._added = 0
        self._updated = 0
        self._offloaded = 0
        self._failed = 0
        self._populating = False

        self.setWindowTitle("Beamtimes")
        self.resize(1500, 700)

        layout = QVBoxLayout(self)
        layout.addLayout(self._create_options_row())
        layout.addWidget(self._create_table(), 1)
        layout.addLayout(self._create_selection_row())
        layout.addWidget(self._create_progress())
        layout.addWidget(self.status_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Rebuilding a sorted table on every FOUND event is expensive;
        # coalesce refreshes during a scan.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(500)
        self._refresh_timer.timeout.connect(self._load_from_db)

        self._bridge = _ScanBridge()
        self._bridge.scanEvent.connect(self._on_scan_event)
        self._scanner = BeamtimeScanner(self._bridge.scanEvent.emit)

        self._load_from_db()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _create_options_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.base_edit = QLineEdit(str(DEFAULT_BASE))

        self.gpfs_only_check = QCheckBox("Show only on GPFS")
        self.gpfs_only_check.setToolTip(
            "Hide beamtimes whose data are not on GPFS, including archived "
            "README-only references and vanished directories. Scanning always "
            "covers all beamtimes."
        )
        self.gpfs_only_check.toggled.connect(lambda _: self._load_from_db())

        self.scan_button = QPushButton("Scan")
        self.scan_button.clicked.connect(self._start_scan)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_scan)

        row.addWidget(QLabel("GPFS base:"))
        row.addWidget(self.base_edit, 1)
        row.addWidget(self.gpfs_only_check)
        row.addWidget(self.scan_button)
        row.addWidget(self.cancel_button)

        return row

    def _create_table(self) -> QTableWidget:
        self.table = QTableWidget(0, 14)
        self.table.setHorizontalHeaderLabels([
            "Listed",
            "Beamtime",
            "Beamline",
            "Year",
            "PI",
            "Title",
            "Description",
            "GPFS",
            "Tape",
            "raw",
            "processed",
            "scratch_cc",
            "metadata",
            "Path",
        ])

        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)

        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(
            self._show_context_menu
        )
        self.table.itemChanged.connect(self._on_item_changed)

        header = self.table.horizontalHeader()
        for column in range(self.COL_PATH):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_PATH, QHeaderView.Stretch)

        return self.table

    def _create_selection_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        select_all = QPushButton("List all shown")
        select_all.clicked.connect(lambda: self._set_all_listed(True))

        select_none = QPushButton("Unlist all shown")
        select_none.clicked.connect(lambda: self._set_all_listed(False))

        select_writable = QPushButton("List only writable")
        select_writable.clicked.connect(self._list_writable)

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

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cell(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        return item

    @classmethod
    def _flag(cls, value: bool | None) -> QTableWidgetItem:
        if value is None:
            return cls._cell("?")
        return cls._cell("yes" if value else "no")

    @staticmethod
    def _year(beamtime) -> str:
        parts = beamtime.core_path.parts if beamtime.core_path else ()
        return parts[-3] if len(parts) >= 3 else ""

    @staticmethod
    def _beamline(beamtime) -> str:
        parts = beamtime.core_path.parts if beamtime.core_path else ()
        return beamtime.beamline or (
            parts[-4] if len(parts) >= 4 else ""
        )

    @staticmethod
    def _pi(beamtime) -> str:
        """Return a compact PI label for the table."""
        lastname = (beamtime.pi_lastname or "").strip()
        username = (beamtime.pi_username or "").strip()
        institute = (beamtime.pi_institute or "").strip()

        if lastname and username:
            value = f"{lastname} ({username})"
        else:
            value = lastname or username

        if institute:
            if value:
                value += f" — {institute}"
            else:
                value = institute

        return value

    @staticmethod
    def _description_first_line(beamtime) -> str:
        description = beamtime.description or ""
        lines = description.splitlines()
        return lines[0].strip() if lines else ""

    @staticmethod
    def _description_tooltip(beamtime) -> str:
        return (beamtime.description or "").strip()

    # ------------------------------------------------------------------
    # Table from database
    # ------------------------------------------------------------------

    def _load_from_db(self) -> None:
        listed = self.project_manager.listed_beamtime_ids()
        entries = []

        for beamtime in self.db.list_beamtimes():
            storage = self.db.get_beamtime_storage(beamtime.id)

            if self.gpfs_only_check.isChecked() and not (
                storage is not None and storage.on_gpfs
            ):
                continue

            entries.append((beamtime, storage))

        # Sort primarily by beamline and year, with beamtime ID as a stable
        # third key.
        entries.sort(
            key=lambda entry: (
                self._beamline(entry[0]).lower(),
                self._year(entry[0]),
                entry[0].beamtime_id,
            )
        )

        self._populating = True
        try:
            self.table.setRowCount(0)
            self.table.setRowCount(len(entries))

            for row, (beamtime, storage) in enumerate(entries):
                self._fill_row(
                    row,
                    beamtime,
                    storage,
                    beamtime.id in listed,
                )
        finally:
            self._populating = False

        if not self._scanner.running:
            self.status_label.setText(
                f"{len(entries)} beamtimes shown, "
                f"{len(listed)} listed."
            )

    def _fill_row(
        self,
        row: int,
        beamtime,
        storage,
        listed: bool,
    ) -> None:
        select = QTableWidgetItem()
        select.setFlags(
            Qt.ItemIsUserCheckable
            | Qt.ItemIsEnabled
            | Qt.ItemIsSelectable
        )
        select.setCheckState(Qt.Checked if listed else Qt.Unchecked)
        select.setData(Qt.UserRole, beamtime.id)
        self.table.setItem(row, self.COL_SELECT, select)

        self.table.setItem(
            row,
            self.COL_ID,
            self._cell(beamtime.beamtime_id),
        )
        self.table.setItem(
            row,
            self.COL_BEAMLINE,
            self._cell(self._beamline(beamtime)),
        )
        self.table.setItem(
            row,
            self.COL_YEAR,
            self._cell(self._year(beamtime)),
        )
        self.table.setItem(
            row,
            self.COL_PI,
            self._cell(self._pi(beamtime)),
        )
        self.table.setItem(
            row,
            self.COL_TITLE,
            self._cell(beamtime.title or ""),
        )
        self.table.setItem(
            row,
            self.COL_DESCRIPTION,
            self._cell(self._description_first_line(beamtime)),
        )

        self.table.setItem(
            row,
            self.COL_GPFS,
            self._flag(storage.on_gpfs if storage else None),
        )
        self.table.setItem(
            row,
            self.COL_TAPE,
            self._flag(storage.on_tape if storage else None),
        )
        self.table.setItem(
            row,
            self.COL_RAW,
            self._flag(storage.raw_exists if storage else None),
        )
        self.table.setItem(
            row,
            self.COL_PROCESSED,
            self._flag(
                storage.processed_exists if storage else None
            ),
        )

        scratch = "?"
        if storage is not None and storage.scratch_cc_exists is not None:
            scratch = "no"
            if storage.scratch_cc_exists:
                scratch = (
                    "writable"
                    if storage.scratch_cc_writable
                    else "read-only"
                )

        self.table.setItem(
            row,
            self.COL_SCRATCH,
            self._cell(scratch),
        )
        self.table.setItem(
            row,
            self.COL_META,
            self._flag(bool(beamtime.metadata_json)),
        )

        path = beamtime.core_path
        path_item = self._cell(str(path) if path else "")
        path_item.setToolTip(str(path) if path else "")
        if path is not None:
            path_item.setData(Qt.UserRole, path)
        self.table.setItem(row, self.COL_PATH, path_item)

        # Preserve the complete description as a tooltip while showing only
        # its first line in the table.
        description_tooltip = self._description_tooltip(beamtime)
        if description_tooltip:
            self.table.item(
                row, self.COL_DESCRIPTION
            ).setToolTip(description_tooltip)

        if beamtime.title:
            self.table.item(row, self.COL_TITLE).setToolTip(
                beamtime.title
            )

        if self._pi(beamtime):
            self.table.item(row, self.COL_PI).setToolTip(
                self._pi(beamtime)
            )

        tooltip = description_tooltip
        if storage is not None and storage.on_gpfs is False:
            if storage.last_on_gpfs:
                last_seen = storage.last_on_gpfs.isoformat(
                    sep=" ",
                    timespec="seconds",
                )
                tooltip = (
                    f"Last on GPFS: {last_seen}\n\n"
                    f"{tooltip}"
                ).strip()

        if tooltip:
            for column in (
                self.COL_ID,
                self.COL_GPFS,
                self.COL_TAPE,
            ):
                self.table.item(row, column).setToolTip(tooltip)

        # Row colors:
        # - yellow: not on GPFS
        # - green: on GPFS and writable scratch_cc
        # - red: on GPFS and non-writable or missing scratch_cc
        color = None
        if storage is not None and storage.on_gpfs is False:
            color = COLOR_NOT_ON_GPFS
        elif storage is not None and storage.on_gpfs:
            if storage.scratch_cc_writable:
                color = COLOR_WRITABLE
            else:
                color = COLOR_READ_ONLY

        if color is not None:
            brush = QBrush(color)
            for column in range(self.table.columnCount()):
                item = self.table.item(row, column)
                if item is not None:
                    item.setBackground(brush)
                    item.setForeground(QBrush(Qt.black))

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._populating or item.column() != self.COL_SELECT:
            return

        beamtime_id = item.data(Qt.UserRole)
        if beamtime_id is None:
            return

        listed = item.checkState() == Qt.Checked

        try:
            self.project_manager.set_beamtime_listed(
                beamtime_id,
                listed,
            )
        except Exception as exc:
            log.warning(
                "Cannot update listing of %s: %s",
                beamtime_id,
                exc,
            )
            self.status_label.setText(
                f"Cannot update listing: {exc}"
            )

    def _set_listed_rows(self, predicate) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.COL_SELECT)
            if item is None:
                continue

            state = Qt.Checked if predicate(row) else Qt.Unchecked
            if item.checkState() != state:
                item.setCheckState(state)

    def _set_all_listed(self, listed: bool) -> None:
        self._set_listed_rows(lambda _row: listed)

    def _list_writable(self) -> None:
        def writable(row: int) -> bool:
            scratch = self.table.item(row, self.COL_SCRATCH)
            gpfs = self.table.item(row, self.COL_GPFS)

            return (
                scratch is not None
                and scratch.text() == "writable"
                and gpfs is not None
                and gpfs.text() == "yes"
            )

        self._set_listed_rows(writable)

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _start_scan(self) -> None:
        base = Path(
            self.base_edit.text().strip() or str(DEFAULT_BASE)
        )

        if self._scanner.running:
            self.status_label.setText(
                "Previous scan is still stopping"
            )
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
                self._known_paths[
                    path_key(beamtime.core_path)
                ] = beamtime.id

        self._known_ids = {
            beamtime.beamtime_id
            for beamtime in self.db.list_beamtimes()
        }

        self._added = 0
        self._updated = 0
        self._offloaded = 0
        self._failed = 0

        self.scan_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setVisible(True)
        self.status_label.setText(f"Scanning {base}...")

        # Always scan all candidates. The checkbox only filters the table.
        self._scanner.start(
            base=base,
            known_on_gpfs=[
                Path(path)
                for path in self._known_paths
            ],
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

        if (
            event.kind == ScanEventKind.FOUND
            and event.candidate is not None
        ):
            self._store_candidate(event.candidate)
            self._refresh_timer.start()
            return

        if (
            event.kind == ScanEventKind.OFFLOADED
            and event.current is not None
        ):
            beamtime_id = self._known_paths.get(
                path_key(event.current)
            )
            if beamtime_id is not None:
                self.project_manager.mark_beamtime_offloaded(
                    beamtime_id
                )
                self._offloaded += 1
                self._refresh_timer.start()
            return

        if event.kind in (
            ScanEventKind.FINISHED,
            ScanEventKind.CANCELLED,
            ScanEventKind.FAILED,
        ):
            self._refresh_timer.stop()
            self.progress.setVisible(False)
            self.scan_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self._load_from_db()

            summary = (
                f"{self._added} new, "
                f"{self._updated} updated, "
                f"{self._offloaded} no longer on GPFS, "
                f"{self._failed} failed"
            )

            if event.kind == ScanEventKind.FAILED:
                self.status_label.setText(
                    f"Scan failed: {event.message}; {summary}"
                )
            elif event.kind == ScanEventKind.CANCELLED:
                self.status_label.setText(
                    f"Scan cancelled; {summary}"
                )
            else:
                self.status_label.setText(
                    f"Scan finished ({event.scanned} directories); "
                    f"{summary}"
                )

    def _store_candidate(
        self,
        candidate: BeamtimeCandidate,
    ) -> None:
        is_new = candidate.beamtime_id not in self._known_ids

        try:
            self.project_manager.scan_beamtime(
                candidate.path,
                candidate=candidate,
            )
        except Exception as exc:
            self._failed += 1
            log.warning(
                "Cannot scan beamtime %s: %s",
                candidate.path,
                exc,
            )
            return

        self._known_ids.add(candidate.beamtime_id)

        if is_new:
            self._added += 1
        else:
            self._updated += 1

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _show_context_menu(self, position) -> None:
        item = self.table.itemAt(position)
        if item is None:
            return

        path_item = self.table.item(
            item.row(),
            self.COL_PATH,
        )
        path = path_item.data(Qt.UserRole) if path_item else None

        if not isinstance(path, Path):
            return

        menu = QMenu(self)
        action = menu.addAction("Open Terminal Here")
        action.triggered.connect(
            lambda: open_terminal(
                path,
                on_error=self.status_label.setText,
            )
        )
        menu.exec(
            self.table.viewport().mapToGlobal(position)
        )

    def done(self, result: int) -> None:
        self._refresh_timer.stop()
        self._scanner.cancel()
        self._scanner.wait(timeout=2.0)
        super().done(result)
