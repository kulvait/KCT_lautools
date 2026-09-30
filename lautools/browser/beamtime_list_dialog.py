from __future__ import annotations

import logging
from pathlib import Path
import re

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
    QInputDialog,
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

COLOR_NOT_ON_GPFS = QColor(255, 245, 157)  # yellow
COLOR_WRITABLE = QColor(200, 230, 201)  # green
COLOR_READ_ONLY = QColor(255, 205, 210)  # red

LABEL_COLUMN_WIDTH = 150
PI_COLUMN_WIDTH = 120
MODALITY_COLUMN_WIDTH = 120
TITLE_COLUMN_WIDTH = 320
DESCRIPTION_COLUMN_WIDTH = 250
PATH_COLUMN_WIDTH = 340


class _NumericItem(QTableWidgetItem):
    """Table item that sorts its UserRole value numerically when possible."""

    def __lt__(self, other: QTableWidgetItem) -> bool:
        left = self.data(Qt.UserRole)
        right = other.data(Qt.UserRole)

        try:
            return int(left) < int(right)
        except (TypeError, ValueError):
            return self.text().casefold() < other.text().casefold()


class _ScanBridge(QObject):
    """Deliver worker-thread events to the GUI thread."""

    scanEvent = Signal(object)


class BeamtimeListDialog(QDialog):
    """Database-backed beamtime list updated by explicit GPFS scans.

    The tick in the first column mirrors lautools_app_listed_beamtime and is
    written immediately when toggled.
    """

    (
        COL_SELECT,
        COL_ID,
        COL_BEAMLINE,
        COL_YEAR,
        COL_GPFS,
        COL_TITLE,
        COL_PI,
        COL_MODALITY,
        COL_DESCRIPTION,
        COL_RETENTION,
        COL_TAPE,
        COL_RAW,
        COL_PROCESSED,
        COL_SCRATCH,
        COL_META,
        COL_PATH,
    ) = range(16)

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
        self.setWindowFlags(
            self.windowFlags()
            | Qt.CustomizeWindowHint
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinMaxButtonsHint
            | Qt.WindowCloseButtonHint
        )
        self.setSizeGripEnabled(True)
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

        # Avoid rebuilding and re-sorting the table for every individual
        # FOUND event while a scan is running.
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

        self.gpfs_only_check = QCheckBox("Show on GPFS")
        self.gpfs_only_check.setToolTip(
            "Hide beamtimes whose data are not on GPFS, including archived "
            "references and vanished directories. Scanning always examines "
            "all available beamtime directories."
        )
        self.gpfs_only_check.toggled.connect(
            lambda _checked: self._load_from_db()
        )

        self.scan_button = QPushButton("Scan")
        self.scan_button.setToolTip(
            "Crawl the GPFS base for beamtimes and update the database."
        )
        self.scan_button.clicked.connect(self._start_scan)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setToolTip(
            "Re-inspect beamtimes already in the database without crawling "
            "GPFS. New beamtimes are not discovered."
        )
        self.refresh_button.clicked.connect(self._start_refresh)
 
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_scan)

        row.addWidget(QLabel("GPFS base:"))
        row.addWidget(self.base_edit, 1)
        row.addWidget(self.gpfs_only_check)
        row.addWidget(self.scan_button)
        row.addWidget(self.refresh_button)
        row.addWidget(self.cancel_button)

        return row

    def _create_table(self) -> QTableWidget:
        self.table = QTableWidget(0, 16)
        self.table.setHorizontalHeaderLabels([
            "Listed",
            "Beamtime ID",
            "Beamline",
            "Year",
            "GPFS",
            "Title",
            "PI",
            "Modality",
            "Description",
            "Retention",
            "Tape",
            "raw",
            "processed",
            "scratch_cc",
            "metadata",
            "Path",
        ])

        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.verticalHeader().setVisible(False)

        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(
            self._show_context_menu
        )
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)

        self.table.itemChanged.connect(self._on_item_changed)

        header = self.table.horizontalHeader()
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.setStretchLastSection(False)

        # All columns remain manually adjustable.
        for column in range(self.table.columnCount()):
            header.setSectionResizeMode(
                column,
                QHeaderView.Interactive,
            )

        self.table.setColumnWidth(self.COL_SELECT, LABEL_COLUMN_WIDTH)
        self.table.setColumnWidth(self.COL_ID, 115)
        self.table.setColumnWidth(self.COL_BEAMLINE, 85)
        self.table.setColumnWidth(self.COL_YEAR, 65)
        self.table.setColumnWidth(self.COL_GPFS, 65)
        self.table.setColumnWidth(self.COL_TITLE, TITLE_COLUMN_WIDTH)
        self.table.setColumnWidth(self.COL_PI, PI_COLUMN_WIDTH)
        self.table.setColumnWidth(
            self.COL_MODALITY,
            MODALITY_COLUMN_WIDTH,
        )
        self.table.setColumnWidth(
            self.COL_DESCRIPTION,
            DESCRIPTION_COLUMN_WIDTH,
        )
        self.table.setColumnWidth(self.COL_RETENTION, 60)
        self.table.setColumnWidth(self.COL_TAPE, 60)
        self.table.setColumnWidth(self.COL_RAW, 60)
        self.table.setColumnWidth(self.COL_PROCESSED, 85)
        self.table.setColumnWidth(self.COL_SCRATCH, 100)
        self.table.setColumnWidth(self.COL_META, 80)
        self.table.setColumnWidth(self.COL_PATH, PATH_COLUMN_WIDTH)

        # The initial table order is provided by _load_from_db. Once enabled,
        # the user can sort by clicking any column header.
        self.table.setSortingEnabled(True)
        header.setSortIndicator(
            self.COL_BEAMLINE,
            Qt.AscendingOrder,
        )

        return self.table

    def _create_selection_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        select_all = QPushButton("List all shown")
        select_all.clicked.connect(
            lambda: self._set_all_listed(True)
        )

        select_none = QPushButton("Unlist all shown")
        select_none.clicked.connect(
            lambda: self._set_all_listed(False)
        )

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
    # Display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cell(
        text: str,
        alignment=Qt.AlignVCenter | Qt.AlignLeft,
    ) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(alignment)
        return item

    @classmethod
    def _flag(cls, value: bool | None) -> QTableWidgetItem:
        text = "?" if value is None else "yes" if value else "no"
        return cls._cell(text, Qt.AlignCenter)

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
        lastname = (beamtime.pi_lastname or "").strip()
        username = (beamtime.pi_username or "").strip()
        institute = (beamtime.pi_institute or "").strip()
        if lastname and username:
            result = f"{lastname} ({username})"
        else:
            result = lastname or username
        if not result and institute:
            result = institute
        return result

    @classmethod
    def _default_label(cls, beamtime) -> str:
        parts = (
            cls._beamline(beamtime),
            cls._year(beamtime),
            beamtime.beamtime_id,
        )
        return "_".join(part for part in parts if part)

    @classmethod
    def _label(cls, beamtime) -> str:
        return beamtime.label or cls._default_label(beamtime)

    @staticmethod
    def _retention(beamtime) -> str:
        return (beamtime.retention_period or "").strip()

    @staticmethod
    def _description_first_line(beamtime) -> str:
        description = (beamtime.description or "").strip()
        if not description:
            return ""

        for line in description.splitlines():
            line = line.strip()
            if line:
                return line

        return ""

    @staticmethod
    def _modality(beamtime) -> str:
        """Return beamline setup without Hereon ownership labels."""
        text = (beamtime.beamline_setup or "").strip()
        if not text:
            return ""
        # Remove "Hereon" together with an adjacent separator.
        # Handles:
        #   Hereon - Microtomography (EH4)
        #   Microtomography (Hereon - EH4)
        #   Microtomography (EH4 - Hereon)
        #   Energy dispersive diffraction (LEDDI type diffractometer - Hereon)
        text = re.sub(
            r"\bhereon\b\s*[-–—:]?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        # Clean up separators that may now be left behind.
        text = re.sub(r"\(\s*[-–—:]\s*", "(", text)
        text = re.sub(r"\s*[-–—:]\s*\)", ")", text)
        # General whitespace/punctuation cleanup.
        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip(" \t-–—:")


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

        # Default order before the user selects a header sort.
        entries.sort(
            key=lambda entry: (
                self._beamline(entry[0]).casefold(),
                self._year(entry[0]),
                entry[0].beamtime_id,
            )
        )

        header = self.table.horizontalHeader()
        sort_column = header.sortIndicatorSection()
        sort_order = header.sortIndicatorOrder()

        self._populating = True
        self.table.setSortingEnabled(False)

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
            self.table.setSortingEnabled(True)
            self.table.sortItems(sort_column, sort_order)
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
        listed_item = QTableWidgetItem(self._label(beamtime))
        listed_item.setFlags(
            Qt.ItemIsUserCheckable
            | Qt.ItemIsEnabled
            | Qt.ItemIsSelectable
        )
        listed_item.setCheckState(
            Qt.Checked if listed else Qt.Unchecked
        )
        listed_item.setData(Qt.UserRole, beamtime.id)
        listed_item.setToolTip(f"Double-click or right-click to rename.\n Default label: {self._default_label(beamtime)}")
        self.table.setItem(row, self.COL_SELECT, listed_item)

        beamtime_id = beamtime.beamtime_id
        id_item = _NumericItem(beamtime_id)
        id_item.setData(
            Qt.UserRole,
            int(beamtime_id) if beamtime_id.isdigit() else beamtime_id,
        )
        id_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, self.COL_ID, id_item)

        self.table.setItem(
            row,
            self.COL_BEAMLINE,
            self._cell(
                self._beamline(beamtime),
                Qt.AlignCenter,
            ),
        )

        year = self._year(beamtime)
        year_item = _NumericItem(year)
        year_item.setData(
            Qt.UserRole,
            int(year) if year.isdigit() else year,
        )
        year_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, self.COL_YEAR, year_item)

        self.table.setItem(
            row,
            self.COL_GPFS,
            self._flag(
                storage.on_gpfs if storage is not None else None
            ),
        )

        title = (beamtime.title or "").strip()
        title_item = self._cell(title)
        if title:
            title_item.setToolTip(title)
        self.table.setItem(row, self.COL_TITLE, title_item)

        pi = self._pi(beamtime)
        pi_item = self._cell(pi)
        if pi:
            pi_item.setToolTip(pi)
        self.table.setItem(row, self.COL_PI, pi_item)

        original_modality = (beamtime.beamline_setup or "").strip()
        modality = self._modality(beamtime)
        modality_item = self._cell(modality)
        if original_modality:
            modality_item.setToolTip(original_modality)
        self.table.setItem(
            row,
            self.COL_MODALITY,
            modality_item,
        )

        description = self._description_first_line(beamtime)
        description_item = self._cell(description)
        if beamtime.description:
            description_item.setToolTip(beamtime.description)
        self.table.setItem(
            row,
            self.COL_DESCRIPTION,
            description_item,
        )
        self.table.setItem(row, self.COL_RETENTION, self._cell(self._retention(beamtime), Qt.AlignCenter, ),)

        self.table.setItem(
            row,
            self.COL_TAPE,
            self._flag(
                storage.on_tape if storage is not None else None
            ),
        )
        self.table.setItem(
            row,
            self.COL_RAW,
            self._flag(
                storage.raw_exists if storage is not None else None
            ),
        )
        self.table.setItem(
            row,
            self.COL_PROCESSED,
            self._flag(
                storage.processed_exists
                if storage is not None
                else None
            ),
        )

        scratch = "?"
        if (
            storage is not None
            and storage.scratch_cc_exists is not None
        ):
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
            self._cell(scratch, Qt.AlignCenter),
        )

        self.table.setItem(
            row,
            self.COL_META,
            self._flag(bool(beamtime.metadata_json)),
        )

        path = beamtime.core_path
        path_text = str(path) if path else ""
        path_item = self._cell(path_text)
        path_item.setToolTip(path_text)

        if path is not None:
            path_item.setData(Qt.UserRole, path)

        self.table.setItem(
            row,
            self.COL_PATH,
            path_item,
        )

        tooltip = (beamtime.description or "").strip()

        if (
            storage is not None
            and storage.on_gpfs is False
            and storage.last_on_gpfs
        ):
            last_seen = storage.last_on_gpfs.isoformat(
                sep=" ",
                timespec="seconds",
            )
            tooltip = (
                f"Last on GPFS: {last_seen}\n\n{tooltip}"
            ).strip()

        if tooltip:
            for column in (
                self.COL_ID,
                self.COL_GPFS,
                self.COL_TAPE,
            ):
                item = self.table.item(row, column)
                if item is not None:
                    item.setToolTip(tooltip)

        color = None

        if storage is not None and storage.on_gpfs is False:
            color = COLOR_NOT_ON_GPFS
        elif storage is not None and storage.on_gpfs:
            color = (
                COLOR_WRITABLE
                if storage.scratch_cc_writable
                else COLOR_READ_ONLY
            )

        if color is not None:
            brush = QBrush(color)

            for column in range(self.table.columnCount()):
                item = self.table.item(row, column)

                if item is not None:
                    item.setBackground(brush)
                    item.setForeground(QBrush(Qt.black))

    # ------------------------------------------------------------------
    # Renaming labels
    # ------------------------------------------------------------------

    def _on_cell_double_clicked(self, row: int, column: int) -> None:
        if column == self.COL_SELECT:
            self._rename_label(row)

    def _rename_label(self, row: int) -> None:
        item = self.table.item(row, self.COL_SELECT)
        if item is None:
            return

        beamtime_id = item.data(Qt.UserRole)
        beamtime = self.db.get_beamtime(beamtime_id)
        if beamtime is None:
            return

        text, accepted = QInputDialog.getText(
            self,
            "Rename beamtime",
            f"Label for {beamtime.beamtime_id}\n"
            f"(empty restores {self._default_label(beamtime)}):",
            text=self._label(beamtime),
        )
        if not accepted:
            return

        text = text.strip()
        try:
            self.project_manager.set_beamtime_label(
                beamtime_id,
                text or None,
            )
        except Exception as exc:
            log.warning("Cannot rename beamtime %s: %s", beamtime_id, exc)
            self.status_label.setText(f"Cannot rename: {exc}")
            return
        self._load_from_db()
    
    def _reset_label_defult(self, row: int) -> None:
        item = self.table.item(row, self.COL_SELECT)
        if item is None:
            return
        beamtime_id = item.data(Qt.UserRole)
        beamtime = self.db.get_beamtime(beamtime_id)
        try:
            self.project_manager.set_beamtime_label(beamtime_id, self._default_label(beamtime),
            )
        except Exception as exc:
            log.warning("Cannot rename beamtime %s: %s", beamtime_id, exc)
            self.status_label.setText(f"Cannot rename: {exc}")
            return
        self._load_from_db()

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def _on_item_changed(
        self,
        item: QTableWidgetItem,
    ) -> None:
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

            state = (
                Qt.Checked
                if predicate(row)
                else Qt.Unchecked
            )

            if item.checkState() != state:
                item.setCheckState(state)

    def _set_all_listed(self, listed: bool) -> None:
        self._set_listed_rows(
            lambda _row: listed
        )

    def _list_writable(self) -> None:
        def writable(row: int) -> bool:
            scratch = self.table.item(
                row,
                self.COL_SCRATCH,
            )
            gpfs = self.table.item(
                row,
                self.COL_GPFS,
            )

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
    def _collect_known(self) -> list:
        """Snapshot database state before a scan or refresh."""
        beamtimes = self.db.list_beamtimes()
        self._known_paths = {}
        for beamtime in beamtimes:
            storage = self.db.get_beamtime_storage(beamtime.id)
            if (
                beamtime.core_path is not None
                and storage is not None
                and storage.on_gpfs is True
            ):
                self._known_paths[path_key(beamtime.core_path)] = beamtime.id
        self._known_ids = {beamtime.beamtime_id for beamtime in beamtimes}
        self._added = self._updated = self._offloaded = self._failed = 0
        self._skipped = 0
        return beamtimes

    def _set_busy(self, busy: bool) -> None:
        self.scan_button.setEnabled(not busy)
        self.refresh_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.progress.setVisible(busy)

    def _start_scan(self) -> None:
        base = Path(self.base_edit.text().strip() or str(DEFAULT_BASE))

        if self._scanner.running:
            self.status_label.setText("Previous scan is still stopping")
            return

        if not base.is_dir():
            self.status_label.setText(
                f"Not a directory: {base}"
            )
            return
        self._collect_known()
        self._operation = "Scan"
        self._set_busy(True)
        self.status_label.setText(f"Scanning {base}...")

        self._scanner.start(
            base=base,
            known_on_gpfs=[
                Path(path)
                for path in self._known_paths
            ],
        )

    def _start_refresh(self) -> None:
        if self._scanner.running:
            self.status_label.setText("Previous operation is still stopping")
            return

        beamtimes = self._collect_known()
        paths = [
            beamtime.core_path
            for beamtime in beamtimes
            if beamtime.core_path is not None
        ]
        self._skipped = len(beamtimes) - len(paths)  # no stored path
        if not paths:
            self.status_label.setText("No beamtimes with a known path to refresh")
            return

        self._operation = "Refresh"
        self._set_busy(True)
        self.status_label.setText(f"Refreshing {len(paths)} beamtimes...")
        self._scanner.start_refresh(paths, known_on_gpfs=[Path(path) for path in self._known_paths],)

    def _cancel_scan(self) -> None:
        self._scanner.cancel()
        self.status_label.setText(f"Cancelling {self._operation.lower()}...")

    def _on_scan_event(
        self,
        event: ScanEvent,
    ) -> None:
        if event.kind == ScanEventKind.PROGRESS:
            if event.scanned % 25 == 0:
               verb = "Checked" if self._operation == "Scan" else "Refreshed"
               self.status_label.setText(
                   f"{verb} {event.scanned} directories, "
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
            self._set_busy(False)
            self._load_from_db()
            if self._operation == "Refresh":
                # Rows not recognized and not demonstrably gone were left as-is.
                self._skipped += max(
                    0, event.scanned - event.found - self._offloaded
                )
            summary = (
                f"{self._added} new, "
                f"{self._updated} updated, "
                f"{self._offloaded} no longer on GPFS, "
                f"{self._failed} failed"
            )
            if self._operation == "Refresh":
                summary += f", {self._skipped} skipped (unreadable or unknown)"

            op = self._operation
            if event.kind == ScanEventKind.FAILED:
                self.status_label.setText(
                    f"{op} failed: {event.message}; {summary}"
                )
            elif event.kind == ScanEventKind.CANCELLED:
                self.status_label.setText(
                    f"{op} cancelled; {summary}"
                )
            else:
                self.status_label.setText(
                    f"{op} finished "
                    f"({event.scanned} directories); {summary}"
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
        menu = QMenu(self)
        rename_action = menu.addAction("Rename label")
        rename_action.triggered.connect(lambda: self._rename_label(item.row()))
        rename_to_default = menu.addAction("Restore default label")
        rename_to_default.triggered.connect(lambda: self._reset_label_defult(item.row()))
        path_item = self.table.item(item.row(), self.COL_PATH)
        path = path_item.data(Qt.UserRole) if path_item else None
        if isinstance(path, Path):
            menu.addSeparator()
            action = menu.addAction("Open Terminal Here")
            action.triggered.connect(lambda: open_terminal(path, on_error=self.status_label.setText))
        menu.exec(self.table.viewport().mapToGlobal(position))

    def done(self, result: int) -> None:
        self._refresh_timer.stop()
        self._scanner.cancel()
        self._scanner.wait(timeout=2.0)
        super().done(result)
