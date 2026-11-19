"""
PySide6 tree widget for displaying pipeline status.
Accepts configuration from other widgets.
"""

from pathlib import Path
import json
import shutil
from typing import Any, Dict, List, Optional
import shlex
import os
import logging

from PySide6.QtCore import Qt, QTimer, QProcess, QThread, QObject, Signal
import time

from PySide6.QtWidgets import (
    QComboBox,
    QStyle,
    QStyleOptionButton,
    QStylePainter,
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QLabel,
    QPushButton,
    QCheckBox,
    QMenu,
    QMessageBox,
)
from PySide6.QtGui import QColor, QPalette, QPen

from lautools.browser.pipeline_manager import PipelineManager
from laupy.pipeline_entries import PipelineEntriesStore, cancel_selected, entry_slurm_id, retire_selected
from laupy.flow import resubmit_slurm_job
from laupy.slurm import scancel
from lautools.browser.text_viewer import TextSource, open_text_viewer

from lautools.browser.utils import open_files_mousepad, open_terminal, open_files_vim

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

if not log.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s : %(message)s",
        datefmt="%d.%m.%Y %H:%M:%S",
    )
    ch.setFormatter(formatter)
    log.addHandler(ch)

log.propagate = False


# Color mapping for job states
STATE_COLORS = {
    "PENDING": QColor(255, 255, 0),      # Yellow
    "RUNNING": QColor(0, 255, 0),        # Green
    "COMPLETED": QColor(128, 0, 128),    # Magenta
    "FAILED": QColor(255, 0, 0),         # Red
    "CANCELLED": QColor(255, 0, 0),      # Red
    "TIMEOUT": QColor(255, 0, 0),        # Red
    "REQUEUED": QColor(255, 165, 0),     # Orange
    "COMPLETING": QColor(255, 165, 0),   # Orange
}

# Auto refresh choices (label, seconds; 0 = off). Every refresh runs one
# batched sacct + squeue query for the whole working directory; SLURM sites
# discourage tight polling loops and pipeline jobs run for minutes to hours.
AUTO_REFRESH_CHOICES = [("Off", 0), ("10 s", 10), ("30 s", 30), ("1 min", 60), ("2 min", 120)]
DEFAULT_AUTO_REFRESH_S = 30
AUTO_REFRESH_SETTING = "pipeline_status.auto_refresh_s"
COUNTDOWN_TICK_MS = 100
# Showing the tab: a click-like flash of the Refresh button and a refresh after a short delay
SHOW_REFRESH_DELAY_MS = 150
FLASH_DURATION_MS = 910


class CountdownButton(QPushButton):
    """Push button whose background fills up as the next auto refresh approaches.

    progress: 0..1 fraction of the interval elapsed, or None for no fill.
    busy: a band sweeps across the button while a refresh is running.
    """

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self._progress = None
        self._busy = False
        self._flash_until = 0.0

    def flash(self, duration_ms: int = FLASH_DURATION_MS):
        """Click-like feedback: a 3 px outline for a moment."""
        self._flash_until = time.monotonic() + duration_ms / 1000.0
        self.update()
        QTimer.singleShot(duration_ms + 20, self.update)

    def set_countdown(self, progress, busy: bool = False):
        if progress != self._progress or busy or busy != self._busy:
            self._progress, self._busy = progress, busy
            self.update()

    def paintEvent(self, event):
        painter = QStylePainter(self)
        option = QStyleOptionButton()
        self.initStyleOption(option)
        painter.drawControl(QStyle.CE_PushButtonBevel, option)
        inner = self.style().subElementRect(QStyle.SE_PushButtonContents, option, self)
        # Active colour group: the button is disabled while a refresh runs, the band must stay visible
        fill = QColor(self.palette().color(QPalette.Active, QPalette.Highlight))
        if self._busy:
            fill.setAlpha(110)
            # A band one third of the width sweeps left to right, ~1.2 passes per second
            band = max(inner.width() // 3, 1)
            start = inner.x() - band + int((time.monotonic() * 1.2 % 1.0) * (inner.width() + band))
            left, right = max(start, inner.x()), min(start + band, inner.x() + inner.width())
            if right > left:
                painter.fillRect(left, inner.y(), right - left, inner.height(), fill)
        elif self._progress is not None and self.isEnabled():
            fill.setAlpha(70)
            painter.fillRect(inner.x(), inner.y(), int(inner.width() * min(max(self._progress, 0.0), 1.0)), inner.height(), fill)
        painter.drawControl(QStyle.CE_PushButtonLabel, option)
        if time.monotonic() < self._flash_until:
            pen = QPen(QColor(self.palette().color(QPalette.Active, QPalette.Highlight)), 3)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 3, 3)  # 3 px line fully inside the button

# Colors used for the busy/status indicator bar
STATUS_BAR_COLORS = {
    "idle": "#4CAF50",      # Green
    "running": "#2196F3",   # Blue
    "error": "#F44336",     # Red
}


class _PipelineRefreshWorker(QObject):
    """
    Worker executed on a background QThread that fetches pipeline entries
    via PipelineManager without blocking the UI thread.
    """

    finished = Signal(list)
    error = Signal(str)

    def __init__(self, pipeline_manager: PipelineManager, show_completed: bool,
                 show_retired: bool, update_slurm_info: bool):
        super().__init__()
        self.pipeline_manager = pipeline_manager
        self.show_completed = show_completed
        self.show_retired = show_retired
        self.update_slurm_info = update_slurm_info

    def run(self):
        try:
            entries = self.pipeline_manager.get_pipeline_entries(
                show_completed=self.show_completed,
                show_retired=self.show_retired,
                update_slurm_info=self.update_slurm_info,
            )
            self.finished.emit(entries)
        except Exception as e:
            self.error.emit(str(e))


class PipelineTreeWidget(QWidget):
    """
    Tree widget displaying pipeline status with filtering.
    
    Configuration can be passed from external widgets:
    - show_completed: bool
    - show_retired: bool
    - auto_refresh_interval: int (seconds, 0 = off)

    Auto refresh runs only while the widget is visible (its tab is current
    and the window is not minimized); stale data is refreshed as soon as it
    becomes visible again. The Refresh button background fills up towards
    the next refresh. The interval is stored in ``settings`` (an object with
    get_app_setting/set_app_setting, e.g. the lautools database) if given.
    """
    
    def __init__(self, parent=None, *, environment_provider=None, settings=None):
        super().__init__(parent)
        self.environment_provider = environment_provider
        self.settings = settings
        self.pipeline_manager: Optional[PipelineManager] = None
        self.current_working_dir: Optional[Path] = None
        
        # Configuration from external widgets
        self.config = {
            "show_completed": False,
            "show_retired": False,
            "update_slurm_info": True,
        }

        # Background refresh bookkeeping
        self._refresh_thread: Optional[QThread] = None
        self._refresh_worker: Optional[_PipelineRefreshWorker] = None
        self._refresh_pending = False  # coalesce refresh() calls while running
        self._closing = False  # Flag to indicate if the widget is closing
        
        # Auto refresh: countdown from the end of the last refresh
        self._auto_refresh_s = self._stored_auto_refresh()
        self._last_refresh_end = time.monotonic()
        self._show_refresh_pending = False  # the show-triggered refresh owns the next refresh

        self._setup_ui()

        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(COUNTDOWN_TICK_MS)
        self._countdown_timer.timeout.connect(self._on_countdown_tick)
        self._countdown_timer.start()
        
    def _setup_ui(self):
        """Create the UI components."""
        layout = QVBoxLayout(self)
        
        # Toolbar with controls
        toolbar_layout = QHBoxLayout()
        
        self.show_completed_check = QCheckBox("Show Completed")
        self.show_completed_check.stateChanged.connect(self._on_show_completed_changed)
        
        self.show_retired_check = QCheckBox("Show Retired")
        self.show_retired_check.stateChanged.connect(self._on_show_retired_changed)
        
        self.refresh_btn = CountdownButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        
        self.auto_refresh_combo = QComboBox()
        for label, seconds in AUTO_REFRESH_CHOICES:
            self.auto_refresh_combo.addItem(label, seconds)
        self.auto_refresh_combo.setCurrentIndex(self.auto_refresh_combo.findData(self._auto_refresh_s))
        self.auto_refresh_combo.setToolTip("Auto refresh interval; paused while this tab is not visible")
        self.auto_refresh_combo.currentIndexChanged.connect(self._on_auto_refresh_changed)
        
        toolbar_layout.addWidget(QLabel("Filter:"))
        toolbar_layout.addWidget(self.show_completed_check)
        toolbar_layout.addWidget(self.show_retired_check)
        toolbar_layout.addStretch()

        # Status/busy indicator bar (colored QLabel used as a small LED-like bar)
        self.status_bar = QLabel()
        self.status_bar.setFixedWidth(90)
        self.status_bar.setFixedHeight(20)
        self.status_bar.setAlignment(Qt.AlignCenter)
        self._set_status_bar("idle", "Idle")
        toolbar_layout.addWidget(self.status_bar)

        toolbar_layout.addWidget(self.refresh_btn)
        toolbar_layout.addWidget(QLabel("Auto:"))
        toolbar_layout.addWidget(self.auto_refresh_combo)
        
        layout.addLayout(toolbar_layout)
        
        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setColumnCount(6)
        self.tree.setHeaderLabels([
            "Subdir",
            "Step/Node",
            "Slurm ID",
            "Name",
            "State",
            "Details"
        ])
        self.tree.setColumnWidth(0, 50)
        self.tree.setColumnWidth(1, 60)
        self.tree.setColumnWidth(2, 80)
        self.tree.setColumnWidth(3, 120)
        self.tree.setColumnWidth(4, 100)
        self.tree.setColumnWidth(5, 300)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_tree_context_menu)
        
        layout.addWidget(self.tree)

        # Status label
        self.status_label = QLabel("No pipeline loaded")
        layout.addWidget(self.status_label)

    def _set_status_bar(self, state: str, text: str):
        """Update the colored status bar (idle=green, running=blue, error=red)."""
        color = STATUS_BAR_COLORS.get(state, STATUS_BAR_COLORS["idle"])
        self.status_bar.setText(text)
        self.status_bar.setStyleSheet(
            f"background-color: {color}; color: white; border-radius: 4px; font-weight: bold;"
        )
        
    def set_working_directory(self, working_dir: Path):
        """Set the working directory to load pipeline from."""
        self.current_working_dir = Path(working_dir)
        self.global_working_dir = self.current_working_dir.parent
        self.pipeline_manager = PipelineManager(self.current_working_dir)
        self.refresh()
    
    def set_config(self, **kwargs):
        """
        Update widget configuration from external widgets.
        
        Parameters
        ----------
        show_completed : bool
            Show completed jobs
        show_retired : bool
            Show retired jobs
        update_slurm_info : bool
            Query SLURM for latest status
        """
        self.config.update(kwargs)
        # Update checkboxes if they exist
        if "show_completed" in kwargs:
            self.show_completed_check.setChecked(kwargs["show_completed"])
        if "show_retired" in kwargs:
            self.show_retired_check.setChecked(kwargs["show_retired"])
    
    def _on_show_completed_changed(self, state):
        """Handle show_completed checkbox change."""
        self.config["show_completed"] = self.show_completed_check.isChecked()
        self.refresh()
    
    def _on_show_retired_changed(self, state):
        """Handle show_retired checkbox change."""
        self.config["show_retired"] = self.show_retired_check.isChecked()
        self.refresh()
    
    def refresh(self):
        """
        Refresh the tree with current pipeline status.

        The actual (potentially slow) data fetch runs on a background
        QThread so the UI stays responsive. If a refresh is already in
        flight, this schedules one more refresh to run right after it
        finishes instead of starting overlapping threads.
        """
        if self._closing:
            return  # Don't start new refreshes if the widget is closing
        if self.pipeline_manager is None:
            self.status_label.setText("No working directory selected")
            return

        if self._refresh_thread is not None and self._refresh_thread.isRunning():
            # A refresh is already running; remember to run again once done.
            self._refresh_pending = True
            return

        self._start_refresh_thread()

    def _start_refresh_thread(self):
        if self._closing:
            return  # Don't start new refreshes if the widget is closing
        self._refresh_pending = False
        self._set_status_bar("running", "Refreshing…")
        self.refresh_btn.setEnabled(False)

        thread = QThread(self)
        worker = _PipelineRefreshWorker(
            self.pipeline_manager,
            self.config["show_completed"],
            self.config["show_retired"],
            self.config["update_slurm_info"],
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_refresh_finished)
        worker.error.connect(self._on_refresh_error)
        # Make sure the thread quits and objects are cleaned up either way.
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._refresh_thread = thread
        self._refresh_worker = worker
        thread.start()

    def _on_thread_finished(self):
        self._refresh_thread = None
        self._refresh_worker = None
        self._last_refresh_end = time.monotonic()  # countdown restarts after each refresh
        self.refresh_btn.setEnabled(True)
        if self._refresh_pending:
            # Another refresh was requested while this one was running.
            self._start_refresh_thread()

    def _on_refresh_finished(self, entries: List[Dict[str, Any]]):
        if self._closing:
            return  # Don't start new refreshes if the widget is closing
        self._populate_tree(entries)
        self.status_label.setText(
            f"Loaded {len(entries)} pipeline entries from "
            f"{self.current_working_dir.name}"
        )
        self._set_status_bar("idle", "Idle")

    def _on_refresh_error(self, message: str):
        if self._closing:
            log.error(f"Error refreshing pipeline while closing: {message}")
            return
        self.status_label.setText(f"Error loading pipeline: {message}")
        log.error(f"Error refreshing pipeline: {message}")
        self._set_status_bar("error", "Error")
    
    def _populate_tree(self, entries: List[Dict[str, Any]]):
        """Populate the tree widget with entries."""
        self.tree.clear()
        if not entries:
            self.tree.addTopLevelItem(QTreeWidgetItem(["No entries to display"]))
            return
        for entry in entries:
            item = self._create_entry_item(entry)
            self.tree.addTopLevelItem(item)
        # Resize columns to content
        for i in range(self.tree.columnCount()):
            self.tree.resizeColumnToContents(i)
    
    def _create_entry_item(self, entry: Dict[str, Any]) -> QTreeWidgetItem:
        """Create a tree item for a pipeline entry."""
        processing_unit = entry.get("processing_unit_name", "N/A")
        processing_unit_subdir = entry.get("processing_unit_subdir", "N/A")
        step = entry.get("step", "N/A")
        job_id = entry.get("job_id", "N/A")
        slurm_info = entry.get("slurm_info", {})
        slurm_id = entry.get("slurm_id")
        dag_node_name = entry.get("dag_node_name", str(step))
        command = entry.get("command", "N/A")
        if slurm_id is None:
            slurm_id = slurm_info.get("JobID")
        job_name = slurm_info.get("JobName", "N/A")
        job_state = slurm_info.get("State", "UNKNOWN")
        stdout_file = slurm_info.get("StdOut", None)
        stderr_file = slurm_info.get("StdErr", None)
        
        item = QTreeWidgetItem([
            str(processing_unit),
            str(dag_node_name),
            str(slurm_id),
            job_name,
            job_state,
            self._format_details(entry),
        ])
        
        # Set background color based on state
        color = STATE_COLORS.get(job_state, QColor(255, 255, 255))
        # Dim the color for better readability
        color.setAlpha(100)
        for col in range(item.columnCount()):
            item.setBackground(col, color)

        #Add processing unit directory as a tooltip and sub-item
        if processing_unit is not None:
            item.setToolTip(0, str(processing_unit_subdir))
            processing_unit_item = QTreeWidgetItem(["", "", "Execution dir", str(processing_unit)])
            item.addChild(processing_unit_item)

        # If stdout and stderr files are present, add right click context menu to open them
        if stdout_file:
            stdout_item = QTreeWidgetItem(["", "", "StdOut", str(stdout_file)])
            item.addChild(stdout_item)
            
        if stderr_file:
            stderr_item = QTreeWidgetItem(["", "", "StdErr", str(stderr_file)])
            item.addChild(stderr_item)

        # Add sub-items for more details
        dag_id_item = QTreeWidgetItem(["", "", "Job ID", str(job_id)])
        item.addChild(dag_id_item)
        command_item = QTreeWidgetItem(["", "", "Command", str(command)])
        item.addChild(command_item)
        if job_state == "PENDING":
            reason = slurm_info.get("Reason", "N/A")
            if reason and reason not in ("N/A", "None", ""):
                reason_item = QTreeWidgetItem(
                    ["", "", "Reason", reason]
                )
                item.addChild(reason_item)
        elif job_state == "RUNNING":
            elapsed = slurm_info.get("Elapsed", "N/A")
            time_limit = slurm_info.get("Timelimit", "N/A")
            node_list = slurm_info.get("NodeList", "N/A")
            
            elapsed_item = QTreeWidgetItem(
                ["", "", "Elapsed", elapsed]
            )
            limit_item = QTreeWidgetItem(
                ["", "", "Time Limit", time_limit]
            )
            nodes_item = QTreeWidgetItem(
                ["", "", "Nodes", node_list]
            )
            item.addChild(elapsed_item)
            item.addChild(limit_item)
            item.addChild(nodes_item)
        
        # Add dependencies if present
        dependencies = entry.get("dependencies", [])
        if dependencies:
            deps_str = ", ".join(str(d) for d in dependencies)
            deps_item = QTreeWidgetItem(
                ["", "", "Dependencies", deps_str]
            )
            item.addChild(deps_item)
        
        # Mark if retired
        if entry.get("retired", False):
            retired_item = QTreeWidgetItem(
                ["", "", "Status", "RETIRED"]
            )
            item.addChild(retired_item)

        item.setData(0, Qt.UserRole, entry)  # Store the entry data for context menu actions
        return item

    def _show_tree_context_menu(self, position):
        """Show actions for opening the selected job's SLURM log files."""
        item = self.tree.itemAt(position)
        if item is None:
            return
        # Detail rows are children of a pipeline-entry item.
        while item.parent() is not None:
            item = item.parent()

        entry = item.data(0, Qt.UserRole)
        if not isinstance(entry, dict):
            return
        
        slurm_info = entry.get("slurm_info", {})
        job_name = slurm_info.get("JobName", "N/A")
        job_state = slurm_info.get("State", "UNKNOWN")
        processing_unit_subdir = entry.get("processing_unit_subdir", None)
        stdout_file = slurm_info.get("StdOut", None)
        stderr_file = slurm_info.get("StdErr", None)
        scriptname = entry.get("script_name", None)
        script_path = self.global_working_dir / "sbatch" / scriptname if scriptname else None
        script_path = script_path.resolve() if script_path else None
        log.info(f"script_path: {script_path}, scriptname: {scriptname}, global_working_dir: {self.global_working_dir}")
        menu = QMenu(self)
        #log.info(f"Context menu for job {job_name} (ID: {entry.get('job_id', 'N/A')}) with state {job_state} stdout: {stdout_file}, stderr: {stderr_file}")
        # Logs open in the built-in viewer; a running job's logs are followed as they grow.
        follow = job_state == "RUNNING"
        logs = [(label, path) for label, path in (("StdOut", stdout_file), ("StdErr", stderr_file)) if path]
        if len(logs) == 2:
            menu.addAction("Open Both StdOut and StdErr", lambda: self._open_logs(entry, logs, follow))
        for label, path in logs:
            menu.addAction(f"{label}: {Path(path).name}", lambda l=label, p=path: self._open_logs(entry, [(l, p)], follow))
        if logs and shutil.which("mousepad"):
            menu.addAction("Open logs in mousepad", lambda: open_files_mousepad([p for _, p in logs], on_error=lambda msg: self.status_label.setText(msg)))
        if menu.actions():
            menu.addSeparator()
        if processing_unit_subdir is not None:
            menu.addAction("Open Terminal Here", lambda: open_terminal(processing_unit_subdir, on_error=lambda msg: self.status_label.setText(msg)))
        if menu.actions():
            menu.addSeparator()
        if job_state in ("RUNNING", "PENDING"):
            menu.addAction("Cancel Job", lambda: self._cancel_job(entry))
        if job_state in ("FAILED", "CANCELLED", "TIMEOUT"):
            menu.addAction("Retire Job", lambda: self._retire_job(entry))
        if job_state in ("FAILED", "CANCELLED", "TIMEOUT"):
            menu.addAction("Resubmit Job", lambda: self._resubmit_job(entry))
        if script_path is not None and script_path.exists():
            menu.addAction("Open %s" % scriptname, lambda: open_files_vim([str(script_path)], on_error=lambda msg: self.status_label.setText(msg)))
        menu.addSeparator()
        menu.addAction("Show DAG entry JSON...", lambda: self._show_entry_json(entry))
        # If there are added actions, show the menu
        if menu.actions():
            menu.exec(self.tree.viewport().mapToGlobal(position))

    @staticmethod
    def _entry_label(entry: Dict[str, Any]) -> str:
        node = entry.get("dag_node_name", entry.get("step", ""))
        return f"{entry.get('processing_unit_name', '')} / {node} / {entry_slurm_id(entry)}"

    def _open_logs(self, entry: Dict[str, Any], logs, follow: bool):
        sources = [TextSource(title=f"{label}: {Path(path).name}", path=Path(path), header=str(path)) for label, path in logs]
        open_text_viewer(sources, title=f"Logs {self._entry_label(entry)}", follow=follow)

    def _show_entry_json(self, entry: Dict[str, Any]):
        """Show the complete DAG entry, including the refreshed slurm_info."""
        processing_unit_subdir = entry.get("processing_unit_subdir")
        header = f"DAG file: {Path(processing_unit_subdir) / 'pipeline' / 'dag.json'}" if processing_unit_subdir else ""
        # default=str keeps non-JSON values (Path, ...) inspectable.
        text = json.dumps(entry, indent=2, ensure_ascii=False, default=str)
        label = self._entry_label(entry)
        open_text_viewer([TextSource(title=f"DAG entry {label}", text=text, header=header)], title=f"DAG entry {label}")

    def _cancel_job(self, entry: Dict[str, Any]):
        """Cancel a SLURM job and record it as CANCELLED in its pipeline entries."""
        slurm_id = entry_slurm_id(entry) or entry.get("job_id")
        processing_unit_subdir = entry.get("processing_unit_subdir")
        job_id = entry.get("job_id")
        if processing_unit_subdir and job_id is not None:
            try:
                results = next(iter(cancel_selected({processing_unit_subdir: [job_id]}).values()), [])
            except Exception:
                log.exception("Failed to cancel job %s in %s", slurm_id, processing_unit_subdir)
                self.status_label.setText(f"Failed to cancel job {slurm_id}")
                return
            error = results[0]["error"] if results else "job not found in pipeline entries"
        elif slurm_id is not None:
            error = scancel(slurm_id)
        else:
            error = "no SLURM/job ID found"
        if error:
            log.error("Failed to cancel SLURM job %s: %s", slurm_id, error)
            self.status_label.setText(f"Failed to cancel job {slurm_id}: {error}")
            return
        log.info("Cancelled SLURM job %s", slurm_id)
        self.status_label.setText(f"Cancelled job {slurm_id}")
        self.refresh()

    def _retire_job(self, entry: Dict[str, Any]):
        """Mark a pipeline entry as retired."""
        slurm_id = entry_slurm_id(entry) or entry.get("job_id")
        job_id = entry.get("job_id")
        processing_unit_subdir = entry.get("processing_unit_subdir")
        if job_id is None or not processing_unit_subdir:
            log.error("Cannot retire job: no job ID or processing_unit_subdir in entry: %r", entry)
            self.status_label.setText(f"Failed to retire job {slurm_id}: incomplete pipeline entry")
            return
        try:
            retired = next(iter(retire_selected({processing_unit_subdir: [job_id]}, reason="lautools-browser").values()), [])
        except Exception:
            log.exception("Failed to retire job %s in %s", slurm_id, processing_unit_subdir)
            self.status_label.setText(f"Failed to retire job {slurm_id}")
            return
        if not retired:
            log.error("Cannot retire job %s: not found in pipeline entries of %s", slurm_id, processing_unit_subdir)
            self.status_label.setText(f"Failed to retire job {slurm_id}: job not found")
            return
        log.info("Retired SLURM job %s", slurm_id)
        self.status_label.setText(f"Retired job {slurm_id}")
        self.refresh()

    def _resubmit_job(self, entry: Dict[str, Any]):
        """Requeue/resubmit a SLURM job."""
        job_id = entry.get("job_id")
        slurm_id = entry.get("slurm_id") or job_id
        if job_id is None:
            log.error("Cannot requeue job: no SLURM/job ID found in entry: %r", entry)
            self.status_label.setText("Failed to requeue job: no job ID found")
            return
        processing_unit_subdir = entry.get("processing_unit_subdir")
        if not processing_unit_subdir:
            log.error("Cannot requeue job %s: no processing_unit_subdir in entry: %r", slurm_id, entry,)
            self.status_label.setText(f"Failed to requeue job {slurm_id}: no processing unit subdir")
            return
        try:
            env = None
            if self.environment_provider is not None:
                env = self.environment_provider()
            store = PipelineEntriesStore([processing_unit_subdir])
            PIPELINE_ENTRIES = store[processing_unit_subdir]
            old_entry = next((e for e in PIPELINE_ENTRIES if e.get("job_id") == job_id), None)
            if old_entry is None:
                log.error("Cannot requeue job %s: not found in pipeline entries of %s", job_id, processing_unit_subdir,)
                self.status_label.setText(f"Failed to requeue job {slurm_id}: job not found")
                return
            # One query for the node and its ancestors: the new job depends on their active jobs
            store.refresh(limit_to_ancestors_including_node=old_entry["dag_node_name"],
                          node_dependencies=old_entry.get("dag_node_deps", []),
                          update_retired=False, update_negative_step=False, filter_terminal_states=True)
            new_entry = resubmit_slurm_job(PIPELINE_ENTRIES, old_entry, env=env)
            new_slurm_id = new_entry.get("slurm_id") if new_entry else None
            if new_entry is None:
                log.error("Failed to requeue job %s: resubmit_slurm_job returned None", slurm_id,)
                self.status_label.setText(f"Failed to requeue job {slurm_id}")
                return
            PIPELINE_ENTRIES.append(new_entry)
            store.store()
        except Exception:
            log.exception("Failed to requeue job %s in %s", slurm_id, processing_unit_subdir,)
            self.status_label.setText(f"Failed to requeue job {slurm_id}")
            return
        log.info("Requeued original SLURM job %s as new job %s", slurm_id, new_slurm_id)
        self.status_label.setText(f"Requeued job {slurm_id} as {new_slurm_id}")
        self.refresh()

    def _format_details(self, entry: Dict[str, Any]) -> str:
        """Format details string based on job state."""
        slurm_info = entry.get("slurm_info", {})
        state = slurm_info.get("State", "UNKNOWN")
        
        if state == "PENDING":
            reason = slurm_info.get("Reason", "")
            if reason and reason not in ("N/A", "None", ""):
                return f"Reason: {reason}"
            return "Waiting to start"
        elif state == "RUNNING":
            elapsed = slurm_info.get("Elapsed", "N/A")
            time_limit = slurm_info.get("Timelimit", "N/A")
            return f"Elapsed: {elapsed} / {time_limit}"
        elif state in ("FAILED", "CANCELLED", "TIMEOUT"):
            return f"Job terminated"
        elif state == "COMPLETED":
            elapsed = slurm_info.get("Elapsed", "N/A")
            return f"Completed in {elapsed}"
        else:
            return f"State: {state}"
    
    # ------------------------------------------------------------------
    # Auto refresh
    # ------------------------------------------------------------------

    def _stored_auto_refresh(self) -> int:
        valid = {seconds for _, seconds in AUTO_REFRESH_CHOICES}
        if self.settings is not None:
            try:
                value = int(self.settings.get_app_setting(AUTO_REFRESH_SETTING, DEFAULT_AUTO_REFRESH_S))
                if value in valid:
                    return value
            except Exception:
                log.exception("Cannot read the auto refresh setting")
        return DEFAULT_AUTO_REFRESH_S

    def _on_auto_refresh_changed(self, index: int):
        self._auto_refresh_s = self.auto_refresh_combo.itemData(index) or 0
        if self.settings is not None:
            try:
                self.settings.set_app_setting(AUTO_REFRESH_SETTING, self._auto_refresh_s)
            except Exception:
                log.exception("Cannot store the auto refresh setting")
        self._on_countdown_tick()

    def showEvent(self, event):
        """Tab switched to (or window shown): flash the Refresh button and refresh right away.

        Every time, also when the last refresh was moments ago; a refresh
        already running is not started twice.
        """
        super().showEvent(event)
        if self.pipeline_manager is None:
            return
        self._show_refresh_pending = True
        QTimer.singleShot(SHOW_REFRESH_DELAY_MS, self._refresh_on_show)

    def _refresh_on_show(self):
        self._show_refresh_pending = False
        if not self._is_shown() or self.pipeline_manager is None:
            return  # switched away again during the delay
        self.refresh_btn.flash()
        if not self._refresh_running():
            self.refresh()

    def _is_shown(self) -> bool:
        """Visible to the user: tab current and window not minimized."""
        return self.isVisible() and not self.window().isMinimized()

    def _refresh_running(self) -> bool:
        return self._refresh_thread is not None and self._refresh_thread.isRunning()

    def _on_countdown_tick(self):
        interval = self._auto_refresh_s
        button = self.refresh_btn
        if self._refresh_running():
            button.set_countdown(None, busy=True)
            button.setToolTip("Refreshing…")
            return
        if interval <= 0 or self.pipeline_manager is None:
            button.set_countdown(None)
            button.setToolTip("Refresh now (auto refresh off)")
            return
        age = time.monotonic() - self._last_refresh_end
        if self._show_refresh_pending:
            return  # the tab was just shown; _refresh_on_show refreshes after its short delay
        if not self._is_shown():
            # No SLURM queries for a hidden view; refreshed when shown again if stale.
            button.set_countdown(min(age / interval, 1.0))
            button.setToolTip("Refresh now (auto refresh paused while not visible)")
            return
        if age >= interval:
            self.refresh()
            return
        button.set_countdown(age / interval)
        button.setToolTip(f"Refresh now (next auto refresh in {interval - age:.0f} s)")

    def clear(self):
        """Clear the tree and reset status."""
        self.pipeline_manager = None
        self.current_working_dir = None
        self.tree.clear()
        self.tree.addTopLevelItem(QTreeWidgetItem(["No working directory selected"]))
        self.status_label.setText("No working directory selected")
        self.refresh_btn.set_countdown(None)
        self._set_status_bar("idle", "Idle")
