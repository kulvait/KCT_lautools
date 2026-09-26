"""
PySide6 tree widget for displaying pipeline status.
Accepts configuration from other widgets.
"""

from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional
import shlex
import os
import logging

from PySide6.QtCore import Qt, QTimer, QProcess, QThread, QObject, Signal
from PySide6.QtWidgets import (
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
from PySide6.QtGui import QColor

from lautools.browser.pipeline_manager import PipelineManager
from laupy.flow import load_dag, save_dag, clean_dag
from laupy.flow import update_dag_entries, resubmit_slurm_job

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
    - auto_refresh_interval: int (milliseconds)
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
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
        
        self._setup_ui()
        
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh)
        
    def _setup_ui(self):
        """Create the UI components."""
        layout = QVBoxLayout(self)
        
        # Toolbar with controls
        toolbar_layout = QHBoxLayout()
        
        self.show_completed_check = QCheckBox("Show Completed")
        self.show_completed_check.stateChanged.connect(self._on_show_completed_changed)
        
        self.show_retired_check = QCheckBox("Show Retired")
        self.show_retired_check.stateChanged.connect(self._on_show_retired_changed)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        
        auto_refresh_btn = QPushButton("Auto Refresh (5s)")
        auto_refresh_btn.clicked.connect(self._toggle_auto_refresh)
        self.auto_refresh_btn = auto_refresh_btn
        
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
        toolbar_layout.addWidget(auto_refresh_btn)
        
        layout.addLayout(toolbar_layout)
        
        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setColumnCount(6)
        self.tree.setHeaderLabels([
            "Execution Unit",
            "Step",
            "Job ID",
            "Name",
            "State",
            "Details"
        ])
        self.tree.setColumnWidth(0, 50)
        self.tree.setColumnWidth(1, 50)
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
        if self.pipeline_manager is None:
            self.status_label.setText("No working directory selected")
            return

        if self._refresh_thread is not None and self._refresh_thread.isRunning():
            # A refresh is already running; remember to run again once done.
            self._refresh_pending = True
            return

        self._start_refresh_thread()

    def _start_refresh_thread(self):
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
        self.refresh_btn.setEnabled(True)
        if self._refresh_pending:
            # Another refresh was requested while this one was running.
            self._start_refresh_thread()

    def _on_refresh_finished(self, entries: List[Dict[str, Any]]):
        self._populate_tree(entries)
        self.status_label.setText(
            f"Loaded {len(entries)} pipeline entries from "
            f"{self.current_working_dir.name}"
        )
        self._set_status_bar("idle", "Idle")

    def _on_refresh_error(self, message: str):
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
        execution_unit = entry.get("execution_unit_name", "N/A")
        execution_unit_dir = entry.get("execution_unit_dir", "N/A")
        step = entry.get("step", "N/A")
        job_id = entry.get("job_id", "N/A")
        slurm_info = entry.get("slurm_info", {})
        job_name = slurm_info.get("JobName", "N/A")
        job_state = slurm_info.get("State", "UNKNOWN")
        stdout_file = slurm_info.get("StdOut", None)
        stderr_file = slurm_info.get("StdErr", None)
        
        item = QTreeWidgetItem([
            str(execution_unit),
            str(step),
            str(job_id),
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

        #Add execution unit directory as a tooltip and sub-item
        if execution_unit is not None:
            item.setToolTip(0, str(execution_unit_dir))
            execution_unit_item = QTreeWidgetItem(["", "", "Execution dir", str(execution_unit)])
            item.addChild(execution_unit_item)

        # If stdout and stderr files are present, add right click context menu to open them
        if stdout_file:
            stdout_item = QTreeWidgetItem(["", "", "StdOut", str(stdout_file)])
            item.addChild(stdout_item)
            
        if stderr_file:
            stderr_item = QTreeWidgetItem(["", "", "StdErr", str(stderr_file)])
            item.addChild(stderr_item)

        # Add sub-items for more details
        if job_state == "PENDING":
            reason = slurm_info.get("Reason", "N/A")
            if reason and reason not in ("N/A", "None", ""):
                reason_item = QTreeWidgetItem(
                    ["", "", "", "", "Reason", reason]
                )
                item.addChild(reason_item)
        elif job_state == "RUNNING":
            elapsed = slurm_info.get("Elapsed", "N/A")
            time_limit = slurm_info.get("Timelimit", "N/A")
            node_list = slurm_info.get("NodeList", "N/A")
            
            elapsed_item = QTreeWidgetItem(
                ["", "", "", "", "Elapsed", elapsed]
            )
            limit_item = QTreeWidgetItem(
                ["", "", "", "", "Time Limit", time_limit]
            )
            nodes_item = QTreeWidgetItem(
                ["", "", "", "", "Nodes", node_list]
            )
            item.addChild(elapsed_item)
            item.addChild(limit_item)
            item.addChild(nodes_item)
        
        # Add dependencies if present
        dependencies = entry.get("dependencies", [])
        if dependencies:
            deps_str = ", ".join(str(d) for d in dependencies)
            deps_item = QTreeWidgetItem(
                ["", "", "", "", "Dependencies", deps_str]
            )
            item.addChild(deps_item)
        
        # Mark if retired
        if entry.get("retired", False):
            retired_item = QTreeWidgetItem(
                ["", "", "", "", "Status", "RETIRED"]
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
        unit_dir = entry.get("execution_unit_dir", None)
        stdout_file = slurm_info.get("StdOut", None)
        stdout_basename = Path(stdout_file).name if stdout_file else None
        stderr_file = slurm_info.get("StdErr", None)
        stderr_basename = Path(stderr_file).name if stderr_file else None
        scriptname = entry.get("script_name", None)
        script_path = self.global_working_dir / "sbatch" / scriptname if scriptname else None
        script_path = script_path.resolve() if script_path else None
        log.info(f"script_path: {script_path}, scriptname: {scriptname}, global_working_dir: {self.global_working_dir}")
        menu = QMenu(self)
        #log.info(f"Context menu for job {job_name} (ID: {entry.get('job_id', 'N/A')}) with state {job_state} stdout: {stdout_file}, stderr: {stderr_file}")
        if stdout_file is not None and stderr_file is not None:
            menu.addAction("Open Both StdOut and StdErr", lambda: (open_files_mousepad([stdout_file, stderr_file], on_error=lambda msg: self.status_label.setText(msg))))
        if stdout_file is not None:
            menu.addAction(f"StdOut: {stdout_basename}", lambda: open_files_mousepad([stdout_file], on_error=lambda msg: self.status_label.setText(msg)))
        if stderr_file is not None:
            menu.addAction(f"StdErr: {stderr_basename}", lambda: open_files_mousepad([stderr_file], on_error=lambda msg: self.status_label.setText(msg)))
        if menu.actions():
            menu.addSeparator()
        if unit_dir is not None:
            menu.addAction("Open Terminal Here", lambda: open_terminal(unit_dir, on_error=lambda msg: self.status_label.setText(msg)))
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
        # If there are added actions, show the menu
        if menu.actions():
            menu.exec(self.tree.viewport().mapToGlobal(position))

    def _cancel_job(self, entry: Dict[str, Any]):
        """Cancel a SLURM job."""
        slurm_id = entry.get("slurm_id") or entry.get("job_id")
        if slurm_id is None:
            log.error("Cannot cancel job: no SLURM/job ID found in entry: %r", entry)
            self.status_label.setText("Failed to cancel job: no job ID found")
            return
        try:
            subprocess.run(["scancel", str(slurm_id)], check=True, capture_output=True, text=True,)
        except FileNotFoundError:
            log.error("Cannot cancel job %s: 'scancel' command not found", slurm_id)
            self.status_label.setText(f"Failed to cancel job {slurm_id}: scancel not found")
            return
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            log.error("Failed to cancel SLURM job %s (return code %s): %s", slurm_id, exc.returncode, stderr or "no error message",)
            self.status_label.setText(f"Failed to cancel job {slurm_id}")
            return
        except OSError as exc:
            log.error("Failed to cancel SLURM job %s: %s", slurm_id, exc)
            self.status_label.setText(f"Failed to cancel job {slurm_id}")
            return
        log.info("Cancelled SLURM job %s", slurm_id)
        self.status_label.setText(f"Cancelled job {slurm_id}")
        self.refresh()
    
    
    def _retire_job(self, entry: Dict[str, Any]):
        """Mark a SLURM job as retired."""
        slurm_id = entry.get("slurm_id") or entry.get("job_id")
        job_id = entry.get("job_id")
        if job_id is None:
            log.error("Cannot retire job: no SLURM/job ID found in entry: %r", entry)
            self.status_label.setText("Failed to cancel job: no job ID found")
            return
        execution_unit_dir = entry.get("execution_unit_dir")
        if not execution_unit_dir:
            log.error( "Cannot retire job %s: no execution_unit_dir in entry: %r", slurm_id, entry,)
            self.status_label.setText(f"Failed to retire job {slurm_id}: no execution directory")
            return
        try:
            dag_entries = load_dag(execution_unit_dir)
            update_dag_entries(dag_entries, update_retired=False, update_negative_step=False, filter_terminal_states=True,)
            dag_entry = next((e for e in dag_entries if e.get("job_id") == job_id), None)
            if dag_entry is None:
                log.error("Cannot retire job %s: job not found in DAG %s", slurm_id, execution_unit_dir,)
                self.status_label.setText(f"Failed to retire job {slurm_id}: job not found")
                return
            dag_entry["retired"] = True
            save_dag(execution_unit_dir, dag_entries)
        except Exception:
            log.exception("Failed to retire job %s in %s", slurm_id, execution_unit_dir,)
            self.status_label.setText(f"Failed to retire job {slurm_id}")
            return
        log.info("Retired SLURM job %s", slurm_id)
        self.status_label.setText(f"Retired job {slurm_id}")
        self.refresh()
    
    def _resubmit_job(self, entry: Dict[str, Any]):
        """Requeue/resubmit a SLURM job."""
        job_id = entry.get("job_id")
        if job_id is None:
            log.error("Cannot requeue job: no SLURM/job ID found in entry: %r", entry)
            self.status_label.setText("Failed to requeue job: no job ID found")
            return
        execution_unit_dir = entry.get("execution_unit_dir")
        if not execution_unit_dir:
            log.error("Cannot requeue job %s: no execution_unit_dir in entry: %r", slurm_id, entry,)
            self.status_label.setText(f"Failed to requeue job {slurm_id}: no execution directory")
            return
        try:
            dag_entries = load_dag(execution_unit_dir)
            update_dag_entries(dag_entries, update_retired=False, update_negative_step=False, filter_terminal_states=True,)
            dag_entry = next((e for e in dag_entries if e.get("job_id") == job_id), None)
            if dag_entry is None:
                log.error("Cannot requeue job %s: job not found in DAG %s", job_id, execution_unit_dir,)
                self.status_label.setText(f"Failed to requeue job {slurm_id}: job not found")
                return
            new_entry = resubmit_slurm_job(dag_entries, dag_entry)
            if new_entry is None:
                log.error("Failed to requeue job %s: resubmit_slurm_job returned None", slurm_id,)
                self.status_label.setText(f"Failed to requeue job {slurm_id}")
                return
            dag_entries.append(new_entry)
            save_dag(execution_unit_dir, dag_entries)
        except Exception:
            log.exception("Failed to requeue job %s in %s", slurm_id, execution_unit_dir,)
            self.status_label.setText(f"Failed to requeue job {slurm_id}")
            return
        log.info("Requeued SLURM job %s", slurm_id)
        self.status_label.setText(f"Requeued job {slurm_id}")
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
    
    def _toggle_auto_refresh(self):
        """Toggle auto-refresh timer."""
        if self.refresh_timer.isActive():
            self.refresh_timer.stop()
            self.auto_refresh_btn.setText("Auto Refresh (5s)")
        else:
            self.refresh_timer.start(5000)  # 5 seconds
            self.auto_refresh_btn.setText("Auto Refresh (ON)")

    def clear(self):
        """Clear the tree and reset status."""
        self.pipeline_manager = None
        self.current_working_dir = None
        self.tree.clear()
        self.tree.addTopLevelItem(QTreeWidgetItem(["No working directory selected"]))
        self.status_label.setText("No working directory selected")
        self.refresh_timer.stop()
        self.auto_refresh_btn.setText("Auto Refresh (5s)")
        self._set_status_bar("idle", "Idle")
