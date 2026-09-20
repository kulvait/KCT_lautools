"""
PySide6 tree widget for displaying pipeline status.
Accepts configuration from other widgets.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import logging

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QLabel,
    QPushButton,
    QCheckBox,
)
from PySide6.QtGui import QColor

from pipeline_manager import PipelineManager

log = logging.getLogger(__name__)

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
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        
        auto_refresh_btn = QPushButton("Auto Refresh (5s)")
        auto_refresh_btn.clicked.connect(self._toggle_auto_refresh)
        self.auto_refresh_btn = auto_refresh_btn
        
        toolbar_layout.addWidget(QLabel("Filter:"))
        toolbar_layout.addWidget(self.show_completed_check)
        toolbar_layout.addWidget(self.show_retired_check)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(refresh_btn)
        toolbar_layout.addWidget(auto_refresh_btn)
        
        layout.addLayout(toolbar_layout)
        
        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels([
            "Step",
            "Job ID",
            "Name",
            "State",
            "Details"
        ])
        self.tree.setColumnWidth(0, 50)
        self.tree.setColumnWidth(1, 80)
        self.tree.setColumnWidth(2, 120)
        self.tree.setColumnWidth(3, 100)
        self.tree.setColumnWidth(4, 300)
        
        layout.addWidget(self.tree)
        
        # Status label
        self.status_label = QLabel("No pipeline loaded")
        layout.addWidget(self.status_label)
        
    def set_working_directory(self, working_dir: Path):
        """Set the working directory to load pipeline from."""
        self.current_working_dir = Path(working_dir)
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
        """Refresh the tree with current pipeline status."""
        if self.pipeline_manager is None:
            self.status_label.setText("No working directory selected")
            return
        
        try:
            entries = self.pipeline_manager.get_pipeline_entries(
                show_completed=self.config["show_completed"],
                show_retired=self.config["show_retired"],
                update_slurm_info=self.config["update_slurm_info"],
            )
            
            self._populate_tree(entries)
            self.status_label.setText(
                f"Loaded {len(entries)} pipeline entries from "
                f"{self.current_working_dir.name}"
            )
        except Exception as e:
            self.status_label.setText(f"Error loading pipeline: {e}")
            log.error(f"Error refreshing pipeline: {e}")
    
    def _populate_tree(self, entries: List[Dict[str, Any]]):
        """Populate the tree widget with entries."""
        self.tree.clear()
        
        if not entries:
            self.tree.addTopLevelItem(
                QTreeWidgetItem(["No entries to display"])
            )
            return
        
        for entry in entries:
            item = self._create_entry_item(entry)
            self.tree.addTopLevelItem(item)
        
        # Resize columns to content
        for i in range(self.tree.columnCount()):
            self.tree.resizeColumnToContents(i)
    
    def _create_entry_item(self, entry: Dict[str, Any]) -> QTreeWidgetItem:
        """Create a tree item for a pipeline entry."""
        step = entry.get("step", "N/A")
        job_id = entry.get("job_id", "N/A")
        slurm_info = entry.get("slurm_info", {})
        job_name = slurm_info.get("JobName", "N/A")
        job_state = slurm_info.get("State", "UNKNOWN")
        
        item = QTreeWidgetItem([
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
        
        # Add sub-items for more details
        if job_state == "PENDING":
            reason = slurm_info.get("Reason", "N/A")
            if reason and reason not in ("N/A", "None", ""):
                reason_item = QTreeWidgetItem(
                    ["", "", "", "Reason", reason]
                )
                item.addChild(reason_item)
        elif job_state == "RUNNING":
            elapsed = slurm_info.get("Elapsed", "N/A")
            time_limit = slurm_info.get("Timelimit", "N/A")
            node_list = slurm_info.get("NodeList", "N/A")
            
            elapsed_item = QTreeWidgetItem(
                ["", "", "", "Elapsed", elapsed]
            )
            limit_item = QTreeWidgetItem(
                ["", "", "", "Time Limit", time_limit]
            )
            nodes_item = QTreeWidgetItem(
                ["", "", "", "Nodes", node_list]
            )
            item.addChild(elapsed_item)
            item.addChild(limit_item)
            item.addChild(nodes_item)
        
        # Add dependencies if present
        dependencies = entry.get("dependencies", [])
        if dependencies:
            deps_str = ", ".join(str(d) for d in dependencies)
            deps_item = QTreeWidgetItem(
                ["", "", "", "Dependencies", deps_str]
            )
            item.addChild(deps_item)
        
        # Mark if retired
        if entry.get("retired", False):
            retired_item = QTreeWidgetItem(
                ["", "", "", "Status", "RETIRED"]
            )
            item.addChild(retired_item)
        
        return item
    
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
