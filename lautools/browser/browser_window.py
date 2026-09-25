from pathlib import Path
import subprocess
import logging

from lautools import resources_pyside

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lautools.browser.project_config_dialog import ProjectConfigDialog
from lautools.browser.project_manager import ProjectManager
from lautools.browser.pipeline_tree_widget import PipelineTreeWidget

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


class BrowserWindow(QMainWindow):
    def __init__(self, db):
        super().__init__()

        self.db = db
        self.project_manager = ProjectManager(self.db)
        self.current_location = None
        self.current_working_directory = None

        self.setWindowTitle("Laupy")
        self.resize(1100, 700)

        self._create_menu()
        self._create_ui()
        self._create_status_bar()

        self._restore_last_selected_project()
        if self.current_location is None:
            self.status_label.setText("No project selected")
        self._update_action_states()

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _create_menu(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")

        self.open_action = file_menu.addAction("Open Project...")
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self.open_location)

        self.close_action = file_menu.addAction("Close Project")
        self.close_action.setShortcut("Ctrl+W")
        self.close_action.triggered.connect(self.close_location)

        file_menu.addSeparator()

        self.exit_action = file_menu.addAction("Exit App")
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.triggered.connect(self.close)

        project_menu = menu_bar.addMenu("&Project")

        self.configure_action = project_menu.addAction("Configure...")
        self.configure_action.triggered.connect(self.configure_project)

        self.open_terminal_action = project_menu.addAction("Open Terminal")
        self.open_terminal_action.triggered.connect(self.open_terminal)

        project_menu.addSeparator()

        self.create_wd_action = project_menu.addAction("Create wd")
        self.create_wd_action.triggered.connect(
            self.create_working_directory
        )

        self.create_custom_wd_action = project_menu.addAction(
            "Create wd with custom suffix..."
        )
        self.create_custom_wd_action.triggered.connect(
            self.create_working_directory_with_suffix
        )

        self.switch_menu = menu_bar.addMenu("&Switch")
        self.switch_menu.aboutToShow.connect(self._populate_switch_menu)

    def _populate_switch_menu(self):
        self.switch_menu.clear()

        locations = self.db.list_recent_locations() if self.db else []

        if not locations:
            action = self.switch_menu.addAction("(No saved projects)")
            action.setEnabled(False)
            return

        for location in locations:
            action = self.switch_menu.addAction(location.name)
            action.setToolTip(str(location.path))
            action.triggered.connect(
                lambda checked=False, loc=location: self.open_recent_location(loc)
            )

    def open_recent_location(self, location):
        self.db.update_last_access(location.id)

        if hasattr(self.db, "set_last_selected"):
            self.db.set_last_selected(location.id)

        refreshed = self.db.get_location(location.id)

        self.current_location = refreshed
        self.current_working_directory = None

        self._update_current_location_ui()
        self._load_location(refreshed)

    # ------------------------------------------------------------------
    # Main UI
    # ------------------------------------------------------------------

    def _create_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_title = QLabel("Working directories")
        left_title.setStyleSheet(
            "font-weight: bold; padding: 4px;"
        )

        left_layout.addWidget(left_title)

        self.location_list = QListWidget()
        self.location_list.setMinimumWidth(250)

        left_layout.addWidget(self.location_list)

        splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        self.tasks_tab = self._create_tasks_tab()
        self.measurements_tab = self._create_measurements_tab()
        self.pipeline_tab = self._create_pipeline_tab()
        self.status_tab = self._create_status_tab()

        self.tabs.addTab(self.tasks_tab, "Tasks")
        self.tabs.addTab(self.measurements_tab, "Measurements")
        self.tabs.addTab(self.pipeline_tab, "Pipeline")
        self.tabs.addTab(self.status_tab, "Status")
        # make Status first opened tab
        self.tabs.setCurrentWidget(self.status_tab)

        right_layout.addWidget(self.tabs)

        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        self.location_list.itemClicked.connect(
            self.select_working_directory
        )
        self.location_list.itemDoubleClicked.connect(
            self.select_working_directory
        )

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------

    def _create_tasks_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.tasks_label = QLabel(
            "Tasks\n\n"
            "Tasks associated with the selected working directory "
            "will appear here."
        )
        self.tasks_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        layout.addWidget(self.tasks_label)
        layout.addStretch()

        return widget

    def _create_measurements_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.measurements_label = QLabel(
            "Measurements\n\n"
            "Measurements for the selected working directory "
            "will appear here."
        )
        self.measurements_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        layout.addWidget(self.measurements_label)
        layout.addStretch()

        return widget

    def _create_pipeline_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.pipeline_tab_label = QLabel(
            "Pipeline\n\n"
            "Pipeline information for the selected working directory "
            "will appear here."
        )
        self.pipeline_tab_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self.pipeline_tab_label)
        layout.addStretch()
        return widget

    def _create_status_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.pipeline_status_tree = PipelineTreeWidget()
        layout.addWidget(self.pipeline_status_tree)
        return widget
        
    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_label = QLabel("Ready")
        self.location_status = QLabel("No project selected")

        self.status_bar.addWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.location_status)

    # ------------------------------------------------------------------
    # Project / working directories
    # ------------------------------------------------------------------

    def _restore_last_selected_project(self):
        if not hasattr(self.db, "get_last_selected_location"):
            log.info(
                "Database does not support last selected location retrieval."
            )
            return

        location = self.db.get_last_selected_location()
        if location is None:
            log.info("No last selected location found in the database.")
            return

        self.current_location = location
        log.info(
            f"Restored last selected project: {self.current_location.name}"
        )
        self.current_working_directory = None
        self._update_current_location_ui()
        self._load_location(location)

    def refresh_locations(self):
        self.location_list.clear()

        if self.current_location is None:
            return

        project_path = self.current_location.path

        try:
            working_dirs = sorted(
                [
                    entry for entry in project_path.iterdir()
                    if entry.is_dir() and entry.name.startswith("wd")
                ],
                key=lambda p: p.name,
            )
        except OSError as exc:
            self.status_label.setText(
                f"Cannot list working directories: {exc}"
            )
            return

        if not working_dirs:
            self.status_label.setText(
                f"No working directories starting with 'wd' in {project_path}"
            )
            return

        for working_dir in working_dirs:
            item = QListWidgetItem(working_dir.name)
            item.setData(Qt.UserRole, working_dir)
            item.setToolTip(str(working_dir))
            self.location_list.addItem(item)

        self.status_label.setText(
            f"Loaded {len(working_dirs)} working directories"
        )

    def select_working_directory(self, item):
        working_dir = item.data(Qt.UserRole)

        if not working_dir:
            return

        self.current_working_directory = working_dir

        self.location_status.setText(str(self.current_location.path))
        self.status_label.setText(
            f"Selected working directory: {working_dir.name}"
        )

        self.tasks_label.setText(
            f"Tasks\n\nSelected working directory:\n{working_dir}"
        )
        self.measurements_label.setText(
            f"Measurements\n\nSelected working directory:\n{working_dir}"
        )
        # Load pipeline for this working directory
        self.pipeline_status_tree.set_working_directory(working_dir)

    def _load_location(self, location):
        """
        Load tasks, measurements and pipeline data for `location`.
        """
        self.refresh_locations()
        self._reset_tab_texts()
        self._update_action_states()

    def _update_window_title(self):
        if self.current_location is None:
            self.setWindowTitle("Laupy")
        else:
            self.setWindowTitle(
                f"Laupy - {self.current_location.name}"
            )

    # ------------------------------------------------------------------
    # File actions
    # ------------------------------------------------------------------

    def open_location(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Open Project Directory",
        )

        if not directory:
            return

        path = Path(directory).resolve()

        self._add_location(path)

        location = self.db.get_location_by_path(path)
        if location is None:
            self.status_label.setText("Could not open project")
            return

        self.db.update_last_access(location.id)

        if hasattr(self.db, "set_last_selected"):
            self.db.set_last_selected(location.id)

        self.current_location = self.db.get_location(location.id)
        self.current_working_directory = None

        self._update_current_location_ui()
        self._load_location(self.current_location)

    def close_location(self):
        self.current_location = None
        self.current_working_directory = None

        self.location_list.clear()
        self.location_status.setText("No project selected")
        self.status_label.setText("Ready")
        self.tabs.setCurrentIndex(0)
        self._reset_tab_texts()
        self._update_current_location_ui()

    # ------------------------------------------------------------------
    # Project actions
    # ------------------------------------------------------------------

    def configure_project(self):
        if self.current_location is None:
            self.status_label.setText("No active project to configure")
            return

        project_info = self.project_manager.get_project_info(self.current_location)
        dialog = ProjectConfigDialog(self.project_manager, self.current_location, project_info, parent=self)

        if dialog.exec() != QDialog.Accepted:
            return

        new_name = dialog.project_name()
        if new_name and new_name != self.current_location.name:
            self.db.rename_location(self.current_location.id, new_name)

        if hasattr(dialog, "project_description") and hasattr(
            self.db, "update_description"
        ):
            new_description = dialog.project_description()
            self.db.update_description(
                self.current_location.id,
                new_description or None,
            )

        self.current_location = self.db.get_location(
            self.current_location.id
        )
        self._update_current_location_ui()
        self.status_label.setText(
            f"Updated project: {self.current_location.name}"
        )

    def create_working_directory(self):
        self._create_named_working_directory("wd")

    def create_working_directory_with_suffix(self):
        if self.current_location is None:
            self.status_label.setText("No active project")
            return

        suffix, ok = QInputDialog.getText(
            self,
            "Create working directory",
            "Suffix for wd directory:",
        )
        if not ok:
            return

        suffix = suffix.strip()
        if not suffix:
            self.status_label.setText("Empty suffix, nothing created")
            return

        safe_suffix = suffix.replace(" ", "_")
        self._create_named_working_directory(f"wd_{safe_suffix}")

    def _create_named_working_directory(self, directory_name: str):
        if self.current_location is None:
            self.status_label.setText("No active project")
            return

        path = self.current_location.path / directory_name

        if path.exists():
            QMessageBox.information(
                self,
                "Working directory exists",
                f"{path} already exists.",
            )
            return

        try:
            path.mkdir(parents=False, exist_ok=False)
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Cannot create working directory",
                str(exc),
            )
            self.status_label.setText(
                f"Failed to create {directory_name}"
            )
            return

        self.refresh_locations()
        self.status_label.setText(
            f"Created working directory: {directory_name}"
        )

    def open_terminal(self):
        if self.current_location is None:
            self.status_label.setText("No active project")
            return

        project_path = self.current_location.path
        try:
            subprocess.Popen(
                ["xfce4-terminal", "--working-directory", str(project_path)]
            )
            self.status_label.setText(
                f"Opened terminal in {project_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Cannot open terminal",
                f"Failed to open terminal: {e}",
            )
            self.status_label.setText("Failed to open terminal")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_location(self, path):
        self.db.add_location(path)

    def _update_current_location_ui(self):
        if self.current_location is None:
            self.location_status.setText("No project selected")
            self.status_label.setText("Ready")
        else:
            self.location_status.setText(str(self.current_location.path))
            self.status_label.setText(
                f"Selected project: {self.current_location.name}"
            )

        self._update_window_title()
        self._update_action_states()

    def _update_action_states(self):
        has_location = self.current_location is not None
        self.close_action.setEnabled(has_location)
        self.open_terminal_action.setEnabled(has_location)
        self.configure_action.setEnabled(has_location)
        self.create_wd_action.setEnabled(has_location)
        self.create_custom_wd_action.setEnabled(has_location)

    def _reset_tab_texts(self):
        self.tasks_label.setText(
            "Tasks\n\n"
            "Tasks associated with the selected working directory "
            "will appear here."
        )
        self.measurements_label.setText(
            "Measurements\n\n"
            "Measurements for the selected working directory "
            "will appear here."
        )
        self.pipeline_status_tree.clear()
