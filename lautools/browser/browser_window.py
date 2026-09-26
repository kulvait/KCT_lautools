from pathlib import Path
import subprocess
import logging

from lautools import resources_pyside

from PySide6.QtCore import Qt
from PySide6.QtGui import QActionGroup
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

from lautools.browser.create_wd_dialogs import (
    ProcessLogDialog,
    SampleSelectionDialog,
    script_command,
)

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
        # File menu
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

        # Project menu
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

        # Switch menu
        self.switch_menu = menu_bar.addMenu("&Switch")
        self.switch_menu.aboutToShow.connect(self._populate_switch_menu)
        # Workspace menu
        self.workspace_menu = menu_bar.addMenu("&Workspace")
        self.workspace_menu.aboutToShow.connect(self._populate_workspace_menu)

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

    def _list_working_directories(self):
        if self.current_location is None:
            return []
        try:
            return sorted(
                (e for e in self.current_location.path.iterdir()
                 if e.is_dir() and e.name.startswith("wd")),
                key=lambda p: p.name,
            )
        except OSError as exc:
            self.status_label.setText(f"Cannot list working directories: {exc}")
            return []

    def _populate_workspace_menu(self):
        self.workspace_menu.clear()
        if self.current_location is None:
            a = self.workspace_menu.addAction("(No project selected)")
            a.setEnabled(False)
            return
        working_dirs = self._list_working_directories()
        if not working_dirs:
            a = self.workspace_menu.addAction("(No wd* directories)")
            a.setEnabled(False)
        else:
            group = QActionGroup(self.workspace_menu)
            group.setExclusive(True)
            for wd in working_dirs:
                a = self.workspace_menu.addAction(wd.name)
                a.setCheckable(True)
                a.setChecked(
                    self.current_working_directory is not None
                    and wd == self.current_working_directory
                )
                a.setToolTip(str(wd))
                group.addAction(a)
                a.triggered.connect(
                    lambda checked=False, p=wd: self.select_working_directory(p)
                )

        self.workspace_menu.addSeparator()
        if "wd" not in [wd.name for wd in working_dirs]:
            create_wd_action = self.workspace_menu.addAction("Create wd")
            create_wd_action.triggered.connect(self.create_working_directory)
        create_custom_wd_action = self.workspace_menu.addAction("Create wd with custom suffix...")
        create_custom_wd_action.triggered.connect(self.create_working_directory_with_suffix)

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

        self.left_title = QLabel()
        if self.current_working_directory is not None:
            self.left_title.setText(f"Subdirectories of {self.current_working_directory.name}")
        else:
            self.left_title.setText("No working directory selected")
        self.left_title.setStyleSheet("font-weight: bold; padding: 4px;")
        left_layout.addWidget(self.left_title)

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

        self.location_list.itemClicked.connect(self.select_subdirectory)
        self.location_list.itemDoubleClicked.connect(self.select_subdirectory)

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

    def _load_location(self, location):
        self._reset_tab_texts()
        self.refresh_locations()
        self._restore_working_directory()
        self._update_action_states()

    def _restore_last_selected_project(self):
        if not hasattr(self.db, "get_last_selected_location"):
            log.info("Database does not support last selected location retrieval.")
            return

        location = self.db.get_last_selected_location()
        if location is None:
            log.info("No last selected location found in the database.")
            return
        self.current_location = location
        log.info(f"Restored last selected project: {self.current_location.name}")

        wd = self.db.get_working_directory(self.current_location.id)
        if wd is not None:
            wd_path = self.current_location.path / wd
            if wd_path.is_dir():
                self.current_working_directory = wd_path
                log.info(f"Restored last selected working directory: {self.current_working_directory.name}")
            else:
                self.current_working_directory = None
                log.warning(f"Last selected working directory '{wd}' does not exist in project '{self.current_location.name}'.")
        self._update_current_location_ui()
        self._load_location(location)
        if self.current_working_directory is not None:
                self.pipeline_status_tree.set_working_directory(self.current_working_directory)

    def refresh_locations(self):
        self.location_list.clear()
        if self.current_location is None:
            return
        wd = self.current_working_directory
        if wd is None:
            self.left_title.setText("Subdirectories")
            return

        self.left_title.setText(f"Subdirectories of {wd.name}")

        project_path = self.current_location.path

        try:
            wd = self.current_working_directory
            if wd is not None and wd.exists() and wd.is_dir():
                subdirs = sorted( [entry for entry in wd.iterdir() if entry.is_dir()], key=lambda p: p.name)
                for working_dir in subdirs:
                    item = QListWidgetItem(working_dir.name)
                    item.setData(Qt.UserRole, working_dir)
                    item.setToolTip(str(working_dir))
                    self.location_list.addItem(item)
                self.status_label.setText(f"Loaded {len(subdirs)} subdirectories of {wd.name}")
            else:
                self.status_label.setText("No working directory selected or it does not exist.")
        except OSError as exc:
            self.status_label.setText(f"Cannot list working directories: {exc}")
            return

        


    def select_working_directory(self, working_dir, persist=True):
        if self.current_location is None or working_dir is None:
            return
        self.current_working_directory = working_dir
        if persist:
            log.info(f"Persisting working directory selection: {working_dir.name}")
            self.db.set_working_directory(self.current_location.id, working_dir.name)

        self.location_status.setText(
            f"{self.current_location.path}  [{working_dir.name}]"
        )
        self.tasks_label.setText(
            f"Tasks\n\nSelected working directory:\n{working_dir}"
        )
        self.measurements_label.setText(
            f"Measurements\n\nSelected working directory:\n{working_dir}"
        )
        self.pipeline_status_tree.set_working_directory(working_dir)
        self.refresh_locations()
        self._update_window_title()

    def select_subdirectory(self, item):
        sub = item.data(Qt.UserRole)
        if sub:
            self.status_label.setText(f"Selected: {sub}")

    def _restore_working_directory(self):
        if self.current_location is None:
            return
        name = self.db.get_working_directory(self.current_location.id)
        if not name:
            return
        path = self.current_location.path / name
        if path.is_dir():
            self.select_working_directory(path, persist=False)
        else:
            self.db.set_working_directory(self.current_location.id, None)

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
        elif self.current_working_directory is None:
            self.setWindowTitle(f"Laupy - {self.current_location.name}")
        else:
            self.setWindowTitle(f"Laupy - {self.current_location.name} [{self.current_working_directory.name}]")

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

    def _list_raw_samples(self, raw_dir):
        program, args = script_command("--list", raw_dir)
        try:
            result = subprocess.run(
                [program, *args], capture_output=True, text=True, timeout=120
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Cannot run listing: {exc}")
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout or "Listing failed")
        return [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    def _create_named_working_directory(self, directory_name: str):
        if self.current_location is None:
            self.status_label.setText("No active project")
            return

        path = self.current_location.path / directory_name
        if path.exists():
            QMessageBox.information(
                self, "Working directory exists", f"{path} already exists."
            )
            return

        raw_dir = self.current_location.path / "raw"
        if not raw_dir.is_dir():
            QMessageBox.critical(
                self, "No raw directory", f"{raw_dir} does not exist."
            )
            return

        # 1) List available samples
        self.status_label.setText(f"Listing samples in {raw_dir}...")
        try:
            samples = self._list_raw_samples(raw_dir)
        except RuntimeError as exc:
            QMessageBox.critical(self, "Cannot list samples", str(exc))
            self.status_label.setText("Sample listing failed")
            return
        if not samples:
            QMessageBox.information(
                self, "No samples", f"No samples found in {raw_dir}."
            )
            self.status_label.setText("No samples found")
            return

        # 2) Let the user choose (all preselected)
        dialog = SampleSelectionDialog(samples, directory_name, parent=self)
        if dialog.exec() != QDialog.Accepted:
            self.status_label.setText("Working directory creation cancelled")
            return
        selected = dialog.selected_samples()
        if not selected:
            self.status_label.setText("No samples selected, nothing created")
            return

        # 3) Create wd and run the script asynchronously
        try:
            path.mkdir(parents=False, exist_ok=False)
        except OSError as exc:
            QMessageBox.critical(self, "Cannot create working directory", str(exc))
            self.status_label.setText(f"Failed to create {directory_name}")
            return

        self._run_create_process(raw_dir, path, selected)

    def _run_create_process(self, raw_dir, path, samples):
        program, args = script_command(
            "--samples", *samples, "--", raw_dir, path
        )

        def on_finished(ok):
            if ok:
                self.status_label.setText(
                    f"Created working directory {path.name} with {len(samples)} samples"
                )
            else:
                self.status_label.setText(f"Errors while populating {path.name}")
            self.select_working_directory(path)

        self.status_label.setText(f"Populating {path.name}...")
        self._create_wd_log = ProcessLogDialog(
            f"Creating {path.name}", program, args,
            on_finished=on_finished, parent=self,
        )
        self._create_wd_log.setModal(False)
        self._create_wd_log.show()


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
