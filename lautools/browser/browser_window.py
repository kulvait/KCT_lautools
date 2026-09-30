from __future__ import annotations

import logging
from pathlib import Path
import subprocess

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
    QMenu,
    QStatusBar,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lautools.browser.create_wd_dialogs import (
    ProcessLogDialog,
    SampleSelectionDialog,
    script_command,
)
from lautools.browser.beamtime_list_dialog import BeamtimeListDialog
from lautools.browser.pipeline_tree_widget import PipelineTreeWidget
from lautools.browser.project_config_dialog import ProjectConfigDialog
from lautools.browser.project_manager import ProjectManager
from lautools.browser.utils import (
    open_files_mousepad,
    open_files_vim,
    open_hdf5view,
    open_terminal,
    open_thunar,
)
from lautools.browser.size_service_qt import SizeServiceBridge
from lautools.size_service import SizeEventKind, SizeService


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

if not log.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s : %(message)s",
        datefmt="%d.%m.%Y %H:%M:%S",
    )
    handler.setFormatter(formatter)
    log.addHandler(handler)

log.propagate = False


class BrowserWindow(QMainWindow):
    """
    Main Laupy browser window.

    Projects are persisted in the database and shown in the Switch menu.
    Workspaces are discovered from the filesystem. Database workspace rows are
    used only for history, optional metadata and future size measurements.
    """

    def __init__(self, db, size_service=None):
        super().__init__()

        self.db = db
        self.project_manager = ProjectManager(db)

        self.current_project = None
        self.current_workspace = None

        self.size_service = size_service or SizeService(db.db_path, workers=2)
        self.size_service.start()
        self.size_bridge = SizeServiceBridge(self.size_service, parent=self)
        self.size_bridge.sizeEvent.connect(self._on_size_event)

        self.setWindowTitle("Laupy")
        self.resize(1100, 700)

        self._create_menu()
        self._create_ui()
        self._create_status_bar()

        self._restore_last_project()

        if self.current_project is None:
            self.status_label.setText("No project selected")

        self._update_action_states()

    # ------------------------------------------------------------------
    # Compatibility aliases
    # ------------------------------------------------------------------

    @property
    def current_location(self):
        """
        Compatibility alias.

        Older code calls the current project ``current_location``. It now
        contains a LaupyProject instance.
        """
        return self.current_project

    @current_location.setter
    def current_location(self, value):
        self.current_project = value

    @property
    def current_working_directory(self) -> Path | None:
        """
        Compatibility alias returning the selected workspace path.
        """
        if self.current_workspace is None:
            return None
        return self.current_workspace.path

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _create_menu(self):
        menu_bar = self.menuBar()
        # File
        file_menu = menu_bar.addMenu("&File")
        self.open_action = file_menu.addAction("Open Project...")
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self.open_project)
        self.close_action = file_menu.addAction("Close Project")
        self.close_action.setShortcut("Ctrl+W")
        self.close_action.triggered.connect(self.close_project)
        file_menu.addSeparator()
        self.exit_action = file_menu.addAction("Exit App")
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.triggered.connect(self.close)
        # Beamtime
        self.beamtime_menu = menu_bar.addMenu("&Beamtime")
        self.beamtime_menu.aboutToShow.connect(self._populate_beamtime_menu)

        project_menu = menu_bar.addMenu("&Project")

        self.configure_action = project_menu.addAction("Configure...")
        self.configure_action.triggered.connect(self.configure_project)

        self.open_terminal_action = project_menu.addAction("Open Terminal")
        self.open_terminal_action.triggered.connect(
            self._open_project_terminal
        )

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

        self.workspace_menu = menu_bar.addMenu("&Workspace")
        self.workspace_menu.aboutToShow.connect(
            self._populate_workspace_menu
        )

    def _open_project_terminal(self):
        if self.current_project is None:
            self.status_label.setText("No project selected")
            return

        target = self.current_working_directory
        if target is None:
            target = self.current_project.path

        open_terminal(
            target,
            on_error=lambda error: self.status_label.setText(
                f"Error: {error}"
            ),
        )

    # ------------------------------------------------------------------
    # Project switch menu
    # ------------------------------------------------------------------

    def _get_switchable_projects(self):
        """
        Return database-listed projects that still exist on disk.

        The database remains the source for saved projects, but an unavailable
        project is not shown in the switch menu.
        """
        try:
            projects = self.db.list_listed_projects()
        except Exception:
            log.exception("Cannot load listed projects")
            projects = []

        if not projects:
            try:
                projects = self.db.list_projects()
            except Exception:
                log.exception("Cannot load projects")
                return []

        available = []

        for project in projects:
            try:
                if project.path.is_dir():
                    available.append(project)
            except OSError:
                log.warning(
                    "Cannot inspect project path: %s",
                    project.path,
                )

        return available

    def _populate_switch_menu(self):
        self.switch_menu.clear()

        projects = self._get_switchable_projects()

        if not projects:
            action = self.switch_menu.addAction("(No saved projects)")
            action.setEnabled(False)
            return

        current_id = (
            self.current_project.id
            if self.current_project is not None
            else None
        )

        action_group = QActionGroup(self.switch_menu)
        action_group.setExclusive(True)

        for project in projects:
            action = self.switch_menu.addAction(project.name)
            action.setCheckable(True)
            action.setChecked(project.id == current_id)
            action.setToolTip(str(project.path))
            action_group.addAction(action)

            action.triggered.connect(
                lambda checked=False, project_id=project.id:
                    self.switch_to_project(project_id)
            )

    def switch_to_project(self, project_id: int):
        project = self.db.get_project(project_id)

        if project is None:
            self.status_label.setText("Project no longer exists")
            return

        if not project.path.is_dir():
            QMessageBox.warning(
                self,
                "Project unavailable",
                f"The project directory does not exist:\n{project.path}",
            )
            self.status_label.setText(
                f"Project path is unavailable: {project.path}"
            )
            return

        self._activate_project(project)

    # ------------------------------------------------------------------
    # Workspace discovery
    # ------------------------------------------------------------------

    def _scan_workspaces_from_disk(self):
        """
        Return workspace database objects for directories currently on disk.

        The filesystem is authoritative. Existing database rows are ignored if
        their directories have disappeared. New directories are registered in
        the database so that they can participate in history.
        """
        if self.current_project is None:
            return []

        project_path = self.current_project.path.resolve()

        try:
            paths = sorted(
                (
                    entry
                    for entry in project_path.iterdir()
                    if entry.is_dir() and entry.name.startswith("wd")
                ),
                key=lambda path: path.name.lower(),
            )
        except OSError as exc:
            self.status_label.setText(
                f"Cannot list working directories: {exc}"
            )
            return []

        workspaces = []

        for path in paths:
            try:
                workspace = self.project_manager.register_workspace(
                    self.current_project,
                    path,
                )
            except Exception:
                log.exception(
                    "Cannot register workspace directory: %s",
                    path,
                )
                continue

            workspaces.append(workspace)

        return workspaces

    def _populate_workspace_menu(self):
        self.workspace_menu.clear()

        if self.current_project is None:
            action = self.workspace_menu.addAction(
                "(No project selected)"
            )
            action.setEnabled(False)
            return

        # Always scan the filesystem when the menu is opened.
        workspaces = self._scan_workspaces_from_disk()

        current_workspace_id = (
            self.current_workspace.id
            if self.current_workspace is not None
            else None
        )

        if not workspaces:
            action = self.workspace_menu.addAction(
                "(No wd* directories)"
            )
            action.setEnabled(False)
        else:
            action_group = QActionGroup(self.workspace_menu)
            action_group.setExclusive(True)

            for workspace in workspaces:
                action = self.workspace_menu.addAction(workspace.name)
                action.setCheckable(True)
                action.setChecked(
                    workspace.id == current_workspace_id
                )
                action.setToolTip(str(workspace.path))
                action_group.addAction(action)

                action.triggered.connect(
                    lambda checked=False, workspace_id=workspace.id:
                    self.select_workspace_by_id(workspace_id)
                )

        self.workspace_menu.addSeparator()

        workspace_names = {workspace.name for workspace in workspaces}

        if "wd" not in workspace_names:
            create_action = self.workspace_menu.addAction("Create wd")
            create_action.triggered.connect(
                self.create_working_directory
            )

        create_custom_action = self.workspace_menu.addAction(
            "Create wd with custom suffix..."
        )
        create_custom_action.triggered.connect(
            self.create_working_directory_with_suffix
        )

    # ------------------------------------------------------------------
    # Main UI
    # ------------------------------------------------------------------

    def _create_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.left_title = QLabel("No working directory selected")
        self.left_title.setStyleSheet(
            "font-weight: bold; padding: 4px;"
        )
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

        self.tabs.setCurrentWidget(self.status_tab)

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        self.location_list.itemClicked.connect(
            self.select_subdirectory
        )
        self.location_list.itemDoubleClicked.connect(
            self.select_subdirectory
        )
        self.location_list.setContextMenuPolicy(
            Qt.CustomContextMenu
        )
        self.location_list.customContextMenuRequested.connect(
            self._show_folder_context_menu
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
        self.measurements_label.setAlignment(
            Qt.AlignTop | Qt.AlignLeft
        )

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
        self.pipeline_tab_label.setAlignment(
            Qt.AlignTop | Qt.AlignLeft
        )

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
    # Project activation and restoration
    # ------------------------------------------------------------------

    def _activate_project(self, project):
        self.current_project = project
        self.current_workspace = None

        try:
            self.db.add_listed_project(project.id)
            self.project_manager.record_project_open(project)
        except Exception:
            log.exception(
                "Cannot record project open: %s",
                project.path,
            )

        # Metadata refresh only. It does not calculate directory sizes.
        try:
            self.project_manager.refresh_project_metadata(project)
        except Exception:
            log.exception(
                "Cannot refresh project metadata: %s",
                project.path,
            )

        refreshed = self.db.get_project(project.id)
        if refreshed is not None:
            self.current_project = refreshed

        self._update_current_project_ui()
        self._load_project()
        self._restore_workspace()

    def _restore_last_project(self):
        """
        Restore the most recent project unless the latest history event was
        an explicit close event.
        """
        try:
            project = self.project_manager.current_project()
        except Exception:
            log.exception("Cannot restore project from application history")
            return

        if project is None:
            log.info("No project to restore from application history")
            return

        if not project.path.is_dir():
            log.warning(
                "Last project no longer exists: %s",
                project.path,
            )
            return

        self.current_project = project
        self._update_current_project_ui()
        self._load_project()
        self._restore_workspace()

    def _restore_workspace(self):
        if self.current_project is None:
            return

        try:
            workspace = self.db.last_workspace(
                self.current_project.id
            )
        except Exception:
            log.exception("Cannot restore last workspace")
            return

        if workspace is None:
            return

        # Filesystem is authoritative. Do not restore a deleted workspace.
        if not workspace.path.is_dir():
            log.info(
                "Previously selected workspace is no longer on disk: %s",
                workspace.path,
            )
            return

        try:
            self.current_workspace = self.project_manager.register_workspace(
                self.current_project,
                workspace.path,
            )
        except Exception:
            log.exception(
                "Cannot restore workspace: %s",
                workspace.path,
            )
            return

        self.select_workspace(
            self.current_workspace,
            persist=False,
        )

    def _load_project(self):
        self._reset_tab_texts()
        self.refresh_locations()
        self._update_action_states()

    # ------------------------------------------------------------------
    # Project open / close
    # ------------------------------------------------------------------

    def open_project(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Open Project Directory",
        )

        if not directory:
            return

        path = Path(directory).resolve()

        if not path.is_dir():
            self.status_label.setText(
                f"Not a directory: {path}"
            )
            return

        try:
            project = self.project_manager.register_project(path)
        except Exception as exc:
            log.exception("Cannot register project: %s", path)
            self.status_label.setText(
                f"Could not open project: {exc}"
            )
            return

        self._activate_project(project)

    # Compatibility alias for existing signal connections.
    open_location = open_project

    def close_project(self):
        try:
            # A null project/workspace row is intentional and represents
            # that the application was closed without an active project.
            self.project_manager.record_project_close()
        except Exception:
            log.exception("Cannot record project close")

        self.current_project = None
        self.current_workspace = None

        self.location_list.clear()
        self.left_title.setText("No working directory selected")
        self.location_status.setText("No project selected")
        self.status_label.setText("Ready")

        self.tabs.setCurrentIndex(0)
        self._reset_tab_texts()
        self._update_current_project_ui()

    # Compatibility alias for existing signal connections.
    close_location = close_project

    # ------------------------------------------------------------------
    # Workspace selection
    # ------------------------------------------------------------------

    def select_workspace_by_id(self, workspace_id: int):
        workspace = self.db.get_workspace(workspace_id)

        if workspace is None:
            self.status_label.setText(
                "Working directory is not registered"
            )
            return

        # Database rows are not authoritative.
        if not workspace.path.is_dir():
            self.status_label.setText(
                "Working directory no longer exists on disk"
            )
            self.refresh_locations()
            return

        if self.current_project is None:
            self.status_label.setText("No project selected")
            return

        if workspace.project_id != self.current_project.id:
            self.status_label.setText(
                "Working directory belongs to another project"
            )
            return

        self.select_workspace(workspace)

    def select_workspace(self, workspace, persist=True):
        if self.current_project is None:
            return

        if workspace is None or not workspace.path.is_dir():
            return

        if workspace.project_id != self.current_project.id:
            self.status_label.setText(
                "Working directory belongs to another project"
            )
            return

        self.current_workspace = workspace

        if persist:
            try:
                self.project_manager.record_project_open(
                    self.current_project,
                    workspace,
                )
            except Exception:
                log.exception(
                    "Cannot record workspace open: %s",
                    workspace.path,
                )

        self.location_status.setText(
            f"{self.current_project.path}  [{workspace.name}]"
        )

        self.tasks_label.setText(
            "Tasks\n\n"
            f"Selected working directory:\n{workspace.path}"
        )

        self.measurements_label.setText(
            "Measurements\n\n"
            f"Selected working directory:\n{workspace.path}"
        )

        self.pipeline_status_tree.set_working_directory(
            workspace.path
        )

        self.refresh_locations()
        self._update_window_title()

    def select_working_directory(self, working_dir, persist=True):
        """
        Compatibility method accepting a workspace Path.
        """
        if self.current_project is None:
            return

        path = Path(working_dir).resolve()

        try:
            workspace = self.project_manager.register_workspace(
                self.current_project,
                path,
            )
        except Exception as exc:
            log.exception(
                "Cannot register working directory: %s",
                path,
            )
            self.status_label.setText(
                f"Cannot select working directory: {exc}"
            )
            return

        self.select_workspace(workspace, persist=persist)

    # ------------------------------------------------------------------
    # Directory listing
    # ------------------------------------------------------------------

    def refresh_locations(self):
        """
        Refresh the left-hand list of subdirectories below the selected
        workspace.

        This does not calculate sizes and does not update workspace metadata.
        """
        self.location_list.clear()

        if self.current_project is None:
            self.left_title.setText("No project selected")
            return

        workspace_path = self.current_working_directory

        if workspace_path is None:
            self.left_title.setText("Subdirectories")
            return

        self.left_title.setText(
            f"Subdirectories of {workspace_path.name}"
        )

        if not workspace_path.is_dir():
            self.status_label.setText(
                "Selected working directory does not exist."
            )
            return

        try:
            subdirectories = sorted(
                (
                    entry
                    for entry in workspace_path.iterdir()
                    if entry.is_dir()
                ),
                key=lambda path: path.name.lower(),
            )
        except OSError as exc:
            self.status_label.setText(
                f"Cannot list subdirectories: {exc}"
            )
            return

        for subdirectory in subdirectories:
            item = QListWidgetItem(subdirectory.name)
            item.setData(Qt.UserRole, subdirectory)
            item.setToolTip(str(subdirectory))
            self.location_list.addItem(item)

        self.status_label.setText(
            f"Loaded {len(subdirectories)} subdirectories of "
            f"{workspace_path.name}"
        )

    def select_subdirectory(self, item):
        subdirectory = item.data(Qt.UserRole)
        if subdirectory:
            self.status_label.setText(
                f"Selected: {subdirectory}"
            )

    # ------------------------------------------------------------------
    # Project configuration
    # ------------------------------------------------------------------

    def configure_project(self):
        if self.current_project is None:
            self.status_label.setText(
                "No active project to configure"
            )
            return

        project_info = self.project_manager.get_project_info(
            self.current_project
        )

        dialog = ProjectConfigDialog(
            self.project_manager,
            self.current_project,
            project_info,
            size_service=self.size_service,
            parent=self,
        )

        if dialog.exec() != QDialog.Accepted:
            return

        new_name = dialog.project_name()
        new_description = None

        if hasattr(dialog, "project_description"):
            new_description = dialog.project_description()

        if new_name:
            self.db.connection.execute(
                """
                UPDATE laupy_project
                SET name = ?
                WHERE id = ?
                """,
                (new_name, self.current_project.id),
            )

        if new_description is not None:
            self.db.connection.execute(
                """
                UPDATE laupy_project
                SET description = ?
                WHERE id = ?
                """,
                (
                    new_description.strip() or None,
                    self.current_project.id,
                ),
            )

        self.db.connection.commit()

        refreshed = self.db.get_project(self.current_project.id)
        if refreshed is not None:
            self.current_project = refreshed

        self._update_current_project_ui()
        self.status_label.setText(
            f"Updated project: {self.current_project.name}"
        )

    # ------------------------------------------------------------------
    # Working directory creation
    # ------------------------------------------------------------------

    def create_working_directory(self):
        self._create_named_working_directory("wd")

    def create_working_directory_with_suffix(self):
        if self.current_project is None:
            self.status_label.setText("No active project")
            return

        suffix, accepted = QInputDialog.getText(
            self,
            "Create working directory",
            "Suffix for wd directory:",
        )

        if not accepted:
            return

        suffix = suffix.strip()

        if not suffix:
            self.status_label.setText(
                "Empty suffix, nothing created"
            )
            return

        directory_name = f"wd_{suffix.replace(' ', '_')}"
        self._create_named_working_directory(directory_name)

    def _list_raw_samples(self, raw_dir):
        program, args = script_command("--list", raw_dir)

        try:
            result = subprocess.run(
                [program, *args],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"Cannot run sample listing: {exc}"
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr
                or result.stdout
                or "Sample listing failed"
            )

        return [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
            and not line.lstrip().startswith("#")
        ]

    def _create_named_working_directory(self, directory_name: str):
        if self.current_project is None:
            self.status_label.setText("No active project")
            return

        project_path = self.current_project.path
        workspace_path = project_path / directory_name

        if workspace_path.exists():
            message = (
                f"Working directory already exists:\n"
                f"{workspace_path}"
            )
            log.warning(message)
            self.status_label.setText(message)
            return

        raw_dir = project_path / "raw"

        if not raw_dir.is_dir():
            QMessageBox.critical(
                self,
                "No raw directory",
                f"{raw_dir} does not exist.",
            )
            return

        self.status_label.setText(
            f"Listing samples in {raw_dir}..."
        )

        try:
            samples = self._list_raw_samples(raw_dir)
        except RuntimeError as exc:
            log.error("Sample listing failed: %s", exc)
            self.status_label.setText(
                "Sample listing failed"
            )
            return

        if not samples:
            QMessageBox.information(
                self,
                "No samples",
                f"No samples found in {raw_dir}.",
            )
            self.status_label.setText("No samples found")
            return

        dialog = SampleSelectionDialog(
            samples,
            directory_name,
            parent=self,
        )

        if dialog.exec() != QDialog.Accepted:
            self.status_label.setText(
                "Working directory creation cancelled"
            )
            return

        selected_samples = dialog.selected_samples()

        if not selected_samples:
            self.status_label.setText(
                "No samples selected, nothing created"
            )
            return

        try:
            workspace_path.mkdir(
                parents=False,
                exist_ok=False,
            )
        except OSError as exc:
            message = (
                f"Failed to create working directory "
                f"{workspace_path}: {exc}"
            )
            log.error(message)
            self.status_label.setText(message)
            return

        self._run_create_process(
            raw_dir,
            workspace_path,
            selected_samples,
        )

    def _run_create_process(self, raw_dir, workspace_path, samples):
        program, args = script_command(
            "--samples",
            *samples,
            "--",
            raw_dir,
            workspace_path,
        )

        def on_finished(success):
            if success:
                self.status_label.setText(
                    f"Created working directory "
                    f"{workspace_path.name} with "
                    f"{len(samples)} samples"
                )
            else:
                self.status_label.setText(
                    f"Errors while populating "
                    f"{workspace_path.name}"
                )

            # The directory now exists on disk. Register it and select it.
            self.select_working_directory(workspace_path)

        self.status_label.setText(
            f"Populating {workspace_path.name}..."
        )

        log_file = workspace_path / "createStructure.log"

        self._create_wd_log = ProcessLogDialog(
            f"Creating {workspace_path.name}",
            program,
            args,
            on_finished=on_finished,
            parent=self,
            log_file=log_file,
        )
        self._create_wd_log.setModal(False)
        self._create_wd_log.show()

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _show_folder_context_menu(self, position):
        item = self.location_list.itemAt(position)

        if item is None:
            return

        folder_path = item.data(Qt.UserRole)

        if not isinstance(folder_path, Path):
            return

        def on_error(error):
            self.status_label.setText(f"Error: {error}")

        menu = QMenu(self)

        menu.addAction(
            "Open Terminal Here",
            lambda: open_terminal(
                folder_path,
                on_error=on_error,
            ),
        )

        menu.addAction(
            "Open Thunar Here",
            lambda: open_thunar(
                folder_path,
                on_error=on_error,
            ),
        )

        menu.addSeparator()

        params_file = folder_path / "params"
        if params_file.is_file():
            menu.addAction(
                "Open params",
                lambda: open_files_mousepad(
                    [params_file],
                    on_error=on_error,
                ),
            )

        param_json_file = folder_path / "param.json"
        if param_json_file.is_file():
            menu.addAction(
                "Open param.json",
                lambda: open_files_mousepad(
                    [param_json_file],
                    on_error=on_error,
                ),
            )

        h5_file = folder_path / "h5"
        if h5_file.is_file():
            menu.addAction(
                "Open h5",
                lambda: open_hdf5view(
                    str(h5_file),
                    on_error=on_error,
                ),
            )

        dag_json = folder_path / "pipeline" / "dag.json"
        if dag_json.is_file():
            menu.addAction(
                "Open dag.json",
                lambda: open_files_vim(
                    [str(dag_json)],
                    on_error=on_error,
                ),
            )

        menu.exec(
            self.location_list.viewport().mapToGlobal(position)
        )

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _update_current_project_ui(self):
        if self.current_project is None:
            self.location_status.setText(
                "No project selected"
            )
            self.status_label.setText("Ready")
        else:
            self.location_status.setText(
                str(self.current_project.path)
            )
            self.status_label.setText(
                f"Selected project: "
                f"{self.current_project.name}"
            )

        self._update_window_title()
        self._update_action_states()

    # Compatibility alias.
    _update_current_location_ui = _update_current_project_ui

    def _update_window_title(self):
        if self.current_project is None:
            self.setWindowTitle("Laupy")
            return

        if self.current_workspace is None:
            self.setWindowTitle(
                f"Laupy - {self.current_project.name}"
            )
            return

        self.setWindowTitle(
            f"Laupy - {self.current_project.name} "
            f"[{self.current_workspace.name}]"
        )

    def _update_action_states(self):
        has_project = self.current_project is not None

        self.close_action.setEnabled(has_project)
        self.open_terminal_action.setEnabled(has_project)
        self.configure_action.setEnabled(has_project)
        self.create_wd_action.setEnabled(has_project)
        self.create_custom_wd_action.setEnabled(has_project)

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

    def refresh_sizes(self):
        """Explicit user action; never triggered by navigation."""
        if self.current_project is None:
            return

        linked = self.db.list_beamtimes_for_project(self.current_project.id)
        scratch_areas = [
            beamtime.core_path / "scratch_cc"
            for beamtime in linked
            if beamtime.core_path is not None
        ]

        # One scratch_cc scan also covers the project and its workspaces
        # when the project lives below scratch_cc.
        covered = any(
            self.current_project.path.is_relative_to(area)
            for area in scratch_areas
        )
        self.size_service.request_many(scratch_areas)
        if not covered:
            self.size_service.request(self.current_project.path)

        self.status_label.setText("Size refresh requested")

    def _on_size_event(self, event):
        if event.kind == SizeEventKind.PROGRESS:
            self.status_label.setText(
                f"Counting {event.path.name}: {event.files_scanned} files"
            )
        elif event.kind == SizeEventKind.FAILED:
            self.status_label.setText(
                f"Size count failed for {event.path}: {event.message}"
            )
        elif event.kind in (SizeEventKind.FINISHED, SizeEventKind.SKIPPED):
            if (
                self.current_project is not None
                and self.current_project.id in event.updated_projects
            ):
                self.current_project = self.db.get_project(
                    self.current_project.id
                )
            self.status_label.setText(f"Sizes updated: {event.path}")

    def find_beamtimes(self):
        """Scan GPFS for accessible beamtimes and store the chosen ones."""
        dialog = BeamtimeListDialog(self.project_manager, parent=self)
        dialog.exec()

        try:
            count = len(self.db.list_beamtimes())
        except Exception:
            log.exception("Cannot count saved beamtimes")
            return

        self.status_label.setText(f"{count} beamtime(s) saved")
    # ------------------------------------------------------------------
    # Beamtime menu
    # ------------------------------------------------------------------

    @staticmethod
    def _beamtime_year(beamtime) -> str:
        parts = beamtime.core_path.parts if beamtime.core_path else ()
        return parts[-3] if len(parts) >= 3 else ""

    @staticmethod
    def _beamtime_beamline(beamtime) -> str:
        parts = beamtime.core_path.parts if beamtime.core_path else ()
        return beamtime.beamline or (parts[-4] if len(parts) >= 4 else "")

    def _beamtime_label(self, beamtime) -> str:
        """beamline_year_beamtimeID, skipping unknown parts."""
        parts = [
            self._beamtime_beamline(beamtime),
            self._beamtime_year(beamtime),
            beamtime.beamtime_id,
        ]
        return "_".join(part for part in parts if part)

    def _menu_beamtimes(self):
        """Beamtimes ticked as listed; all beamtimes if none are ticked."""
        try:
            beamtimes = self.db.list_listed_beamtimes()
            if not beamtimes:
                beamtimes = self.db.list_beamtimes()
        except Exception:
            log.exception("Cannot load beamtimes")
            return []

        return sorted(
            beamtimes,
            key=lambda b: (
                self._beamtime_beamline(b).casefold(),
                self._beamtime_year(b),
                b.beamtime_id,
            ),
        )

    def _populate_beamtime_menu(self):
        self.beamtime_menu.clear()

        beamtimes = self._menu_beamtimes()
        if not beamtimes:
            action = self.beamtime_menu.addAction("(No saved beamtimes)")
            action.setEnabled(False)
            return

        for beamtime in beamtimes:
            submenu = self.beamtime_menu.addMenu(
                self._beamtime_label(beamtime)
            )
            tooltip = beamtime.title or str(beamtime.core_path or "")
            submenu.menuAction().setToolTip(tooltip)
            submenu.setToolTipsVisible(True)
            # Fill lazily: scratch_cc is listed only when hovered.
            submenu.aboutToShow.connect(
                lambda menu=submenu, bt=beamtime:
                    self._populate_beamtime_submenu(menu, bt)
            )
        self.beamtime_menu.addSeparator()
        self.find_beamtimes_action = self.beamtime_menu.addAction("List Beamtimes...")
        self.find_beamtimes_action.triggered.connect(self.find_beamtimes)
        self.beamtime_menu.addAction(self.find_beamtimes_action)

    def _beamtime_project_entries(self, beamtime) -> list[tuple[str, Path]]:
        """Linked projects plus scratch_cc folders containing "kct".

        Returns (label, path) pairs, deduplicated by path.
        """
        entries: dict[str, tuple[str, Path]] = {}

        try:
            linked = self.db.list_projects_for_beamtime(beamtime.id)
        except Exception:
            log.exception("Cannot load projects for %s", beamtime.beamtime_id)
            linked = []

        for project in linked:
            entries[str(project.path)] = (project.name, project.path)

        if beamtime.core_path is not None:
            scratch = beamtime.core_path / "scratch_cc"
            try:
                folders = [
                    entry for entry in scratch.iterdir()
                    if entry.is_dir() and "kct" in entry.name.casefold()
                ]
            except OSError:
                folders = []  # scratch_cc missing or unreadable

            for folder in folders:
                entries.setdefault(str(folder), (folder.name, folder))

        return sorted(entries.values(), key=lambda e: e[0].casefold())

    def _populate_beamtime_submenu(self, menu: QMenu, beamtime) -> None:
        menu.clear()

        entries = self._beamtime_project_entries(beamtime)
        if not entries:
            action = menu.addAction("(No projects or kct folders)")
            action.setEnabled(False)
        else:
            current = (
                str(self.current_project.path)
                if self.current_project is not None else None
            )
            for label, path in entries:
                action = menu.addAction(label)
                action.setToolTip(str(path))
                action.setCheckable(True)
                action.setChecked(str(path) == current)
                action.triggered.connect(
                    lambda checked=False, p=path, bt=beamtime:
                        self.open_beamtime_project(bt, p)
                )

        if beamtime.core_path is not None:
            scratch = beamtime.core_path / "scratch_cc"
            menu.addSeparator()
            terminal = menu.addAction("Open Terminal in scratch_cc")
            terminal.setEnabled(scratch.is_dir())
            terminal.triggered.connect(
                lambda checked=False, p=scratch: open_terminal(
                    p,
                    on_error=lambda error: self.status_label.setText(
                        f"Error: {error}"
                    ),
                )
            )

    def open_beamtime_project(self, beamtime, path: Path) -> None:
        """Open a folder as a project and link it to its beamtime."""
        if not path.is_dir():
            log.warning("Beamtime project path unavailable/not directory: %s", path)
            self.status_label.setText(f"Project path is unavailable: {path}")
            return

        try:
            project = self.project_manager.register_project(path)
            self.project_manager.link_beamtime(project, beamtime)
        except Exception as exc:
            log.exception("Cannot open beamtime project: %s", path)
            self.status_label.setText(f"Could not open project: {exc}")
            return

        self._activate_project(project)

    def open_beamtime_scratch(self, beamtime):
        """Open the beamtime's scratch_cc as a project directory."""
        if beamtime.core_path is None:
            self.status_label.setText("Beamtime has no known path")
            return

        scratch = beamtime.core_path / "scratch_cc"
        if not scratch.is_dir():
            QMessageBox.warning(
                self,
                "scratch_cc unavailable",
                f"The directory does not exist:\n{scratch}",
            )
            return

        try:
            project = self.project_manager.register_project(scratch)
            self.project_manager.link_beamtime(project, beamtime)
        except Exception as exc:
            log.exception("Cannot open beamtime scratch: %s", scratch)
            self.status_label.setText(f"Could not open beamtime: {exc}")
            return

        self._activate_project(project)

    def closeEvent(self, event):
        self.size_bridge.detach()
        self.size_service.stop()
        super().closeEvent(event)


