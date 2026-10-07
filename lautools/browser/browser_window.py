from __future__ import annotations

import logging
import html
import os
from pathlib import Path
import subprocess

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices
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
)
from lautools.working_directory_creator import WDCreator
from lautools.browser.settings_dialog import SettingsDialog
from lautools.browser.project_config_dialog import ProjectConfigDialog
from lautools.browser.beamtime_info_dialog import BeamtimeInfoDialog
from lautools.browser.beamtime_list_dialog import BeamtimeListDialog
from lautools.browser.pipeline_tree_widget import PipelineTreeWidget
from lautools.browser.project_manager import ProjectManager
from lautools.browser.new_project_from_recipe_dialog import (NewProjectFromRecipeDialog,)
from lautools.browser.utils import (
    open_files_mousepad,
    open_files_vim,
    open_hdf5view,
    open_terminal,
    open_thunar,
)
from lautools import about as lautools_about
from lautools.browser.size_service_qt import SizeServiceBridge
from lautools.size_service import SizeEventKind, SizeService

PROJECT_REPOSITORY_URL = "https://github.com/kulvait/KCT_lautools"

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
        settings_action = file_menu.addAction("Settings...")
        settings_action.triggered.connect(self.open_settings)
        file_menu.addSeparator()
        self.exit_action = file_menu.addAction("Exit App")
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.triggered.connect(self.close)
        # Beamtime
        self.beamtime_menu = menu_bar.addMenu("&Beamtime")
        self.beamtime_menu.aboutToShow.connect(self._populate_beamtime_menu)
        # Project
        self.project_menu = menu_bar.addMenu("&Project")
        self.project_menu.setToolTipsVisible(True)
        self.project_menu.aboutToShow.connect(self._populate_project_menu)
        # Workspace
        self.workspace_menu = menu_bar.addMenu("&Workspace")
        self.workspace_menu.setToolTipsVisible(True)
        self.workspace_menu.aboutToShow.connect(self._populate_workspace_menu)
        # Switch
        self.switch_menu = menu_bar.addMenu("&Switch")
        self.switch_menu.aboutToShow.connect(self._populate_switch_menu)
        # Help
        help_menu = menu_bar.addMenu("&Help")
        repository_action = help_menu.addAction("Project Repository")
        repository_action.triggered.connect(self._open_project_repository)
        license_action = help_menu.addAction("Show GNU GPLv3 License")
        license_action.triggered.connect(self._show_license)
        help_menu.addSeparator()
        about_action = help_menu.addAction("About Lautools")
        about_action.triggered.connect(self._show_about)

    #-------------------------------------------------------------------
    # File menu
    #-------------------------------------------------------------------

    def open_settings(self):
        SettingsDialog(self.db, parent=self).exec()

    # ------------------------------------------------------------------
    # Project Menu
    # ------------------------------------------------------------------

    def _populate_project_menu(self):
        self.project_menu.clear()

        self.project_configure_action = self.project_menu.addAction("Project info...")
        self.project_configure_action.setEnabled(self.current_project is not None)
        self.project_configure_action.triggered.connect(self.configure_project)
        
        # On error when opening vim, terminal, or thunar, display the error in the status bar.
        def on_error(error):
            self.status_label.setText(f"Error: {error}")
        # Add open actions in terminal or thunar for project, workspace, recipe, and upstream recipe
        self.current_project_recipes = self.get_project_recipes()
        if self.current_project is not None:
            self._add_open_action(self.project_menu, "Open Terminal", self.current_project.path, open_terminal)
            self._add_open_action(self.project_menu, "Open Thunar", self.current_project.path, open_thunar)
            if self.current_workspace is not None:
                workspace_name = self.current_workspace.name or "Workspace"
                self._add_open_action(self.project_menu, f"Open Terminal in {workspace_name}", self.current_workspace.path, open_terminal)
                self._add_open_action(self.project_menu, f"Open Thunar in {workspace_name}", self.current_workspace.path, open_thunar)
            if self.current_project_recipes.get("workbench_instance_path") is not None:
                recipe_name = self.current_project_recipes["workbench_instance"].name or ""
                self._add_open_action(self.project_menu, f"Open Terminal for recipe {recipe_name}", self.current_project_recipes["workbench_instance_path"], open_terminal)
                self._add_open_action(self.project_menu, f"Open Thunar for recipe {recipe_name}", self.current_project_recipes["workbench_instance_path"], open_thunar)
            if self.current_project_recipes.get("upstream_instance_path") is not None:
                upstream_name = self.current_project_recipes["upstream_instance"].name or ""
                self._add_open_action(self.project_menu, f"Open Terminal for upstream recipe {upstream_name}", self.current_project_recipes["upstream_instance_path"], open_terminal)
                self._add_open_action(self.project_menu, f"Open Thunar for upstream recipe {upstream_name}", self.current_project_recipes["upstream_instance_path"], open_thunar)
            if (self.current_project.path / "INFO").is_file():
                self.project_menu.addAction("Open INFO in vim").triggered.connect(lambda: open_files_vim([str(self.current_project.path / "INFO")], on_error=on_error))
                self.project_menu.addAction("Open INFO in mousepad").triggered.connect(lambda: open_files_mousepad([str(self.current_project.path / "INFO")], on_error=on_error))
            else:
                log.info("INFO file not found in project path: %s", (self.current_project.path / "INFO"))
        self.project_menu.addSeparator()
        wd_dirs = self.current_on_disk_wd_dirs()
        self._add_workspace_creation_actions(self.project_menu, wd_dirs)

    def _add_workspace_creation_actions(self, menu: QMenu, wd_dirs: list[str]):
        if "wd" not in wd_dirs:
            create_action = menu.addAction("Create wd")
            create_action.triggered.connect(self.create_working_directory)
        create_custom_action = menu.addAction("Create wd with custom suffix...")
        create_custom_action.triggered.connect(self.create_working_directory_with_suffix)

    def get_project_recipes(self) -> Dict[str, LaupyRecipeInstance]:
        """
        Return a dictionary mapping recipe names to LaupyRecipeInstance objects for the current project.
        """
        if self.current_project is None:
            return {}
        try:
            recipes = {}
            project_to_recipe = self.db.get_project_to_recipe_instance_for_project(self.current_project.id)
            if project_to_recipe is not None:
                recipes["workbench_instance_exist"] = True
                recipes["workbench_instance_id"] = project_to_recipe.recipe_instance_id
                recipes["workbench_instance"] = (self.db.get_recipe_instance(project_to_recipe.recipe_instance_id))
            else:
                recipes["workbench_instance_exist"] = False
                recipes["workbench_instance_id"] = None
                recipes["workbench_instance"] = None
            if recipes["workbench_instance"] is not None:
                recipes["workbench_instance_path"] = self.project_manager.recipe_instance_path(recipes["workbench_instance"])
                if recipes["workbench_instance"].cloned_from_instance_id is not None:
                    source = self.db.get_recipe_instance(recipes["workbench_instance"].cloned_from_instance_id)
                    if source is not None:
                        recipes["upstream_instance"] = source
                        recipes["upstream_instance_path"] = self.project_manager.recipe_instance_path(source)
                    else:
                        recipes["upstream_instance"] = None
                        recipes["upstream_instance_path"] = None
            return recipes
        except Exception:
            log.exception("Cannot resolve project recipes")
            return {}

    def _add_open_action(self, menu: QMenu, label: str, path: Path | None, opener):
        action = menu.addAction(label)
        action.setEnabled(self._folder_available(path))
        action.setToolTip(str(path) if path is not None else "No associated folder")
        action.triggered.connect(lambda checked=False, p=path, launch=opener: launch(p, on_error=lambda error: self.status_label.setText(f"Error: {error}")))
        return action

    @staticmethod
    def _folder_available(path: Path | None) -> bool:
        if path is None:
            return False
        try:
            return path.is_dir()
        except OSError:
            return False

    def _open_project_terminal(self):
        if self.current_project is None:
            self.status_label.setText("No project selected")
            return
        open_terminal(target,  on_error=lambda error: self.status_label.setText(f"Error: {error}"),)

    def current_on_disk_wd_dirs(self):
        """
        Return a list of strings that identify folders in project directory starting with "wd" that are currently on disk.
    
        The filesystem is authoritative. Existing database rows are ignored if
        their directories have disappeared. New directories are registered in
        the database so that they can participate in history.
        """
        try:
            project_path = self.current_project.path.resolve()
            wd_paths = sorted( entry.name for entry in project_path.iterdir() if entry.is_dir() and entry.name.startswith("wd"))
            return wd_paths
        except Exception:
            log.exception("Cannot resolve current project path")
            return []

    def get_project_workspaces(self, register_disk_workspaces=True):
        """
        Return workspace information for the current project. Include wokspaces in the database and project subfolders starting with "wd" that are currently on disk.
        
        If ``register_disk_workspaces`` is True, wd subfolders that are not yet registered in the database will be registered and included in the returned list.
    
        Each returned dictionary contains:
    
        ``workspace``
            The database workspace object, or None if the directory only exists
            on disk.

        ``workspace_path_exists``
            True if the workspace path exists on disk.
    
        ``wd_subfolder_name``
            The workspace directory name on disk, or None if the workspace only
            exists in the database.
    
        """
        if self.current_project is None:
            return []
        workspaces_in_db = self.db.list_workspaces_for_project(self.current_project.id)
        wd_subfolder_names = self.current_on_disk_wd_dirs()
        # Create collections for path comparison
        workspaces_in_db_by_path = {ws.path.resolve(): ws for ws in workspaces_in_db}
        wd_subfolder_names_by_path = { (self.current_project.path / name).resolve(): name for name in wd_subfolder_names }
        all_paths = ( workspaces_in_db_by_path.keys() | wd_subfolder_names_by_path.keys() )
        workspaces = []
        for path in all_paths:
            workspace = workspaces_in_db_by_path.get(path) # None if the path is not in workspaces_in_db_by_path
            wd_subfolder_name = wd_subfolder_names_by_path.get(path) # None if the path is not in wd_subfolder_names_by_path
            if workspace is None and register_disk_workspaces:
                try:
                    workspace = self.project_manager.register_workspace(self.current_project, path,)
                except Exception:
                    log.exception("Cannot register workspace directory: %s of project %s", path, self.current_project.path,)
            workspaces.append({
                "workspace": workspace,
                "wd_subfolder_name": wd_subfolder_name,
                "workspace_path_exists": path.is_dir()
            })
        return workspaces


    #------------------------------------------------------------------
    # Workspace menu
    #------------------------------------------------------------------

    def _populate_workspace_menu(self):
        self.workspace_menu.clear()

        if self.current_project is None:
            action = self.workspace_menu.addAction("(No project selected)")
            action.setEnabled(False)
            return

        # Always scan the filesystem when the menu is opened.
        workspaces = self.get_project_workspaces(register_disk_workspaces=True)
        current_workspace_id = (self.current_workspace.id if self.current_workspace is not None else None )

        if not workspaces or len(workspaces) == 0:
            action = self.workspace_menu.addAction("(No wd* directories)")
            action.setEnabled(False)
        else:
            action_group = QActionGroup(self.workspace_menu)
            action_group.setExclusive(True)
            for workspace_info in workspaces:
                workspace = workspace_info["workspace"]
                action = self.workspace_menu.addAction(workspace.name)
                action.setCheckable(True)
                action.setChecked(workspace.id == current_workspace_id)
                action.setToolTip(str(workspace.path))
                action_group.addAction(action)
                if not workspace_info["workspace_path_exists"]:
                    action.setEnabled(False)
                    action.setToolTip(f"{workspace.path} (no longer exists on disk)")
                else:
                    action.triggered.connect(lambda checked=False, workspace_id=workspace.id:self.select_workspace_by_id(workspace_id))
        
        self.workspace_menu.addSeparator()
        workspace_names = {ws_info["workspace"].name for ws_info in workspaces if ws_info["workspace"] is not None}
        self._add_workspace_creation_actions(self.workspace_menu, workspace_names)

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
        directory = QFileDialog.getExistingDirectory(self, "Open Project Directory", )
        if not directory:
            return
        path = Path(directory).resolve()
        if not path.is_dir():
            self.status_label.setText(f"Not a directory: {path}")
            return

        try:
            project = self.project_manager.register_project(path)
        except Exception as exc:
            log.exception("Cannot register project: %s", path)
            self.status_label.setText(f"Could not open project: {exc}")
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

        self.status_label.setText(f"Listing samples in {raw_dir}...")
        wdc = None
        try:
            wdc = WDCreator(self.current_project, db=self.db)
            samples = wdc.getSampleList(raw_dir)
        except (RuntimeError, ValueError) as exc:
            log.error("Sample listing failed: %s", exc)
            self.status_label.setText("Sample listing failed")
            return

        if not samples:
            QMessageBox.information(self, "No samples found", f"No samples found in {raw_dir}.")
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
            message = (f"Failed to create working directory {workspace_path}: {exc}")
            log.error(message)
            self.status_label.setText(message)
            return

        self._run_create_process(
            wdc,
            raw_dir,
            workspace_path,
            selected_samples,
        )

    def _run_create_process(self, creator, raw_dir, workspace_path, samples):
        try:
            program, args = creator.getCreationCommand(
                workspace_path,
                samples,
                raw_dir=raw_dir,
            )
        except ValueError as exc:
            self.status_label.setText(f"Cannot create working directory: {exc}")
            QMessageBox.critical(
                self,
                "Working directory creation failed",
                str(exc),
            )
            return
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
        dialog = BeamtimeListDialog(self.project_manager, size_service=self.size_service, parent=self)
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
        """Stored label, falling back to beamline_year_beamtimeID."""
        if beamtime.label:
            return beamtime.label
        return "_".join(part for part in [self._beamtime_beamline(beamtime), self._beamtime_year(beamtime), beamtime.beamtime_id ] if part)

    def _current_beamtimes(self):
        """Return beamtimes linked to the currently opened project."""
        if self.current_project is None:
            return []

        try:
            return self.db.list_beamtimes_for_project(
                self.current_project.id
            )
        except Exception:
            log.exception(
                "Cannot load beamtimes for current project %s",
                self.current_project.id,
            )
            return []

    def _menu_beamtimes(self):
        """Return listed beamtimes plus the currently opened beamtime.

        If no beamtimes are explicitly listed, retain the existing fallback
        of showing every beamtime in the database.
        """
        try:
            beamtimes = self.db.list_listed_beamtimes()

            if not beamtimes:
                beamtimes = self.db.list_beamtimes()

            # The current project's beamtime must be visible even when it is
            # not selected in lautools_app_listed_beamtime.
            by_id = {
                beamtime.id: beamtime
                for beamtime in beamtimes
            }

            for beamtime in self._current_beamtimes():
                by_id[beamtime.id] = beamtime

            beamtimes = list(by_id.values())

        except Exception:
            log.exception("Cannot load beamtimes")
            return []

        return sorted(
            beamtimes,
            key=lambda beamtime: (
                self._beamtime_beamline(beamtime).casefold(),
                self._beamtime_year(beamtime),
                beamtime.beamtime_id,
            ),
        )

    def _populate_beamtime_menu(self):
        self.beamtime_menu.clear()

        beamtimes = self._menu_beamtimes()
        current_beamtimes = self._current_beamtimes()

        # Normally a project belongs to one beamtime. If several links exist,
        # mark the first one according to the menu's stable beamtime ordering.
        current_ids = {
            beamtime.id
            for beamtime in current_beamtimes
        }
        current_beamtime_id = next(
            (
                beamtime.id
                for beamtime in beamtimes
                if beamtime.id in current_ids
            ),
            None,
        )

        if not beamtimes:
            action = self.beamtime_menu.addAction(
                "(No saved beamtimes)"
            )
            action.setEnabled(False)
        else:
            # Exclusive checkable menu actions are rendered like the dot used
            # by the Switch menu.
            action_group = QActionGroup(self.beamtime_menu)
            action_group.setExclusive(True)

            for beamtime in beamtimes:
                submenu = self.beamtime_menu.addMenu(self._beamtime_label(beamtime)) #Name submenu after the beamtime label
                menu_action = submenu.menuAction()
                menu_action.setCheckable(True)
                menu_action.setChecked(beamtime.id == current_beamtime_id)
                action_group.addAction(menu_action)
                tooltip = (beamtime.title or str(beamtime.core_path or "").strip())
                menu_action.setToolTip(tooltip)
                submenu.setToolTipsVisible(True)

                # scratch_cc is inspected lazily only when this submenu opens.
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
                if str(folder) not in entries:
                    entries.setdefault(str(folder), (folder.name, folder))
        return sorted(entries.values(), key=lambda e: e[0].casefold())

    def _populate_beamtime_submenu(self, menu: QMenu, beamtime) -> None:
        menu.clear()
        info_dialog_action = menu.addAction("Beamtime Info...")
        info_dialog_action.triggered.connect(lambda checked=False, bt=beamtime: self.open_beamtime_dialog(bt))
        if beamtime.core_path is not None:
            terminal = menu.addAction("Terminal")
            terminal.setEnabled(beamtime.core_path.is_dir())
            terminal.triggered.connect(lambda checked=False, p=beamtime.core_path: open_terminal(p, on_error=lambda error: self.status_label.setText(f"Error: {error}")))
            scratch = beamtime.core_path / "scratch_cc"
            terminal_scratch = menu.addAction("Terminal in scratch_cc")
            terminal_scratch.setEnabled(scratch.is_dir())
            terminal_scratch.triggered.connect(lambda checked=False, p=scratch: open_terminal(p, on_error=lambda error: self.status_label.setText(f"Error: {error}")))
        menu.addSeparator()
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
        create_action = menu.addAction("New project from recipe...")
        create_action.setEnabled(
            beamtime.core_path is not None
            and (beamtime.core_path / "scratch_cc").is_dir()
        )
        create_action.triggered.connect(
            lambda checked=False, bt=beamtime:
                self.new_project_from_recipe(bt)
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

    def new_project_from_recipe(self, beamtime) -> None:
        """Create a new project from a recipe in the beamtime's scratch_cc."""
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
        dialog = NewProjectFromRecipeDialog(
            self.project_manager,
            beamtime,
            scratch,
            size_service=self.size_service,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        new_project_path = dialog.selected_project_path()
        if new_project_path is None:
            self.status_label.setText("No project created")
            return
        try:
            project = self.project_manager.register_project(new_project_path)
            self.project_manager.link_beamtime(project, beamtime)
        except Exception as exc:
            log.exception("Cannot open new project: %s", new_project_path)
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

    def open_beamtime_dialog(self, beamtime) -> None:
        refreshed = self.db.get_beamtime(beamtime.id)
        if refreshed is None:
            self.status_label.setText(
                "Beamtime no longer exists"
            )
            return

        dialog = BeamtimeInfoDialog(
            self.project_manager,
            refreshed,
            size_service=self.size_service,
            parent=self,
        )

        if dialog.exec() != QDialog.Accepted:
            return

        refreshed = self.db.get_beamtime(beamtime.id)
        label = (
            refreshed.label
            if refreshed is not None
            else beamtime.beamtime_id
        )
        self.status_label.setText(f"Updated beamtime: {label}")

    def closeEvent(self, event):
        self.size_bridge.detach()
        self.size_service.stop()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def _open_project_repository(self) -> None:
        opened = QDesktopServices.openUrl(
            QUrl(PROJECT_REPOSITORY_URL)
        )

        if not opened:
            self.status_label.setText(
                "Could not open the project repository"
            )

    @staticmethod
    def _license_path() -> Path | None:
        """Locate LICENSE when running from the source repository."""
        module_path = Path(__file__).resolve()

        # browser_window.py normally lives at:
        # <repository>/lautools/browser/browser_window.py
        candidates = [
            module_path.parents[2] / "LICENSE",
            Path.cwd() / "LICENSE",
        ]

        for candidate in candidates:
            try:
                if candidate.is_file():
                    return candidate
            except OSError:
                continue

        return None

    def _show_license(self) -> None:
        license_path = self._license_path()

        if license_path is None:
            QMessageBox.warning(
                self,
                "License unavailable",
                "The LICENSE file could not be located.\n\n"
                "This project is licensed under GNU GPL v3.0.",
            )
            return

        opened = open_files_mousepad(
            [str(license_path)],
            on_error=lambda error: self.status_label.setText(
                f"Error: {error}"
            ),
        )

        if not opened:
            self.status_label.setText(
                f"Could not open license file: {license_path}"
            )

    def _show_about(self) -> None:
        version = html.escape(lautools_about())
        repository = html.escape(PROJECT_REPOSITORY_URL)

        message = QMessageBox(self)
        message.setWindowTitle("About Lautools")
        message.setIcon(QMessageBox.Information)
        message.setTextFormat(Qt.RichText)
        message.setTextInteractionFlags(
            Qt.TextBrowserInteraction
        )
        message.setStandardButtons(QMessageBox.Ok)

        message.setText(
            f"<h3>{version}</h3>"
            "<p>"
            "Tools for tomography data preprocessing, reconstruction "
            "workflows, and beamtime project management."
            "</p>"
            "<p>"
            "The development of this package was supported by "
            "<b>Hi ACTS Use Case Initiatives 2026</b> within the project "
            "<i>Advanced reconstruction pipeline for tomography "
            "experiments at PETRA III</i>."
            "</p>"
            "<p>"
            "<b>Licensing</b><br>"
            "GNU GPL v3.0."
            "</p>"
            "<p>"
            "Copyright &copy; 2026 Vojtěch Kulvait"
            "</p>"
            f'<p><a href="{repository}">{repository}</a></p>'
        )

        # Enable the repository hyperlink inside the message box.
        for label in message.findChildren(QLabel):
            label.setOpenExternalLinks(True)

        message.exec()
