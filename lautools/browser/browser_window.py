from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStatusBar,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from project_config_dialog import ProjectConfigDialog
from project_manager import ProjectManager


class BrowserWindow(QMainWindow):
    def __init__(self, db):
        super().__init__()

        self.db = db
        self.project_manager = ProjectManager()
        self.current_location = None

        self.setWindowTitle("Laupy")
        self.resize(1100, 700)

        self._create_menu()
        self._create_ui()
        self._create_status_bar()

        self.refresh_locations()
        self._update_action_states()

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _create_menu(self):
        menu_bar = self.menuBar()

        project_menu = menu_bar.addMenu("&Project")

        self.open_action = project_menu.addAction("&Open...")
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self.open_location)

        self.close_action = project_menu.addAction("&Close")
        self.close_action.setShortcut("Ctrl+W")
        self.close_action.triggered.connect(self.close_location)

        self.configure_action = project_menu.addAction("&Configure...")
        self.configure_action.triggered.connect(self.configure_project)

        project_menu.addSeparator()

        self.recent_menu = project_menu.addMenu("&Recent")
        self.recent_menu.aboutToShow.connect(self._populate_recent_menu)

        project_menu.addSeparator()

        exit_action = project_menu.addAction("E&xit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)

    def _populate_recent_menu(self):
        self.recent_menu.clear()

        locations = self.db.list_recent_locations() if self.db else []

        if not locations:
            action = self.recent_menu.addAction("(No locations)")
            action.setEnabled(False)
            return

        for location in locations:
            action = self.recent_menu.addAction(location.name)
            action.setToolTip(str(location.path))
            action.setData(location)
            action.triggered.connect(
                lambda checked=False, loc=location: self.open_recent_location(loc)
            )

    def open_recent_location(self, location):
        self.db.update_last_access(location.id)
        refreshed = self.db.get_location(location.id)

        self.current_location = refreshed
        self.refresh_locations()
        self._select_location_in_list(refreshed.id)
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

        # --------------------------------------------------------------
        # Left: working directories
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Right: application tabs
        # --------------------------------------------------------------

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        self.tasks_tab = self._create_tasks_tab()
        self.measurements_tab = self._create_measurements_tab()
        self.pipeline_tab = self._create_pipeline_tab()

        self.tabs.addTab(self.tasks_tab, "Tasks")
        self.tabs.addTab(self.measurements_tab, "Measurements")
        self.tabs.addTab(self.pipeline_tab, "Pipeline")

        right_layout.addWidget(self.tabs)

        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        self.location_list.itemClicked.connect(
            self.select_location
        )

        self.location_list.itemDoubleClicked.connect(
            self.select_location
        )

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------

    def _create_tasks_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        label = QLabel(
            "Tasks\n\n"
            "Tasks associated with the selected working directory "
            "will appear here."
        )
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        layout.addWidget(label)
        layout.addStretch()

        return widget

    def _create_measurements_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        label = QLabel(
            "Measurements\n\n"
            "Measurements for the selected working directory "
            "will appear here."
        )
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        layout.addWidget(label)
        layout.addStretch()

        return widget

    def _create_pipeline_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        label = QLabel(
            "Pipeline\n\n"
            "The processing pipeline for the selected working "
            "directory will appear here."
        )
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        layout.addWidget(label)
        layout.addStretch()

        return widget

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_label = QLabel("Ready")
        self.location_status = QLabel("No location selected")

        self.status_bar.addWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.location_status)

    # ------------------------------------------------------------------
    # Locations
    # ------------------------------------------------------------------

    def refresh_locations(self):
        self.location_list.clear()

        locations = self.db.list_locations()

        for location in locations:
            item = QListWidgetItem(f"{location.name}    {location.path}")
            item.setData(Qt.UserRole, location)
            self.location_list.addItem(item)

        if self.current_location is not None:
            self._select_location_in_list(self.current_location.id)

    def select_location(self, item):
        location = item.data(Qt.UserRole)

        if not location:
            return

        self.current_location = location
        self._update_current_location_ui()
        self._load_location(location)

    def _load_location(self, location):
        """
        Load tasks, measurements and pipeline data for `location`.

        This is where your database/application logic should be connected.
        """

        # TODO:
        # self.load_tasks(location)
        # self.load_measurements(location)
        # self.load_pipeline(location)

        self._update_action_states()

    # ------------------------------------------------------------------
    # Project actions
    # ------------------------------------------------------------------

    def open_location(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Open Working Directory",
        )

        if not directory:
            return

        path = Path(directory).resolve()

        self._add_location(path)

        location = self.db.get_location_by_path(path)
        if location is None:
            self.status_label.setText("Could not open location")
            return

        self.db.update_last_access(location.id)
        self.current_location = self.db.get_location(location.id)

        self.refresh_locations()
        self._select_location_in_list(self.current_location.id)
        self._update_current_location_ui()
        self._load_location(self.current_location)

    def close_location(self):
        self.current_location = None

        self.location_list.clearSelection()

        self.location_status.setText(
            "No location selected"
        )
        self.status_label.setText("Ready")
        self.tabs.setCurrentIndex(0)
        self._update_action_states()

    def configure_project(self):
        if self.current_location is None:
            self.status_label.setText("No active project to configure")
            return

        project_info = self.project_manager.build_project_info(
            self.current_location
        )
        dialog = ProjectConfigDialog(project_info, self)

        if dialog.exec() != QDialog.Accepted:
            return

        new_name = dialog.project_name()
        if new_name and new_name != self.current_location.name:
            self.db.rename_location(self.current_location.id, new_name)
            self.current_location = self.db.get_location(
                self.current_location.id
            )
            self.refresh_locations()
            self._update_current_location_ui()
            self.status_label.setText(
                f"Renamed project to: {self.current_location.name}"
            )

    def show_recent_locations(self):
        self.status_label.setText(
            "Recent locations"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_location(self, path):
        self.db.add_location(path)

    def _update_current_location_ui(self):
        if self.current_location is None:
            self.location_status.setText("No location selected")
            self.status_label.setText("Ready")
        else:
            self.location_status.setText(str(self.current_location.path))
            self.status_label.setText(
                f"Selected: {self.current_location.name}"
            )

        self._update_action_states()

    def _update_action_states(self):
        has_location = self.current_location is not None
        self.close_action.setEnabled(has_location)
        self.configure_action.setEnabled(has_location)

    def _select_location_in_list(self, location_id: int):
        for index in range(self.location_list.count()):
            item = self.location_list.item(index)
            location = item.data(Qt.UserRole)
            if location and location.id == location_id:
                self.location_list.setCurrentItem(item)
                return
