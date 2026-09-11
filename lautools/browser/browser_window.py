from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QSplitter,
    QStatusBar,
)


class BrowserWindow(QMainWindow):
    def __init__(self, db):
        super().__init__()

        self.db = db
        self.current_location = None

        self.setWindowTitle("Laupy")
        self.resize(1100, 700)

        self._create_menu()
        self._create_ui()
        self._create_status_bar()

        self.refresh_locations()

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
            # Keep the Location object associated with the action.
            action.setData(location)
    
            action.triggered.connect(
                lambda checked=False, loc=location: self.open_recent_location(loc)
            )

    def open_recent_location(self, location):
        self.db.update_last_access(location.id)  # Update last access time in the database.
        self.current_location = location.path
    
        self.location_status.setText(
            str(location.path)
        )
    
        self.status_label.setText(
            f"Selected: {location.name}"
        )
    
        self._load_location(location.path)

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

        # Give the right side more space.
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        self.location_list.itemClicked.connect(
            self.select_location
        )

        self.location_list.itemDoubleClicked.connect(
            lambda _: self.open_location()
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

        # Keep your existing database implementation here.
        #
        # Example:
        #
        # locations = self.db.get_locations()
        #
        # for location in locations:
        #     item = QListWidgetItem(str(location))
        #     item.setData(Qt.UserRole, location)
        #     self.location_list.addItem(item)

        locations = self._get_locations()

        for location in locations:
            item = QListWidgetItem(f"{location.name}    {location.path}")
            item.setData(Qt.UserRole, location)
            self.location_list.addItem(item)

    def _get_locations(self):
        """
        Adapt this method to your database API.
        """
        try:
            return self.db.get_locations()
        except AttributeError:
            return []

    def select_location(self, item):
        location = item.data(Qt.UserRole)

        if not location:
            return

        self.current_location = Path(location)

        self.location_status.setText(
            str(self.current_location)
        )

        self.status_label.setText(
            f"Selected: {self.current_location.name}"
        )

        self._load_location(self.current_location)

    def _load_location(self, location):
        """
        Load tasks, measurements and pipeline data for `location`.

        This is where your database/application logic should be connected.
        """

        # TODO:
        # self.load_tasks(location)
        # self.load_measurements(location)
        # self.load_pipeline(location)

        pass

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

        path = Path(directory)

        self._add_location(path)
        self.refresh_locations()

        # Select the newly opened directory.
        for index in range(self.location_list.count()):
            item = self.location_list.item(index)

            if item.data(Qt.UserRole) == str(path):
                self.location_list.setCurrentItem(item)
                self.select_location(item)
                break

    def close_location(self):
        self.current_location = None

        self.location_list.clearSelection()

        self.location_status.setText(
            "No location selected"
        )

        self.status_label.setText("Ready")

        # Reset the tabs if necessary.
        self.tabs.setCurrentIndex(0)

    def show_recent_locations(self):
        """
        Replace this with a dialog/menu populated from your database.
        """

        self.status_label.setText(
            "Recent locations"
        )

        # TODO:
        # Show recent locations from self.db.

    # ------------------------------------------------------------------
    # Database integration
    # ------------------------------------------------------------------

    def _add_location(self, path):
        """
        Adapt to your database API.
        """
        try:
            self.db.add_location(str(path))
        except AttributeError:
            pass
 
