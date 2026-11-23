"""Editor of the submission context: every submitslurm option for all submissions, free text,
and the job environment (PATH and variables) as submitslurm will export it.

The options and the environment preview come from laupy.submitslurm_options, so the editor
follows the submitslurm parser.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from laupy.cli import check_python_environment
from laupy.submitslurm_options import (
    context_option_specs,
    merge_arguments,
    parse_merged,
    preview_environment,
    quote_arguments,
)
from lautools.browser.submitslurm_context import (
    MODE_DEFAULT,
    MODE_FORCE,
    MODE_OFF,
    OptionSetting,
    SubmitslurmContext,
)

# Variables set per job by submitslurm; not editable here
FIXED_VARIABLES = {"PIPELINE_SCRIPTNAME", "PIPELINE_JOBNAME", "LD_PRELOAD", "PATH"}
GREY = "color: #808080;"
RED = "color: #c0392b;"


def _default_text(spec) -> str:
    default = spec.default
    if default in (None, False, [], ""):
        return ""
    if isinstance(default, list):
        return " ".join(str(v) for v in default)
    return str(default)


class _OptionRow:
    """Widgets of one option: mode combo and value widget."""

    def __init__(self, spec, setting: OptionSetting, changed):
        self.spec = spec
        self.mode = QComboBox()
        self.mode.addItem("off", MODE_OFF)
        if spec.kind == "flag":
            self.mode.addItem("on", MODE_DEFAULT)
        else:
            self.mode.addItem("default", MODE_DEFAULT)
            self.mode.addItem("force", MODE_FORCE)
        mode = MODE_DEFAULT if spec.kind == "flag" and setting.mode == MODE_FORCE else setting.mode
        self.mode.setCurrentIndex(max(self.mode.findData(mode), 0))
        self.mode.currentIndexChanged.connect(changed)
        self.checks: list[QCheckBox] = []
        default = _default_text(spec)
        if spec.kind == "flag":
            self.value = QLabel("")
        elif spec.kind == "choice":
            self.value = QComboBox()
            self.value.addItems(list(spec.choices))
            current = setting.value if setting.value in spec.choices else default
            if current in spec.choices:
                self.value.setCurrentText(current)
            self.value.currentIndexChanged.connect(lambda *_: self._touched(changed))
        elif spec.kind == "multi_choice":
            self.value = QWidget()
            box = QHBoxLayout(self.value)
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(4)
            selected = setting.value if isinstance(setting.value, list) else list(spec.default or [])
            for choice in spec.choices:
                check = QCheckBox(choice)
                check.setChecked(choice in selected)
                check.toggled.connect(lambda *_: self._touched(changed))
                self.checks.append(check)
                box.addWidget(check)
            box.addStretch()
        else:
            self.value = QLineEdit("" if setting.value is None else str(setting.value))
            placeholder = spec.metavar or ""
            if default:
                placeholder = f"submitslurm default: {default}"
            if spec.kind == "append":
                placeholder = (placeholder + "  (several: separate by spaces, quote as in the shell)").strip()
            self.value.setPlaceholderText(placeholder)
            self.value.textEdited.connect(lambda *_: self._touched(changed))

    def _touched(self, changed):
        # Editing the value of a switched-off option switches it on (as a default)
        if self.mode.currentData() == MODE_OFF:
            self.mode.setCurrentIndex(self.mode.findData(MODE_DEFAULT))
        changed()

    def setting(self) -> OptionSetting:
        kind = self.spec.kind
        if kind == "flag":
            value = None
        elif kind == "choice":
            value = self.value.currentText()
        elif kind == "multi_choice":
            value = [check.text() for check in self.checks if check.isChecked()]
        else:
            value = self.value.text()
        return OptionSetting(self.mode.currentData(), value)


class SubmitslurmContextDialog(QDialog):
    """Edits a copy of the context; context() gives the result after OK."""

    def __init__(self, context: SubmitslurmContext, project_root: Path | None, shell_env: dict[str, str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Submission context (submitslurm)")
        self.resize(1000, 780)
        self._context = context.copy()
        self._root = Path(project_root) if project_root else Path.cwd()
        self._shell_env = dict(shell_env)
        self._updating = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(150)
        self._refresh_timer.timeout.connect(self._refresh)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._options_tab(), "Options")
        self.tabs.addTab(self._free_text_tab(), "Free text")
        self.tabs.addTab(self._environment_tab(), "Job environment")

        example_row = QHBoxLayout()
        example_row.addWidget(QLabel("Example call: submitslurm"))
        self.example_edit = QLineEdit("--pipeline-step 1 SCRIPT.sbatch")
        self.example_edit.setToolTip("A call as written in a script; shown merged with the context, and its job environment")
        self.example_edit.textChanged.connect(self._schedule)
        example_row.addWidget(self.example_edit, stretch=1)
        self.merged_label = QLabel()
        self.merged_label.setWordWrap(True)
        self.merged_label.setTextFormat(Qt.PlainText)
        self.merged_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet(RED)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs, stretch=1)
        layout.addLayout(example_row)
        layout.addWidget(self.merged_label)
        layout.addWidget(self.error_label)
        layout.addWidget(self.buttons)
        self._refresh()

    # ------------------------------------------------------------ tabs
    def _options_tab(self) -> QWidget:
        content = QWidget()
        grid = QGridLayout(content)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(3)
        for column, title in enumerate(("Option", "Mode", "Value")):
            grid.addWidget(QLabel(f"<b>{title}</b>"), 0, column)
        self.rows: dict[str, _OptionRow] = {}
        row_index = 1
        for spec in context_option_specs():
            row = _OptionRow(spec, self._context.setting(spec.dest), self._schedule)
            label = QLabel(", ".join(spec.option_strings))
            for widget in (label, row.mode, row.value):
                widget.setToolTip(spec.help)
            grid.addWidget(label, row_index, 0)
            grid.addWidget(row.mode, row_index, 1)
            grid.addWidget(row.value, row_index, 2)
            self.rows[spec.dest] = row
            row_index += 1
            if spec.dest == "job_path":
                self.job_path_project = QCheckBox("when off: the directories of the project PATH settings (not the inherited PATH)")
                self.job_path_project.setChecked(self._context.job_path_from_project)
                self.job_path_project.toggled.connect(self._schedule)
                grid.addWidget(self.job_path_project, row_index, 2)
                row_index += 1
            if spec.dest == "python":
                self.python_status = QLabel()
                grid.addWidget(self.python_status, row_index, 2)
                row_index += 1
        grid.setColumnStretch(2, 1)
        grid.setRowStretch(row_index, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        widget = QWidget()
        box = QVBoxLayout(widget)
        note = QLabel("Options added to every submitslurm call of the Pipeline tab: default = used when the call "
                      "does not give the option, force = replaces the call's value. Per-call options (-a, "
                      "--pipeline-step, --dependencies, the script) and pipeline maintenance actions are not here. "
                      "Hover an option for its help.")
        note.setWordWrap(True)
        note.setStyleSheet(GREY)
        box.addWidget(note)
        box.addWidget(scroll, stretch=1)
        return widget

    def _free_text_tab(self) -> QWidget:
        widget = QWidget()
        box = QVBoxLayout(widget)
        note = QLabel("Further submitslurm arguments, added to every call after the options "
                      "(shell quoting). The call's own options come later and win where an option is given once.")
        note.setWordWrap(True)
        note.setStyleSheet(GREY)
        box.addWidget(note)
        self.free_text_edit = QPlainTextEdit(self._context.free_text)
        self.free_text_edit.textChanged.connect(self._schedule)
        box.addWidget(self.free_text_edit, stretch=1)
        return widget

    def _environment_tab(self) -> QWidget:
        widget = QWidget()
        box = QVBoxLayout(widget)
        self.inherit_label = QLabel()
        self.inherit_label.setWordWrap(True)
        box.addWidget(self.inherit_label)

        path_header = QHBoxLayout()
        self.path_title = QLabel()
        path_header.addWidget(self.path_title)
        path_header.addStretch()
        for text, slot in (("Add…", self._path_add), ("Remove", self._path_remove), ("Up", lambda: self._path_move(-1)),
                           ("Down", lambda: self._path_move(1)), ("Reset to computed", self._path_reset)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            path_header.addWidget(button)
        box.addLayout(path_header)
        self.path_tree = QTreeWidget()
        self.path_tree.setHeaderLabels(["Directory (double-click to edit)", "Source"])
        self.path_tree.setRootIsDecorated(False)
        self.path_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.path_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.path_tree.itemChanged.connect(self._path_edited)
        box.addWidget(self.path_tree, stretch=2)

        variables_header = QHBoxLayout()
        variables_header.addWidget(QLabel("<b>Variables</b> (double-click a value to change it; grey ones are set per job)"))
        variables_header.addStretch()
        add = QPushButton("Add variable")
        add.clicked.connect(self._variable_add)
        variables_header.addWidget(add)
        reset = QPushButton("Reset to computed")
        reset.clicked.connect(self._variables_reset)
        variables_header.addWidget(reset)
        box.addLayout(variables_header)
        self.variables_table = QTableWidget(0, 3)
        self.variables_table.setHorizontalHeaderLabels(["Name", "Value", "Source"])
        self.variables_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.variables_table.verticalHeader().setVisible(False)
        self.variables_table.itemChanged.connect(self._variable_edited)
        box.addWidget(self.variables_table, stretch=2)

        box.addWidget(QLabel("<b>--export</b> of the example job"))
        self.export_view = QPlainTextEdit()
        self.export_view.setReadOnly(True)
        self.export_view.setMaximumHeight(80)
        box.addWidget(self.export_view)
        self.warnings_label = QLabel()
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setStyleSheet(RED)
        box.addWidget(self.warnings_label)
        return widget

    # ------------------------------------------------------------ collecting and preview
    def _schedule(self, *_):
        if not self._updating:
            self._refresh_timer.start()

    def _collect(self):
        context = self._context
        options = {}
        for dest, row in self.rows.items():
            setting = row.setting()
            if setting.mode != MODE_OFF or setting.value not in (None, "", []):
                options[dest] = setting
        context.options = options
        context.job_path_from_project = self.job_path_project.isChecked()
        context.free_text = self.free_text_edit.toPlainText()

    def _refresh(self):
        self._collect()
        context = self._context
        python = self.rows["python"].setting()
        if python.mode != MODE_OFF and str(python.value or "").strip():
            try:
                check_python_environment(str(python.value).strip())
                self.python_status.setText("✓ environment with bin/python")
                self.python_status.setStyleSheet("color: #27ae60;")
            except ValueError as e:
                self.python_status.setText(f"⚠ {e}")
                self.python_status.setStyleSheet(RED)
        else:
            self.python_status.setText("off: LAUPY_PYTHON of the console shell if set (not with --vanilla), else the initPython default")
            self.python_status.setStyleSheet(GREY)
        self.job_path_project.setToolTip(os.pathsep.join(context.project_path_dirs) or "(no directories configured)")
        error = ""
        try:
            defaults, forced, extra = context.arguments()
            merged = merge_arguments(shlex.split(self.example_edit.text()), defaults, forced, extra)
            self.merged_label.setText("Merged: submitslurm " + quote_arguments(merged))
            self._show_preview(preview_environment(parse_merged(merged), self._root, self._shell_env))
        except ValueError as e:
            error = f"⚠ {e}"
        self.error_label.setText(error)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(not error)

    def _show_preview(self, preview):
        self._updating = True
        try:
            if preview.inherits:
                self.inherit_label.setText("Jobs inherit the whole environment of the console shell (--export=ALL); "
                                           "listed are the variables submitslurm sets explicitly. Use --vanilla to isolate the jobs.")
                self.inherit_label.setStyleSheet(GREY)
            else:
                self.inherit_label.setText("--vanilla / --vanilla-strict: jobs get only SLURM_* and the variables listed here.")
                self.inherit_label.setStyleSheet("")
            edited = self._context.path_override is not None
            self.path_title.setText("<b>PATH</b>" + (" (edited, exported as --env PATH)" if edited else " (computed)"))
            self.path_tree.clear()
            for directory, source in preview.path:
                item = QTreeWidgetItem([directory, "edited" if edited else source])
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                if not os.path.isdir(directory):
                    item.setForeground(0, Qt.red)
                    item.setToolTip(0, "directory does not exist")
                self.path_tree.addTopLevelItem(item)
            self.variables_table.setRowCount(0)
            for name, (value, source) in preview.variables.items():
                if name != "PATH":
                    self._add_variable_row(name, value, source, editable_value=name not in FIXED_VARIABLES)
            self.export_view.setPlainText(preview.export)
            self.warnings_label.setText("\n".join(f"⚠ {w}" for w in preview.warnings))
        finally:
            self._updating = False

    def _add_variable_row(self, name, value, source, editable_value=True, editable_name=False):
        row = self.variables_table.rowCount()
        self.variables_table.insertRow(row)
        name_item = QTableWidgetItem(name)
        if not editable_name:
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
        value_item = QTableWidgetItem(value)
        if not editable_value:
            value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)
            value_item.setForeground(Qt.gray)
        source_item = QTableWidgetItem(source)
        source_item.setFlags(source_item.flags() & ~Qt.ItemIsEditable)
        for column, item in enumerate((name_item, value_item, source_item)):
            self.variables_table.setItem(row, column, item)
        return row

    # ------------------------------------------------------------ PATH editing
    def _current_path(self) -> list[str]:
        items = (self.path_tree.topLevelItem(i) for i in range(self.path_tree.topLevelItemCount()))
        return [item.text(0).strip() for item in items if item.text(0).strip()]

    def _set_path_override(self, directories: list[str]):
        if any("," in d or os.pathsep in d for d in directories):
            self.error_label.setText(f"⚠ a PATH directory must not contain ',' or '{os.pathsep}'")
            return
        self._context.path_override = directories
        self._schedule()

    def _path_edited(self, *_):
        if not self._updating:
            self._set_path_override(self._current_path())

    def _path_add(self):
        directory = QFileDialog.getExistingDirectory(self, "Directory for the job PATH", str(self._root))
        if directory:
            directories = self._current_path()
            item = self.path_tree.currentItem()
            index = self.path_tree.indexOfTopLevelItem(item) + 1 if item is not None else len(directories)
            directories.insert(index, directory)
            self._set_path_override(directories)

    def _path_remove(self):
        item = self.path_tree.currentItem()
        if item is not None:
            directories = self._current_path()
            del directories[self.path_tree.indexOfTopLevelItem(item)]
            self._set_path_override(directories)

    def _path_move(self, step):
        item = self.path_tree.currentItem()
        if item is None:
            return
        directories = self._current_path()
        index = self.path_tree.indexOfTopLevelItem(item)
        target = index + step
        if 0 <= target < len(directories):
            directories[index], directories[target] = directories[target], directories[index]
            self._set_path_override(directories)
            QTimer.singleShot(300, lambda: self.path_tree.setCurrentItem(self.path_tree.topLevelItem(target)))

    def _path_reset(self):
        self._context.path_override = None
        self._schedule()

    # ------------------------------------------------------------ variable editing
    def _variable_edited(self, item):
        if self._updating:
            return
        row = item.row()
        name_item, value_item = self.variables_table.item(row, 0), self.variables_table.item(row, 1)
        name = name_item.text().strip() if name_item else ""
        if not name or name == "NAME" or name in FIXED_VARIABLES:
            return
        value = value_item.text() if value_item else ""
        if "," in value:
            self.error_label.setText(f"⚠ {name}: a value must not contain ',' (sbatch --export separator)")
            return
        self._context.env_overrides[name] = value
        self._schedule()

    def _variable_add(self):
        self._updating = True
        row = self._add_variable_row("NAME", "", "new", editable_value=True, editable_name=True)
        self._updating = False
        self.variables_table.editItem(self.variables_table.item(row, 0))

    def _variables_reset(self):
        self._context.env_overrides.clear()
        self._schedule()

    # ------------------------------------------------------------ result
    def context(self) -> SubmitslurmContext:
        self._collect()
        return self._context
