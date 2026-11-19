"""
Checkable job table used by the Jobs menu.

The dialog is both the overview and the confirmation: the user sees every
candidate job, (un)checks what should be affected and presses the action
button once. All selectable rows are checked by default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from lautools.browser.pipeline_tree_widget import STATE_COLORS


@dataclass
class JobRow:
    """One table row. ``key`` is returned for checked rows.

    ``checked`` is the initial tick. A non-empty ``check_warning`` is shown
    (Yes/No) before the user ticks the row; answering No leaves it unticked.
    """
    key: Any
    values: Sequence[str]
    state: str = ""
    selectable: bool = True
    tooltip: str = ""
    checked: bool = True
    check_warning: str = ""
    extra: dict = field(default_factory=dict)


class JobSelectionDialog(QDialog):
    def __init__(self, title: str, description: str, headers: Sequence[str], rows: Sequence[JobRow],
                 action_label: str, *, state_column: int | None = None, destructive: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 500)
        self._action_label = action_label
        self._rows = list(rows)
        self._row_of: dict[int, JobRow] = {}
        self._suppress_warnings = False

        layout = QVBoxLayout(self)
        label = QLabel(description)
        label.setWordWrap(True)
        layout.addWidget(label)

        self.tree = QTreeWidget()
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(False)
        self.tree.setHeaderLabels(list(headers))
        self._items: list[QTreeWidgetItem] = []
        for row in self._rows:
            item = QTreeWidgetItem([str(v) if v is not None else "" for v in row.values])
            flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
            if row.selectable:
                flags |= Qt.ItemIsUserCheckable
                item.setCheckState(0, Qt.Checked if row.checked else Qt.Unchecked)
            else:
                item.setCheckState(0, Qt.Unchecked)
                item.setDisabled(True)
            item.setFlags(flags)
            if row.tooltip:
                for col in range(len(headers)):
                    item.setToolTip(col, row.tooltip)
            if state_column is not None:
                color = STATE_COLORS.get(row.state)
                if color is not None:
                    color = QColor(color)  # do not mutate the shared palette
                    color.setAlpha(80)
                    item.setBackground(state_column, color)
            self.tree.addTopLevelItem(item)
            self._items.append(item)
            self._row_of[id(item)] = row
        header = self.tree.header()
        header.setSectionResizeMode(QHeaderView.Interactive)
        for col in range(len(headers)):
            self.tree.resizeColumnToContents(col)
        header.setStretchLastSection(True)
        # Keep the given (topological) order until the user clicks a header.
        header.setSortIndicator(-1, Qt.AscendingOrder)
        self.tree.setSortingEnabled(True)
        self.tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.tree)

        selection_row = QHBoxLayout()
        all_button = QPushButton("Select all")
        all_button.clicked.connect(lambda: self._set_all(Qt.Checked))
        none_button = QPushButton("Select none")
        none_button.clicked.connect(lambda: self._set_all(Qt.Unchecked))
        for button in (all_button, none_button):
            button.setAutoDefault(False)  # Enter goes to the dialog's default button only
        selection_row.addWidget(all_button)
        selection_row.addWidget(none_button)
        selection_row.addStretch()
        layout.addLayout(selection_row)

        buttons = QDialogButtonBox()
        self.action_button = buttons.addButton(action_label, QDialogButtonBox.AcceptRole)
        cancel_button = buttons.addButton(QDialogButtonBox.Cancel)
        if destructive:
            self.action_button.setStyleSheet("QPushButton:enabled { color: white; background-color: #c62828; font-weight: bold; }")
            # The safe choice is the default; Enter must not cancel jobs.
            cancel_button.setDefault(True)
            self.action_button.setAutoDefault(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._update_action_button()

    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        row = self._row_of.get(id(item))
        if (column == 0 and row is not None and row.check_warning and not self._suppress_warnings
                and item.checkState(0) == Qt.Checked):
            answer = QMessageBox.warning(self, "Confirm selection", row.check_warning,
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                self._set_check(item, Qt.Unchecked)
        self._update_action_button()

    def _set_check(self, item: QTreeWidgetItem, state):
        self._suppress_warnings = True
        try:
            item.setCheckState(0, state)
        finally:
            self._suppress_warnings = False

    def _set_all(self, state):
        selectable = [(item, row) for item, row in zip(self._items, self._rows) if row.selectable]
        warned = [(item, row) for item, row in selectable if row.check_warning]
        for item, row in selectable:
            if state == Qt.Unchecked or not row.check_warning:
                self._set_check(item, state)
        if state == Qt.Checked:
            # Ask once for all rows that need confirmation instead of per row.
            pending = [(item, row) for item, row in warned if item.checkState(0) != Qt.Checked]
            if pending:
                answer = QMessageBox.warning(
                    self, "Confirm selection",
                    f"Also select {len(pending)} job(s) that need confirmation?\n\n{pending[0][1].check_warning}",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if answer == QMessageBox.Yes:
                    for item, _ in pending:
                        self._set_check(item, Qt.Checked)
        self._update_action_button()

    def _update_action_button(self, *args):
        count = len(self.selected_keys())
        self.action_button.setText(f"{self._action_label} ({count})")
        self.action_button.setEnabled(count > 0)

    def selected_keys(self) -> list:
        return [row.key for item, row in zip(self._items, self._rows)
                if row.selectable and item.checkState(0) == Qt.Checked]
