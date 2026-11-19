"""
Compare a project's recipe copy with its upstream cookbook recipe.

Project > Compare recipe with upstream... shows which files differ, their
diffs, and copies single files in either direction. It also checks whether
the cookbook git repository is behind its remote and offers a clean
(fast-forward) merge.

@author: Vojtech Kulvait
@license: GNU GPL v3.0
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import shiboken6
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from lautools.browser.text_viewer import TextSource, open_text_viewer
from lautools.browser.utils import open_terminal
from lautools.recipe_sync import (
    CookbookGitStatus,
    ProjectRecipeState,
    ProjectRecipe,
    RecipeFileStatus,
    backport_file,
    compare_recipes,
    cookbook_git_status,
    fast_forward_cookbook,
    project_recipe_state,
    remember_project_recipe,
    resolve_project_recipe,
    unified_diff,
    update_file_from_upstream,
)

log = logging.getLogger(__name__)

STATUS_COLORS = {
    RecipeFileStatus.MODIFIED: QColor(255, 200, 120),
    RecipeFileStatus.ONLY_IN_PROJECT: QColor(170, 220, 170),
    RecipeFileStatus.ONLY_IN_UPSTREAM: QColor(170, 200, 240),
}


# ----------------------------------------------------------------------
# Cookbook git check in the background (git fetch may take a while)
# ----------------------------------------------------------------------

class _TaskSignals(QObject):
    finished = Signal(object)


class _Task(QRunnable):
    def __init__(self, function: Callable[[], object], signals: _TaskSignals):
        super().__init__()
        self.function = function
        self.signals = signals

    def run(self):
        result = self.function()
        try:
            self.signals.finished.emit(result)
        except RuntimeError:
            pass  # application shutting down: the receiver is gone


_running_tasks: set[_TaskSignals] = set()


def run_in_background(function: Callable[[], object], owner: QObject, callback: Callable[[object], None]) -> None:
    """Run function() on the thread pool and pass its result to callback in the GUI thread.

    The callback is skipped if owner (e.g. a dialog) was deleted in the meantime.
    function must not touch widgets or the database connection of the GUI thread.
    """
    signals = _TaskSignals()  # no parent: must outlive a closed dialog until the task ends
    _running_tasks.add(signals)

    def finished(result):
        _running_tasks.discard(signals)
        if shiboken6.isValid(owner):
            callback(result)

    signals.finished.connect(finished)
    QThreadPool.globalInstance().start(_Task(function, signals))


def check_cookbook_in_background(directory: Path, owner: QObject, callback: Callable[[CookbookGitStatus], None]) -> None:
    """cookbook_git_status(directory) with a fetch, on the thread pool."""
    run_in_background(lambda: cookbook_git_status(Path(directory), fetch=True), owner, callback)


def offer_cookbook_fast_forward(parent: QWidget, status: CookbookGitStatus) -> bool:
    """If the cookbook can be fast-forwarded, ask the user and merge. Returns True if merged."""
    if not status.can_fast_forward:
        return False
    answer = QMessageBox.question(
        parent, "Cookbook not up to date",
        f"Your cookbook\n{status.repo_root}\nis {status.behind} commit(s) behind {status.upstream_ref} "
        "and has no local changes.\n\nDo you want to update it with a clean (fast-forward) merge?",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
    )
    if answer != QMessageBox.Yes:
        return False
    try:
        fast_forward_cookbook(status.repo_root)
    except Exception as exc:
        QMessageBox.warning(parent, "Merge failed", str(exc))
        return False
    return True


# ----------------------------------------------------------------------
# Dialog
# ----------------------------------------------------------------------

class RecipeSyncDialog(QDialog):
    def __init__(self, db, project, parent=None):
        super().__init__(parent)
        self.db = db
        self.project = project
        self.recipe: ProjectRecipe = resolve_project_recipe(db, project)
        self.setWindowTitle(f"Recipe of {project.name} vs. upstream")
        self.resize(1000, 640)
        self._build()
        self._load_files()
        if self.recipe.upstream_recipe_dir is not None:
            self._check_cookbook()

    # -- layout ----------------------------------------------------------

    def _build(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.project_recipe_label = self._path_label(self.recipe.project_recipe_dir)
        self.upstream_recipe_label = self._path_label(self.recipe.upstream_recipe_dir)
        form.addRow("Project recipe:", self.project_recipe_label)
        form.addRow("Upstream recipe:", self.upstream_recipe_label)

        link_row = QHBoxLayout()
        self.link_label = QLabel()
        self.remember_button = QPushButton("Remember this link")
        self.remember_button.setToolTip("Store the inferred project → recipe → upstream link in the database")
        self.remember_button.clicked.connect(self._remember_link)
        link_row.addWidget(self.link_label, 1)
        link_row.addWidget(self.remember_button)
        form.addRow("Link:", self._wrap(link_row))

        git_row = QHBoxLayout()
        self.git_label = QLabel("Cookbook git: not checked")
        self.git_label.setWordWrap(True)
        self.check_button = QPushButton("Check upstream")
        self.check_button.clicked.connect(self._check_cookbook)
        git_row.addWidget(self.git_label, 1)
        git_row.addWidget(self.check_button)
        form.addRow("Cookbook:", self._wrap(git_row))
        layout.addLayout(form)

        self.problems_label = QLabel("\n".join(self.recipe.problems))
        self.problems_label.setStyleSheet("color: #b00000;")
        self.problems_label.setWordWrap(True)
        self.problems_label.setVisible(bool(self.recipe.problems))
        layout.addWidget(self.problems_label)

        filter_row = QHBoxLayout()
        self.show_identical = QCheckBox("Show identical files")
        self.show_identical.toggled.connect(self._fill_table)
        self.counts_label = QLabel()
        filter_row.addWidget(self.show_identical)
        filter_row.addStretch(1)
        filter_row.addWidget(self.counts_label)
        layout.addLayout(filter_row)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Status", "File"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.doubleClicked.connect(self._show_diff)
        layout.addWidget(self.table, 1)

        actions = QHBoxLayout()
        self.diff_button = QPushButton("Show diff")
        self.diff_button.clicked.connect(self._show_diff)
        self.backport_button = QPushButton("Backport to upstream…")
        self.backport_button.setToolTip("Copy the selected project files into the cookbook recipe")
        self.backport_button.clicked.connect(self._backport)
        self.update_button = QPushButton("Update project from upstream…")
        self.update_button.setToolTip("Copy the selected upstream files into the project recipe")
        self.update_button.clicked.connect(self._update_from_upstream)
        self.upstream_terminal_button = QPushButton("Terminal in upstream")
        self.upstream_terminal_button.clicked.connect(
            lambda: open_terminal(self.recipe.upstream_recipe_dir, on_error=self._error))
        for button in (self.diff_button, self.backport_button, self.update_button, self.upstream_terminal_button):
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._update_link_label()
        self._update_buttons()

    @staticmethod
    def _wrap(row_layout) -> QWidget:
        widget = QWidget()
        row_layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(row_layout)
        return widget

    @staticmethod
    def _path_label(path: Path | None) -> QLabel:
        label = QLabel(str(path) if path is not None else "— (not found)")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return label

    def _update_link_label(self):
        if self.recipe.linked_in_database:
            self.link_label.setText("stored in the database")
            self.remember_button.setVisible(False)
        else:
            self.link_label.setText("inferred from the project's symlinks and the recipe name")
            self.remember_button.setVisible(True)
            self.remember_button.setEnabled(self.recipe.upstream_recipe_dir is not None)

    def _error(self, message: str):
        self.status_label.setText(f"Error: {message}")

    # -- files -------------------------------------------------------------

    def _load_files(self):
        self.diffs = []
        if self.recipe.project_recipe_dir is not None and self.recipe.upstream_recipe_dir is not None:
            try:
                self.diffs = compare_recipes(self.recipe.project_recipe_dir, self.recipe.upstream_recipe_dir)
            except OSError as exc:
                self._error(f"Cannot compare recipes: {exc}")
        self._fill_table()

    def _fill_table(self):
        shown = [d for d in self.diffs if self.show_identical.isChecked() or d.status != RecipeFileStatus.IDENTICAL]
        self.table.setRowCount(len(shown))
        for row, diff in enumerate(shown):
            status_item = QTableWidgetItem(diff.status.value)
            status_item.setData(Qt.UserRole, diff)
            file_item = QTableWidgetItem(diff.relative_path)
            color = STATUS_COLORS.get(diff.status)
            if color is not None:
                for item in (status_item, file_item):
                    item.setBackground(QBrush(color))
                    item.setForeground(QBrush(QColor("black")))
            self.table.setItem(row, 0, status_item)
            self.table.setItem(row, 1, file_item)
        counts = {status: sum(1 for d in self.diffs if d.status == status) for status in RecipeFileStatus}
        self.counts_label.setText(", ".join(f"{n} {status.value}" for status, n in counts.items() if n))
        self._update_buttons()

    def _selected(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        return [self.table.item(row, 0).data(Qt.UserRole) for row in rows]

    def _update_buttons(self):
        selected = self._selected()
        have_both = self.recipe.project_recipe_dir is not None and self.recipe.upstream_recipe_dir is not None
        self.diff_button.setEnabled(have_both and len(selected) > 0)
        self.backport_button.setEnabled(have_both and any(
            d.status in (RecipeFileStatus.MODIFIED, RecipeFileStatus.ONLY_IN_PROJECT) for d in selected))
        self.update_button.setEnabled(have_both and any(
            d.status in (RecipeFileStatus.MODIFIED, RecipeFileStatus.ONLY_IN_UPSTREAM) for d in selected))
        self.upstream_terminal_button.setEnabled(self.recipe.upstream_recipe_dir is not None)

    def _show_diff(self, *_):
        selected = self._selected()
        if not selected:
            return
        sources = [TextSource(title=Path(d.relative_path).name,
                              text=unified_diff(self.recipe.project_recipe_dir, self.recipe.upstream_recipe_dir, d.relative_path),
                              header=f"{d.relative_path}: '-' upstream, '+' project")
                   for d in selected]
        open_text_viewer(sources, title=f"Recipe diff – {self.project.name}")

    def _copy(self, statuses, copy_file, direction: str, target_dir: Path):
        files = [d.relative_path for d in self._selected() if d.status in statuses]
        if not files:
            return
        listing = "\n".join(files[:20]) + (f"\n… and {len(files) - 20} more" if len(files) > 20 else "")
        answer = QMessageBox.question(
            self, direction, f"Overwrite in\n{target_dir}\nthese {len(files)} file(s)?\n\n{listing}",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
        if answer != QMessageBox.Yes:
            return
        failed = []
        for relative in files:
            try:
                copy_file(self.recipe.project_recipe_dir, self.recipe.upstream_recipe_dir, relative)
            except OSError as exc:
                failed.append(f"{relative}: {exc}")
        self._load_files()
        if failed:
            QMessageBox.warning(self, direction, "Some files were not copied:\n" + "\n".join(failed))
        return len(files) - len(failed)

    def _backport(self):
        copied = self._copy((RecipeFileStatus.MODIFIED, RecipeFileStatus.ONLY_IN_PROJECT), backport_file,
                            "Backport to upstream", self.recipe.upstream_recipe_dir)
        if copied:
            self.status_label.setText(f"Backported {copied} file(s). Review, commit and push them in the cookbook "
                                      "(Terminal in upstream: git diff, git commit, git push).")
            self._check_cookbook(fetch_message=False)

    def _update_from_upstream(self):
        copied = self._copy((RecipeFileStatus.MODIFIED, RecipeFileStatus.ONLY_IN_UPSTREAM), update_file_from_upstream,
                            "Update project from upstream", self.recipe.project_recipe_dir)
        if copied:
            self.status_label.setText(f"Updated {copied} file(s) of the project recipe from upstream.")

    # -- database link -----------------------------------------------------

    def _remember_link(self):
        try:
            remember_project_recipe(self.db, self.project, self.recipe)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot remember link", str(exc))
            return
        self.recipe = resolve_project_recipe(self.db, self.project)
        self._update_link_label()
        self.status_label.setText("Link stored; the Project menu now offers the upstream recipe.")

    # -- cookbook git --------------------------------------------------------

    def _check_cookbook(self, *_, fetch_message: bool = True):
        if self.recipe.upstream_recipe_dir is None:
            return
        self.check_button.setEnabled(False)
        if fetch_message:
            self.git_label.setText("Cookbook git: fetching the remote…")
        check_cookbook_in_background(self.recipe.upstream_recipe_dir, self, self._cookbook_checked)

    def _cookbook_checked(self, status: CookbookGitStatus):
        self.check_button.setEnabled(True)
        self.git_label.setText(status.summary())
        if offer_cookbook_fast_forward(self, status):
            self.status_label.setText(f"Cookbook updated from {status.upstream_ref}.")
            merged = cookbook_git_status(status.directory, fetch=False)
            merged.fetched = True  # fetched just before the merge
            self.git_label.setText(merged.summary())
            self._load_files()


# ----------------------------------------------------------------------
# Compact status: "Cookbook: up to date · Recipe: 2 files changed"
# ----------------------------------------------------------------------

class RecipeStatusButton(QToolButton):
    """Flat button showing whether the cookbook and the project recipe are up to date.

    The check (recipe comparison and git fetch, at most every few minutes)
    runs in the background. Clicking opens the RecipeSyncDialog; the state is
    checked again when it closes. Used in the main window status bar and in
    the Configure Project dialog.
    """
    stateChanged = Signal(object)  # ProjectRecipeState or None

    def __init__(self, db, parent=None, hide_without_recipe: bool = True):
        super().__init__(parent)
        self.db = db
        self.hide_without_recipe = hide_without_recipe
        self.project = None
        self.state: ProjectRecipeState | None = None
        self._generation = 0  # ignores results of checks started for a previous project
        self.setAutoRaise(True)
        self.setCursor(Qt.PointingHandCursor)
        self.clicked.connect(self.open_compare_dialog)
        self.setVisible(False)

    def set_project(self, project, fetch: bool = True) -> None:
        """Show the state of project (None hides the button) and start a check."""
        self.project = project
        self.state = None
        self._generation += 1
        if project is None:
            self.setVisible(False)
            self.stateChanged.emit(None)
            return
        try:
            recipe = resolve_project_recipe(self.db, project)  # database access: GUI thread
        except Exception as exc:
            log.exception("Cannot resolve project recipe")
            self._show("Recipe: check failed", str(exc), attention=True)
            return
        if recipe.project_recipe_dir is None:  # not created from a recipe
            if self.hide_without_recipe:
                self.setVisible(False)
            else:
                self._show("— (no recipe symlinks)", "\n".join(recipe.problems), attention=False)
                self.setEnabled(False)
            return
        self.setEnabled(True)
        self._show("Cookbook: checking… · Recipe: checking…", "Checking the cookbook and the project recipe", attention=False)
        generation = self._generation
        run_in_background(lambda: project_recipe_state(recipe, fetch=fetch), self,
                          lambda state: self._checked(generation, state))

    def refresh(self, fetch: bool = False) -> None:
        self.set_project(self.project, fetch=fetch)

    def _checked(self, generation: int, state: ProjectRecipeState) -> None:
        if generation != self._generation:
            return
        self.state = state
        self._show(state.short_text(), state.tooltip() + "\n\nClick to compare the recipe with upstream.",
                   attention=state.needs_attention)
        self.stateChanged.emit(state)

    def _show(self, text: str, tooltip: str, attention: bool) -> None:
        self.setText(text)
        self.setToolTip(tooltip)
        self.setStyleSheet("QToolButton { color: #c05000; font-weight: bold; }" if attention else "")
        self.setVisible(True)

    def open_compare_dialog(self) -> None:
        if self.project is None:
            return
        RecipeSyncDialog(self.db, self.project, parent=self.window()).exec()
        self.refresh(fetch=False)  # the dialog fetched already; files may have been copied
