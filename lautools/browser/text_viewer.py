"""
Built-in text viewer/editor (PySide6), replacing external editors such as
mousepad for SLURM logs, parameter files and DAG entries.

* Independent top-level window; the browser stays usable while it is open.
* Several documents as tabs (e.g. StdOut and StdErr of one job).
* Files are read on a worker thread, so large logs on GPFS do not freeze the UI.
* Follow mode re-reads a growing log and keeps the view at its end.
* Read-only by default; Edit enables editing and saving of files.
* F11 toggles full screen, Ctrl+F finds, Ctrl+wheel / Ctrl+± zooms.
"""
from __future__ import annotations

import os
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QFontDatabase, QKeySequence, QTextCursor, QTextDocument
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Larger files are shown from their end (logs are read bottom-up) and cannot be edited.
MAX_LOAD_BYTES = 32 * 1024 * 1024
FOLLOW_INTERVAL_MS = 2000


@dataclass
class TextSource:
    """A file on disk, or inline text when ``path`` is None."""
    title: str
    path: Optional[Path] = None
    text: str = ""
    header: str = ""


Source = Union[str, Path, TextSource]


def _as_source(source: Source) -> TextSource:
    if isinstance(source, TextSource):
        return source
    path = Path(source)
    return TextSource(title=path.name, path=path, header=str(path))


# ----------------------------------------------------------------------
# Background file loading
# ----------------------------------------------------------------------

class _LoadSignals(QObject):
    loaded = Signal(int, str, bool, object)  # token, text, truncated, (size, mtime)
    failed = Signal(int, str)


class _LoadTask(QRunnable):
    def __init__(self, path: Path, token: int, signals: _LoadSignals):
        super().__init__()
        self.path, self.token, self.signals = path, token, signals

    def run(self):
        try:
            stat = self.path.stat()
            with open(self.path, "rb") as handle:
                truncated = stat.st_size > MAX_LOAD_BYTES
                if truncated:
                    handle.seek(stat.st_size - MAX_LOAD_BYTES)
                data = handle.read()
            if truncated:
                data = data.split(b"\n", 1)[-1]  # drop the partial first line
            text = data.decode("utf-8", errors="replace")
            self.signals.loaded.emit(self.token, text, truncated, (stat.st_size, stat.st_mtime))
        except OSError as exc:
            self.signals.failed.emit(self.token, str(exc))


# ----------------------------------------------------------------------
# One document
# ----------------------------------------------------------------------

class _DocumentTab(QWidget):
    stateChanged = Signal()

    def __init__(self, source: TextSource, parent=None):
        super().__init__(parent)
        self.source = source
        self.truncated = False
        self.loading = False
        self._token = 0
        self._file_state = None  # (size, mtime) of the loaded content
        self._signals = _LoadSignals(self)
        self._signals.loaded.connect(self._on_loaded)
        self._signals.failed.connect(self._on_failed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.info = QLabel(source.header)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        self.editor = QPlainTextEdit()
        self.editor.setReadOnly(True)
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.editor.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.editor.modificationChanged.connect(lambda _: self.stateChanged.emit())
        layout.addWidget(self.editor)

        if source.path is None:
            self._set_text(source.text)
        else:
            self.reload()

    # -- properties --------------------------------------------------------

    @property
    def is_file(self) -> bool:
        return self.source.path is not None

    @property
    def can_edit(self) -> bool:
        return self.is_file and not self.truncated and not self.loading

    @property
    def modified(self) -> bool:
        return self.editor.document().isModified()

    def tab_title(self) -> str:
        return ("* " if self.modified else "") + self.source.title

    # -- loading -----------------------------------------------------------

    def reload(self, *, keep_end: bool = False):
        if not self.is_file:
            return
        self._token += 1
        self.loading = True
        self._keep_end = keep_end
        if not self.editor.toPlainText():
            self.editor.setPlainText("Loading...")
        QThreadPool.globalInstance().start(_LoadTask(self.source.path, self._token, self._signals))
        self.stateChanged.emit()

    def _on_loaded(self, token: int, text: str, truncated: bool, file_state):
        if token != self._token:
            return  # a newer reload is underway
        self.loading = False
        self.truncated = truncated
        self._file_state = file_state
        self._set_text(text, keep_end=self._keep_end)
        note = f" — showing the last {MAX_LOAD_BYTES // (1024 * 1024)} MiB of {file_state[0] / 2**20:.1f} MiB" if truncated else ""
        self.info.setText(f"{self.source.header}{note}")
        self.stateChanged.emit()

    def _on_failed(self, token: int, message: str):
        if token != self._token:
            return
        self.loading = False
        self.info.setText(f"{self.source.header} — cannot read: {message}")
        self._set_text("")
        self.stateChanged.emit()

    def _set_text(self, text: str, keep_end: bool = False):
        bar = self.editor.verticalScrollBar()
        position = bar.value()
        self.editor.setPlainText(text)
        self.editor.document().setModified(False)
        bar.setValue(bar.maximum() if keep_end else position)

    def changed_on_disk(self) -> bool:
        if not self.is_file or self.loading:
            return False
        try:
            stat = self.source.path.stat()
        except OSError:
            return False
        return (stat.st_size, stat.st_mtime) != self._file_state

    # -- editing -----------------------------------------------------------

    def set_editing(self, editing: bool):
        self.editor.setReadOnly(not (editing and self.can_edit))

    def save(self) -> Optional[str]:
        """Write the text back atomically. Returns an error message or None."""
        path = self.source.path
        try:
            handle, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                out.write(self.editor.toPlainText())
            if path.exists():
                shutil.copymode(path, tmp)
            os.replace(tmp, path)
        except OSError as exc:
            try:
                os.unlink(tmp)
            except (OSError, UnboundLocalError):
                pass
            return str(exc)
        self.editor.document().setModified(False)
        stat = path.stat()
        self._file_state = (stat.st_size, stat.st_mtime)
        return None


# ----------------------------------------------------------------------
# Window
# ----------------------------------------------------------------------

class TextViewerWindow(QMainWindow):
    def __init__(self, sources: Iterable[Source], *, title: str = "", editable: bool = False,
                 follow: bool = False, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(1000, 700)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self._create_toolbar()  # before tabs: their signals update the toolbar actions
        for source in sources:
            tab = _DocumentTab(_as_source(source))
            tab.stateChanged.connect(self._update_actions)
            self.tabs.addTab(tab, tab.tab_title())
            self.tabs.setTabToolTip(self.tabs.count() - 1, tab.source.header)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.tabs)
        layout.addWidget(self._create_find_bar())
        self.setCentralWidget(central)
        self.tabs.currentChanged.connect(lambda _: self._update_actions())
        self.setWindowTitle(title or (self._current().source.title if self._current() else "Viewer"))

        self.follow_timer = QTimer(self)
        self.follow_timer.setInterval(FOLLOW_INTERVAL_MS)
        self.follow_timer.timeout.connect(self._follow_tick)
        self.edit_action.setChecked(editable)
        if follow and not editable:
            self.follow_action.setChecked(True)
            self.follow_timer.start()
            for tab in self._tabs():
                tab._keep_end = True  # initial load lands at the end
        self._update_actions()

    # -- UI construction ---------------------------------------------------

    def _action(self, toolbar, text, shortcut=None, slot=None, checkable=False, tip=""):
        action = QAction(text, self)
        action.setCheckable(checkable)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if tip or shortcut:
            action.setToolTip(f"{tip or text} ({QKeySequence(shortcut).toString()})" if shortcut else tip)
        if slot:
            action.triggered.connect(slot)
        toolbar.addAction(action)
        return action

    def _create_toolbar(self):
        bar = self.addToolBar("View")
        bar.setMovable(False)
        self.reload_action = self._action(bar, "Reload", "F5", lambda: self._current() and self._current().reload())
        self.follow_action = self._action(bar, "Follow", None, self._toggle_follow, True,
                                          "Re-read growing files and keep the view at their end")
        self.wrap_action = self._action(bar, "Wrap", None, self._toggle_wrap, True, "Wrap long lines")
        bar.addSeparator()
        self.edit_action = self._action(bar, "Edit", None, self._toggle_edit, True, "Allow editing of files")
        self.save_action = self._action(bar, "Save", QKeySequence.Save, self._save_current)
        bar.addSeparator()
        self._action(bar, "Find", QKeySequence.Find, self._show_find)
        self._action(bar, "A+", QKeySequence.ZoomIn, lambda: self._zoom(1), tip="Larger font")
        self._action(bar, "A−", QKeySequence.ZoomOut, lambda: self._zoom(-1), tip="Smaller font")
        self._action(bar, "Full screen", "F11", self._toggle_full_screen)

    def _create_find_bar(self) -> QWidget:
        self.find_bar = QWidget()
        row = QHBoxLayout(self.find_bar)
        row.setContentsMargins(0, 0, 0, 0)
        self.find_edit = QLineEdit()
        self.find_edit.setPlaceholderText("Find (Enter: next, Shift+Enter: previous)")
        self.find_edit.returnPressed.connect(lambda: self._find(backward=False))
        self.find_case = QCheckBox("Match case")
        previous = QPushButton("Previous")
        previous.clicked.connect(lambda: self._find(backward=True))
        following = QPushButton("Next")
        following.clicked.connect(lambda: self._find(backward=False))
        close = QPushButton("✕")
        close.setFlat(True)
        close.clicked.connect(self.find_bar.hide)
        for widget in (self.find_edit, self.find_case, previous, following, close):
            row.addWidget(widget)
        shift_enter = QAction(self.find_edit)
        shift_enter.setShortcut(QKeySequence("Shift+Return"))
        shift_enter.triggered.connect(lambda: self._find(backward=True))
        self.find_edit.addAction(shift_enter)
        escape = QAction(self.find_bar)
        escape.setShortcut(QKeySequence("Escape"))
        escape.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        escape.triggered.connect(self.find_bar.hide)
        self.find_bar.addAction(escape)
        self.find_bar.hide()
        return self.find_bar

    # -- helpers -----------------------------------------------------------

    def _current(self) -> Optional[_DocumentTab]:
        return self.tabs.currentWidget()

    def _tabs(self) -> list[_DocumentTab]:
        return [self.tabs.widget(i) for i in range(self.tabs.count())]

    def _update_actions(self):
        for index, tab in enumerate(self._tabs()):
            self.tabs.setTabText(index, tab.tab_title())
            tab.set_editing(self.edit_action.isChecked())
        tab = self._current()
        self.reload_action.setEnabled(tab is not None and tab.is_file)
        self.edit_action.setEnabled(any(t.is_file for t in self._tabs()))
        self.save_action.setEnabled(tab is not None and tab.is_file and tab.modified)
        if tab is not None and self.edit_action.isChecked() and tab.is_file and not tab.can_edit and tab.truncated:
            self.statusBar().showMessage("File is too large to edit; showing its end read-only")

    # -- actions -----------------------------------------------------------

    def _toggle_follow(self, on: bool):
        if on:
            self.follow_timer.start()
            for tab in self._tabs():
                tab.reload(keep_end=True)
        else:
            self.follow_timer.stop()

    def _follow_tick(self):
        for tab in self._tabs():
            # Never overwrite unsaved edits.
            if not tab.modified and tab.changed_on_disk():
                tab.reload(keep_end=True)

    def _toggle_wrap(self, on: bool):
        mode = QPlainTextEdit.WidgetWidth if on else QPlainTextEdit.NoWrap
        for tab in self._tabs():
            tab.editor.setLineWrapMode(mode)

    def _toggle_edit(self, on: bool):
        if on and self.follow_action.isChecked():
            self.follow_action.setChecked(False)
            self._toggle_follow(False)
        self._update_actions()

    def _save_current(self):
        tab = self._current()
        if tab is None or not tab.is_file or not tab.modified:
            return
        if tab.changed_on_disk():
            answer = QMessageBox.question(self, "File changed on disk",
                                          f"{tab.source.path}\nchanged on disk since it was loaded. Overwrite it?")
            if answer != QMessageBox.Yes:
                return
        error = tab.save()
        if error:
            QMessageBox.critical(self, "Save failed", f"Cannot save {tab.source.path}:\n{error}")
        else:
            self.statusBar().showMessage(f"Saved {tab.source.path}", 5000)
        self._update_actions()

    def _show_find(self):
        self.find_bar.show()
        self.find_edit.setFocus()
        self.find_edit.selectAll()

    def _find(self, backward: bool):
        tab = self._current()
        text = self.find_edit.text()
        if tab is None or not text:
            return
        flags = QTextDocument.FindFlag(0)
        if backward:
            flags |= QTextDocument.FindBackward
        if self.find_case.isChecked():
            flags |= QTextDocument.FindCaseSensitively
        if not tab.editor.find(text, flags):
            # Wrap around once.
            cursor = tab.editor.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End if backward else QTextCursor.MoveOperation.Start)
            tab.editor.setTextCursor(cursor)
            if not tab.editor.find(text, flags):
                self.statusBar().showMessage(f"'{text}' not found", 3000)

    def _zoom(self, step: int):
        for tab in self._tabs():
            tab.editor.zoomIn(step)

    def _toggle_full_screen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    # -- closing -----------------------------------------------------------

    def _confirm_discard(self, tabs: list[_DocumentTab]) -> bool:
        modified = [tab for tab in tabs if tab.modified]
        if not modified:
            return True
        names = "\n".join(str(tab.source.path) for tab in modified)
        answer = QMessageBox.question(self, "Unsaved changes", f"Discard unsaved changes?\n\n{names}",
                                      QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Cancel)
        return answer == QMessageBox.Discard

    def _close_tab(self, index: int):
        tab = self.tabs.widget(index)
        if not self._confirm_discard([tab]):
            return
        self.tabs.removeTab(index)
        tab.deleteLater()
        if self.tabs.count() == 0:
            self.close()

    def closeEvent(self, event):
        if not self._confirm_discard(self._tabs()):
            event.ignore()
            return
        self.follow_timer.stop()
        super().closeEvent(event)


# Top-level viewers have no Qt parent (so they are independent windows that do
# not stay above the browser); keep references so they are not garbage collected.
_open_windows: set = set()


def open_text_viewer(sources: Iterable[Source], *, title: str = "", editable: bool = False,
                     follow: bool = False) -> TextViewerWindow:
    """Open files (paths) and/or inline TextSource documents in a new viewer window."""
    window = TextViewerWindow(sources, title=title, editable=editable, follow=follow)
    _open_windows.add(window)
    window.destroyed.connect(lambda *_: _open_windows.discard(window))
    window.show()
    return window
