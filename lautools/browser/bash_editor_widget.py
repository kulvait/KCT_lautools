"""
Bash code editor with an embedded persistent bash session.

Upper part: code editor (line numbers, simple bash highlighting).
Lower part: console showing the output of a persistent bash process.

Code is executed by writing it into a temporary snippet file and sourcing it
in the persistent shell, so variables, cd, functions, exports persist between
executions, multi-line constructs work, and stdin is redirected from
/dev/null so commands cannot consume the shell's control input.
"""
from __future__ import annotations

import logging
import shlex
import subprocess
import tempfile
import uuid
from pathlib import Path

from PySide6.QtCore import (
    QObject,
    QProcess,
    QProcessEnvironment,
    QRect,
    QRegularExpression,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QFontDatabase,
    QKeySequence,
    QPainter,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
    QTextFormat,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from lautools.browser.bash_script_model import (
    INFO_HEADER_END,
    INFO_HEADER_FIRST_LINE,
    SEPARATOR_RE,
    ShellFunction,
    comment_start,
    executable_text,
    function_at,
    info_header_end,
    parse_functions,
)

from lautools.browser.submitslurm_context import SubmitslurmContext, WRAPPER


def _fmt(color=None, bold=False, italic=False, background=None):
    f = QTextCharFormat()
    if color:
        f.setForeground(QColor(color))
    if background:
        f.setBackground(QColor(background))
    if bold:
        f.setFontWeight(QFont.Bold)
    f.setFontItalic(italic)
    return f

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Persistent bash session
# ----------------------------------------------------------------------

class BashSession(QObject):
    """Persistent non-interactive bash driven through QProcess."""
    started = Signal()              # bash process started
    output = Signal(str)            # stdout+stderr text
    info = Signal(str)              # messages from the session itself
    commandFinished = Signal(int)   # exit status of the last snippet
    busyChanged = Signal(bool)
    cwdChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._marker = f"__LAUTOOLS_DONE_{uuid.uuid4().hex}__"
        self._tmpdir = tempfile.TemporaryDirectory(prefix="lautools_bash_")
        self._counter = 0
        self._buffer = ""
        self._busy = False
        self._queue: list[str] = []
        self._cwd: Path | None = None

    # -- lifecycle -----------------------------------------------------

    def is_running(self) -> bool:
        return (
            self._proc is not None
            and self._proc.state() != QProcess.NotRunning
        )

    def is_busy(self) -> bool:
        return self._busy

    def start(self, cwd: Path) -> bool:
        self.stop()
        self._cwd = Path(cwd)
        proc = QProcess(self)
        proc.setProgram("bash")
        proc.setArguments(["--noprofile", "--norc"])
        proc.setWorkingDirectory(str(cwd))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("TERM", "dumb")
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(self._on_ready_read)
        proc.finished.connect(self._on_finished)
        proc.start()
        if not proc.waitForStarted(3000):
            self.info.emit(f"Cannot start bash: {proc.errorString()}")
            return False
        self._proc = proc
        self._buffer = ""
        self._queue.clear()
        self._set_busy(False)
        self.info.emit(f"bash started in {cwd}")
        self.cwdChanged.emit(str(cwd))
        self.started.emit()
        return True

    def stop(self):
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        proc.finished.disconnect(self._on_finished)
        if proc.state() != QProcess.NotRunning:
            proc.write(b"exit\n")
            proc.closeWriteChannel()
            if not proc.waitForFinished(1000):
                proc.kill()
                proc.waitForFinished(1000)
        proc.deleteLater()
        self._queue.clear()
        self._set_busy(False)

    def restart(self):
        if self._cwd is not None:
            self.start(self._cwd)

    def interrupt(self):
        """Send SIGINT to children of the shell (running external commands)."""
        if not self.is_running():
            return
        pid = self._proc.processId()
        try:
            subprocess.run(["pkill", "-INT", "-P", str(pid)], check=False)
            self.info.emit("SIGINT sent to running commands")
        except OSError as exc:
            self.info.emit(f"Cannot interrupt: {exc}")

    # -- execution -----------------------------------------------------

    def execute(self, code: str, silent: bool = False):
        if not code.strip():
            return
        if not self.is_running():
            if self._cwd is None or not self.start(self._cwd):
                return
        if self._busy:
            self._queue.append((code, silent))
            self.info.emit(f"queued ({len(self._queue)} waiting)")
            return
        self._send(code, silent)



    def _send(self, code: str, silent: bool = False):
        self._current_silent = silent
        self._counter += 1
        snippet = Path(self._tmpdir.name) / f"snippet_{self._counter:05d}.sh"
        snippet.write_text(code if code.endswith("\n") else code + "\n")
        command = (
            f"source {shlex.quote(str(snippet))} < /dev/null\n"
            f"printf '%s %d %s\\n' {self._marker} \"$?\" \"$PWD\"\n"
        )
        self._set_busy(True)
        self._proc.write(command.encode())

    # -- process callbacks ---------------------------------------------

    def _on_ready_read(self):
        data = bytes(self._proc.readAllStandardOutput())
        self._buffer += data.decode(errors="replace")

        while True:
            idx = self._buffer.find(self._marker)
            if idx < 0:
                break
            end = self._buffer.find("\n", idx)
            if end < 0:
                break  # marker line not complete yet
            before = self._buffer[:idx]
            status_line = self._buffer[idx + len(self._marker):end].strip()
            self._buffer = self._buffer[end + 1:]
            if before:
                self.output.emit(before)
            status, _, cwd = status_line.partition(" ")
            if cwd:
                self._cwd = Path(cwd)
                self.cwdChanged.emit(cwd)
            self._set_busy(False)
            # in _on_ready_read, where commandFinished is emitted:
            if not self._current_silent:
                self.commandFinished.emit(int(status) if status.lstrip("-").isdigit() else -1)
            if self._queue:
                self._send(*self._queue.pop(0))

        # Flush everything except a tail that may be the beginning of a marker.
        keep = 0
        idx = self._buffer.find(self._marker)
        if idx >= 0:
            keep = len(self._buffer) - idx
        else:
            for k in range(min(len(self._marker), len(self._buffer)), 0, -1):
                if self._marker.startswith(self._buffer[-k:]):
                    keep = k
                    break
        flush = self._buffer[: len(self._buffer) - keep]
        if flush:
            self.output.emit(flush)
            self._buffer = self._buffer[len(flush):]

    def _on_finished(self, exit_code, _status):
        if self._buffer:
            self.output.emit(self._buffer)
            self._buffer = ""
        self.info.emit(
            f"bash exited with code {exit_code}; it will be restarted "
            "on the next execution"
        )
        self._queue.clear()
        self._set_busy(False)

    def _set_busy(self, busy: bool):
        if busy != self._busy:
            self._busy = busy
            self.busyChanged.emit(busy)


# ----------------------------------------------------------------------
# Editor
# ----------------------------------------------------------------------


class InfoScriptHighlighter(QSyntaxHighlighter):
    """Bash highlighting aware of the laupy INFO header and submitslurm."""

    NORMAL, HEADER, HEADER_END = 0, 1, 2
    KEYWORDS = (
        "if then else elif fi for while until do done case esac in "
        "function select return export local source declare readonly "
        "set unset shift break continue"
    ).split()

    def __init__(self, document):
        super().__init__(document)
        header_bg = "#f2f2f2"
        self.f_header = _fmt("#8c8c8c", italic=True, background=header_bg)
        self.f_header_key = _fmt("#5a5a5a", bold=True, background=header_bg)
        self.f_header_value = _fmt("#6e6e6e", background=header_bg)
        self.f_separator = _fmt("#b0b0b0")
        self.f_comment = _fmt("#8c8c8c", italic=True)
        self.f_keyword = _fmt("#0033b3", bold=True)
        self.f_string = _fmt("#067d17")
        self.f_variable = _fmt("#871094")
        self.f_submitslurm = _fmt("#b34700", bold=True, background="#fff1e0")
        self.f_option = _fmt("#9e4fc2")
        self.f_step = _fmt("#d00000", bold=True)
        self.f_sbatch = _fmt("#005f9e", bold=True)
        self.f_function = _fmt("#00627a", bold=True)

        R = QRegularExpression
        self.re_header_kv = R(r"^([A-Za-z][^:]{0,40}):(\s.*)?$")
        self.re_keyword = R(r"\b(?:%s)\b" % "|".join(self.KEYWORDS))
        self.re_func_def = R(
            r"^\s*(?:function\s+)?([A-Za-z_][\w:.-]*)\s*\(\s*\)"
            r"|^\s*function\s+([A-Za-z_][\w:.-]*)"
        )
        self.re_submitslurm = R(r"\bsubmitslurm\b")
        self.re_option = R(r"(?<![\w\"'-])--?[A-Za-z][\w-]*")
        self.re_step = R(r"--pipeline-step\s+(\d+)")
        self.re_sbatch = R(r"[\w./-]+\.sbatch\b")
        self.re_string = R(r'"(?:[^"\\]|\\.)*"' + r"|'[^']*'")
        self.re_variable = R(
            r"\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*|\$[0-9#?@*$!-]"
        )
        self.re_calls: QRegularExpression | None = None
        self._function_names: frozenset[str] = frozenset()

    def set_function_names(self, names):
        names = frozenset(names)
        if names == self._function_names:
            return
        self._function_names = names
        self.re_calls = (
            QRegularExpression(
                r"(?<![\w.-])(?:%s)(?![\w.-])"
                % "|".join(QRegularExpression.escape(n) for n in sorted(names))
            )
            if names else None
        )
        self.rehighlight()

    # ------------------------------------------------------------------

    def _apply(self, regex, char_format, text, limit, group=0):
        it = regex.globalMatch(text)
        while it.hasNext():
            m = it.next()
            start = m.capturedStart(group)
            if 0 <= start < limit:
                length = min(m.capturedLength(group), limit - start)
                self.setFormat(start, length, char_format)

    def _header_line(self, text):
        if SEPARATOR_RE.match(text):
            self.setFormat(0, len(text), self.f_separator)
            return
        m = self.re_header_kv.match(text)
        if m.hasMatch():
            self.setFormat(0, m.capturedEnd(1) + 1, self.f_header_key)
            self.setFormat(m.capturedEnd(1) + 1, len(text), self.f_header_value)
        else:
            self.setFormat(0, len(text), self.f_header)

    def highlightBlock(self, text):
        prev = self.previousBlockState()

        if self.currentBlock().blockNumber() == 0 and text.startswith(INFO_HEADER_FIRST_LINE):
            self._header_line(text)
            self.setCurrentBlockState(self.HEADER)
            return
        if prev == self.HEADER:
            self._header_line(text)
            self.setCurrentBlockState(
                self.HEADER_END if text.strip() == INFO_HEADER_END else self.HEADER
            )
            return

        self.setCurrentBlockState(self.NORMAL)
        if SEPARATOR_RE.match(text):
            self.setFormat(0, len(text), self.f_separator)
            return

        ci = comment_start(text)
        limit = len(text) if ci < 0 else ci

        self._apply(self.re_keyword, self.f_keyword, text, limit)
        is_submit = self.re_submitslurm.match(text[:limit]).hasMatch() or \
            self.re_submitslurm.globalMatch(text[:limit]).hasNext()
        if is_submit:
            self._apply(self.re_option, self.f_option, text, limit)
        # Strings after options: arguments passed via -a "..." stay green.
        self._apply(self.re_string, self.f_string, text, limit)
        self._apply(self.re_variable, self.f_variable, text, limit)
        if is_submit:
            self._apply(self.re_submitslurm, self.f_submitslurm, text, limit)
            self._apply(self.re_step, self.f_step, text, limit, group=1)
            self._apply(self.re_sbatch, self.f_sbatch, text, limit)
        self._apply(self.re_func_def, self.f_function, text, limit, group=1)
        self._apply(self.re_func_def, self.f_function, text, limit, group=2)
        if self.re_calls is not None:
            self._apply(self.re_calls, self.f_function, text, limit)

        if ci >= 0:
            self.setFormat(ci, len(text) - ci, self.f_comment)


class BashHighlighter(QSyntaxHighlighter):
    KEYWORDS = (
        "if then else elif fi for while until do done case esac in "
        "function select return export local source declare readonly "
        "set unset shift break continue"
    ).split()

    def __init__(self, document):
        super().__init__(document)

        def fmt(color, bold=False, italic=False):
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            if bold:
                f.setFontWeight(QFont.Bold)
            f.setFontItalic(italic)
            return f

        self._rules = [
            (QRegularExpression(r"\b(?:%s)\b" % "|".join(self.KEYWORDS)),
             fmt("#0033b3", bold=True)),
            (QRegularExpression(
                r"\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*|\$[0-9#?@*$!-]"),
             fmt("#871094")),
            (QRegularExpression(r'"(?:[^"\\]|\\.)*"' + r"|'[^']*'"),
             fmt("#067d17")),
            (QRegularExpression(r"(?:^|\s)#.*$"),
             fmt("#8c8c8c", italic=True)),
        ]

    def highlightBlock(self, text):
        for regex, char_format in self._rules:
            it = regex.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), char_format)


class _LineNumberArea(QWidget):
    def __init__(self, editor: "CodeEditor"):
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self):
        return QSize(self._editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self._editor.paint_line_numbers(event)


class CodeEditor(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))
        self._area = _LineNumberArea(self)
        self.blockCountChanged.connect(self._update_width)
        self.updateRequest.connect(self._update_area)
        self.cursorPositionChanged.connect(self._highlight_current_line)
        self._update_width()
        self._highlight_current_line()
        #self._highlighter = BashHighlighter(self.document())
        self._highlighter = InfoScriptHighlighter(self.document())

    def line_number_area_width(self) -> int:
        digits = len(str(max(1, self.blockCount())))
        return 12 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_width(self, *_):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _update_area(self, rect, dy):
        if dy:
            self._area.scroll(0, dy)
        else:
            self._area.update(0, rect.y(), self._area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_width()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._area.setGeometry(
            QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height())
        )

    def paint_line_numbers(self, event):
        painter = QPainter(self._area)
        painter.fillRect(event.rect(), self.palette().alternateBase())
        painter.setPen(self.palette().placeholderText().color())
        block = self.firstVisibleBlock()
        number = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())
        height = self.fontMetrics().height()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.drawText(0, top, self._area.width() - 6, height,
                                 Qt.AlignRight, str(number + 1))
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            number += 1

    def _highlight_current_line(self):
        selection = QTextEdit.ExtraSelection()
        selection.format.setBackground(self.palette().alternateBase())
        selection.format.setProperty(QTextFormat.FullWidthSelection, True)
        selection.cursor = self.textCursor()
        selection.cursor.clearSelection()
        self.setExtraSelections([selection])


# ----------------------------------------------------------------------
# Composite widget
# ----------------------------------------------------------------------

class BashEditorWidget(QWidget):
    """Editor (top) + persistent bash console (bottom)."""

    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project_path: Path | None = None
        self._file_path: Path | None = None

        self.session = BashSession(self)
        self.session.output.connect(lambda t: self._append(t))
        self.session.info.connect(lambda t: self._append(f"[{t}]\n", "#8c8c8c"))
        self.session.commandFinished.connect(self._on_command_finished)
        self.session.busyChanged.connect(self._on_busy_changed)
        self.session.cwdChanged.connect(lambda cwd: self.cwd_label.setText(f"cwd: {cwd}"))
        self._functions: list[ShellFunction] = []
        self._loaded_functions: dict[str, str] = {}   # name -> code in shell
        self._header_end = 0
        self.session.started.connect(self._on_shell_started)
        self.ss_context = SubmitslurmContext(values={"-w": "wd"}, forced=set())
        self._create_ui()
        self._update_title()



    # -- UI ------------------------------------------------------------

    def _create_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QToolBar(self)
        self.open_action = self._action(toolbar, "Open…", self.open_dialog)
        self.save_action = self._action(toolbar, "Save", self.save, QKeySequence.Save)
        self._action(toolbar, "Save As…", self.save_as)
        self._action(toolbar, "Reload", self.reload)
        toolbar.addSeparator()
        self._action(toolbar, "▶ Line", self.run_current_line)
        self._action(toolbar, "▶ Selection", self.run_selection)
        self._action(toolbar, "▶ File", self.run_file, "Ctrl+Shift+Return")
        toolbar.addSeparator()
        self._action(toolbar, "▶ Function", self.run_function_at_cursor)
        self._action(toolbar, "⟳ Load functions", self.load_all_functions, "Ctrl+Shift+L")
        self.auto_functions_check = QCheckBox("Auto-load functions")
        self.auto_functions_check.setChecked(True)
        self.auto_functions_check.setToolTip(
            "Before each execution, (re)define functions whose code changed"
        )
        toolbar.addWidget(self.auto_functions_check)
        self.functions_label = QLabel("functions: 0")
        toolbar.addWidget(self.functions_label)
        # Ctrl+Return: selection if present, otherwise current line.
        self._action(self, "Run line/selection", self.run_line_or_selection, "Ctrl+Return")
        toolbar.addSeparator()
        self.interrupt_action = self._action(toolbar, "■ Interrupt", self.session.interrupt)
        self.interrupt_action.setEnabled(False)
        self._action(toolbar, "↻ Restart shell", self.session.restart)
        self._action(toolbar, "Clear", lambda: self.console.clear())

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        self.file_label = QLabel()
        self.cwd_label = QLabel("cwd: –")
        toolbar.addWidget(self.file_label)
        toolbar.addWidget(QLabel("   "))
        toolbar.addWidget(self.cwd_label)
        layout.addWidget(toolbar)

        splitter = QSplitter(Qt.Vertical, self)
        ctx_row = QHBoxLayout()
        self.ctx_enabled = QCheckBox("submitslurm context:")
        self.ctx_enabled.setChecked(True)
        ctx_row.addWidget(self.ctx_enabled)
        self.ctx_edits, self.ctx_force = {}, {}
        for key, label in (("-w", "-w"), ("-p", "-p"), ("-f", "-f")):
            ctx_row.addWidget(QLabel(label))
            edit = QLineEdit(self.ss_context.values.get(key, ""))
            edit.setMaximumWidth(110)
            edit.setPlaceholderText("(none)")
            self.ctx_edits[key] = edit
            ctx_row.addWidget(edit)
            force = QCheckBox("force")
            force.setToolTip("Override the option even if the call sets it")
            self.ctx_force[key] = force
            ctx_row.addWidget(force)
            edit.editingFinished.connect(self._ctx_changed)
            force.toggled.connect(self._ctx_changed)
        self.ctx_enabled.toggled.connect(self._ctx_changed)
        ctx_row.addStretch()
        layout.addLayout(ctx_row)

        self.editor = CodeEditor(self)
        self.editor.document().modificationChanged.connect(lambda _m: self._update_title())
        self._parse_timer = QTimer(self)
        self._parse_timer.setSingleShot(True)
        self._parse_timer.setInterval(300)
        self._parse_timer.timeout.connect(self._refresh_functions)
        self.editor.document().contentsChanged.connect(self._parse_timer.start)
        splitter.addWidget(self.editor)

        console_widget = QWidget()
        console_layout = QVBoxLayout(console_widget)
        console_layout.setContentsMargins(0, 0, 0, 0)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.console.setMaximumBlockCount(20000)
        console_layout.addWidget(self.console)

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("$"))
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Type a bash command and press Enter")
        self.command_input.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.command_input.returnPressed.connect(self._run_command_input)
        input_row.addWidget(self.command_input)
        console_layout.addLayout(input_row)
        splitter.addWidget(console_widget)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

    def _ctx_changed(self, *_):
        self.ss_context.enabled = self.ctx_enabled.isChecked()
        self.ss_context.values = {k: e.text() for k, e in self.ctx_edits.items()}
        self.ss_context.forced = {k for k, c in self.ctx_force.items() if c.isChecked()}
        self.apply_context()

    def _update_context_label(self):
        # keep the edit in sync when set_workspace() changed -w programmatically
        edit = self.ctx_edits.get("-w")
        if edit is not None and edit.text() != self.ss_context.values.get("-w", ""):
            edit.setText(self.ss_context.values.get("-w", ""))

    def _action(self, container, text, slot, shortcut=None) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
            # Do not clash with main window shortcuts (Ctrl+O, Ctrl+W, ...).
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        action.triggered.connect(lambda checked=False: slot())
        container.addAction(action)
        if container is not self:
            self.addAction(action)
        return action

    def _append(self, text: str, color: str | None = None):
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.End)
        char_format = QTextCharFormat()
        if color:
            char_format.setForeground(QColor(color))
        cursor.insertText(text, char_format)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    def _ensure_newline(self):
        text = self.console.toPlainText()
        if text and not text.endswith("\n"):
            self._append("\n")

    def _update_title(self):
        name = self._file_path.name if self._file_path else "(untitled)"
        if self.editor.document().isModified():
            name += " *"
        self.file_label.setText(name)
        self.file_label.setToolTip(str(self._file_path or ""))

    # -- project / file handling ---------------------------------------
    def _on_shell_started(self):
        self._loaded_functions.clear()
        # Order matters: wrapper and context first, then user functions.
        self.session.execute(WRAPPER + self.ss_context.to_bash(), silent=True)
        self.sync_functions()

    def apply_context(self):
        """Push the current context into the running shell (no echo)."""
        if self.session.is_running():
            self.session.execute(self.ss_context.to_bash(), silent=True)
        self._update_context_label()

    def set_workspace(self, name: str | None):
        """Called by the main window when a wd* directory is selected."""
        if name:
            self.ss_context.values["-w"] = name
        else:
            self.ss_context.values.pop("-w", None)
        self.apply_context()

    def set_project(self, project_path: Path | None):
        self._project_path = Path(project_path) if project_path else None
        self.console.clear()
        # Clear the old document first so the new shell does not
        # receive functions from the previous project.
        self._set_document("", None)
        if self._project_path is None:
            self.session.stop()
            self.cwd_label.setText("cwd: –")
            return
        self.session.start(self._project_path)
        info = self._project_path / "INFO"
        if info.is_file():
            self.open_file(info)

    def maybe_save(self) -> bool:
        """Return False if the user cancelled."""
        if not self.editor.document().isModified():
            return True
        answer = QMessageBox.question(
            self, "Unsaved changes",
            f"Save changes to {self._file_path or 'untitled script'}?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
        )
        if answer == QMessageBox.Cancel:
            return False
        if answer == QMessageBox.Save:
            return self.save()
        return True

    def open_dialog(self):
        if not self.maybe_save():
            return
        start = str(self._file_path.parent if self._file_path else self._project_path or "")
        path, _ = QFileDialog.getOpenFileName(self, "Open script", start)
        if path:
            self.open_file(Path(path))

    def open_file(self, path: Path) -> bool:
        try:
            text = Path(path).read_text(errors="replace")
        except OSError as exc:
            self.statusMessage.emit(f"Cannot open {path}: {exc}")
            return False
        self._set_document(text, Path(path))
        self._refresh_functions()
        if self.session.is_running():
            self.sync_functions()   # make INFO functions callable right away
        self.statusMessage.emit(f"Opened {path}")
        return True

    def reload(self):
        if self._file_path and self.maybe_save():
            self.open_file(self._file_path)

    def save(self) -> bool:
        if self._file_path is None:
            return self.save_as()
        try:
            self._file_path.write_text(self.editor.toPlainText())
        except OSError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return False
        self.editor.document().setModified(False)
        self.statusMessage.emit(f"Saved {self._file_path}")
        return True

    def save_as(self) -> bool:
        start = str(self._file_path or self._project_path or "")
        path, _ = QFileDialog.getSaveFileName(self, "Save script as", start)
        if not path:
            return False
        self._file_path = Path(path)
        return self.save()

    def _set_document(self, text: str, path: Path | None):
        self.editor.setPlainText(text)
        self.editor.document().setModified(False)
        self._file_path = path
        self._update_title()

    # -- execution -----------------------------------------------------

    def _execute(self, code: str, sync_functions: bool = True):
        if self._project_path is None:
            self.statusMessage.emit("No project selected – no shell available")
            return
        if sync_functions:
            self.sync_functions()   # queued before the command itself
        self._ensure_newline()
        echo = "".join(f"$ {line}\n" for line in code.rstrip("\n").splitlines())
        self._append(echo, "#0033b3")
        self.session.execute(code)

    def _advance_cursor_after(self, block):
        cursor = self.editor.textCursor()
        cursor.setPosition(block.position())
        if block.next().isValid():
            cursor.movePosition(QTextCursor.NextBlock)
        self.editor.setTextCursor(cursor)

    def run_line_or_selection(self):
        if self.editor.textCursor().hasSelection():
            self.run_selection()
        else:
            self.run_current_line()
    def run_current_line(self):
        """Run current line; inside a function body, redefine that function."""
        self._refresh_functions()
        block = self.editor.textCursor().block()
        line = block.blockNumber()

        if line < self._header_end:
            self.statusMessage.emit("INFO header is not executable")
            self._advance_cursor_after(block)
            return

        f = function_at(self._functions, line)
        if f is not None:
            self._define_functions([f], "redefining")
            self._advance_cursor_after(
                self.editor.document().findBlockByNumber(f.end)
            )
            return

        lines = [block.text()]
        while lines[-1].rstrip().endswith("\\") and block.next().isValid():
            block = block.next()
            lines.append(block.text())
        code = "\n".join(lines)
        if code.strip() and not SEPARATOR_RE.match(code):
            self._execute(code)
        self._advance_cursor_after(block)

    def run_selection(self):
        cursor = self.editor.textCursor()
        if not cursor.hasSelection():
            self.statusMessage.emit("Nothing selected")
            return
        self._refresh_functions()
        text = cursor.selectedText().replace("\u2029", "\n")
        first = self.editor.document().findBlock(cursor.selectionStart()).blockNumber()
        if first < self._header_end:   # drop header lines from the selection
            text = "\n".join(text.split("\n")[self._header_end - first:])
        self._execute(text)

    def run_file(self):
        """Run everything after the INFO header (including unsaved edits)."""
        self._refresh_functions()
        self._execute(executable_text(self.editor.toPlainText()),
                      sync_functions=False)
        # Running the file defines every function in it.
        for f in self._functions:
            self._loaded_functions[f.name] = f.code
        self._refresh_functions()

    def _run_command_input(self):
        command = self.command_input.text()
        self.command_input.clear()
        if command.strip():
            self._execute(command)

    def _on_command_finished(self, status: int):
        self._ensure_newline()
        color = "#8c8c8c" if status == 0 else "#c00000"
        self._append(f"[exit {status}]\n", color)

    def _on_busy_changed(self, busy: bool):
        self.interrupt_action.setEnabled(busy)
        self.statusMessage.emit("Running…" if busy else "Ready")

    def shutdown(self):
        self.session.stop()

    # -- functions -----------------------------------------------------

    def _refresh_functions(self):
        self._parse_timer.stop()
        text = self.editor.toPlainText()
        self._header_end = info_header_end(text.split("\n"))
        self._functions = parse_functions(text)
        self.editor._highlighter.set_function_names(f.name for f in self._functions)
        stale = len(self._stale_functions())
        label = f"functions: {len(self._functions)}"
        if stale:
            label += f" ({stale} not loaded)"
        self.functions_label.setText(label)

    def _stale_functions(self) -> list[ShellFunction]:
        return [f for f in self._functions
                if self._loaded_functions.get(f.name) != f.code]

    def _define_functions(self, functions: list[ShellFunction], reason: str):
        if not functions or self._project_path is None:
            return
        self._ensure_newline()
        self._append(
            f"[{reason}: {', '.join(f.name for f in functions)}]\n", "#8c8c8c"
        )
        self.session.execute("\n\n".join(f.code for f in functions))
        for f in functions:
            self._loaded_functions[f.name] = f.code
        self._refresh_functions()

    def sync_functions(self):
        """Define functions that are new or changed since the last load."""
        self._refresh_functions()
        if self.auto_functions_check.isChecked():
            self._define_functions(self._stale_functions(), "loading functions")

    def load_all_functions(self):
        self._refresh_functions()
        if not self._functions:
            self.statusMessage.emit("No functions found in the script")
            return
        self._define_functions(self._functions, "loading functions")

    def run_function_at_cursor(self):
        self._refresh_functions()
        line = self.editor.textCursor().blockNumber()
        f = function_at(self._functions, line)
        if f is None:
            self.statusMessage.emit("Cursor is not inside a function definition")
            return
        self._define_functions([f], "redefining")

    def _on_shell_started(self):
        # New shell: nothing is defined any more.
        self._loaded_functions.clear()
        self.sync_functions()
