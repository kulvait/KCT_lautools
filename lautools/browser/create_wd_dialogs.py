import sys

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

SCRIPT_MODULE = "lautools.scripts.createWorkingDirectoryForMicrotomography"


def script_command(*args):
    """Program + argument list to invoke the script with the current interpreter."""
    return sys.executable, ["-m", SCRIPT_MODULE, *[str(a) for a in args]]


class SampleSelectionDialog(QDialog):
    def __init__(self, samples, target_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Select samples for {target_name}")
        self.resize(450, 550)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"{len(samples)} samples found. Select samples to include:"))

        self.list_widget = QListWidget()
        for name in samples:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        row = QHBoxLayout()
        all_btn = QPushButton("Select all")
        none_btn = QPushButton("Select none")
        all_btn.clicked.connect(lambda: self._set_all(Qt.Checked))
        none_btn.clicked.connect(lambda: self._set_all(Qt.Unchecked))
        row.addWidget(all_btn)
        row.addWidget(none_btn)
        row.addStretch()
        layout.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_all(self, state):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(state)

    def selected_samples(self):
        return [
            self.list_widget.item(i).text()
            for i in range(self.list_widget.count())
            if self.list_widget.item(i).checkState() == Qt.Checked
        ]


class ProcessLogDialog(QDialog):
    """Runs a QProcess asynchronously and streams its output."""

    def __init__(self, title, program, args, on_finished=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(800, 500)
        self._on_finished = on_finished

        layout = QVBoxLayout(self)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Monospace"))
        layout.addWidget(self.output)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.buttons.rejected.connect(self.close)
        self.buttons.button(QDialogButtonBox.Close).setEnabled(False)
        layout.addWidget(self.buttons)

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)

        self.output.appendPlainText("$ " + " ".join([program, *args]) + "\n")
        self.process.start(program, args)

    def _read_output(self):
        data = self.process.readAllStandardOutput().data().decode(errors="replace")
        self.output.insertPlainText(data)
        self.output.ensureCursorVisible()

    def _finished(self, exit_code, exit_status):
        ok = exit_status == QProcess.NormalExit and exit_code == 0
        self.output.appendPlainText(
            "\nFinished successfully." if ok else f"\nFailed (exit code {exit_code})."
        )
        self.buttons.button(QDialogButtonBox.Close).setEnabled(True)
        if self._on_finished:
            self._on_finished(ok)

    def _error(self, error):
        if error == QProcess.FailedToStart:
            self.output.appendPlainText(f"\nFailed to start: {self.process.errorString()}")
            self.buttons.button(QDialogButtonBox.Close).setEnabled(True)
            if self._on_finished:
                self._on_finished(False)

    def closeEvent(self, event):
        if self.process.state() != QProcess.NotRunning:
            event.ignore()  # don't close while running
        else:
            super().closeEvent(event)
