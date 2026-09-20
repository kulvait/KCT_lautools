from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "Not cached"

    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]

    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0

    return f"{num_bytes} B"


class RefreshSizesWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, project_manager, location):
        super().__init__()
        self.project_manager = project_manager
        self.location = location

    def run(self):
        try:
            updated_project_info = (
                self.project_manager.refresh_project_sizes_threadsafe(
                    self.location,
                    progress_callback=self.progress.emit,
                )
            )
            self.finished.emit(updated_project_info)
        except Exception as exc:
            self.failed.emit(str(exc))


class ProjectConfigDialog(QDialog):
    def __init__(
        self,
        project_manager,
        location,
        project_info,
        parent=None,
    ):
        super().__init__(parent)
        self.project_manager = project_manager
        self.location = location
        self.project_info = project_info

        self.refresh_thread = None
        self.refresh_worker = None

        self.setWindowTitle("Configure Project")
        self.resize(780, 900)

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.name_edit = QLineEdit(project_info.name)
        form.addRow("Project name:", self.name_edit)

        path_label = QLabel(str(project_info.path))
        path_label.setWordWrap(True)
        form.addRow("Project path:", path_label)

        self.project_size_top_label = QLabel(
            format_bytes(project_info.project_size_bytes)
        )
        form.addRow("Project size:", self.project_size_top_label)

        self.last_inspected_label = QLabel(
            project_info.last_inspected or "Never"
        )
        form.addRow("Cache updated:", self.last_inspected_label)

        self.description_edit = QPlainTextEdit(
            project_info.description or ""
        )
        self.description_edit.setMaximumHeight(120)
        form.addRow("Description:", self.description_edit)

        top_widget = QWidget()
        top_widget.setLayout(form)
        layout.addWidget(top_widget)

        layout.addWidget(self._create_beamtime_box())
        layout.addWidget(self._create_inspection_box())

        button_row = QHBoxLayout()
        self.refresh_sizes_button = QPushButton("Refresh sizes")
        self.refresh_sizes_button.clicked.connect(
            self.refresh_sizes_in_background
        )
        self.refresh_status_label = QLabel("")
        self.refresh_status_label.setWordWrap(True)

        button_row.addWidget(self.refresh_sizes_button)
        button_row.addWidget(self.refresh_status_label, 1)
        layout.addLayout(button_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_fields()

    def project_name(self) -> str:
        return self.name_edit.text().strip()

    def project_description(self) -> str:
        return self.description_edit.toPlainText().strip()

    def _create_beamtime_box(self):
        box = QGroupBox("Beamtime metadata")
        form = QFormLayout(box)

        self.beamtime_id_label = QLabel("")
        self.beamtime_id_label.setWordWrap(True)

        self.beamline_label = QLabel("")
        self.beamline_alias_label = QLabel("")
        self.setup_label = QLabel("")
        self.facility_label = QLabel("")
        self.proposal_id_label = QLabel("")
        self.proposal_type_label = QLabel("")
        self.event_start_label = QLabel("")
        self.event_end_label = QLabel("")
        self.generated_label = QLabel("")

        self.core_path_label = QLabel("")
        self.core_path_label.setWordWrap(True)

        self.applicant_widget = QPlainTextEdit()
        self.applicant_widget.setReadOnly(True)
        self.applicant_widget.setMaximumHeight(150)

        form.addRow("Beamtime ID:", self.beamtime_id_label)
        form.addRow("Beamline:", self.beamline_label)
        form.addRow("Beamline alias:", self.beamline_alias_label)
        form.addRow("Setup:", self.setup_label)
        form.addRow("Facility:", self.facility_label)
        form.addRow("Proposal ID:", self.proposal_id_label)
        form.addRow("Proposal type:", self.proposal_type_label)
        form.addRow("Event start:", self.event_start_label)
        form.addRow("Event end:", self.event_end_label)
        form.addRow("Generated:", self.generated_label)
        form.addRow("Core path:", self.core_path_label)
        form.addRow("Applicant:", self.applicant_widget)

        return box

    def _create_inspection_box(self):
        box = QGroupBox("Beamtime inspection")
        form = QFormLayout(box)

        self.beamtime_root_label = QLabel("")
        self.beamtime_root_label.setWordWrap(True)

        self.raw_exists_label = QLabel("")
        self.processed_exists_label = QLabel("")
        self.scratch_cc_exists_label = QLabel("")
        self.shared_exists_label = QLabel("")
        self.raw_subdir_count_label = QLabel("")
        self.raw_size_label = QLabel("")
        self.processed_size_label = QLabel("")
        self.scratch_cc_size_label = QLabel("")

        self.raw_sample_widget = QPlainTextEdit()
        self.raw_sample_widget.setReadOnly(True)
        self.raw_sample_widget.setMaximumHeight(90)

        form.addRow("Beamtime root:", self.beamtime_root_label)
        form.addRow("raw exists:", self.raw_exists_label)
        form.addRow("processed exists:", self.processed_exists_label)
        form.addRow("scratch_cc exists:", self.scratch_cc_exists_label)
        form.addRow("shared exists:", self.shared_exists_label)
        form.addRow("Samples in raw:", self.raw_subdir_count_label)
        form.addRow("raw size:", self.raw_size_label)
        form.addRow("processed size:", self.processed_size_label)
        form.addRow("scratch_cc size:", self.scratch_cc_size_label)
        form.addRow("raw sample dirs:", self.raw_sample_widget)

        return box

    def _update_fields(self):
        self.project_size_top_label.setText(
            format_bytes(self.project_info.project_size_bytes)
        )
        self.last_inspected_label.setText(
            self.project_info.last_inspected or "Never"
        )

        self._update_beamtime_fields()
        self._update_inspection_fields()

    def _update_beamtime_fields(self):
        beamtime = self.project_info.beamtime_info
        if beamtime is None:
            self.beamtime_id_label.setText("No beamtime metadata detected")
            self.beamline_label.setText("")
            self.beamline_alias_label.setText("")
            self.setup_label.setText("")
            self.facility_label.setText("")
            self.proposal_id_label.setText("")
            self.proposal_type_label.setText("")
            self.event_start_label.setText("")
            self.event_end_label.setText("")
            self.generated_label.setText("")
            self.core_path_label.setText("")
            self.applicant_widget.setPlainText("")
            return

        self.beamtime_id_label.setText(beamtime.beamtime_id or "")
        self.beamline_label.setText(beamtime.beamline or "")
        self.beamline_alias_label.setText(beamtime.beamline_alias or "")
        self.setup_label.setText(beamtime.beamline_setup or "")
        self.facility_label.setText(beamtime.facility or "")
        self.proposal_id_label.setText(beamtime.proposal_id or "")
        self.proposal_type_label.setText(beamtime.proposal_type or "")
        self.event_start_label.setText(beamtime.event_start or "")
        self.event_end_label.setText(beamtime.event_end or "")
        self.generated_label.setText(beamtime.generated or "")
        self.core_path_label.setText(
            str(beamtime.core_path) if beamtime.core_path else ""
        )

        applicant = beamtime.applicant
        if applicant is None:
            self.applicant_widget.setPlainText("")
            return

        applicant_text = "\n".join(
            [
                f"username: {applicant.username or ''}",
                f"lastname: {applicant.lastname or ''}",
                f"institute: {applicant.institute or ''}",
                f"email: {applicant.email or ''}",
                f"userId: {applicant.user_id or ''}",
            ]
        )
        self.applicant_widget.setPlainText(applicant_text)

    def _update_inspection_fields(self):
        inspection = self.project_info.inspection
        if inspection is None:
            self.beamtime_root_label.setText(
                "No beamtime directory inspection available"
            )
            self.raw_exists_label.setText("")
            self.processed_exists_label.setText("")
            self.scratch_cc_exists_label.setText("")
            self.shared_exists_label.setText("")
            self.raw_subdir_count_label.setText("")
            self.raw_size_label.setText("Not cached")
            self.processed_size_label.setText("Not cached")
            self.scratch_cc_size_label.setText("Not cached")
            self.raw_sample_widget.setPlainText("")
            return

        self.beamtime_root_label.setText(str(inspection.beamtime_root))
        self.raw_exists_label.setText(str(inspection.raw_exists))
        self.processed_exists_label.setText(str(inspection.processed_exists))
        self.scratch_cc_exists_label.setText(str(inspection.scratch_cc_exists))
        self.shared_exists_label.setText(str(inspection.shared_exists))
        self.raw_subdir_count_label.setText(str(inspection.raw_subdir_count))
        self.raw_size_label.setText(format_bytes(inspection.raw_size_bytes))
        self.processed_size_label.setText(
            format_bytes(inspection.processed_size_bytes)
        )
        self.scratch_cc_size_label.setText(
            format_bytes(inspection.scratch_cc_size_bytes)
        )
        self.raw_sample_widget.setPlainText(
            ", ".join(inspection.raw_subdir_samples)
        )

    def refresh_sizes_in_background(self):
        if self.refresh_thread is not None and self.refresh_thread.isRunning():
            return

        self.refresh_sizes_button.setEnabled(False)
        self.refresh_status_label.setText("Refreshing cached sizes...")

        self.refresh_thread = QThread(self)
        self.refresh_worker = RefreshSizesWorker(
            self.project_manager,
            self.location,
        )
        self.refresh_worker.moveToThread(self.refresh_thread)

        self.refresh_thread.started.connect(self.refresh_worker.run)
        self.refresh_worker.progress.connect(
            self.refresh_status_label.setText
        )
        self.refresh_worker.finished.connect(self._on_refresh_finished)
        self.refresh_worker.failed.connect(self._on_refresh_failed)

        self.refresh_worker.finished.connect(self.refresh_thread.quit)
        self.refresh_worker.failed.connect(self.refresh_thread.quit)
        self.refresh_thread.finished.connect(self.refresh_thread.deleteLater)
        self.refresh_thread.finished.connect(self._cleanup_worker)

        self.refresh_thread.start()

    def _on_refresh_finished(self, updated_project_info):
        if updated_project_info is not None:
            self.project_info = updated_project_info

        self._update_fields()
        self.refresh_status_label.setText("Size cache refreshed")
        self.refresh_sizes_button.setEnabled(True)

    def _on_refresh_failed(self, message: str):
        self.refresh_status_label.setText(f"Refresh failed: {message}")
        self.refresh_sizes_button.setEnabled(True)

    def _cleanup_worker(self):
        self.refresh_thread = None
        self.refresh_worker = None
