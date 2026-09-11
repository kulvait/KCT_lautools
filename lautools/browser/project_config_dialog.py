from __future__ import annotations

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
        return "N/A"

    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]

    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0

    return f"{num_bytes} B"


class ProjectConfigDialog(QDialog):
    def __init__(
        self,
        project_info,
        refresh_callback=None,
        parent=None,
    ):
        super().__init__(parent)
        self.project_info = project_info
        self.refresh_callback = refresh_callback

        self.setWindowTitle("Configure Project")
        self.resize(760, 620)

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.name_edit = QLineEdit(project_info.name)
        form.addRow("Project name:", self.name_edit)

        path_label = QLabel(str(project_info.path))
        path_label.setWordWrap(True)
        form.addRow("Project path:", path_label)

        form.addRow(QLabel("Project Size:"), QLabel(format_bytes(project_info.project_size_bytes)))

        self.description_edit = QPlainTextEdit(
            project_info.description or ""
        )
        self.description_edit.setMaximumHeight(100)
        form.addRow("Description:", self.description_edit)

        top_widget = QWidget()
        top_widget.setLayout(form)
        layout.addWidget(top_widget)

        layout.addWidget(self._create_beamtime_box())
        layout.addWidget(self._create_inspection_box())

        button_row = QHBoxLayout()
        self.refresh_sizes_button = QPushButton("Refresh sizes")
        self.refresh_sizes_button.clicked.connect(self.refresh_sizes)
        self.refresh_status_label = QLabel("")
        button_row.addWidget(self.refresh_sizes_button)
        button_row.addWidget(self.refresh_status_label, 1)
        layout.addLayout(button_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_inspection_fields()

    def project_name(self) -> str:
        return self.name_edit.text().strip()

    def project_description(self) -> str:
        return self.description_edit.toPlainText().strip()

    def _create_beamtime_box(self):
        box = QGroupBox("Beamtime metadata")
        form = QFormLayout(box)

        beamtime = self.project_info.beamtime_info
        if beamtime is None:
            form.addRow(QLabel("No beamtime metadata detected."))
            return box

        form.addRow("Beamtime ID:", QLabel(beamtime.beamtime_id or ""))
        form.addRow("Beamline:", QLabel(beamtime.beamline or ""))
        form.addRow("Beamline alias:", QLabel(beamtime.beamline_alias or ""))
        form.addRow("Setup:", QLabel(beamtime.beamline_setup or ""))
        form.addRow("Facility:", QLabel(beamtime.facility or ""))
        form.addRow("Proposal ID:", QLabel(beamtime.proposal_id or ""))
        form.addRow("Proposal type:", QLabel(beamtime.proposal_type or ""))
        form.addRow("Event start:", QLabel(beamtime.event_start or ""))
        form.addRow("Event end:", QLabel(beamtime.event_end or ""))
        form.addRow("Generated:", QLabel(beamtime.generated or ""))
        form.addRow(
            "Core path:",
            QLabel(str(beamtime.core_path) if beamtime.core_path else ""),
        )

        applicant = beamtime.applicant
        if applicant is not None:
            applicant_text = "\n".join(
                [
                    f"username: {applicant.username or ''}",
                    f"lastname: {applicant.lastname or ''}",
                    f"institute: {applicant.institute or ''}",
                    f"email: {applicant.email or ''}",
                    f"userId: {applicant.user_id or ''}",
                ]
            )
            applicant_widget = QPlainTextEdit(applicant_text)
            applicant_widget.setReadOnly(True)
            applicant_widget.setMaximumHeight(100)
            form.addRow("Applicant:", applicant_widget)

        return box

    def _create_inspection_box(self):
        box = QGroupBox("Beamtime inspection")
        form = QFormLayout(box)

        self.beamtime_root_label = QLabel("")
        self.raw_exists_label = QLabel("")
        self.processed_exists_label = QLabel("")
        self.scratch_cc_exists_label = QLabel("")
        self.shared_exists_label = QLabel("")
        self.raw_subdir_count_label = QLabel("")
        self.raw_size_label = QLabel("")
        self.processed_size_label = QLabel("")
        self.scratch_cc_size_label = QLabel("")
        self.project_size_label = QLabel("")
        self.raw_sample_widget = QPlainTextEdit()
        self.raw_sample_widget.setReadOnly(True)
        self.raw_sample_widget.setMaximumHeight(80)

        form.addRow("Beamtime root:", self.beamtime_root_label)
        form.addRow("raw exists:", self.raw_exists_label)
        form.addRow("processed exists:", self.processed_exists_label)
        form.addRow("scratch_cc exists:", self.scratch_cc_exists_label)
        form.addRow("shared exists:", self.shared_exists_label)
        form.addRow("raw subdir count:", self.raw_subdir_count_label)
        form.addRow("raw size:", self.raw_size_label)
        form.addRow("processed size:", self.processed_size_label)
        form.addRow("scratch_cc size:", self.scratch_cc_size_label)
        form.addRow("project size:", self.project_size_label)
        form.addRow("raw sample dirs:", self.raw_sample_widget)

        return box

    def _update_inspection_fields(self):
        inspection = self.project_info.inspection

        if inspection is None:
            self.beamtime_root_label.setText("No beamtime directory inspection available.")
            self.raw_exists_label.setText("")
            self.processed_exists_label.setText("")
            self.scratch_cc_exists_label.setText("")
            self.shared_exists_label.setText("")
            self.raw_subdir_count_label.setText("")
            self.raw_size_label.setText("")
            self.processed_size_label.setText("")
            self.scratch_cc_size_label.setText("")
            self.project_size_label.setText("")
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
        self.project_size_label.setText(
            format_bytes(inspection.project_size_bytes)
        )
        self.raw_sample_widget.setPlainText(
            ", ".join(inspection.raw_subdir_samples)
        )

    def refresh_sizes(self):
        if self.refresh_callback is None:
            self.refresh_status_label.setText("No refresh callback configured")
            return

        self.refresh_status_label.setText("Refreshing sizes...")
        self.repaint()

        def progress(message: str):
            self.refresh_status_label.setText(message)
            self.repaint()

        updated_project_info = self.refresh_callback(progress)
        if updated_project_info is not None:
            self.project_info = updated_project_info

        self._update_inspection_fields()
        self.refresh_status_label.setText("Size refresh finished")
