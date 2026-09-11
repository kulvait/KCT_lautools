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
    QVBoxLayout,
    QWidget,
)


class ProjectConfigDialog(QDialog):
    def __init__(self, project_info, parent=None):
        super().__init__(parent)
        self.project_info = project_info

        self.setWindowTitle("Configure Project")
        self.resize(700, 500)

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.name_edit = QLineEdit(project_info.name)
        form.addRow("Project name:", self.name_edit)

        path_label = QLabel(str(project_info.path))
        path_label.setWordWrap(True)
        form.addRow("Project path:", path_label)

        top_widget = QWidget()
        top_widget.setLayout(form)
        layout.addWidget(top_widget)

        layout.addWidget(self._create_beamtime_box())
        layout.addWidget(self._create_inspection_box())

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def project_name(self) -> str:
        return self.name_edit.text().strip()

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

        inspection = self.project_info.inspection
        if inspection is None:
            form.addRow(QLabel("No beamtime directory inspection available."))
            return box

        form.addRow("Beamtime root:", QLabel(str(inspection.beamtime_root)))
        form.addRow("raw exists:", QLabel(str(inspection.raw_exists)))
        form.addRow("processed exists:", QLabel(str(inspection.processed_exists)))
        form.addRow("scratch_cc exists:", QLabel(str(inspection.scratch_cc_exists)))
        form.addRow("shared exists:", QLabel(str(inspection.shared_exists)))
        form.addRow("raw subdir count:", QLabel(str(inspection.raw_subdir_count)))

        sample_text = ", ".join(inspection.raw_subdir_samples)
        sample_widget = QPlainTextEdit(sample_text)
        sample_widget.setReadOnly(True)
        sample_widget.setMaximumHeight(80)
        form.addRow("raw sample dirs:", sample_widget)

        return box
