from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path


@dataclass
class ApplicantInfo:
    username: str | None = None
    lastname: str | None = None
    institute: str | None = None
    email: str | None = None
    user_id: str | None = None


@dataclass
class BeamtimeInfo:
    beamtime_id: str
    beamline: str | None = None
    beamline_alias: str | None = None
    beamline_setup: str | None = None
    facility: str | None = None
    proposal_id: str | None = None
    proposal_type: str | None = None
    event_start: str | None = None
    event_end: str | None = None
    generated: str | None = None
    core_path: Path | None = None
    applicant: ApplicantInfo | None = None


@dataclass
class BeamtimeInspection:
    beamtime_root: Path
    raw_exists: bool
    processed_exists: bool
    scratch_cc_exists: bool
    shared_exists: bool
    raw_subdir_count: int
    raw_subdir_samples: list[str]


@dataclass
class ProjectInfo:
    location_id: int
    name: str
    path: Path
    beamtime_info: BeamtimeInfo | None = None
    inspection: BeamtimeInspection | None = None


class BeamtimeManager:
    """Parse and inspect PETRA beamtime directories."""

    def find_beamtime_root(self, project_path: Path) -> Path | None:
        """
        Detect paths like:
        /asap3/petra3/gpfs/p05/2025/data/11023208/scratch_cc/kct_P05SANDSILT

        and return:
        /asap3/petra3/gpfs/p05/2025/data/11023208
        """
        parts = project_path.resolve().parts

        for i in range(len(parts) - 6):
            # expect ... /gpfs/<beamline>/<year>/data/<beamtime_id>/...
            if (
                parts[i] == "gpfs"
                and i + 4 < len(parts)
                and parts[i + 3] == "data"
            ):
                beamtime_id = parts[i + 4]
                if beamtime_id.isdigit():
                    return Path(*parts[: i + 5])

        return None

    def load_beamtime_info(self, beamtime_root: Path) -> BeamtimeInfo | None:
        beamtime_id = beamtime_root.name
        metadata_file = beamtime_root / f"beamtime-metadata-{beamtime_id}.json"

        if not metadata_file.exists():
            return None

        try:
            data = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        applicant_raw = data.get("applicant", {}) or {}
        applicant = ApplicantInfo(
            username=applicant_raw.get("username"),
            lastname=applicant_raw.get("lastname"),
            institute=applicant_raw.get("institute"),
            email=applicant_raw.get("email"),
            user_id=applicant_raw.get("userId"),
        )

        return BeamtimeInfo(
            beamtime_id=str(data.get("beamtimeId", beamtime_id)),
            beamline=data.get("beamline"),
            beamline_alias=data.get("beamlineAlias"),
            beamline_setup=data.get("beamlineSetup"),
            facility=data.get("facility"),
            proposal_id=data.get("proposalId"),
            proposal_type=data.get("proposalType"),
            event_start=data.get("eventStart"),
            event_end=data.get("eventEnd"),
            generated=data.get("generated"),
            core_path=Path(data["corePath"]) if data.get("corePath") else beamtime_root,
            applicant=applicant,
        )

    def inspect_beamtime(self, beamtime_root: Path) -> BeamtimeInspection:
        raw_dir = beamtime_root / "raw"
        processed_dir = beamtime_root / "processed"
        scratch_cc_dir = beamtime_root / "scratch_cc"
        shared_dir = beamtime_root / "shared"

        raw_subdirs: list[str] = []
        if raw_dir.exists() and raw_dir.is_dir():
            try:
                raw_subdirs = sorted(
                    entry.name
                    for entry in raw_dir.iterdir()
                    if entry.is_dir()
                )
            except OSError:
                raw_subdirs = []

        return BeamtimeInspection(
            beamtime_root=beamtime_root,
            raw_exists=raw_dir.exists(),
            processed_exists=processed_dir.exists(),
            scratch_cc_exists=scratch_cc_dir.exists(),
            shared_exists=shared_dir.exists(),
            raw_subdir_count=len(raw_subdirs),
            raw_subdir_samples=raw_subdirs[:12],
        )


class ProjectManager:
    def __init__(self):
        self.beamtime_manager = BeamtimeManager()

    def build_project_info(self, location) -> ProjectInfo:
        path = location.path.resolve()
        beamtime_root = self.beamtime_manager.find_beamtime_root(path)

        beamtime_info = None
        inspection = None

        if beamtime_root is not None:
            beamtime_info = self.beamtime_manager.load_beamtime_info(beamtime_root)
            inspection = self.beamtime_manager.inspect_beamtime(beamtime_root)

        return ProjectInfo(
            location_id=location.id,
            name=location.name,
            path=path,
            beamtime_info=beamtime_info,
            inspection=inspection,
        )
