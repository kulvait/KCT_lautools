from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

from db import LaupyDB, ProjectCache


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
    raw_size_bytes: int | None = None
    processed_size_bytes: int | None = None
    scratch_cc_size_bytes: int | None = None


@dataclass
class ProjectInfo:
    location_id: int
    name: str
    path: Path
    description: str | None = None
    project_size_bytes: int | None = None
    last_inspected: str | None = None
    beamtime_info: BeamtimeInfo | None = None
    inspection: BeamtimeInspection | None = None


class BeamtimeManager:
    def find_beamtime_root(self, project_path: Path) -> Path | None:
        parts = project_path.resolve().parts

        for i in range(len(parts) - 6):
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
            beamline_setup=data.get("beamtimeSetup") or data.get("beamlineSetup"),
            facility=data.get("facility"),
            proposal_id=data.get("proposalId"),
            proposal_type=data.get("proposalType"),
            event_start=data.get("eventStart"),
            event_end=data.get("eventEnd"),
            generated=data.get("generated"),
            core_path=(
                Path(data["corePath"])
                if data.get("corePath")
                else beamtime_root
            ),
            applicant=applicant,
        )

    def inspect_beamtime_structure(self, beamtime_root: Path) -> BeamtimeInspection:
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

    def dir_size_bytes(
        self,
        path: Path,
        progress_callback=None,
    ) -> int | None:
        if not path.exists() or not path.is_dir():
            return None

        total = 0
        try:
            if progress_callback is not None:
                progress_callback(f"Counting size of {path}")

            for item in path.rglob("*"):
                try:
                    if item.is_file():
                        total += item.stat().st_size
                except OSError:
                    continue
        except OSError:
            return None

        return total


class ProjectManager:
    def __init__(self, db: LaupyDB):
        self.db = db
        self.beamtime_manager = BeamtimeManager()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_project_info(self, location) -> ProjectInfo:
        cache = self.db.get_project_cache(location.id)
        if cache is None:
            return self.refresh_project_metadata(location)

        return self._compose_project_info(location, cache)

    def refresh_project_metadata(
        self,
        location,
        progress_callback=None,
    ) -> ProjectInfo:
        path = location.path.resolve()

        if progress_callback is not None:
            progress_callback("Refreshing beamtime metadata")

        existing_cache = self.db.get_project_cache(location.id)
        cache = existing_cache or ProjectCache(location_id=location.id)

        beamtime_root = self.beamtime_manager.find_beamtime_root(path)
        cache.beamtime_root = beamtime_root

        beamtime_info = None
        inspection = None

        if beamtime_root is not None:
            beamtime_info = self.beamtime_manager.load_beamtime_info(
                beamtime_root
            )
            inspection = self.beamtime_manager.inspect_beamtime_structure(
                beamtime_root
            )

        self._store_beamtime_info_in_cache(cache, beamtime_info)
        self._store_inspection_structure_in_cache(cache, inspection)

        if cache.last_inspected is None:
            cache.last_inspected = datetime.now().isoformat(timespec="seconds")

        self.db.upsert_project_cache(location.id, cache)
        return self._compose_project_info(location, cache)

    def refresh_project_sizes(
        self,
        location,
        progress_callback=None,
    ) -> ProjectInfo:
        path = location.path.resolve()

        existing_cache = self.db.get_project_cache(location.id)
        if existing_cache is None:
            self.refresh_project_metadata(location)
            existing_cache = self.db.get_project_cache(location.id)

        cache = existing_cache or ProjectCache(location_id=location.id)

        if progress_callback is not None:
            progress_callback("Refreshing cached sizes")

        cache.project_size_bytes = self.beamtime_manager.dir_size_bytes(
            path,
            progress_callback=progress_callback,
        )

        beamtime_root = cache.beamtime_root
        if beamtime_root is not None:
            raw_dir = beamtime_root / "raw"
            processed_dir = beamtime_root / "processed"
            scratch_cc_dir = beamtime_root / "scratch_cc"

            cache.raw_size_bytes = self.beamtime_manager.dir_size_bytes(
                raw_dir,
                progress_callback=progress_callback,
            )
            cache.processed_size_bytes = self.beamtime_manager.dir_size_bytes(
                processed_dir,
                progress_callback=progress_callback,
            )
            cache.scratch_cc_size_bytes = self.beamtime_manager.dir_size_bytes(
                scratch_cc_dir,
                progress_callback=progress_callback,
            )

        cache.last_inspected = datetime.now().isoformat(timespec="seconds")
        self.db.upsert_project_cache(location.id, cache)

        return self._compose_project_info(location, cache)

    def refresh_project_cache(
        self,
        location,
        progress_callback=None,
    ) -> ProjectInfo:
        self.refresh_project_metadata(
            location,
            progress_callback=progress_callback,
        )
        return self.refresh_project_sizes(
            location,
            progress_callback=progress_callback,
        )

    def refresh_project_sizes_threadsafe(
        self,
        location,
        progress_callback=None,
    ) -> ProjectInfo:
        thread_db = LaupyDB(self.db.db_path)
        try:
            thread_location = thread_db.get_location(location.id)
            thread_manager = ProjectManager(thread_db)
            return thread_manager.refresh_project_sizes(
                thread_location,
                progress_callback=progress_callback,
            )
        finally:
            thread_db.connection.close()

    def refresh_project_cache_threadsafe(
        self,
        location,
        progress_callback=None,
    ) -> ProjectInfo:
        thread_db = LaupyDB(self.db.db_path)
        try:
            thread_location = thread_db.get_location(location.id)
            thread_manager = ProjectManager(thread_db)
            return thread_manager.refresh_project_cache(
                thread_location,
                progress_callback=progress_callback,
            )
        finally:
            thread_db.connection.close()

    def invalidate_project_cache(self, location_id: int) -> None:
        self.db.clear_project_cache(location_id)

    # ------------------------------------------------------------------
    # Composition helpers
    # ------------------------------------------------------------------

    def _compose_project_info(
        self,
        location,
        cache: ProjectCache | None,
    ) -> ProjectInfo:
        beamtime_info = None
        inspection = None
        project_size_bytes = None
        last_inspected = None

        if cache is not None:
            beamtime_info = self._beamtime_info_from_cache(cache)
            inspection = self._inspection_from_cache(cache)
            project_size_bytes = cache.project_size_bytes
            last_inspected = cache.last_inspected

        return ProjectInfo(
            location_id=location.id,
            name=location.name,
            path=location.path.resolve(),
            description=getattr(location, "description", None),
            project_size_bytes=project_size_bytes,
            last_inspected=last_inspected,
            beamtime_info=beamtime_info,
            inspection=inspection,
        )

    def _beamtime_info_from_cache(
        self,
        cache: ProjectCache,
    ) -> BeamtimeInfo | None:
        if cache.beamtime_id is None and cache.beamtime_root is None:
            return None

        applicant = None
        applicant_values = [
            cache.applicant_username,
            cache.applicant_lastname,
            cache.applicant_institute,
            cache.applicant_email,
            cache.applicant_user_id,
        ]
        if any(value is not None for value in applicant_values):
            applicant = ApplicantInfo(
                username=cache.applicant_username,
                lastname=cache.applicant_lastname,
                institute=cache.applicant_institute,
                email=cache.applicant_email,
                user_id=cache.applicant_user_id,
            )

        return BeamtimeInfo(
            beamtime_id=cache.beamtime_id or "",
            beamline=cache.beamline,
            beamline_alias=cache.beamline_alias,
            beamline_setup=cache.beamline_setup,
            facility=cache.facility,
            proposal_id=cache.proposal_id,
            proposal_type=cache.proposal_type,
            event_start=cache.event_start,
            event_end=cache.event_end,
            generated=cache.generated,
            core_path=cache.beamtime_root,
            applicant=applicant,
        )

    def _inspection_from_cache(
        self,
        cache: ProjectCache,
    ) -> BeamtimeInspection | None:
        if cache.beamtime_root is None:
            return None

        return BeamtimeInspection(
            beamtime_root=cache.beamtime_root,
            raw_exists=bool(cache.raw_exists) if cache.raw_exists is not None else False,
            processed_exists=bool(cache.processed_exists) if cache.processed_exists is not None else False,
            scratch_cc_exists=bool(cache.scratch_cc_exists) if cache.scratch_cc_exists is not None else False,
            shared_exists=bool(cache.shared_exists) if cache.shared_exists is not None else False,
            raw_subdir_count=cache.raw_subdir_count or 0,
            raw_subdir_samples=cache.raw_subdir_samples or [],
            raw_size_bytes=cache.raw_size_bytes,
            processed_size_bytes=cache.processed_size_bytes,
            scratch_cc_size_bytes=cache.scratch_cc_size_bytes,
        )

    def _store_beamtime_info_in_cache(
        self,
        cache: ProjectCache,
        beamtime_info: BeamtimeInfo | None,
    ) -> None:
        if beamtime_info is None:
            cache.beamtime_id = None
            cache.beamline = None
            cache.beamline_alias = None
            cache.beamline_setup = None
            cache.facility = None
            cache.proposal_id = None
            cache.proposal_type = None
            cache.event_start = None
            cache.event_end = None
            cache.generated = None
            cache.applicant_username = None
            cache.applicant_lastname = None
            cache.applicant_institute = None
            cache.applicant_email = None
            cache.applicant_user_id = None
            return

        cache.beamtime_id = beamtime_info.beamtime_id
        cache.beamline = beamtime_info.beamline
        cache.beamline_alias = beamtime_info.beamline_alias
        cache.beamline_setup = beamtime_info.beamline_setup
        cache.facility = beamtime_info.facility
        cache.proposal_id = beamtime_info.proposal_id
        cache.proposal_type = beamtime_info.proposal_type
        cache.event_start = beamtime_info.event_start
        cache.event_end = beamtime_info.event_end
        cache.generated = beamtime_info.generated

        applicant = beamtime_info.applicant
        if applicant is None:
            cache.applicant_username = None
            cache.applicant_lastname = None
            cache.applicant_institute = None
            cache.applicant_email = None
            cache.applicant_user_id = None
        else:
            cache.applicant_username = applicant.username
            cache.applicant_lastname = applicant.lastname
            cache.applicant_institute = applicant.institute
            cache.applicant_email = applicant.email
            cache.applicant_user_id = applicant.user_id

    def _store_inspection_structure_in_cache(
        self,
        cache: ProjectCache,
        inspection: BeamtimeInspection | None,
    ) -> None:
        if inspection is None:
            cache.raw_exists = None
            cache.processed_exists = None
            cache.scratch_cc_exists = None
            cache.shared_exists = None
            cache.raw_subdir_count = None
            cache.raw_subdir_samples = None
            return

        cache.raw_exists = inspection.raw_exists
        cache.processed_exists = inspection.processed_exists
        cache.scratch_cc_exists = inspection.scratch_cc_exists
        cache.shared_exists = inspection.shared_exists
        cache.raw_subdir_count = inspection.raw_subdir_count
        cache.raw_subdir_samples = inspection.raw_subdir_samples
