"""Working-directory operations independent of the browser GUI."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import subprocess
import sys

from lautools.db import Beamtime, LaupyDB, LaupyProject


class WDCreator:
    """Select the appropriate working-directory script for a DB object.

    A project requires a DB instance to resolve its beamtime associations.
    A Beamtime object can be used directly, without a DB instance.
    """

    _SCRIPT_MODULES = {
        "microtomography": (
            "lautools.scripts.createWorkingDirectoryForMicrotomography"
        ),
        "nanotomography": (
            "lautools.scripts.createWorkingDirectoryForNanotomography"
        ),
    }

    def __init__(
        self,
        target: LaupyProject | Beamtime,
        db: LaupyDB | None = None,
    ):
        self.target = target
        self.modality = self._resolve_modality(target, db)
        self._script_module = self._SCRIPT_MODULES[self.modality]

    @staticmethod
    def _beamtime_modality(beamtime: Beamtime) -> str:
        setup = (beamtime.beamline_setup or "").lower()
        matches = [
            modality
            for modality in WDCreator._SCRIPT_MODULES
            if modality in setup
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Cannot determine tomography modality for beamtime "
                f"{beamtime.beamtime_id}: "
                f"setup={beamtime.beamline_setup!r}"
            )

        return matches[0]

    @classmethod
    def _resolve_modality(
        cls,
        target: LaupyProject | Beamtime,
        db: LaupyDB | None,
    ) -> str:
        if isinstance(target, Beamtime):
            beamtimes = [target]
        elif isinstance(target, LaupyProject):
            if target.id is None:
                # An unsaved project cannot have database associations.
                beamtimes = []
            else:
                if db is None:
                    raise ValueError(
                        "A DB instance is required to resolve the "
                        "project's beamtime associations."
                    )

                try:
                    beamtimes = db.list_beamtimes_for_project(target.id)
                except Exception as exc:
                    raise RuntimeError(
                        f"Cannot load beamtimes for project {target.id}: {exc}"
                    ) from exc
        else:
            raise TypeError("target must be a LaupyProject or Beamtime")

        if not beamtimes:
            return "microtomography"

        modalities = {
            cls._beamtime_modality(beamtime)
            for beamtime in beamtimes
        }

        if len(modalities) != 1:
            raise ValueError(
                "The project is associated with both microtomography "
                "and nanotomography beamtimes."
            )

        return modalities.pop()

    @property
    def raw_directory(self) -> Path:
        """Default raw directory for the supplied project or beamtime."""
        if isinstance(self.target, LaupyProject):
            return Path(self.target.path) / "raw"

        if self.target.core_path is None:
            raise ValueError(
                f"Beamtime {self.target.beamtime_id} has no known path."
            )

        return Path(self.target.core_path) / "raw"

    def _command(self, *args: object) -> tuple[str, list[str]]:
        """Keep script invocation details private."""
        return sys.executable, [
            "-u",
            "-m",
            self._script_module,
            *[str(arg) for arg in args],
        ]

    def getSampleList(
        self,
        raw_dir: str | Path | None = None,
        *,
        timeout: float = 120,
    ) -> list[str]:
        """Return sample names using the selected script's --list mode.

        The script's stdout must contain only names and #-prefixed comments.
        Diagnostics should be written to stderr.
        """
        raw_path = (
            self.raw_directory
            if raw_dir is None
            else Path(raw_dir)
        )

        if not raw_path.is_dir():
            raise RuntimeError(f"Raw directory does not exist: {raw_path}")

        program, args = self._command("--list", "--", raw_path)

        try:
            result = subprocess.run(
                [program, *args],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"Cannot run sample listing: {exc}"
            ) from exc

        if result.returncode != 0:
            details = (
                result.stderr.strip()
                or result.stdout.strip()
                or f"exit code {result.returncode}"
            )
            raise RuntimeError(f"Sample listing failed: {details}")

        return [
            name
            for line in result.stdout.splitlines()
            if (name := line.strip())
            and not name.startswith("#")
        ]

    def getCreationCommand(
        self,
        working_dir: str | Path,
        samples: Sequence[str],
        *,
        raw_dir: str | Path | None = None,
    ) -> tuple[str, list[str]]:
        """Build the creation command for an asynchronous process runner.

        This does not start a process or create any directories.
        """
        if isinstance(samples, str) or not samples:
            raise ValueError("Provide a non-empty sequence of sample names.")

        if any(not name or name.startswith("-") for name in samples):
            raise ValueError(
                "Sample names must be non-empty and cannot start with '-'."
            )

        raw_path = (
            self.raw_directory
            if raw_dir is None
            else Path(raw_dir)
        )

        return self._command(
            "--samples",
            *samples,
            "--",
            raw_path,
            Path(working_dir),
        )
