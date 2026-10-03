from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
import shutil

from lautools.browser.project_config_dialog import (
    _format_person,
    format_bytes,
    format_flag,
    format_time,
)


log = logging.getLogger(__name__)

RESERVED_BEAMTIME_NAMES = ("raw", "processed", "scratch_cc", "shared")
INFO_FILENAME = "INFO"


def _directory_name(value: str) -> str:
    name = value.strip()
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\0" in name
    ):
        raise ValueError("Enter a directory name without path separators.")
    return name


def _existing_beamtime_targets(beamtime) -> dict[str, Path]:
    core_path = getattr(beamtime, "core_path", None)
    if core_path is None:
        raise ValueError("Beamtime has no core_path.")
    core_path = Path(core_path).expanduser().resolve(strict=True)
    if not core_path.is_dir():
        raise ValueError(f"Beamtime core path is not a directory: {core_path}")

    targets: dict[str, Path] = {}
    for name in RESERVED_BEAMTIME_NAMES:
        candidate = core_path / name
        if candidate.is_dir():
            targets[name] = candidate
    return targets


def _validate_project_dir(beamtime, project_dir: Path) -> Path:
    core_path = getattr(beamtime, "core_path", None)
    if core_path is None:
        raise ValueError("Beamtime has no core_path.")
    core_path = Path(core_path).expanduser().resolve(strict=True)
    scratch = (core_path / "scratch_cc").resolve(strict=True)

    project_dir = Path(os.path.abspath(Path(project_dir).expanduser()))
    if project_dir.is_symlink():
        raise ValueError("The project directory must not be a symlink.")
    project_dir = project_dir.resolve()

    if project_dir == scratch or not project_dir.is_relative_to(scratch):
        raise ValueError(
            "The project directory must be strictly inside beamtime scratch_cc."
        )
    if not project_dir.parent.is_dir():
        raise ValueError(
            "The project's parent directory must already exist."
        )

    if os.path.lexists(project_dir):
        if not project_dir.is_dir() or any(project_dir.iterdir()):
            raise ValueError(
                "Select a new directory or an existing empty directory."
            )

    return project_dir


def validate_recipe_copy(
    beamtime,
    recipes_root: Path,
    recipe: Path,
    workbench: Path,
    project_dir: Path,
    copy_name: str,
) -> tuple[Path, Path, Path]:
    recipes_root = Path(recipes_root).resolve(strict=True)
    workbench = Path(workbench).resolve(strict=True)
    project_dir = _validate_project_dir(beamtime, project_dir)

    if not recipes_root.is_dir():
        raise ValueError(f"Not a directory: {recipes_root}")
    if not workbench.is_dir():
        raise ValueError(f"Not a directory: {workbench}")

    core_path = Path(beamtime.core_path).expanduser().resolve(strict=True)
    scratch = (core_path / "scratch_cc").resolve(strict=True)

    recipe = Path(recipe)
    if recipe.is_symlink():
        raise ValueError("Recipe folders must not themselves be symlinks.")
    recipe = recipe.resolve(strict=True)

    if recipe.parent != recipes_root or not recipe.is_dir():
        raise ValueError("Select an immediate recipe subfolder.")

    if workbench == scratch or workbench.is_relative_to(scratch):
        raise ValueError(
            "The workbench must be outside this beamtime's scratch area."
        )

    destination = workbench / _directory_name(copy_name)
    if os.path.lexists(destination):
        raise ValueError(
            f"The workbench destination already exists:\n{destination}"
        )

    for left, right in (
        (recipe, destination),
        (recipe, project_dir),
        (destination, project_dir),
    ):
        if (
            left == right
            or left.is_relative_to(right)
            or right.is_relative_to(left)
        ):
            raise ValueError(
                "Recipe, workbench copy and project must not overlap."
            )

    return recipe, destination, project_dir


def _validate_recipe_instance_directory(recipe_instance_directory: Path) -> Path:
    recipe_instance_directory = Path(recipe_instance_directory).resolve(strict=True)
    if not recipe_instance_directory.is_dir():
        raise ValueError(f"Not a content directory: {recipe_instance_directory}")

    reserved = sorted(
        entry.name
        for entry in recipe_instance_directory.iterdir()
        if entry.name in RESERVED_BEAMTIME_NAMES
    )
    if reserved:
        raise ValueError(
            "Recipe/workbench content must not contain reserved top-level names: "
            + ", ".join(reserved)
        )

    return recipe_instance_directory


def _beamtime_summary_text(
    beamtime,
    project_dir: Path,
    recipe_instance_directory: Path | None,
) -> str:
    core_path = Path(beamtime.core_path).expanduser().resolve(strict=True)
    created_at = datetime.now().replace(microsecond=0).isoformat(sep=" ")

    beamline = getattr(beamtime, "beamline", None) or "UNKNOWN"
    proposal_id = getattr(beamtime, "proposal_id", None) or beamtime.beamtime_id
    title = getattr(beamtime, "title", None) or ""
    description = getattr(beamtime, "description", None) or ""
    beamtime_pi = _format_person(beamtime, "pi")
    beamtime_leader = _format_person(beamtime, "pi")
    beamtime_applicant = _format_person(beamtime, "pi")

    raw_dir = core_path / "raw"
    if raw_dir.is_dir():
        try:
            sample_count = sum(
                1 for entry in raw_dir.iterdir() if entry.is_dir()
            )
        except OSError:
            sample_count = None
    else:
        sample_count = None

    lines = [
        f"laupy project INFO file created by lautools on: {created_at}",
        f"Project dir: {project_dir}"]
    lines.append(f"Beamline: {beamline}")
    if beamtime.beamline_setup is not None:
        lines.append(f"Beamline setup: {beamtime.beamline_setup}")
    if beamtime.label is not None:
        lines.append(f"Beamtime label: {beamtime.label}")
    if title:
        lines.append(f"Title: {title}")
    lines.append(f"Beamtime root dir: {core_path}")
    lines.append(f"Beamtime ID: {beamtime.beamtime_id}")
    lines.append(f"Proposal ID: {proposal_id}")
    if beamtime_pi != "-":
         lines.append(f"Beamtime PI: {beamtime_pi}")
    if beamtime_leader != "-" and beamtime_leader != beamtime_pi:
         lines.append(f"Beamtime leader: {beamtime_leader}")
    if beamtime_applicant != "-" and beamtime_applicant not in {beamtime_pi, beamtime_leader}:
         lines.append(f"Beamtime applicant: {beamtime_applicant}")
    if description:
        lines.extend([
            "Beamtime description:",
            description,
        ])
    if recipe_instance_directory is not None:
        lines.append(f"Workbench content dir: {recipe_instance_directory}")
    if sample_count is not None:
        lines.append(f"Number of samples: {sample_count}")

    return "\n".join(lines) + "\n"


def _write_info_file(
    beamtime,
    project_dir: Path,
    recipe_instance_directory: Path | None,
) -> None:
    project_summary = _beamtime_summary_text(
        beamtime=beamtime,
        project_dir=project_dir,
        recipe_instance_directory=recipe_instance_directory,
    )
    if recipe_instance_directory is None:
        (project_dir / INFO_FILENAME).write_text(project_summary, encoding="utf-8")
        return
    info_path = recipe_instance_directory / INFO_FILENAME
    recipe_info_content = ""
    info_parts_separator = [
        "",
        "-------------------",
        "Original INFO content follows:",
        "-------------------",
    ]
    if info_path.exists():
        if not info_path.is_file():
            raise ValueError(f"INFO exists but is not a regular file: {info_path}")
        recipe_info_content = info_path.read_text(encoding="utf-8")
        recipe_info_content = "\n".join(info_parts_separator) + "\n" + recipe_info_content

    info_path.write_text(project_summary + recipe_info_content, encoding="utf-8")


def create_project(
    beamtime,
    project_dir: Path,
    recipe_instance_directory: Path | None = None,
) -> Path:
    """
    Create a project directory with beamtime links and optional workbench links.

    - Always creates links to existing beamtime top-level dirs:
      raw, processed, scratch_cc, shared
    - Optionally links top-level entries from recipe_instance_directory
    - Creates or updates INFO in recipe_instance_directory, or creates INFO in project_dir
      when no recipe_instance_directory is provided
    """
    project_dir = _validate_project_dir(beamtime, project_dir)
    content = None
    if recipe_instance_directory is not None:
        content = _validate_recipe_instance_directory(recipe_instance_directory)

    beamtime_targets = _existing_beamtime_targets(beamtime)

    planned_links: list[tuple[str, Path, bool]] = []
    for name, target in sorted(beamtime_targets.items()):
        planned_links.append((name, target, True))

    if content is not None:
        for entry in sorted(content.iterdir(), key=lambda p: p.name):
            if (
                entry.name in RESERVED_BEAMTIME_NAMES
                or entry.name == INFO_FILENAME
            ):
                continue
            planned_links.append((entry.name, entry, entry.is_dir()))

    created_project = False
    created_links: list[tuple[Path, str]] = []

    try:
        if not os.path.lexists(project_dir):
            project_dir.mkdir()
            created_project = True
        elif (
            project_dir.is_symlink()
            or not project_dir.is_dir()
            or any(project_dir.iterdir())
        ):
            raise ValueError("The project directory is no longer empty.")

        _write_info_file(
            beamtime=beamtime,
            project_dir=project_dir,
            recipe_instance_directory=content,
        )

        if content is None:
            info_target = project_dir / INFO_FILENAME
            created_links.append((info_target, str(info_target)))

        for name, target, is_directory in planned_links:
            link = project_dir / name
            target_text = str(target)
            os.symlink(
                target_text,
                link,
                target_is_directory=is_directory,
            )
            created_links.append((link, target_text))

        if content is not None and (content / INFO_FILENAME).exists():
            info_link = project_dir / INFO_FILENAME
            target_text = str(content / INFO_FILENAME)
            os.symlink(target_text, info_link, target_is_directory=False)
            created_links.append((info_link, target_text))

        return project_dir

    except Exception as exc:
        cleanup_errors = []

        for link, target_text in reversed(created_links):
            try:
                if link.is_symlink() and os.readlink(link) == target_text:
                    link.unlink()
            except OSError as cleanup_exc:
                cleanup_errors.append(str(cleanup_exc))

        if content is None:
            info_file = project_dir / INFO_FILENAME
            try:
                if info_file.exists() and info_file.is_file():
                    info_file.unlink()
            except OSError as cleanup_exc:
                cleanup_errors.append(str(cleanup_exc))

        if created_project:
            try:
                project_dir.rmdir()
            except OSError as cleanup_exc:
                cleanup_errors.append(str(cleanup_exc))

        details = str(exc)
        if cleanup_errors:
            details += "\n\nCleanup warnings:\n" + "\n".join(cleanup_errors)
        raise RuntimeError(details) from exc


def create_project_from_recipe(
    beamtime,
    recipes_root: Path,
    recipe: Path,
    workbench: Path,
    project_dir: Path,
    copy_name: str,
) -> tuple[Path, Path]:
    """
    Copy a recipe into the workbench and create a project linked to that copy.
    """
    recipe, destination, project_dir = validate_recipe_copy(
        beamtime=beamtime,
        recipes_root=recipes_root,
        recipe=recipe,
        workbench=workbench,
        project_dir=project_dir,
        copy_name=copy_name,
    )

    reserved_destination = False
    try:
        destination.mkdir()
        reserved_destination = True
        shutil.copytree(
            recipe,
            destination,
            dirs_exist_ok=True,
            symlinks=True,
        )
        project_dir = create_project(
            beamtime=beamtime,
            project_dir=project_dir,
            recipe_instance_directory=destination,
        )
        return project_dir, destination

    except Exception as exc:
        details = str(exc)
        if reserved_destination:
            details += (
                "\n\nThe workbench copy was retained, possibly incomplete:"
                f"\n{destination}\n"
                "Inspect it before removing it or retrying with another name."
            )
        raise RuntimeError(details) from exc
