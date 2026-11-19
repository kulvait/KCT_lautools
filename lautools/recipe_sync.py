"""
Keep a project's recipe copy and its upstream cookbook recipe in sync.

A project created from a recipe links (via symlinks) to its own copy of the
recipe in the workbench; that copy was cloned from a recipe in the cookbook:

    <project>/sbatch -> <workbench>/<copy>/sbatch      project recipe
    <cookbook>/<recipe>/sbatch                          upstream recipe

This module resolves that pair, compares the files, copies single files in
either direction (e.g. backporting an improved sbatch script to the cookbook)
and checks whether the cookbook git repository is behind its remote.

@author: Vojtech Kulvait
@license: GNU GPL v3.0
"""
from __future__ import annotations

import difflib
import filecmp
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from pathlib import Path

from lautools.db import LaupyProjectToRecipeInstance, LaupyRecipeInstance

# Entries of a project directory that are links into the project recipe copy
RECIPE_LINK_NAMES = ("sbatch", "INFO", "script", "bin")
# Never compared or copied
IGNORED_NAMES = {".git", "__pycache__", ".ipynb_checkpoints"}
IGNORED_SUFFIXES = (".swp", ".swo", ".pyc", "~")
INFO_FILENAME = "INFO"
# project_creator writes this line between its header and the recipe's INFO
INFO_ORIGINAL_MARKER = "Original INFO content follows:"


# ----------------------------------------------------------------------
# Project recipe and its upstream
# ----------------------------------------------------------------------

@dataclass
class ProjectRecipe:
    """The recipe copy a project uses and the cookbook recipe it came from."""
    project_recipe_dir: Path | None
    upstream_recipe_dir: Path | None
    linked_in_database: bool        # False: inferred from the project's symlinks and recipe names
    project_recipe_instance: LaupyRecipeInstance | None = None
    upstream_recipe_instance: LaupyRecipeInstance | None = None
    problems: list[str] = field(default_factory=list)


def _recipe_dir_from_links(project_dir: Path) -> Path | None:
    """Directory the project's recipe symlinks (sbatch, INFO, ...) point into."""
    for name in RECIPE_LINK_NAMES:
        link = project_dir / name
        if link.is_symlink():
            target = link.resolve()
            if target.exists():
                return target.parent
    return None


def _match_upstream_by_name(project_recipe_dir: Path, cookbook_root: Path) -> Path | None:
    """Cookbook recipe whose name ends the copy name (copies are named <project>_<recipe>)."""
    if not cookbook_root.is_dir():
        return None
    name = project_recipe_dir.name
    candidates = [entry for entry in cookbook_root.iterdir()
                  if entry.is_dir() and not entry.name.startswith(".")
                  and (name == entry.name or name.endswith("_" + entry.name))]
    return max(candidates, key=lambda entry: len(entry.name), default=None)


def _collection_root(db, location_id: int | None) -> Path | None:
    location = db.get_location(location_id) if location_id is not None else None
    if location is None or not location.disk_location:
        return None
    return Path(location.disk_location).expanduser().resolve()


def _instance_path(db, instance: LaupyRecipeInstance | None) -> Path | None:
    if instance is None:
        return None
    root = _collection_root(db, instance.collection_location_id)
    return (root / instance.relative_path).resolve() if root is not None else None


def _instance_for_dir(db, directory: Path) -> LaupyRecipeInstance | None:
    """Registered recipe instance at directory, if its collection is registered."""
    for collection in db.list_recipe_collections():
        root = _collection_root(db, collection.location_id)
        if root is not None and directory.parent == root:
            return db.get_recipe_instance_by_relative_path(collection.location_id, Path(directory.name))
    return None


def resolve_project_recipe(db, project) -> ProjectRecipe:
    """Find the project recipe copy and its upstream cookbook recipe.

    The database link (project -> workbench copy -> cloned from cookbook recipe)
    is used when present. Otherwise, e.g. for projects found by scanning
    scratch_cc, the copy is found from the project's symlinks and the upstream
    recipe by name in the default cookbook.
    """
    problems: list[str] = []
    link = db.get_project_to_recipe_instance_for_project(project.id) if project.id is not None else None
    if link is not None:
        copy = db.get_recipe_instance(link.recipe_instance_id)
        upstream = db.get_recipe_instance(copy.cloned_from_instance_id) if copy and copy.cloned_from_instance_id else None
        copy_dir, upstream_dir = _instance_path(db, copy), _instance_path(db, upstream)
        if upstream is None:
            problems.append("The project recipe copy has no recorded upstream recipe.")
        if copy_dir is not None and upstream_dir is not None:
            return ProjectRecipe(copy_dir, upstream_dir, True, copy, upstream, problems)

    copy_dir = _recipe_dir_from_links(Path(project.path))
    if copy_dir is None:
        problems.append("The project has no recipe symlinks (sbatch, INFO, script, bin).")
        return ProjectRecipe(None, None, False, problems=problems)
    cookbook_root = _collection_root(db, db.get_config().default_cookbook_location_id)
    upstream_dir = _match_upstream_by_name(copy_dir, cookbook_root) if cookbook_root is not None else None
    if cookbook_root is None:
        problems.append("No default cookbook is configured (File > Settings).")
    elif upstream_dir is None:
        problems.append(f"No recipe in the cookbook {cookbook_root} matches the name {copy_dir.name}.")
    return ProjectRecipe(copy_dir, upstream_dir, False,
                         _instance_for_dir(db, copy_dir),
                         _instance_for_dir(db, upstream_dir) if upstream_dir is not None else None,
                         problems)


def remember_project_recipe(db, project, recipe: ProjectRecipe) -> None:
    """Store an inferred ProjectRecipe in the database so it no longer has to be guessed.

    Both directories must be recipe instances of registered collections
    (cookbook and workbench in Settings).
    """
    copy, upstream = recipe.project_recipe_instance, recipe.upstream_recipe_instance
    if (copy is None or upstream is None) and recipe.project_recipe_dir and recipe.upstream_recipe_dir:
        # Register the recipe directories of all collections, then look them up again
        from lautools.collection_manager import CollectionManager
        manager = CollectionManager(db)
        for collection in db.list_recipe_collections():
            try:
                manager.sync_recipe_instances_from_disk(collection.location_id)
            except ValueError:
                pass  # collection directory unavailable
        copy = _instance_for_dir(db, recipe.project_recipe_dir)
        upstream = _instance_for_dir(db, recipe.upstream_recipe_dir)
    if copy is None or upstream is None:
        raise ValueError("Both the recipe copy and the upstream recipe must lie in registered "
                         "cookbook/workbench collections (File > Settings).")
    if copy.cloned_from_instance_id != upstream.id:
        db.update_recipe_instance(replace(copy, cloned_from_instance_id=upstream.id))
    if db.get_project_to_recipe_instance_for_project(project.id) is None:
        db.add_project_to_recipe_instance(LaupyProjectToRecipeInstance(
            id=None, project_id=project.id, recipe_instance_id=copy.id, created_at=datetime.now()))
    db.connection.commit()


# ----------------------------------------------------------------------
# File comparison
# ----------------------------------------------------------------------

class RecipeFileStatus(Enum):
    IDENTICAL = "identical"
    MODIFIED = "modified"
    ONLY_IN_PROJECT = "only in project"
    ONLY_IN_UPSTREAM = "only in upstream"


@dataclass
class RecipeFileDiff:
    relative_path: str
    status: RecipeFileStatus


def _ignored(name: str) -> bool:
    return name in IGNORED_NAMES or name.endswith(IGNORED_SUFFIXES)


def _recipe_files(root: Path | None) -> set[str]:
    files: set[str] = set()
    if root is None or not root.is_dir():
        return files
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _ignored(d)]
        for name in filenames:
            if not _ignored(name):
                files.add(Path(directory, name).relative_to(root).as_posix())
    return files


def strip_info_header(text: str) -> str:
    """INFO of a project copy without the header lautools prepended at project creation."""
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip() == INFO_ORIGINAL_MARKER:
            return "".join(lines[index + 1:])
    return text


def _info_header(text: str) -> str:
    stripped = strip_info_header(text)
    return text[: len(text) - len(stripped)] if stripped != text else ""


def recipe_file_text(root: Path, relative_path: str, *, project_side: bool) -> str:
    """File text as compared: the project INFO without its project header."""
    text = (root / relative_path).read_text(errors="replace")
    if project_side and relative_path == INFO_FILENAME:
        text = strip_info_header(text)
    return text


def compare_recipes(project_recipe_dir: Path, upstream_recipe_dir: Path) -> list[RecipeFileDiff]:
    """Compare all files of the two recipe directories (INFO without the project header)."""
    project_files = _recipe_files(project_recipe_dir)
    upstream_files = _recipe_files(upstream_recipe_dir)
    result = []
    for relative in sorted(project_files | upstream_files):
        if relative not in upstream_files:
            status = RecipeFileStatus.ONLY_IN_PROJECT
        elif relative not in project_files:
            status = RecipeFileStatus.ONLY_IN_UPSTREAM
        elif relative == INFO_FILENAME:
            same = (recipe_file_text(project_recipe_dir, relative, project_side=True)
                    == recipe_file_text(upstream_recipe_dir, relative, project_side=False))
            status = RecipeFileStatus.IDENTICAL if same else RecipeFileStatus.MODIFIED
        else:
            same = filecmp.cmp(project_recipe_dir / relative, upstream_recipe_dir / relative, shallow=False)
            status = RecipeFileStatus.IDENTICAL if same else RecipeFileStatus.MODIFIED
        result.append(RecipeFileDiff(relative, status))
    return result


def unified_diff(project_recipe_dir: Path, upstream_recipe_dir: Path, relative_path: str) -> str:
    """Unified diff upstream -> project of one file ('-' upstream, '+' project)."""
    def lines(root, project_side):
        path = root / relative_path
        return recipe_file_text(root, relative_path, project_side=project_side).splitlines(keepends=True) if path.is_file() else []
    diff = difflib.unified_diff(lines(upstream_recipe_dir, False), lines(project_recipe_dir, True),
                                fromfile=f"upstream/{relative_path}", tofile=f"project/{relative_path}")
    return "".join(diff) or "Files are identical.\n"


def backport_file(project_recipe_dir: Path, upstream_recipe_dir: Path, relative_path: str) -> Path:
    """Copy one file from the project recipe to the upstream recipe (INFO without the project header)."""
    source, target = project_recipe_dir / relative_path, upstream_recipe_dir / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if relative_path == INFO_FILENAME:
        target.write_text(recipe_file_text(project_recipe_dir, relative_path, project_side=True))
    else:
        shutil.copy2(source, target)
    return target


def update_file_from_upstream(project_recipe_dir: Path, upstream_recipe_dir: Path, relative_path: str) -> Path:
    """Copy one file from the upstream recipe to the project recipe (INFO keeps its project header)."""
    source, target = upstream_recipe_dir / relative_path, project_recipe_dir / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if relative_path == INFO_FILENAME and target.is_file():
        header = _info_header(target.read_text(errors="replace"))
        target.write_text(header + source.read_text(errors="replace"))
    else:
        shutil.copy2(source, target)
    return target


# ----------------------------------------------------------------------
# Cookbook git state
# ----------------------------------------------------------------------

GIT_TIMEOUT_S = 30


def _git(directory: Path, *args: str, timeout: int = GIT_TIMEOUT_S) -> str:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")  # never block on a credential prompt
    result = subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True,
                            timeout=timeout, env=env)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


@dataclass
class CookbookGitStatus:
    directory: Path
    repo_root: Path | None = None
    branch: str | None = None
    remotes: list[str] = field(default_factory=list)
    upstream_ref: str | None = None      # e.g. origin/main
    ahead: int = 0                       # local commits not on the remote
    behind: int = 0                      # remote commits not merged locally
    local_changes: list[str] = field(default_factory=list)   # modified tracked files (git status)
    fetched: bool = False
    error: str | None = None

    @property
    def can_fast_forward(self) -> bool:
        """Only one remote, no local commits or changes, and the remote is newer."""
        return (self.error is None and len(self.remotes) == 1 and self.upstream_ref is not None
                and self.ahead == 0 and self.behind > 0 and not self.local_changes)

    def summary(self) -> str:
        if self.error:
            return f"Cookbook git: {self.error}"
        if self.repo_root is None:
            return "Cookbook is not a git repository."
        parts = [f"branch {self.branch or '(detached)'}"]
        if self.upstream_ref is None:
            parts.append("no upstream branch")
        elif self.ahead == 0 and self.behind == 0:
            parts.append(f"up to date with {self.upstream_ref}")
        else:
            if self.behind:
                parts.append(f"{self.behind} commit(s) behind {self.upstream_ref}")
            if self.ahead:
                parts.append(f"{self.ahead} local commit(s) not pushed")
        if self.local_changes:
            parts.append(f"{len(self.local_changes)} uncommitted change(s)")
        if len(self.remotes) > 1:
            parts.append(f"remotes: {', '.join(self.remotes)}")
        if not self.fetched:
            parts.append("remote not fetched")
        return "Cookbook git: " + ", ".join(parts)


def cookbook_git_status(directory: Path, fetch: bool = True) -> CookbookGitStatus:
    """Git state of the repository containing directory; fetches the upstream remote first."""
    status = CookbookGitStatus(directory=Path(directory))
    try:
        try:
            status.repo_root = Path(_git(directory, "rev-parse", "--show-toplevel"))
        except RuntimeError:
            return status  # not a git repository
        status.remotes = [r for r in _git(directory, "remote").splitlines() if r]
        status.branch = _git(directory, "rev-parse", "--abbrev-ref", "HEAD")
        if status.branch == "HEAD":
            status.branch = None
        try:
            status.upstream_ref = _git(directory, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        except RuntimeError:
            status.upstream_ref = None
        if fetch and status.upstream_ref is not None:
            _git(directory, "fetch", "--quiet", status.upstream_ref.split("/", 1)[0])
            status.fetched = True
        if status.upstream_ref is not None:
            behind, ahead = _git(directory, "rev-list", "--left-right", "--count", "@{u}...HEAD").split()
            status.behind, status.ahead = int(behind), int(ahead)
        status.local_changes = [line for line in _git(directory, "status", "--porcelain", "--untracked-files=no").splitlines() if line]
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        status.error = str(exc)
    return status


def fast_forward_cookbook(directory: Path) -> str:
    """Clean merge of the upstream branch: git merge --ff-only @{u}. Raises RuntimeError on failure."""
    return _git(directory, "merge", "--ff-only", "@{u}")


# ----------------------------------------------------------------------
# Combined state for the status bar and the project dialog
# ----------------------------------------------------------------------

# A remote is fetched at most this often per repository by project_recipe_state()
FETCH_INTERVAL_S = 300
_last_fetch: dict[str, float] = {}


@dataclass
class ProjectRecipeState:
    """Is the cookbook up to date with its remote, and does the project recipe differ from upstream?"""
    recipe: ProjectRecipe
    git: CookbookGitStatus | None = None
    diffs: list[RecipeFileDiff] = field(default_factory=list)
    error: str | None = None

    def changed_files(self) -> list[RecipeFileDiff]:
        return [d for d in self.diffs if d.status != RecipeFileStatus.IDENTICAL]

    def cookbook_text(self) -> str:
        git = self.git
        if git is None:
            return "unknown"
        if git.error:
            return "check failed"
        if git.repo_root is None:
            return "not under git"
        parts = []
        if git.behind:
            parts.append(f"{git.behind} new upstream commit{'s' if git.behind != 1 else ''}")
        if git.ahead:
            parts.append(f"{git.ahead} unpushed commit{'s' if git.ahead != 1 else ''}")
        if git.local_changes:
            n = len(git.local_changes)
            parts.append(f"{n} uncommitted change{'s' if n != 1 else ''}")
        if git.upstream_ref is None:
            parts.append("no upstream branch")
        return ", ".join(parts) or "up to date"

    def recipe_text(self) -> str:
        if self.recipe.project_recipe_dir is None:
            return "none"
        if self.recipe.upstream_recipe_dir is None:
            return "no upstream recipe"
        if self.error:
            return "check failed"
        n = len(self.changed_files())
        return f"{n} file{'s' if n != 1 else ''} changed" if n else "up to date"

    def short_text(self) -> str:
        return f"Cookbook: {self.cookbook_text()} · Recipe: {self.recipe_text()}"

    @property
    def needs_attention(self) -> bool:
        """The cookbook is behind, ahead or dirty, or a check failed.

        A project recipe that differs from upstream is normal (adapted scripts)
        and only reported, not highlighted.
        """
        git = self.git
        cookbook = git is not None and (git.error or git.behind or git.ahead or git.local_changes)
        return bool(cookbook or self.error)

    @property
    def cookbook_can_fast_forward(self) -> bool:
        return self.git is not None and self.git.can_fast_forward

    def tooltip(self) -> str:
        lines = [f"Project recipe: {self.recipe.project_recipe_dir or '(not found)'}",
                 f"Upstream recipe: {self.recipe.upstream_recipe_dir or '(not found)'}"
                 + ("" if self.recipe.linked_in_database else " (inferred)")]
        if self.git is not None:
            lines.append(self.git.summary())
            if self.git.behind and self.git.local_changes:
                lines.append("New upstream commits, but the cookbook has local changes: "
                             "commit or stash them before merging.")
            elif self.git.can_fast_forward:
                lines.append("The cookbook can be updated with a clean (fast-forward) merge.")
        changed = self.changed_files()
        if changed:
            lines.append("Changed files:")
            lines += [f"  {d.status.value}: {d.relative_path}" for d in changed[:15]]
            if len(changed) > 15:
                lines.append(f"  … and {len(changed) - 15} more")
        lines += self.recipe.problems
        if self.error:
            lines.append(f"Error: {self.error}")
        return "\n".join(lines)


def project_recipe_state(recipe: ProjectRecipe, fetch: bool = True) -> ProjectRecipeState:
    """Compare the recipes and check the cookbook git state (file and network I/O, no database access).

    The remote is fetched at most every FETCH_INTERVAL_S per repository.
    """
    state = ProjectRecipeState(recipe=recipe)
    if recipe.upstream_recipe_dir is None:
        return state
    try:
        if recipe.project_recipe_dir is not None:
            state.diffs = compare_recipes(recipe.project_recipe_dir, recipe.upstream_recipe_dir)
    except OSError as exc:
        state.error = f"cannot compare recipes: {exc}"
    key = str(recipe.upstream_recipe_dir)
    do_fetch = fetch and time.monotonic() - _last_fetch.get(key, -FETCH_INTERVAL_S) >= FETCH_INTERVAL_S
    state.git = cookbook_git_status(recipe.upstream_recipe_dir, fetch=do_fetch)
    if state.git.fetched:
        _last_fetch[key] = time.monotonic()
    return state
