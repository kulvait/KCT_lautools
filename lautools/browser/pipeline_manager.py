"""
Thin middleware layer between the browser UI and laupy's PIPELINE_ENTRIES management.

All processing unit subdirs of the selected working directory are handled by one
laupy.pipeline_entries.PipelineEntriesStore, so a refresh costs one batched SLURM query instead
of one per processing unit subdir, and only changed pipeline/dag.json files are written.
"""
import logging
import os
import time

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from laupy.pipeline_entries import (
    CANCELLABLE_STATES,
    RETIRABLE_STATES,
    PipelineEntriesStore,
    cancel_selected,
    retire_selected,
    select_pipeline_entries,
)
from laupy.flow import list_processing_unit_subdirs

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

if not log.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s : %(message)s",
        datefmt="%d.%m.%Y %H:%M:%S",
    )
    handler.setFormatter(formatter)
    log.addHandler(handler)

log.propagate = False

class PipelineManager:
    """
    Pipeline entries of one browser working directory (the ``-w`` of the
    ``pipeline`` CLI); each child with a pipeline/ directory is a processing unit subdir.
    """

    def __init__(self, working_dir: Path):
        self.WORKING_DIR = Path(working_dir).resolve()
        self.PIPELINE_ENTRIES: List[Dict[str, Any]] = []   # entries of the last get_pipeline_entries()
        self.LAST_FETCH: Optional[float] = None

    def _get_processing_unit_subdirs(self) -> List[Path]:
        """Return child directories that contain a pipeline directory."""
        try:
            return [child.resolve() for child in list_processing_unit_subdirs(self.WORKING_DIR, require_pipeline=True)]
        except OSError as exc:
            log.warning("Cannot enumerate pipeline directories below %s: %s", self.WORKING_DIR, exc)
            return []

    def _load_store(self) -> Tuple[PipelineEntriesStore, List[str]]:
        """PIPELINE_ENTRIES of all processing unit subdirs; subdirs that cannot be read are reported, not fatal."""
        store = PipelineEntriesStore(topological_sort=True)
        errors = []
        for subdir in self._get_processing_unit_subdirs():
            try:
                store.load(subdir)
            except (OSError, ValueError) as exc:
                log.warning("Cannot load pipeline entries from %s: %s", subdir, exc)
                errors.append(f"{subdir.name}: {exc}")
        return store, errors

    def get_pipeline_entries(
        self,
        show_completed: bool = False,
        show_retired: bool = False,
        update_slurm_info: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Entries of all processing unit subdirs, refreshed from SLURM, filtered for display.

        Mirrors ``pipeline status``: one batched SLURM query for all subdirs;
        entries already stored in a final state are not queried again.
        Refreshed SLURM info is saved, only for subdirs where it changed.
        """
        store, _ = self._load_store()
        if update_slurm_info:
            try:
                store.refresh(update_retired=show_retired, update_negative_step=False, filter_terminal_states=True)
            except Exception:
                log.exception("Cannot update SLURM status for processing unit subdirs")
        try:
            store.store()
        except Exception:
            log.exception("Cannot save refreshed pipeline entries")
        entries = store.all_entries()  # subdirs by name, topological order within a subdir
        self.PIPELINE_ENTRIES = entries
        self.LAST_FETCH = time.time()
        if not show_retired:
            entries = [entry for entry in entries if entry.get("retired", False) is not True]
        if not show_completed:
            entries = [entry for entry in entries if entry.get("slurm_info", {}).get("State") != "COMPLETED"]
        return entries

    # ------------------------------------------------------------------
    # Bulk job management (Jobs menu), delegated to laupy.pipeline_entries
    # ------------------------------------------------------------------

    def collect_candidates(self, states) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Non-retired entries in ``states`` over all processing unit subdirs (one SLURM query). Returns ``(entries, errors)``."""
        store, errors = self._load_store()
        try:
            store.refresh(update_retired=False, update_negative_step=True, filter_terminal_states=True)
        except Exception as exc:
            log.exception("Cannot refresh SLURM status below %s", self.WORKING_DIR)
            errors.append(f"SLURM status not refreshed: {exc}")
        candidates = [entry for _, PIPELINE_ENTRIES in store.items() for entry in select_pipeline_entries(PIPELINE_ENTRIES, states)]
        return candidates, errors

    def retire_candidates(self):
        return self.collect_candidates(RETIRABLE_STATES)

    def cancel_candidates(self):
        return self.collect_candidates(CANCELLABLE_STATES)

    def retire_pipeline_entries(self, entries: List[Dict[str, Any]]) -> List[str]:
        """Mark the given entries retired. Returns error messages."""
        try:
            retire_selected(entries, reason="lautools-browser")
        except Exception as exc:
            log.exception("Cannot retire jobs below %s", self.WORKING_DIR)
            return [str(exc)]
        return []

    def cancel_pipeline_entries(self, entries: List[Dict[str, Any]]) -> List[str]:
        """scancel the given entries (one SLURM query for all) and record them as CANCELLED. Returns error messages."""
        try:
            results = cancel_selected(entries)
        except Exception as exc:
            log.exception("Cannot cancel jobs below %s", self.WORKING_DIR)
            return [str(exc)]
        return [
            f"{os.path.basename(subdir)}: node {r['entry'].get('dag_node_name')}, SLURM {r['slurm_id']}: {r['error']}"
            for subdir, subdir_results in results.items() for r in subdir_results if r["error"]
        ]
