"""
Thin middleware layer for pipeline/DAG operations.
Wraps laupy.flow and laupy.slurm calls with parameters from UI widgets.
"""
import getpass
import logging
import os
import subprocess
import time

from pathlib import Path
from typing import List, Dict, Any, Optional
from laupy.flow import load_dag, save_dag, update_dag_entries

# Defaults mirror `pipeline retire` / `pipeline cancel` in laupy/scripts/pipeline.py.
RETIRE_STATES = ("FAILED", "CANCELLED", "TIMEOUT")
CANCEL_STATES = ("PENDING", "RUNNING", "REQUEUED", "COMPLETING")

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
    Thin wrapper around laupy.flow functions.
    Accepts UI configuration parameters and delegates to laupy.
    """
    
    def __init__(self, working_dir: Path):
        """
        Initialize with a working directory.
        
        Parameters
        ----------
        WORKING_DIR : Path
            Path to the working directory (execution unit)
        """
        self.WORKING_DIR = Path(working_dir).resolve()
        self.DAG_ENTRIES: List[Dict[str, Any]] = []
        self.LAST_DAG_FETCH: Optional[float] = None

    def _get_execution_unit_dirs(self) -> List[Path]:
        """Return child directories that contain a pipeline directory.

        A browser working directory corresponds to the ``-w/--working-dir``
        argument of ``laupy/scripts/pipeline.py``.  Each immediate child is
        therefore an execution unit whose DAG is stored in:

            <WORKING_DIR>/<execution_unit>/pipeline/dag.json
        """
        try:
            return sorted(
                (
                    child.resolve()
                    for child in self.WORKING_DIR.iterdir()
                    if child.is_dir() and (child / "pipeline").is_dir()
                ),
                key=lambda path: path.name.lower(),
            )
        except OSError as exc:
            log.warning(
                "Cannot enumerate pipeline directories below %s: %s",
                self.WORKING_DIR,
                exc,
            )
            return []

    # Fetch DAG entries from disk and update SLURM info.  This is a thin wrapper
    # around laupy.flow.load_dag().
    def fetch_dag_entries(self):
        """Fetch DAG entries from disk and update SLURM info."""
        self.DAG_ENTRIES.clear()
        for execution_unit_dir in self._get_execution_unit_dirs():
            try:
                EXECTUTION_UNIT_DIR = os.path.abspath(execution_unit_dir)
                EXECUTION_UNIT_NAME = os.path.basename(execution_unit_dir)
                dag = load_dag(str(execution_unit_dir), topological_sort=True)
                for dag_entry in dag:
                    if "execution_unit_dir" not in dag_entry:
                        dag_entry["execution_unit_dir"] = EXECTUTION_UNIT_DIR
                    if "execution_unit_name" not in dag_entry:
                        dag_entry["execution_unit_name"] = EXECUTION_UNIT_NAME
                self.DAG_ENTRIES.extend(dag)
            except (OSError, ValueError) as exc:
                log.warning("Cannot load DAG from %s: %s", execution_unit_dir, exc,)
                continue
        self.LAST_DAG_FETCH = time.time()

    def get_pipeline_entries(
        self,
        show_completed: bool = False,
        show_retired: bool = False,
        update_slurm_info: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get filtered pipeline DAG entries.
        
        This is a thin wrapper that:
        1. Loads DAG from disk via laupy.flow.load_dag()
        2. Updates SLURM info via laupy.flow.update_dag_entries()
        3. Applies UI-requested filters
        
        Parameters
        ----------
        show_completed : bool
            Include COMPLETED jobs
        show_retired : bool
            Include retired jobs
        update_slurm_info : bool
            Query SLURM to update job status
            
        Returns
        -------
        List[Dict[str, Any]]
            Filtered DAG entries with populated slurm_info
        """
        # Fetch DAG entries from disk and update SLURM info
        self.fetch_dag_entries()
        # Mirror `laupy pipeline status`: each child of the selected working
        # directory is an execution unit with its own pipeline/dag.json.
        try:
            if update_slurm_info:
                update_dag_entries(
                    self.DAG_ENTRIES,
                    # Include retired entries in the SLURM refresh only
                    # when the corresponding UI filter asks for them.
                    update_retired=show_retired,
                    update_negative_step=False,
                    # Same status-refresh behaviour as the CLI command:
                    # do not re-query cached terminal jobs.
                    filter_terminal_states=False,
                )
            # Update on disk DAG entries with the latest SLURM info.  This is a no-op if
            for execution_unit_dir in self._get_execution_unit_dirs():
                EXECUTION_UNIT_DIR = os.path.abspath(execution_unit_dir)
                dag = [entry for entry in self.DAG_ENTRIES if entry.get("execution_unit_dir") == EXECUTION_UNIT_DIR]
                save_dag(EXECUTION_UNIT_DIR, dag)
        except Exception:
            log.exception("Cannot update SLURM status for execution units")
        if not show_retired:
            entries = [ entry for entry in self.DAG_ENTRIES if entry.get("retired", False) is not True]
        if not show_completed:
            entries = [ entry for entry in entries if entry.get("slurm_info", {}).get("State") != "COMPLETED"]
        #Sort entries by execution unit name, preserve topological sort order within each execution unit
        entries.sort(key=lambda x: (x.get("execution_unit_name", "")))
        return entries

    # ------------------------------------------------------------------
    # Bulk job management (Jobs menu)
    #
    # laupy implements retire/cancel only inline in scripts/pipeline.py, so
    # the selection logic is mirrored here as "collect candidates" followed by
    # "apply to the entries the user selected".
    # ------------------------------------------------------------------

    def collect_candidates(self, states) -> tuple[List[Dict[str, Any]], List[str]]:
        """Return non-retired DAG entries whose refreshed SLURM state is in ``states``.

        Same filter as ``pipeline retire/cancel -w <WORKING_DIR> --state ...``.
        Entries are annotated with ``execution_unit_dir``/``execution_unit_name``.
        Returns ``(entries, errors)``.
        """
        states = set(states)
        candidates, errors = [], []
        for execution_unit_dir in self._get_execution_unit_dirs():
            unit_dir = str(execution_unit_dir)
            try:
                dag = load_dag(unit_dir, topological_sort=True)
                update_dag_entries(dag, update_retired=False, update_negative_step=True, filter_terminal_states=False)
            except Exception as exc:
                log.exception("Cannot load DAG in %s", unit_dir)
                errors.append(f"{execution_unit_dir.name}: {exc}")
                continue
            for entry in dag:
                if entry.get("retired", False):
                    continue
                if entry.get("slurm_info", {}).get("State") in states:
                    entry["execution_unit_dir"] = unit_dir
                    entry["execution_unit_name"] = execution_unit_dir.name
                    candidates.append(entry)
        return candidates, errors

    def retire_candidates(self):
        return self.collect_candidates(RETIRE_STATES)

    def cancel_candidates(self):
        return self.collect_candidates(CANCEL_STATES)

    @staticmethod
    def _update_dags(entries: List[Dict[str, Any]], modify) -> List[str]:
        """Reload each affected DAG, apply ``modify`` to the selected entries, save.

        Entries are matched by ``job_id`` in a fresh copy of the DAG so that
        changes written meanwhile (e.g. resubmissions) are not lost.
        """
        errors = []
        by_unit: Dict[str, set] = {}
        for entry in entries:
            by_unit.setdefault(entry["execution_unit_dir"], set()).add(str(entry.get("job_id")))
        for unit_dir, job_ids in by_unit.items():
            try:
                dag = load_dag(unit_dir)
                for dag_entry in dag:
                    if str(dag_entry.get("job_id")) in job_ids:
                        modify(dag_entry)
                save_dag(unit_dir, dag)
            except Exception as exc:
                log.exception("Cannot update DAG in %s", unit_dir)
                errors.append(f"{os.path.basename(unit_dir)}: {exc}")
        return errors

    def retire_entries(self, entries: List[Dict[str, Any]]) -> List[str]:
        """Mark the given entries retired. Returns error messages."""
        return self._update_dags(entries, lambda e: e.__setitem__("retired", True))

    def cancel_entries(self, entries: List[Dict[str, Any]]) -> List[str]:
        """scancel the given entries and record them as CANCELLED. Returns error messages.

        Like ``pipeline cancel`` the entries are not retired.
        """
        ids = [str(entry_slurm_id(e)) for e in entries if entry_slurm_id(e) is not None]
        errors = [f"{e.get('execution_unit_name')}: node {e.get('dag_node_name')} has no SLURM ID"
                  for e in entries if entry_slurm_id(e) is None]
        if not ids:
            return errors
        error = scancel_jobs(ids)
        if error:
            # Unknown which IDs failed; leave the DAG alone, the next status
            # refresh reads the real states from SLURM.
            return errors + [error]

        def mark_cancelled(dag_entry):
            dag_entry.setdefault("slurm_info", {})["State"] = "CANCELLED"
        return errors + self._update_dags(entries, mark_cancelled)


def entry_slurm_id(entry: Dict[str, Any]):
    return entry.get("slurm_id") or entry.get("slurm_info", {}).get("JobID")


# ----------------------------------------------------------------------
# SLURM operations independent of laupy DAG management
# ----------------------------------------------------------------------

# laupy.slurm only queries jobs by ID, so the user listing is one squeue call.
_SQUEUE_FIELDS = ("JobID", "JobName", "State", "Partition", "Elapsed", "Timelimit", "Reason", "SubmitTime", "WorkDir")
_SQUEUE_FORMAT = "|".join(("%i", "%j", "%T", "%P", "%M", "%l", "%R", "%V", "%Z"))


def list_user_slurm_jobs(user: Optional[str] = None) -> List[Dict[str, str]]:
    """Return all queued/running SLURM jobs of ``user`` (default: current user).

    ``Reason`` holds the pending reason or, for running jobs, the node list.
    """
    user = user or getpass.getuser()
    result = subprocess.run(["squeue", "-h", "-u", user, "-o", _SQUEUE_FORMAT],
                            check=True, capture_output=True, text=True)
    jobs = []
    for line in result.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < len(_SQUEUE_FIELDS):
            continue
        # A "|" inside the job name would shift fields; rejoin it.
        extra = len(parts) - len(_SQUEUE_FIELDS)
        parts = [parts[0], "|".join(parts[1:2 + extra])] + parts[2 + extra:]
        jobs.append(dict(zip(_SQUEUE_FIELDS, parts)))
    return jobs


def protected_slurm_job_ids() -> set:
    """Job IDs that must not be cancelled: the allocation this process runs in."""
    return {os.environ[var].strip() for var in ("SLURM_JOB_ID", "SLURM_JOBID") if os.environ.get(var)}


def is_protected_job(job_id: str, protected: set) -> bool:
    # Array tasks are reported as <array_id>_<task>.
    return str(job_id).split("_")[0] in protected


def scancel_jobs(job_ids: List[str]) -> Optional[str]:
    """Cancel the given job IDs with one scancel call. Returns an error message or None."""
    if not job_ids:
        return None
    try:
        subprocess.run(["scancel", *map(str, job_ids)], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        # scancel still cancels the valid IDs, e.g. when some jobs already ended.
        return (exc.stderr or "").strip() or str(exc)
    except OSError as exc:
        return str(exc)
    return None
