"""
Thin middleware layer for pipeline/DAG operations.
Wraps laupy.flow and laupy.slurm calls with parameters from UI widgets.
"""
import logging
import os
import time

from pathlib import Path
from typing import List, Dict, Any, Optional
from laupy.flow import load_dag, save_dag, update_dag_entries

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

if not log.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s : %(message)s",
        datefmt="%d.%m.%Y %H:%M:%S",
    )
    ch.setFormatter(formatter)
    log.addHandler(ch)

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
                dag = load_dag(str(execution_unit_dir))
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
        if self.LAST_DAG_FETCH is None:
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
        #Sort entries by execution unit name and step number
        entries.sort(key=lambda x: (x.get("execution_unit_name", ""), x.get("step", 0)))
        return entries
