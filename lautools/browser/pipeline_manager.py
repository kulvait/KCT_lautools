"""
Thin middleware layer for pipeline/DAG operations.
Wraps laupy.flow and laupy.slurm calls with parameters from UI widgets.
"""
import logging

from pathlib import Path
from typing import List, Dict, Any, Optional
from laupy.flow import load_dag, update_dag_entries

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
        working_dir : Path
            Path to the working directory (execution unit)
        """
        self.working_dir = Path(working_dir)

    def _get_execution_unit_dirs(self) -> List[Path]:
        """Return child directories that contain a pipeline directory.

        A browser working directory corresponds to the ``-w/--working-dir``
        argument of ``laupy/scripts/pipeline.py``.  Each immediate child is
        therefore an execution unit whose DAG is stored in:

            <working_dir>/<execution_unit>/pipeline/dag.json
        """
        try:
            return sorted(
                (
                    child.resolve()
                    for child in self.working_dir.iterdir()
                    if child.is_dir() and (child / "pipeline").is_dir()
                ),
                key=lambda path: path.name.lower(),
            )
        except OSError as exc:
            log.warning(
                "Cannot enumerate pipeline directories below %s: %s",
                self.working_dir,
                exc,
            )
            return []


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
        entries: List[Dict[str, Any]] = []
        # Mirror `laupy pipeline status`: each child of the selected working
        # directory is an execution unit with its own pipeline/dag.json.
        for execution_unit_dir in self._get_execution_unit_dirs():
            try:
                dag = load_dag(str(execution_unit_dir))
            except (OSError, ValueError) as exc:
                log.warning(
                    "Cannot load DAG from %s: %s",
                    execution_unit_dir,
                    exc,
                )
                continue

            if update_slurm_info and dag:
                try:
                    update_dag_entries(
                        dag,
                        # Include retired entries in the SLURM refresh only
                        # when the corresponding UI filter asks for them.
                        update_retired=show_retired,
                        update_negative_step=False,
                        # Same status-refresh behaviour as the CLI command:
                        # do not re-query cached terminal jobs.
                        filter_terminal_states=True,
                    )
                except Exception:
                    # One unavailable Slurm query must not hide status from
                    # other execution units.
                    log.exception(
                        "Cannot update SLURM status for %s",
                        execution_unit_dir,
                    )

            for dag_entry in dag:
                # Do not mutate the DAG object loaded from disk.  The source
                # fields let the widget distinguish identical steps/jobs from
                # separate execution units.
                entry = dict(dag_entry)
                entry["execution_unit_dir"] = execution_unit_dir
                entry["execution_unit_name"] = execution_unit_dir.name
                entry["execution_unit_relative"] = str(
                    execution_unit_dir.relative_to(self.working_dir)
                )
                entries.append(entry)

        if not show_retired:
            entries = [
                entry
                for entry in entries
                if not entry.get("retired", False)
            ]

        if not show_completed:
            entries = [
                entry
                for entry in entries
                if entry.get("slurm_info", {}).get("State") != "COMPLETED"
            ]

        return entries
