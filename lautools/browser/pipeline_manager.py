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
        # Load DAG from pipeline/dag.json
        dag = load_dag(str(self.working_dir))
        
        # Update SLURM info if requested
        if update_slurm_info:
            update_dag_entries(
                dag,
                update_retired=show_retired,
                update_negative_step=False,
                filter_terminal_states=True
            )
        
        # Apply UI filters
        filtered = dag
        
        if not show_retired:
            filtered = [e for e in filtered if not e.get("retired", False)]
        
        if not show_completed:
            filtered = [e for e in filtered 
                       if e.get("slurm_info", {}).get("State") != "COMPLETED"]
        
        return filtered
