import threading
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class JobState:
    status: str = "PENDING"
    current_round: int = 0
    total_rounds: int = 0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    phase: str = "idle"
    phase_index: int = 0
    experiments: List[Dict[str, Any]] = field(default_factory=list)  # [{"name": str, "round": int}, ...]
    baseline_name: str = "Baseline"
    strategy_name: str = "Strategy"

_JOBS: Dict[str, JobState] = {}
_LOCK = threading.Lock()

def create_job(job_id: str, total_rounds: int, experiment_names: List[str]):
    with _LOCK:
        experiments = [{"name": n, "round": 0} for n in experiment_names]
        _JOBS[job_id] = JobState(
            status="RUNNING",
            total_rounds=total_rounds,
            experiments=experiments,
        )

def set_job_phase(job_id: str, phase: str):
    with _LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].phase = phase

def update_job(job_id: str, phase: str, current_round: int, message: str = ""):
    with _LOCK:
        if job_id in _JOBS:
            job = _JOBS[job_id]
            job.phase = phase
            job.message = message
            job.current_round = current_round
            idx = _phase_to_index(phase)
            if 0 <= idx < len(job.experiments):
                job.experiments[idx]["round"] = current_round
            job.phase_index = idx

def _phase_to_index(phase: str) -> int:
    if phase.startswith("exp_"):
        try:
            return int(phase.split("_")[1])
        except (IndexError, ValueError):
            return 0
    mapping = {"baseline": 0, "strategy": 1}
    return mapping.get(phase, 0)

def finish_job(job_id: str, result: Dict[str, Any]):
    with _LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].status = "DONE"
            _JOBS[job_id].result = result

def fail_job(job_id: str, error: str):
    with _LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].status = "ERROR"
            _JOBS[job_id].error = error

def get_job(job_id: str):
    with _LOCK:
        return _JOBS.get(job_id)
