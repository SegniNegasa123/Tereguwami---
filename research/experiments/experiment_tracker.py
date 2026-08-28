"""
Experiment Run Logger & Metric Aggregator
Part of Tereguwami (ተርጓሚ) Research Infrastructure
"""

import time
import json
import os
from typing import Dict, Any, Optional


class ExperimentTracker:
    """Tracks training loss, validation accuracy, and evaluation metrics across model iterations."""

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir or os.path.dirname(os.path.abspath(__file__))
        self.runs_file = os.path.join(self.log_dir, "experiment_history.json")
        self._history = []
        self._load_history()

    def _load_history(self):
        if os.path.exists(self.runs_file):
            with open(self.runs_file, "r", encoding="utf-8") as f:
                self._history = json.load(f)

    def log_run(
        self,
        experiment_name: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Record an experiment run."""
        record = {
            "run_id": f"RUN_{int(time.time())}",
            "experiment_name": experiment_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hyperparameters": hyperparameters,
            "metrics": metrics
        }
        self._history.append(record)
        with open(self.runs_file, "w", encoding="utf-8") as f:
            json.dump(self._history, f, indent=2)
        return record


experiment_tracker = ExperimentTracker()
