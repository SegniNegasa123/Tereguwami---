"""
Sign Recognition & Signer Independence Evaluation Module (§15)
Part of Tereguwami (ተርጓሚ) Benchmark Evaluation Suite

Evaluates Accuracy, Precision, Recall, and Macro-F1 across signer-dependent and
signer-independent partitions, quantifying the critical generalization gap documented
in the 2025 Ethiopian benchmark study (Scientific Reports).
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class RecognitionEvaluator:
    """Evaluates lexical sign recognition across signer splits."""

    def evaluate_split(
        self,
        y_true: List[str],
        y_pred: List[str],
        split_name: str = "signer_independent_test"
    ) -> Dict[str, Any]:
        """Compute comprehensive classification metrics for a split."""
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        return {
            "split_name": split_name,
            "sample_count": len(y_true),
            "accuracy": round(float(acc * 100.0), 2),
            "precision_macro": round(float(precision * 100.0), 2),
            "recall_macro": round(float(recall * 100.0), 2),
            "f1_macro": round(float(f1 * 100.0), 2)
        }

    def compare_generalization_gap(
        self,
        dep_true: List[str],
        dep_pred: List[str],
        indep_true: List[str],
        indep_pred: List[str]
    ) -> Dict[str, Any]:
        """
        Directly measures the 21-percentage-point generalization collapse
        identified in Ethiopian Sign Language research.
        """
        dep_metrics = self.evaluate_split(dep_true, dep_pred, "signer_dependent_test")
        indep_metrics = self.evaluate_split(indep_true, indep_pred, "signer_independent_test")

        gap = dep_metrics["accuracy"] - indep_metrics["accuracy"]
        return {
            "signer_dependent": dep_metrics,
            "signer_independent": indep_metrics,
            "generalization_gap_points": round(gap, 2),
            "analysis": (
                f"Signer generalization drop: {gap:.1f} percentage points "
                f"({dep_metrics['accuracy']}% dependent vs {indep_metrics['accuracy']}% independent)."
            )
        }


recognition_evaluator = RecognitionEvaluator()
