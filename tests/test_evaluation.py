"""
Unit Tests: Benchmark Evaluation Suites
Part of Tereguwami (ተርጓሚ) Automated Test Suite
"""

import pytest
from evaluation.benchmarks.evaluate_translation import translation_evaluator
from evaluation.benchmarks.evaluate_recognition import recognition_evaluator
from evaluation.benchmarks.ablation_non_manual import non_manual_ablation_evaluator


def test_bleu_evaluation():
    refs = [["እኔ", "ተማሪ", "ነኝ"], ["ዶክተር", "ናቸው"]]
    hyps = [["እኔ", "ተማሪ", "ነኝ"], ["ዶክተር", "ናቸው"]]
    scores = translation_evaluator.compute_bleu(refs, hyps)
    assert scores["bleu_1"] == 100.0
    assert scores["brevity_penalty"] == 1.0


def test_chrf_evaluation():
    refs = ["ዶክተር ብርቱ የራስ ምታት አለኝ።"]
    hyps = ["ዶክተር ብርቱ የራስ ምታት አለኝ።"]
    chrf = translation_evaluator.compute_chrf(refs, hyps)
    assert chrf == 100.0


def test_recognition_gap_evaluation():
    y_true = ["A", "B", "C", "D"] * 10
    dep_pred = ["A", "B", "C", "D"] * 10
    indep_pred = ["A", "B", "X", "X"] * 10  # 50% acc
    res = recognition_evaluator.compare_generalization_gap(y_true, dep_pred, y_true, indep_pred)
    assert res["generalization_gap_points"] == 50.0
    assert res["signer_dependent"]["accuracy"] == 100.0
    assert res["signer_independent"]["accuracy"] == 50.0


def test_non_manual_ablation_evaluation():
    truth = {"polar_questions": ["Q"] * 10}
    hands_only = {"polar_questions": ["DECL"] * 8 + ["Q"] * 2}
    hands_face = {"polar_questions": ["Q"] * 9 + ["DECL"] * 1}
    res = non_manual_ablation_evaluator.evaluate_ablation(hands_only, hands_face, truth)
    cat = res["categories"]["polar_questions"]
    assert cat["gain_percentage_points"] == 70.0
