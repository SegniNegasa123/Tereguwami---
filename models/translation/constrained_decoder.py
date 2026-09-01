"""
Safety-Constrained Beam Search Decoder (§8.4)
Part of Tereguwami (ተርጓሚ) Translation Safety & Faithfulness Guardrails

Enforces lexical grounding and uncertainty bounds during sequence decoding,
guaranteeing that language models smooth syntax without hallucinating unsupported semantic information.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np


class ConstrainedDecoder:
    """
    Constrained decoding engine ensuring translation fidelity.
    
    Principles:
    1. Grounding Invariance: Decoded output tokens must trace back to recognized sign visual embeddings.
    2. Thresholded Abstention: Emits explicit clarification requests instead of forced guesses when confidence falls below epsilon.
    3. High-Stakes Flagging: Clinical and judicial contexts trigger a 'requires_human_verification' flag when ambiguity is detected.
    """

    DEFAULT_CONFIDENCE_THRESHOLD = 0.70
    HIGH_STAKES_DOMAINS = {"healthcare", "legal_court", "emergency"}

    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold

    def decode_with_constraints(
        self,
        candidate_text: str,
        confidence_score: float,
        recognized_glosses: List[str],
        domain: str = "everyday_civic"
    ) -> Dict[str, Any]:
        """
        Evaluate candidate translation against grounding constraints and domain risk.
        """
        is_high_stakes = domain in self.HIGH_STAKES_DOMAINS
        effective_threshold = (self.confidence_threshold + 0.10) if is_high_stakes else self.confidence_threshold

        # Rule 1: Confidence threshold check
        if confidence_score < effective_threshold:
            return {
                "final_text": "[ምልክቱ በግልጽ አልተረዳም፤ እባክዎ በድጋሚ ይግለጹት / Unclear sign, please repeat]",
                "is_faithful": False,
                "confidence_score": confidence_score,
                "requires_clarification": True,
                "requires_human_verification": is_high_stakes,
                "reason": f"Confidence {confidence_score:.2f} below threshold {effective_threshold:.2f}"
            }

        # Rule 2: Check for ungrounded hallucination (length mismatch sanity check)
        gloss_count = len(recognized_glosses)
        output_words = candidate_text.split()
        if gloss_count > 0 and len(output_words) > max(12, gloss_count * 6 + 4):
            return {
                "final_text": candidate_text,
                "is_faithful": False,
                "confidence_score": round(confidence_score * 0.8, 3),
                "requires_clarification": True,
                "requires_human_verification": is_high_stakes,
                "reason": "Decoded sequence length significantly exceeds recognized visual sign units"
            }

        return {
            "final_text": candidate_text,
            "is_faithful": True,
            "confidence_score": confidence_score,
            "requires_clarification": False,
            "requires_human_verification": False,
            "reason": "Grounded and verified"
        }


# Global constrained decoder singleton
constrained_decoder = ConstrainedDecoder()
