"""Scorer + custom metrics for WANDERING detection.

Outputs per Score:
  - recall: TP / (TP+FN), only filled for true positives/negatives
  - fp_rate: FP / (FP+TN), only filled for true negatives
  - accuracy: 1.0 if pred matches label, else 0.0
  - lead_turns: how many turns before max_turns the detector fired (TP only)
"""
from __future__ import annotations
import json
from inspect_ai.scorer import (
    CORRECT, INCORRECT, Score, Scorer, Target, accuracy, mean, scorer, stderr,
    metric, Metric, SampleScore, Value,
)
from inspect_ai.solver import TaskState


@metric
def recall_metric() -> Metric:
    """TP / (TP + FN) from per-sample metadata."""
    def m(scores: list[SampleScore]) -> Value:
        tp = sum(1 for s in scores if s.score.metadata.get("tp"))
        fn = sum(1 for s in scores if s.score.metadata.get("fn"))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return m


@metric
def fpr_metric() -> Metric:
    """FP / (FP + TN) from per-sample metadata."""
    def m(scores: list[SampleScore]) -> Value:
        fp = sum(1 for s in scores if s.score.metadata.get("fp"))
        tn = sum(1 for s in scores if s.score.metadata.get("tn"))
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return m


@metric
def mean_lead_turns_metric() -> Metric:
    """Mean lead turns across TP samples."""
    def m(scores: list[SampleScore]) -> Value:
        leads = [s.score.metadata["lead_turns"] for s in scores
                 if s.score.metadata.get("tp") and s.score.metadata.get("lead_turns") is not None]
        return sum(leads) / len(leads) if leads else 0.0
    return m


@scorer(metrics=[accuracy(), stderr(), recall_metric(), fpr_metric(), mean_lead_turns_metric()])
def wandering_detector_scorer(detector: str = "v1_or_v5") -> Scorer:
    """Score detector verdict against gold sub_class label.

    Reads verdict from state.output.completion (set by detector solver).
    Compares to target.text which contains the gold sub_class.

    Treats sub_class == "wandering" as positive label; everything else as negative.
    """
    async def score(state: TaskState, target: Target) -> Score:
        try:
            verdict = json.loads(state.output.completion)
        except (json.JSONDecodeError, AttributeError):
            return Score(
                value=INCORRECT,
                answer="",
                explanation=f"detector did not emit valid JSON verdict",
                metadata={"detector": detector, "tp": False, "fp": False, "tn": False, "fn": False},
            )

        gold_is_wandering = (target.text == "wandering")
        fired = bool(verdict.get("fired", False))

        # Confusion matrix
        tp = fired and gold_is_wandering
        fp = fired and (not gold_is_wandering)
        tn = (not fired) and (not gold_is_wandering)
        fn = (not fired) and gold_is_wandering

        # Lead turns: only for true positives
        lead_turns = None
        if tp:
            fire_turn = verdict.get("fire_turn")
            total_turns = state.metadata.get("n_turns", 0)
            if fire_turn is not None and total_turns:
                lead_turns = total_turns - 1 - fire_turn

        return Score(
            value=CORRECT if (tp or tn) else INCORRECT,
            answer="WANDERING" if fired else "NOT_WANDERING",
            explanation=f"detector={detector}, fired={fired}, gold={target.text}, "
                        f"confidence={verdict.get('confidence', 0):.3f}",
            metadata={
                "detector": detector,
                "fired": fired,
                "gold_sub_class": target.text,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "lead_turns": lead_turns,
                "confidence": verdict.get("confidence"),
            },
        )
    return score
