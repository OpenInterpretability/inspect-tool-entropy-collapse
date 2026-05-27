"""Tool-Entropy Collapse WANDERING detector eval — main @task definition.

Pattern: monitoring eval (no model invocation). Trajectories pre-recorded;
detector runs inside a passthrough solver, scorer compares to gold label.
"""
from __future__ import annotations
import json
from typing import Literal
from inspect_ai import Task, task
from inspect_ai.solver import Generate, Solver, TaskState, solver

from tool_entropy_collapse.dataset import (
    load_trajectory_dataset, load_trace, load_captures, load_v4_cache,
)
from tool_entropy_collapse.detectors import run_detector, DETECTORS
from tool_entropy_collapse.scorers import wandering_detector_scorer

DetectorName = Literal["v1_forensic", "v4_cross_layer", "v5_tool_entropy", "v1_or_v5"]


# Module-level cache for v4_cross_layer outputs (loaded once per eval run)
_V4_CACHE: dict | None = None


def _get_v4_cache() -> dict:
    global _V4_CACHE
    if _V4_CACHE is None:
        _V4_CACHE = load_v4_cache()
    return _V4_CACHE


@solver
def detector_solver(detector: DetectorName = "v1_or_v5", threshold: float = 0.8) -> Solver:
    """Solver that runs the WANDERING detector on the trajectory + emits JSON verdict.

    Loads the trace from HF for the sample iid, runs the detector, sets
    state.output.completion to JSON of {fired, fire_turn, confidence}.

    For v4_cross_layer: uses precomputed cached outputs (no torch/probes needed).
    For online v4 with raw residuals, see openinterp-swebench-harness main repo.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        iid = state.metadata["iid"]
        trace = load_trace(iid)
        kw = {"threshold": threshold}
        if detector == "v4_cross_layer":
            kw["v4_cache"] = _get_v4_cache()
        verdict = run_detector(detector, trace, **kw)
        clean_verdict = {
            "fired": verdict["fired"],
            "fire_turn": verdict.get("fire_turn"),
            "confidence": verdict.get("confidence", 0.0),
        }
        from inspect_ai.model import ModelOutput, ChatCompletionChoice, ChatMessageAssistant
        state.output = ModelOutput(
            model="detector://" + detector,
            choices=[ChatCompletionChoice(
                message=ChatMessageAssistant(content=json.dumps(clean_verdict)),
                stop_reason="stop",
            )],
        )
        return state
    return solve


@task
def tool_entropy_collapse(
    detector: DetectorName = "v1_or_v5",
    threshold: float = 0.5,  # lowered from 0.8 — paper-grade v5 fires at ~0.4-0.5 tool-entropy
    include_classes: str = "wandering,success",  # comma-separated for CLI override
) -> Task:
    """WANDERING failure-mode detection eval on 99 SWE-bench Pro Qwen3.6-27B trajectories.

    Args:
        detector: which detector to test (v1_forensic / v4_cross_layer / v5_tool_entropy / v1_or_v5)
        threshold: entropy/probability threshold for the detector (default 0.8 for v5)
        include_classes: comma-separated sub-classes to include (default "wandering,success";
            LOCKED is excluded by default since it's externally identical to WANDERING but
            internally distinct — not a meaningful negative class)

    Reports:
        - accuracy (overall classification)
        - recall (TP/(TP+FN) over true WANDERING samples)
        - fp_rate (FP/(FP+TN) over true SUCCESS samples)
        - mean_lead_turns (how early before max_turns the detector fires, for TP)
    """
    classes = tuple(c.strip() for c in include_classes.split(","))
    dataset = load_trajectory_dataset(include_classes=classes)
    return Task(
        dataset=dataset,
        solver=detector_solver(detector=detector, threshold=threshold),
        scorer=wandering_detector_scorer(detector=detector),
    )
