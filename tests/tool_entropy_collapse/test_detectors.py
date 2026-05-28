"""Unit tests for WANDERING detectors.

Run: pytest tests/tool_entropy_collapse/test_detectors.py -v
"""
from __future__ import annotations
import pytest
from tool_entropy_collapse.detectors import (
    v1_forensic, v5_tool_entropy, v1_or_v5,
    tool_entropy_last_n, run_detector, DETECTORS,
)


def make_trace(turns: list[dict]) -> dict:
    return {"instance_id": "test", "turns": turns, "n_turns": len(turns),
            "finish_reason": "max_turns"}


def make_turn(content: str = "", thinking: str = "", tool_calls: list[dict] = None) -> dict:
    return {"content": content, "thinking": thinking,
            "tool_calls": tool_calls or [], "tool_results": []}


class TestV1Forensic:
    def test_no_completion_phrases_does_not_fire(self):
        trace = make_trace([make_turn("Just looking at the code")] * 5)
        r = v1_forensic(trace)
        assert r["fired"] is False

    def test_completion_phrase_fires(self):
        trace = make_trace([
            make_turn(content="I think the answer is 42, this should work")
        ] * 5)
        r = v1_forensic(trace)
        assert r["fired"] is True
        assert r["fire_turn"] is not None

    def test_completion_in_thinking_fires(self):
        trace = make_trace([
            make_turn(thinking="I'm done with this task, submitting solution")
        ] * 5)
        r = v1_forensic(trace)
        assert r["fired"] is True

    def test_empty_trace_does_not_fire(self):
        r = v1_forensic(make_trace([]))
        assert r["fired"] is False


class TestV5ToolEntropy:
    def test_single_tool_repeated_low_entropy_fires(self):
        # 15 turns all calling bash → entropy = 0.0 (single category).
        # This is the CANONICAL WANDERING signature per Tool-Entropy paper §6:
        # an agent collapsed onto a single repeated tool. v5 FIRES (0.0 < threshold),
        # consistent with the validated N=99 result (v5 55% recall / 5% FP).
        turns = [make_turn(tool_calls=[{"name": "bash"}]) for _ in range(15)]
        trace = make_trace(turns)
        r = v5_tool_entropy(trace, threshold=0.5, window=10)
        assert r["fired"] is True
        assert r["tool_entropy_last10"] == 0.0

    def test_no_tool_calls_edge_case(self):
        # Edge case: zero tool calls in the last `window` turns also yields ent=0.0
        # and therefore FIRES under the current (validated) logic. This case does NOT
        # occur in the N=99 validation set — SUCCESS trajectories emit finish_tool and
        # WANDERING/LOCKED loop on tools, so the last-10 window always contains tool
        # calls. Documented here as current behavior; semantically distinguishing
        # "no tools" from "single repeated tool" would be a detector-logic change
        # requiring re-validation against N=99 to confirm 55%/5% is preserved.
        turns = [make_turn(content="thinking only, no tools") for _ in range(15)]
        trace = make_trace(turns)
        r = v5_tool_entropy(trace, threshold=0.5, window=10)
        assert r["fired"] is True  # current validated behavior (ent=0.0 < threshold)
        assert r["tool_entropy_last10"] == 0.0

    def test_diverse_tools_high_entropy_does_not_fire(self):
        # 10 turns each with different tool → high entropy
        tools = ["bash", "str_replace_editor", "python", "view", "grep",
                 "find", "cat", "ls", "head", "tail"]
        turns = [make_turn(tool_calls=[{"name": tools[i]}]) for i in range(10)]
        trace = make_trace(turns * 2)  # 20 turns
        r = v5_tool_entropy(trace, threshold=0.5, window=10)
        assert r["fired"] is False

    def test_too_short_trace_does_not_fire(self):
        trace = make_trace([make_turn(tool_calls=[{"name": "bash"}])] * 5)
        r = v5_tool_entropy(trace, threshold=0.5, window=10)
        assert r["fired"] is False

    def test_entropy_calculation(self):
        # 2 bash + 2 python: entropy = -2*(0.5*log2(0.5)) = 1.0 bit
        turns = [
            make_turn(tool_calls=[{"name": "bash"}]),
            make_turn(tool_calls=[{"name": "bash"}]),
            make_turn(tool_calls=[{"name": "python"}]),
            make_turn(tool_calls=[{"name": "python"}]),
        ]
        trace = make_trace(turns)
        ent = tool_entropy_last_n(trace, window=4)
        assert abs(ent - 1.0) < 1e-6


class TestEnsemble:
    def test_either_fires(self):
        trace = make_trace([
            make_turn(content="this should work", tool_calls=[{"name": "bash"}])
        ] * 15)
        r = v1_or_v5(trace, threshold=0.5)
        # v1 should fire on "should work"
        assert r["fired"] is True
        assert r.get("v1", {}).get("fired") is True


class TestRunDetector:
    def test_dispatch_v1(self):
        trace = make_trace([make_turn(content="I am done")] * 5)
        r = run_detector("v1_forensic", trace)
        assert r["fired"] is True

    def test_unknown_detector_raises(self):
        with pytest.raises(ValueError):
            run_detector("not_a_real_detector", {})

    def test_all_detectors_listed(self):
        expected = {"v1_forensic", "v5_tool_entropy", "v4_cross_layer", "v1_or_v5"}
        assert set(DETECTORS.keys()) >= expected
