"""WANDERING detectors from Tool-Entropy Collapse paper.

Four detectors across three signal channels:
  - v1_forensic: post-hoc text monitor on assistant verbalizations (Tier 1)
  - v4_cross_layer: residual cross-layer probe disagreement L11/L23/L31/L43/L55 (Tier 2)
  - v5_tool_entropy: Shannon entropy of last 10 tool calls (Tier 3)
  - v1_or_v5: logical OR ensemble (Tier 3 combined)

Each detector returns dict: {fired: bool, fire_turn: int|None, confidence: float}
"""
from __future__ import annotations
import math
from collections import Counter
from typing import Literal

# Forensic v1: textual completion verbalizations indicating "done but not finishing"
V1_FORENSIC_PATTERNS = [
    r"\bdone\b", r"\bcomplete\b", r"\bfinished\b", r"\bsubmit",
    r"\bthe answer is\b", r"\bsolution is\b", r"\bI believe\b.*\bcorrect\b",
    r"\bshould (work|pass|be correct)\b",
]


def v1_forensic(trace: dict, threshold_matches: int = 1) -> dict:
    """v1: post-hoc text monitor.

    Fires if assistant verbalizes completion-like phrases in last 5 turns.
    Tier 1 forensics: 35% recall, 0% FP per Tool-Entropy paper.
    """
    import re
    turns = trace.get("turns", [])
    if not turns:
        return {"fired": False, "fire_turn": None, "confidence": 0.0}

    # Check last 5 turns
    late = turns[-5:]
    matches_per_turn = []
    for t in late:
        text = (t.get("content") or "") + " " + (t.get("thinking") or "")
        count = sum(1 for pat in V1_FORENSIC_PATTERNS if re.search(pat, text, re.IGNORECASE))
        matches_per_turn.append(count)

    total = sum(matches_per_turn)
    fired = total >= threshold_matches
    fire_turn = None
    if fired:
        for i, c in enumerate(matches_per_turn):
            if c > 0:
                fire_turn = len(turns) - 5 + i
                break
    return {"fired": fired, "fire_turn": fire_turn, "confidence": min(1.0, total / 5.0)}


def tool_entropy_last_n(trace: dict, window: int = 10) -> float:
    """Shannon entropy of tool-call name distribution in last `window` turns."""
    turns = trace.get("turns", [])
    last_n = turns[-window:]
    names = []
    for t in last_n:
        for tc in (t.get("tool_calls") or []):
            if isinstance(tc, dict):
                names.append(tc.get("name", "unknown"))
    if not names:
        return 0.0
    counter = Counter(names)
    total = sum(counter.values())
    return -sum((c/total) * math.log2(c/total) for c in counter.values() if c > 0)


def v5_tool_entropy(trace: dict, threshold: float = 0.8, window: int = 10) -> dict:
    """v5: tool-use entropy collapse.

    Fires when last-N tool-call entropy < threshold (low diversity = stuck in loop).
    Tier 3 autonomous: 55% recall, 5% FP per Tool-Entropy paper (Qwen3.6-27B).
    """
    turns = trace.get("turns", [])
    if len(turns) < window:
        return {"fired": False, "fire_turn": None, "confidence": 0.0}

    # Scan from earliest possible fire turn (window-th turn) to end
    fired = False
    fire_turn = None
    for fire_idx in range(window, len(turns) + 1):
        subset = {"turns": turns[:fire_idx]}
        ent = tool_entropy_last_n(subset, window=window)
        if ent < threshold and ent > 0:  # ent=0 means no tool calls (different failure)
            fired = True
            fire_turn = fire_idx - 1
            break

    final_ent = tool_entropy_last_n(trace, window=window)
    confidence = max(0.0, 1.0 - final_ent / threshold) if threshold > 0 else 0.0
    return {"fired": fired, "fire_turn": fire_turn, "confidence": confidence}


def v4_cross_layer(trace: dict, captures: dict, threshold_disagreement: float = 0.5) -> dict:
    """v4: residual cross-layer probe disagreement.

    Compares per-turn probe scores at L11/L23/L31/L43/L55 in late-half turns.
    Fires when L11/L55 (edge) disagree with L23/L31/L43 (mid) consensus.

    NOTE: This is a simplified stub. Full implementation needs the trained probes
    from caiovicentino1/agent-probe-guard-qwen36-27b. Currently returns a placeholder.
    """
    # TODO: load probes from HF, compute per-turn per-layer scores, measure
    # late-half range across layers. Per paper: late-half range > threshold = fire.
    n_turns = len(trace.get("turns", []))
    return {"fired": False, "fire_turn": None, "confidence": 0.0,
            "note": "v4 requires probe weights — not yet wired"}


DETECTORS = {
    "v1_forensic": lambda trace, **kw: v1_forensic(trace),
    "v5_tool_entropy": lambda trace, **kw: v5_tool_entropy(trace, threshold=kw.get("threshold", 0.8)),
    "v4_cross_layer": lambda trace, captures, **kw: v4_cross_layer(trace, captures),
}


def v1_or_v5(trace: dict, threshold: float = 0.8) -> dict:
    """Ensemble: fires if v1 OR v5 fires."""
    r1 = v1_forensic(trace)
    r5 = v5_tool_entropy(trace, threshold=threshold)
    fired = r1["fired"] or r5["fired"]
    fire_turn = None
    if fired:
        candidates = [t for t in (r1["fire_turn"], r5["fire_turn"]) if t is not None]
        fire_turn = min(candidates) if candidates else None
    return {"fired": fired, "fire_turn": fire_turn,
            "confidence": max(r1["confidence"], r5["confidence"]),
            "v1": r1, "v5": r5}


DETECTORS["v1_or_v5"] = lambda trace, **kw: v1_or_v5(trace, threshold=kw.get("threshold", 0.8))


def run_detector(detector_name: str, trace: dict, captures: dict | None = None, **kwargs) -> dict:
    """Dispatch to named detector."""
    if detector_name not in DETECTORS:
        raise ValueError(f"Unknown detector: {detector_name}. Available: {list(DETECTORS.keys())}")
    fn = DETECTORS[detector_name]
    if detector_name == "v4_cross_layer":
        return fn(trace, captures or {}, **kwargs)
    return fn(trace, **kwargs)
