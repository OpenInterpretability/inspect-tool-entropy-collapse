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

# Forensic v1: completion verbalizations targeting "task is done" self-referential statements.
# Calibrated against Tool-Entropy paper §6.1 (35% recall / 0% FP).
V1_FORENSIC_PATTERNS = [
    r"\bI[' ]?m\s+(done|finished|complete)\b",
    r"\bI[' ]?ve\s+(completed|finished|solved|fixed|addressed)\b",
    r"\b(solution|task|problem|issue|patch|fix)\s+is\s+(complete|done|solved|finished|ready|good)\b",
    r"\bsubmitting\s+(my|the|this)\s+(solution|answer|patch|fix)\b",
    r"\bI\s+(will|am going to|need to)\s+(submit|finalize|finish)\b",
    r"\bthis\s+(should\s+)?(solve|fix|resolve|address)\s+the\s+(problem|issue|task|bug)\b",
    r"\bready\s+to\s+(submit|finalize|finish)\b",
    r"\b(everything|all)\s+(looks|seems|appears)\s+(good|correct|right)\b",
    r"\b(my|the|this)\s+(implementation|change|patch|fix)\s+(should|will|now)\s+(work|pass|fix|resolve)\b",
    r"\b(let me|I will)\s+finalize\b",
    r"\bI[' ]?m\s+confident\s+(this|that)\b",
]


def v1_forensic(trace: dict, threshold_matches: int = 1) -> dict:
    """v1: post-hoc text monitor on assistant's final-turn completion language.

    Checks the LAST turn's assistant content + thinking for specific self-referential
    completion statements (not generic "done" which is too common in coding context).

    Tier 1 forensics: paper target 35% recall, 0% FP. Per paper, very specific patterns.
    """
    import re
    turns = trace.get("turns", [])
    if not turns:
        return {"fired": False, "fire_turn": None, "confidence": 0.0}

    # Check ONLY the last turn — that's where "I'm done, submitting" lives in WANDERING
    last = turns[-1]
    text = (last.get("content") or "") + " " + (last.get("thinking") or "")
    matches = sum(1 for pat in V1_FORENSIC_PATTERNS if re.search(pat, text, re.IGNORECASE))

    fired = matches >= threshold_matches
    fire_turn = len(turns) - 1 if fired else None
    confidence = min(1.0, matches / 3.0)
    return {"fired": fired, "fire_turn": fire_turn, "confidence": confidence,
            "matched_patterns": matches}


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
    """v5: tool-use entropy collapse (single measurement at trajectory end).

    Per Tool-Entropy paper §6: measure Shannon entropy of tool-call name distribution
    in the LAST `window` turns. Fires when entropy < threshold.

    Paper-reported: W/S median ratio 0.41 on Qwen3.6-27B, Mann-Whitney p=1.0e-6.
    Tier 3 autonomous: 55% recall, 5% FP at threshold=0.8 (paper-grade).

    Single repeated tool (ent=0.0) IS the canonical WANDERING signature — no guard against it.
    """
    turns = trace.get("turns", [])
    if len(turns) < window:
        return {"fired": False, "fire_turn": None, "confidence": 0.0,
                "tool_entropy_last10": None}

    ent = tool_entropy_last_n(trace, window=window)
    fired = ent < threshold
    # If fired, the "fire turn" is conceptually the start of the low-entropy window
    fire_turn = len(turns) - window if fired else None
    confidence = max(0.0, 1.0 - ent / threshold) if threshold > 0 else 0.0
    return {"fired": fired, "fire_turn": fire_turn, "confidence": confidence,
            "tool_entropy_last10": ent}


def v4_cross_layer(trace: dict, v4_cache: dict | None = None,
                    range_threshold: float = 0.30, **kw) -> dict:
    """v4: residual cross-layer probe disagreement (CACHED outputs).

    Loads precomputed v4 detector outputs from features/early_warning_v4_cross_layer.json
    in the HF dataset. Per-trajectory schema:
      - range_late: max-min probe score across L11/L23/L31/L43/L55 in late-half turns
      - std_late, sign_dis_late: variation metrics
      - ranges_per_turn: per-turn range over all layers
      - late_convergence_slope: trend in late half

    Fires when range_late > 0.30 (Tool-Entropy paper Tier 2 threshold).
    Paper: 65% recall, 30% FP @ 15-turn lead.
    """
    iid = trace.get("instance_id", "")
    if v4_cache is None or iid not in v4_cache:
        return {"fired": False, "fire_turn": None, "confidence": 0.0,
                "note": "v4 cache miss"}
    entry = v4_cache[iid]
    range_late = float(entry.get("range_late", 0.0))
    fired = range_late > range_threshold
    # Approximate fire_turn: first turn in late-half where ranges_per_turn exceeds threshold
    fire_turn = None
    if fired:
        rpt = entry.get("ranges_per_turn", [])
        n_turns = len(rpt)
        late_start = n_turns // 2
        for i in range(late_start, n_turns):
            if rpt[i] > range_threshold:
                fire_turn = i
                break
    return {"fired": fired, "fire_turn": fire_turn, "confidence": range_late,
            "range_late": range_late,
            "late_convergence_slope": entry.get("late_convergence_slope")}


DETECTORS = {
    "v1_forensic": lambda trace, **kw: v1_forensic(trace),
    "v5_tool_entropy": lambda trace, **kw: v5_tool_entropy(trace, threshold=kw.get("threshold", 0.8)),
    "v4_cross_layer": lambda trace, **kw: v4_cross_layer(trace, v4_cache=kw.get("v4_cache")),
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


def run_detector(detector_name: str, trace: dict, **kwargs) -> dict:
    """Dispatch to named detector. v4 expects v4_cache kwarg with precomputed outputs."""
    if detector_name not in DETECTORS:
        raise ValueError(f"Unknown detector: {detector_name}. Available: {list(DETECTORS.keys())}")
    return DETECTORS[detector_name](trace, **kwargs)
