# inspect-tool-entropy-collapse

[Inspect AI](https://inspect.aisi.org.uk/) eval for **WANDERING failure-mode detection** in agent trajectories, based on the Tool-Entropy Collapse signature.

Companion to:
- **Tool-Entropy Collapse paper** [(Zenodo DOI 10.5281/zenodo.20368601)](https://zenodo.org/records/20368601)
- **Two Honest Nulls paper #2** (draft in flight)
- **HF dataset** [(caiovicentino1/swebench-pro-qwen36-27b-phase6)](https://huggingface.co/datasets/caiovicentino1/swebench-pro-qwen36-27b-phase6)

## What this eval does

Given a corpus of multi-turn agent trajectories with known WANDERING/SUCCESS/LOCKED ground-truth labels, runs one or more WANDERING detectors and scores them on recall, false-positive rate, and lead-turns (how early before max_turns the detector fires).

This is a **monitoring eval**, not a capability eval: no model is invoked at eval time. The trajectories are pre-recorded; the eval scores a detector function over them. The detectors come from the Tool-Entropy paper:

| Detector | Channel | Tier (per paper) |
|---|---|---|
| `v1_forensic` | Post-hoc text monitor (assistant verbalizations) | Tier 1 forensics (35% recall, 0% FP) |
| `v4_cross_layer` | Cross-layer probe disagreement (L11/L23/L31/L43/L55) | Tier 2 advisory (65% recall, 30% FP, 15-turn lead) |
| `v5_tool_entropy` | Tool-use entropy collapse (Shannon entropy of last 10 tool calls) | Tier 3 autonomous (55% recall, 5% FP) |
| `v1_or_v5` | Logical OR of v1 + v5 | Tier 3 autonomous combined (70% recall, 5% FP) |

## Quick start

```bash
# 1. Install (uses uv for consistency with inspect_evals patterns)
git clone https://github.com/OpenInterpretability/inspect-tool-entropy-collapse.git
cd inspect-tool-entropy-collapse
uv sync

# 2. Run the eval (no model needed)
uv run inspect eval tool_entropy_collapse/tool_entropy_collapse --limit 10

# 3. Run with a specific detector
uv run inspect eval tool_entropy_collapse/tool_entropy_collapse \
    -T detector=v5_tool_entropy \
    --log-format eval --log-dir ./logs

# 4. Sweep detectors
for d in v1_forensic v4_cross_layer v5_tool_entropy v1_or_v5; do
    uv run inspect eval tool_entropy_collapse/tool_entropy_collapse -T detector=$d --log-dir ./logs/$d
done
```

## Dataset

The eval pulls trajectories from [caiovicentino1/swebench-pro-qwen36-27b-phase6](https://huggingface.co/datasets/caiovicentino1/swebench-pro-qwen36-27b-phase6) at a pinned revision SHA.

**N=99 trajectories** from Qwen3.6-27B on SWE-bench Pro:
- SUCCESS: 40
- LOCKED: 39
- WANDERING: 20

Sub-classes follow the operational definitions from the Tool-Entropy paper:
- **SUCCESS** = `finish_reason == "finish_tool"`
- **LOCKED** = `finish_reason == "max_turns"` AND probe stably < 0.30 by mean fraction 0.92 of trajectory
- **WANDERING** = `finish_reason == "max_turns"` AND probe stays > 0.70

⚠️ **Run-stability caveat** (see paper #2): the WANDERING category is not fully run-stable under temperature=1.0. Independent no-hook re-runs of trajectories classified as WANDERING (same RTX 6000 Pro Blackwell, same deterministic seed) emit `finish_tool` up to 35% of the time, while a fresh determinism check showed 0/5 — i.e., run-to-run sampling variance, not a hardware effect (no H100 was used). The detector eval scores against the **original RTX 6000 Pro Blackwell classification**; restricting analysis to the run-stable WANDERING subset is left to follow-up work.

## Metrics

- **`recall`**: TP / (TP + FN) — fraction of WANDERING agents the detector catches
- **`fp_rate`**: FP / (FP + TN) — fraction of SUCCESS agents falsely flagged
- **`accuracy`**: overall classification accuracy (informative but recall+FPR are primary)
- **`lead_turns`**: how many turns before `max_turns` the detector fires (only meaningful for TP)

## Citation

```bibtex
@misc{tool_entropy_2026,
  title={Tool-Entropy Collapse: A Cross-Architecture Signature of Agent WANDERING Failure},
  author={Vicentino, Caio},
  year={2026},
  doi={10.5281/zenodo.20368601},
  url={https://zenodo.org/records/20368601}
}
```

## License

Apache 2.0
