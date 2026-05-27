"""Data-driven calibration of v1_forensic and v4_cross_layer detectors.

v4: sweep `range_threshold` to find value that gives paper-grade 65% recall / 30% FP
v1: examine WANDERING final-turn texts to extract recurring completion phrases
"""
from __future__ import annotations
import json
import re
import sys
from collections import Counter
from huggingface_hub import hf_hub_download

REPO = "caiovicentino1/swebench-pro-qwen36-27b-phase6"
REVISION = "a12189c8b92102084140a2dac5f9a27ddd879ace"


def load_labels():
    p = hf_hub_download(REPO, "features/inflection_results.json", repo_type="dataset", revision=REVISION)
    infl = json.load(open(p))
    labels = {}
    for t in infl["per_trajectory"]:
        if t["label"] == 1:
            labels[t["iid"]] = "success"
        elif t.get("lock_fail_0.40") is not None:
            labels[t["iid"]] = "locked"
        else:
            labels[t["iid"]] = "wandering"
    return labels


def calibrate_v4(labels):
    print("=" * 70)
    print("V4 THRESHOLD SWEEP")
    print("=" * 70)
    p = hf_hub_download(REPO, "features/early_warning_v4_cross_layer.json",
                         repo_type="dataset", revision=REVISION)
    raw = json.load(open(p))
    if isinstance(raw, list):
        cache = {e["iid"]: e for e in raw}
    elif "per_trajectory" in raw:
        cache = {e["iid"]: e for e in raw["per_trajectory"]}
    else:
        cache = raw

    # Build per-sub-class range_late distributions
    wand_ranges = []
    succ_ranges = []
    lock_ranges = []
    for iid, entry in cache.items():
        rl = float(entry.get("range_late", 0.0))
        sub = labels.get(iid)
        if sub == "wandering":
            wand_ranges.append(rl)
        elif sub == "success":
            succ_ranges.append(rl)
        elif sub == "locked":
            lock_ranges.append(rl)

    print(f"\nrange_late distribution (n={len(cache)}):")
    print(f"  WANDERING (n={len(wand_ranges)}): min={min(wand_ranges):.3f}, "
          f"median={sorted(wand_ranges)[len(wand_ranges)//2]:.3f}, "
          f"max={max(wand_ranges):.3f}")
    print(f"  SUCCESS   (n={len(succ_ranges)}): min={min(succ_ranges):.3f}, "
          f"median={sorted(succ_ranges)[len(succ_ranges)//2]:.3f}, "
          f"max={max(succ_ranges):.3f}")
    print(f"  LOCKED    (n={len(lock_ranges)}): min={min(lock_ranges):.3f}, "
          f"median={sorted(lock_ranges)[len(lock_ranges)//2]:.3f}, "
          f"max={max(lock_ranges):.3f}")

    print(f"\nSweep thresholds (WANDERING vs SUCCESS only — LOCKED excluded):")
    print(f"  {'threshold':>10} {'WAND_fire':>10} {'recall':>8} {'SUCC_fire':>10} {'fp':>8}")
    for thr in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        w_fire = sum(1 for r in wand_ranges if r > thr)
        s_fire = sum(1 for r in succ_ranges if r > thr)
        recall = w_fire / max(1, len(wand_ranges))
        fp = s_fire / max(1, len(succ_ranges))
        marker = ""
        if 0.60 <= recall <= 0.70 and 0.25 <= fp <= 0.35:
            marker = " ← PAPER GRADE (65%/30%)"
        print(f"  {thr:>10.2f} {w_fire:>10d} {recall:>8.1%} {s_fire:>10d} {fp:>8.1%}{marker}")


def calibrate_v1(labels):
    print("\n" + "=" * 70)
    print("V1 FORENSIC PATTERN MINING — WANDERING final-turn text")
    print("=" * 70)

    wand_iids = [iid for iid, s in labels.items() if s == "wandering"]
    succ_iids = [iid for iid, s in labels.items() if s == "success"]

    def get_final_text(iid):
        p = hf_hub_download(REPO, f"traces/{iid}.json", repo_type="dataset", revision=REVISION)
        trace = json.load(open(p))
        last = trace["turns"][-1] if trace.get("turns") else {}
        return ((last.get("content") or "") + " " + (last.get("thinking") or "")).strip()

    print(f"\nSampling final-turn text from {len(wand_iids)} WANDERING + {len(succ_iids)} SUCCESS trajectories...")
    wand_texts = [get_final_text(iid) for iid in wand_iids]
    succ_texts = [get_final_text(iid) for iid in succ_iids]

    # Extract n-grams (3-word phrases) from WANDERING texts that are RARE in SUCCESS
    def ngrams(text, n=3):
        words = re.findall(r"\b[a-z]+\b", text.lower())
        return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]

    wand_ngrams = Counter()
    for t in wand_texts:
        for ng in ngrams(t):
            wand_ngrams[ng] += 1
    succ_ngrams = Counter()
    for t in succ_texts:
        for ng in ngrams(t):
            succ_ngrams[ng] += 1

    # Find n-grams that appear in WANDERING but NOT in SUCCESS (or very rarely)
    discriminative = []
    for ng, w_count in wand_ngrams.most_common(200):
        s_count = succ_ngrams.get(ng, 0)
        w_rate = w_count / len(wand_texts)
        s_rate = s_count / max(1, len(succ_texts))
        if w_rate >= 0.10 and s_rate <= 0.05:  # in ≥10% of WANDERING, ≤5% of SUCCESS
            discriminative.append((ng, w_count, s_count, w_rate, s_rate))

    print(f"\nDiscriminative 3-grams (≥10% WANDERING, ≤5% SUCCESS):")
    print(f"  {'phrase':<50} {'W_count':>8} {'S_count':>8} {'W_rate':>8} {'S_rate':>8}")
    for ng, wc, sc, wr, sr in sorted(discriminative, key=lambda x: x[3], reverse=True)[:30]:
        print(f"  {ng:<50} {wc:>8d} {sc:>8d} {wr:>8.1%} {sr:>8.1%}")

    print(f"\nSample raw WANDERING final-turn texts (first 200 chars each):")
    for i, t in enumerate(wand_texts[:5]):
        print(f"\n  --- WANDERING {wand_iids[i][:50]} ---")
        print(f"  {t[:200].replace(chr(10), ' / ')}")


def main():
    labels = load_labels()
    print(f"Loaded {len(labels)} trajectory labels")
    print(f"  WANDERING: {sum(1 for v in labels.values() if v == 'wandering')}")
    print(f"  SUCCESS:   {sum(1 for v in labels.values() if v == 'success')}")
    print(f"  LOCKED:    {sum(1 for v in labels.values() if v == 'locked')}")
    calibrate_v4(labels)
    calibrate_v1(labels)


if __name__ == "__main__":
    main()
