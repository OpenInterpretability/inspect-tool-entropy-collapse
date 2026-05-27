"""Load Phase 6 SWE-bench Pro trajectories from HF dataset.

Pulls trajectories + features + labels from caiovicentino1/swebench-pro-qwen36-27b-phase6
at a pinned revision SHA.
"""
from __future__ import annotations
import json
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from inspect_ai.dataset import MemoryDataset, Sample

REPO_ID = "caiovicentino1/swebench-pro-qwen36-27b-phase6"
REVISION = "a12189c8b92102084140a2dac5f9a27ddd879ace"  # pinned 2026-05-27 (420 files, 910 MB)


def derive_sub_class(traj: dict) -> str:
    """Derive WANDERING/SUCCESS/LOCKED from inflection_results entry."""
    if traj["label"] == 1:
        return "success"
    if traj.get("lock_fail_0.40") is not None:
        return "locked"
    return "wandering"


def load_trajectory_dataset(
    include_classes: tuple[str, ...] = ("wandering", "success"),
    repo_id: str = REPO_ID,
    revision: str = REVISION,
) -> MemoryDataset:
    """Load Phase 6 trajectories filtered to specified sub-classes.

    Args:
        include_classes: which sub-classes to include (default: WANDERING + SUCCESS;
            LOCKED is excluded by default because it's externally identical to WANDERING
            but mechanistically different).
        repo_id: HF dataset repo.
        revision: pinned SHA (required for reproducibility).

    Returns:
        MemoryDataset of Sample objects, where each Sample's metadata holds:
          - iid: SWE-bench Pro instance ID
          - sub_class: "wandering" | "success" | "locked"
          - trace_path: path to downloaded trace JSON
          - captures_path: path to downloaded residual safetensors
          - finish_reason: original finish_reason string
          - n_turns: number of turns
    """
    # Pull metadata files (small)
    infl_path = hf_hub_download(repo_id=repo_id, filename="features/inflection_results.json",
                                 repo_type="dataset", revision=revision)
    infl = json.load(open(infl_path))

    p6_path = hf_hub_download(repo_id=repo_id, filename="phase6_results.json",
                               repo_type="dataset", revision=revision)
    p6 = json.load(open(p6_path))

    samples = []
    for traj in infl["per_trajectory"]:
        iid = traj["iid"]
        sub_class = derive_sub_class(traj)
        if sub_class not in include_classes:
            continue

        p6_entry = p6.get(iid, {})
        trace_filename = Path(p6_entry.get("trace_path", f"traces/{iid}.json")).name
        captures_filename = Path(p6_entry.get("captures_safetensors", f"captures/{iid}.safetensors")).name

        samples.append(Sample(
            id=iid,
            input=f"Detect WANDERING in trajectory {iid}",
            target=sub_class,  # gold label
            metadata={
                "iid": iid,
                "sub_class": sub_class,
                "trace_filename": trace_filename,
                "captures_filename": captures_filename,
                "finish_reason": p6_entry.get("finish_reason"),
                "n_turns": traj["n_turns"],
                "label": traj["label"],
                "score_trajectory": traj.get("score_trajectory", []),
                "lock_fail_0.40": traj.get("lock_fail_0.40"),
                "lock_succ_0.70": traj.get("lock_succ_0.70"),
            },
        ))

    return MemoryDataset(samples=samples, name="phase6_trajectories")


def load_trace(iid: str, repo_id: str = REPO_ID, revision: str = REVISION) -> dict:
    """Download + load a single trace JSON."""
    path = hf_hub_download(repo_id=repo_id, filename=f"traces/{iid}.json",
                            repo_type="dataset", revision=revision)
    return json.load(open(path))


def load_captures(iid: str, repo_id: str = REPO_ID, revision: str = REVISION):
    """Download + load residual safetensors for one trajectory."""
    import safetensors.torch as st
    path = hf_hub_download(repo_id=repo_id, filename=f"captures/{iid}.safetensors",
                            repo_type="dataset", revision=revision)
    return st.load_file(path)
