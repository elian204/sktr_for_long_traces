#!/usr/bin/env python3
"""Run actual-checkpoint B0 validity checks for one OOF DiffAct model."""

from __future__ import annotations

import argparse
import copy
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v2_common import (
    HALO_WIDTHS,
    INNER_FOLDS,
    OUTER_FOLDS,
    atomic_write_json,
    canonical_digest,
    file_sha256,
    load_json,
    verify_manifest,
    verify_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, choices=OUTER_FOLDS, required=True)
    parser.add_argument("--inner-fold", type=int, choices=INNER_FOLDS, required=True)
    parser.add_argument("--device", type=int, required=True)
    return parser.parse_args()


def parse_mapping(path: Path) -> list[str]:
    rows = [line.split(maxsplit=1) for line in path.read_text().splitlines() if line.strip()]
    mapping = {int(index): label for index, label in rows}
    if sorted(mapping) != list(range(48)):
        raise ValueError("Breakfast mapping must contain IDs 0..47")
    return [mapping[index] for index in range(48)]


def reconstruct_incumbent(
    corpus: pd.DataFrame, outer: int, inner: int, case_id: str, length: int
) -> np.ndarray:
    scoped = corpus[
        (corpus.outer_fold.astype(int) == outer)
        & (corpus.inner_fold.astype(int) == inner)
        & (corpus.case_id.astype(str) == case_id)
    ].sort_values(["segment_index", "start"], kind="mergesort")
    starts = scoped.start.to_numpy(dtype=int)
    ends = scoped.end.to_numpy(dtype=int)
    if len(scoped) == 0 or starts[0] != 0 or ends[-1] != length or np.any(starts[1:] != ends[:-1]):
        raise RuntimeError("OOF incumbent segments do not partition the selected case")
    result = np.empty(length, dtype=np.int16)
    for row in scoped.itertuples(index=False):
        result[int(row.start) : int(row.end)] = int(row.predicted_label)
    return result


def execute(study_dir: Path, outer: int, inner: int, physical_device: int) -> None:
    metadata = load_json(study_dir / "study_metadata.json")
    config = load_json(study_dir / "study_config.json")
    manifest = load_json(study_dir / "input_manifest.json")
    verify_source(metadata["source_provenance"])
    paths = verify_manifest(manifest)
    if not config["b0_sampling_allowed"] or not config.get("fable_approval_digest"):
        raise RuntimeError("V2 B0 sampling is review-blocked")
    if config["outer_test_open_allowed"]:
        raise RuntimeError("B0 study unexpectedly authorizes outer-test access")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("V2 B0 fail-closed GPU pinning failed")
    device = torch.device("cuda:0")
    diffact_root = Path(config["diffact_root"])
    sys.path.insert(0, str(diffact_root))
    from main import Trainer

    from masked_diffact_sampler import masked_ddim_sample

    task = next(
        row
        for row in config["tasks"]
        if int(row["outer_fold"]) == outer and int(row["inner_fold"]) == inner
    )
    task_config = load_json(Path(task["config"]))
    if task_config["diffusion_params"]["sampling_timesteps"] != 25:
        raise RuntimeError("B0 requires the trained model's 25-step sampling config")
    if task_config["postprocess"] != {"type": "median", "value": 15}:
        raise RuntimeError("B0 requires the official Breakfast median-15 postprocess")
    event_list = parse_mapping(paths["breakfast/mapping"])
    trainer = Trainer(
        copy.deepcopy(task_config["encoder_params"]),
        copy.deepcopy(task_config["decoder_params"]),
        copy.deepcopy(task_config["diffusion_params"]),
        event_list,
        task_config["sample_rate"],
        task_config["temporal_aug"],
        task_config["set_sampling_seed"],
        task_config["postprocess"],
        device,
    )
    state = torch.load(Path(task["checkpoint"]), map_location="cpu")
    trainer.model.load_state_dict(state, strict=True)
    trainer.model.eval().to(device)

    nested = load_json(paths["features/nested_manifest"])
    clean = dict(nested)
    expected_nested_digest = clean.pop("manifest_digest")
    if canonical_digest(clean) != expected_nested_digest:
        raise RuntimeError("Feature-manifest self-digest drift")
    feature_entries = {str(row["case_id"]): row for row in nested["features"]}
    spans = pd.read_csv(paths["v0/results/flagged_oof_spans.csv"])
    scoped = spans[
        (spans.outer_fold.astype(int) == outer) & (spans.inner_fold.astype(int) == inner)
    ].sort_values(["base_score", "v0_span_id"], ascending=[False, True])
    case_id = str(scoped.iloc[0].case_id)
    case_spans = scoped[scoped.case_id.astype(str) == case_id]
    intervals = [
        (int(row.selected_start), int(row.selected_end))
        for row in case_spans.itertuples(index=False)
    ]
    feature_entry = feature_entries[case_id]
    feature_path = Path(feature_entry["path"])
    if file_sha256(feature_path) != feature_entry["sha256"]:
        raise RuntimeError("Selected feature hash drift")
    feature = np.load(feature_path, allow_pickle=False).astype(np.float32)
    incumbent = reconstruct_incumbent(
        pd.read_csv(paths["v0/oof_segment_corpus"]), outer, inner, case_id, feature.shape[1]
    )
    video_feature = torch.from_numpy(feature).unsqueeze(0).to(device)
    with torch.no_grad():
        if trainer.model.use_instance_norm:
            video_feature = trainer.model.ins_norm(video_feature)
        _, backbone = trainer.model.encoder(video_feature, get_features=True)

    empty = masked_ddim_sample(
        trainer.model,
        backbone,
        incumbent,
        [],
        halo_width=16,
        t_start=500,
        pure_noise=False,
        seed=20260801,
        postprocess=task_config["postprocess"],
    )
    empty_identity = bool(np.array_equal(empty.labels, incumbent))
    replay_a = masked_ddim_sample(
        trainer.model,
        backbone,
        incumbent,
        intervals,
        halo_width=16,
        t_start=500,
        pure_noise=False,
        seed=20260801,
        postprocess=task_config["postprocess"],
    )
    replay_b = masked_ddim_sample(
        trainer.model,
        backbone,
        incumbent,
        intervals,
        halo_width=16,
        t_start=500,
        pure_noise=False,
        seed=20260801,
        postprocess=task_config["postprocess"],
    )
    seeded_replay = bool(
        np.array_equal(replay_a.probabilities, replay_b.probabilities)
        and np.array_equal(replay_a.labels, replay_b.labels)
    )
    exterior_rows: list[dict[str, Any]] = []
    for halo in HALO_WIDTHS:
        output = masked_ddim_sample(
            trainer.model,
            backbone,
            incumbent,
            intervals,
            halo_width=halo,
            t_start=500,
            pure_noise=False,
            seed=20260801 + halo,
            postprocess=task_config["postprocess"],
        )
        non_core = ~output.masks.core
        exterior_rows.append(
            {
                "halo_width": halo,
                "deployed_non_core_changed_frames": int(
                    np.sum(output.labels[non_core] != incumbent[non_core])
                ),
                "pre_restore_non_core_changed_frames": int(
                    np.sum(output.pre_restore_labels[non_core] != incumbent[non_core])
                ),
                "deployed_exterior_invariant": bool(
                    np.array_equal(output.labels[non_core], incumbent[non_core])
                ),
            }
        )
    checks = {
        "empty_mask_identity": empty_identity,
        "seeded_replay": seeded_replay,
        "postprocess_exterior_invariance": all(
            row["deployed_exterior_invariant"] for row in exterior_rows
        ),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "outer_fold": outer,
        "inner_fold": inner,
        "case_id": case_id,
        "core_intervals": intervals,
        "checks": checks,
        "exterior_diagnostics": exterior_rows,
        "replay_probability_sha256": __import__("hashlib").sha256(
            replay_a.probabilities.tobytes()
        ).hexdigest(),
        "checkpoint_sha256": task["checkpoint_sha256"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "outer_test_opened": False,
    }
    result["result_digest"] = canonical_digest(result)
    output = study_dir / "results" / f"outer{outer}_inner{inner}_b0.json"
    atomic_write_json(output, result)
    atomic_write_json(
        study_dir / "status" / f"outer{outer}_inner{inner}_b0.json",
        {"status": result["status"], "result_sha256": file_sha256(output)},
    )
    if result["status"] != "PASS":
        raise RuntimeError(f"V2 B0 validity failed: outer{outer}/inner{inner}")


def main() -> int:
    args = parse_args()
    study = args.study_dir.resolve()
    try:
        execute(study, args.outer_fold, args.inner_fold, args.device)
    except Exception as error:
        atomic_write_json(
            study / "status" / f"outer{args.outer_fold}_inner{args.inner_fold}_b0_failed.json",
            {
                "status": "failed",
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
