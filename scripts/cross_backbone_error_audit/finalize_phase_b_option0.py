#!/usr/bin/env python3
"""Score author-ASFormer exports and freeze the residual Phase-B training bill."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from metrics import aggregate_standard, per_video_metrics
from phase_b_option0_common import (
    ASFORMER_BREAKFAST_PUBLISHED,
    OUTER_FOLDS,
    RECONCILIATION,
    atomic_write_json,
    canonical_digest,
    feature_manifest_entries,
    file_sha256,
    load_json,
    normalize_case,
    parse_mapping,
    read_nonempty_lines,
    residual_plan,
    verify_flat_manifest,
    verify_source_provenance,
)


METRICS = ("acc", "edit", "f1@10", "f1@25", "f1@50")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    return parser.parse_args()


def reconciliation_status(deltas: dict[str, float]) -> str:
    absolute = np.asarray([abs(float(deltas[metric])) for metric in METRICS])
    if (
        float(absolute.max()) <= RECONCILIATION["pass_max_abs_delta_pp"]
        and float(absolute.mean()) <= RECONCILIATION["pass_mean_abs_delta_pp"]
    ):
        return "PASS"
    if (
        float(absolute.max()) <= RECONCILIATION["notes_max_abs_delta_pp"]
        and float(absolute.mean()) <= RECONCILIATION["notes_mean_abs_delta_pp"]
    ):
        return "PASS_WITH_NOTES"
    return "FAIL"


def execute(study_dir: Path) -> None:
    metadata = load_json(study_dir / "study_metadata.json")
    config = load_json(study_dir / "study_config.json")
    manifest = load_json(study_dir / "input_manifest.json")
    verify_source_provenance(metadata["source_provenance"])
    paths = verify_flat_manifest(manifest)
    if config["gpu_training_allowed"] or config["residual_training_launch_allowed"]:
        raise RuntimeError("Option-0 study unexpectedly authorizes training")

    nested = load_json(paths["breakfast/feature_manifest"])
    _, ground_truth = feature_manifest_entries(nested)
    _, name_to_id = parse_mapping(paths["breakfast/mapping"])
    fold_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    all_seen: set[str] = set()
    for fold in OUTER_FOLDS:
        completion = load_json(study_dir / "status" / f"fold{fold}.json")
        if completion.get("status") != "complete" or int(completion["fold"]) != fold:
            raise RuntimeError(f"Fold {fold} export is incomplete")
        export_manifest = study_dir / "exports" / f"fold{fold}" / "export_manifest.csv"
        if file_sha256(export_manifest) != completion["export_manifest_sha256"]:
            raise RuntimeError(f"Fold {fold} export manifest drift")
        records = pd.read_csv(export_manifest)
        expected_cases = [
            normalize_case(value)
            for value in read_nonempty_lines(paths[f"breakfast/fold{fold}/test_bundle"])
        ]
        if set(records.case_id.astype(str)) != set(expected_cases) or len(records) != len(expected_cases):
            raise RuntimeError(f"Fold {fold} exported-case coverage drift")
        cases: list[tuple[np.ndarray, np.ndarray]] = []
        for row in records.itertuples(index=False):
            case_id = str(row.case_id)
            if case_id in all_seen:
                raise RuntimeError(f"Case appears in more than one test fold: {case_id}")
            all_seen.add(case_id)
            probability_path = Path(row.probability_path)
            if file_sha256(probability_path) != row.probability_sha256:
                raise RuntimeError(f"Probability output drift: {case_id}")
            probability = np.load(probability_path, allow_pickle=False)
            gt_entry = ground_truth[case_id]
            gt_path = Path(gt_entry["path"])
            if file_sha256(gt_path) != gt_entry["sha256"]:
                raise RuntimeError(f"Ground-truth input drift: {case_id}")
            target = np.asarray(
                [name_to_id[label] for label in read_nonempty_lines(gt_path)], dtype=np.int64
            )
            if probability.shape != (len(name_to_id), len(target)):
                raise RuntimeError(f"Finalization frame drift: {case_id}")
            prediction = probability.argmax(axis=0).astype(np.int64)
            cases.append((prediction, target))
            video_rows.append(
                {
                    "fold": fold,
                    "case_id": case_id,
                    "frames": len(target),
                    "errors": int(np.sum(prediction != target)),
                    **per_video_metrics(prediction, target, set()),
                }
            )
        fold_rows.append({"fold": fold, "cases": len(cases), **aggregate_standard(cases, set())})

    fold_frame = pd.DataFrame(fold_rows).sort_values("fold")
    videos = pd.DataFrame(video_rows).sort_values(["fold", "case_id"])
    observed = {metric: float(fold_frame[metric].mean()) for metric in METRICS}
    deltas = {
        metric: float(observed[metric] - ASFORMER_BREAKFAST_PUBLISHED[metric])
        for metric in METRICS
    }
    status = reconciliation_status(deltas)
    residual = residual_plan(status)
    decision = {
        "option0_asformer_breakfast_status": status,
        "observed_fold_mean": observed,
        "published": ASFORMER_BREAKFAST_PUBLISHED,
        "delta_pp": deltas,
        "mean_abs_delta_pp": float(np.mean([abs(value) for value in deltas.values()])),
        "max_abs_delta_pp": float(max(abs(value) for value in deltas.values())),
        "mstcn2_official_checkpoint_status": "UNAVAILABLE_IN_OFFICIAL_REPOSITORY",
        "residual_training_plan": residual,
        "residual_training_launch_authorized": False,
        "explicit_approval_required": True,
        "phase_c_allowed": False,
        "sealed_studies_opened": False,
    }
    decision["decision_digest"] = canonical_digest(decision)
    result_dir = study_dir / "results"
    fold_frame.to_csv(result_dir / "asformer_breakfast_per_fold.csv", index=False)
    videos.to_csv(result_dir / "asformer_breakfast_per_video.csv", index=False)
    atomic_write_json(result_dir / "phase_b_option0_decision.json", decision)
    metric_text = " | ".join(
        f"{metric} {observed[metric]:.2f} ({deltas[metric]:+.2f})" for metric in METRICS
    )
    lines = [
        "# Cross-backbone Phase B Option 0 findings",
        "",
        f"**Author-ASFormer Breakfast reconciliation: {status}.** {metric_text}.",
        "",
        "The official ASFormer archive supplies all 13 released fold checkpoints and was evaluated with the untouched author model code. MS-TCN++ exposes no releases, tags, or checkpoint link in its official repository; its Zenodo artifact contains features only.",
        "",
        f"**Residual bill:** {residual['cells']} training cells, approximately {residual['active_gpu_hours']:.1f} active GPU-hours, {residual['buffered_wall_hours']} wall-hours with four lanes, and {residual['historical_checkpoint_storage_gib']:.1f} GiB of historical checkpoint storage.",
        "",
        "No residual training is authorized. The planning estimate must be refreshed and explicitly approved before launch; Phase C remains closed.",
    ]
    (result_dir / "findings.md").write_text("\n".join(lines) + "\n")
    outputs = sorted(path for path in result_dir.iterdir() if path.is_file())
    complete = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest_digest": manifest["manifest_digest"],
        "source_digest": metadata["source_provenance"]["source_digest"],
        "decision_digest": decision["decision_digest"],
        "gpu_training_used": False,
        "sealed_studies_opened": False,
        "output_sha256": {path.name: file_sha256(path) for path in outputs},
    }
    atomic_write_json(result_dir / "phase_b_option0_complete.json", complete)


def main() -> int:
    args = parse_args()
    execute(args.study_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
