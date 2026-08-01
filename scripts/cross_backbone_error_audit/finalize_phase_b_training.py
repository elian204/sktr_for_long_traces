#!/usr/bin/env python3
"""Consolidate and reconcile the 13 approved MS-TCN++ retrainings."""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common import PUBLISHED_METRICS, RECONCILIATION
from metrics import aggregate_standard, per_video_metrics
from phase_b_training_common import (
    DATASETS,
    atomic_write_json,
    canonical_digest,
    file_sha256,
    load_json,
    normalize_case,
    parse_mapping,
    read_nonempty_lines,
    verify_manifest,
    verify_source,
)


METRICS = ("acc", "edit", "f1@10", "f1@25", "f1@50")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    return parser.parse_args()


def reconciliation_status(deltas: dict[str, float]) -> str:
    values = np.asarray([abs(deltas[key]) for key in METRICS])
    if values.max() <= RECONCILIATION["pass_max_abs_delta_pp"] and values.mean() <= RECONCILIATION["pass_mean_abs_delta_pp"]:
        return "PASS"
    if values.max() <= RECONCILIATION["notes_max_abs_delta_pp"] and values.mean() <= RECONCILIATION["notes_mean_abs_delta_pp"]:
        return "PASS_WITH_NOTES"
    return "FAIL"


def ground_truth(data_root: Path, dataset: str, case_id: str, name_to_id: dict[str, int], sample_rate: int) -> np.ndarray:
    labels = read_nonempty_lines(data_root / dataset / "groundTruth" / f"{case_id}.txt")
    return np.asarray([name_to_id[label] for label in labels], dtype=np.int64)[::sample_rate]


def execute(study: Path) -> None:
    metadata = load_json(study / "study_metadata.json")
    config = load_json(study / "study_config.json")
    manifest = load_json(study / "input_manifest.json")
    verify_source(metadata["source_provenance"])
    verify_manifest(manifest, full_hash=False)
    if config["phase_c_allowed"] or not config["gpu_training_allowed"]:
        raise RuntimeError("Invalid Phase-B finalization authorization")
    existing_complete = study / "results" / "phase_b_complete.json"
    if existing_complete.is_file():
        complete = load_json(existing_complete)
        for name, expected in complete["output_sha256"].items():
            if file_sha256(study / "results" / name) != expected:
                raise RuntimeError(f"Completed Phase-B result drift: {name}")
        print("Phase-B finalization already complete")
        return

    completions: dict[tuple[str, int], dict] = {}
    for task in config["tasks"]:
        dataset, fold = str(task["dataset"]), int(task["fold"])
        complete = load_json(study / "status" / f"{dataset}_fold{fold}_complete.json")
        if complete["status"] != "complete":
            raise RuntimeError(f"Incomplete Phase-B task: {dataset}/fold{fold}")
        for relative, expected in complete["output_sha256"].items():
            if file_sha256(study / relative) != expected:
                raise RuntimeError(f"Phase-B output hash drift: {relative}")
        completions[(dataset, fold)] = complete
    if len(completions) != 13:
        raise RuntimeError("Expected 13 Phase-B completions")

    export_root = study / "exports" / "mstcn2"
    if export_root.exists():
        shutil.rmtree(export_root)
    data_root = Path(config["data_root"])
    per_video_rows: list[dict] = []
    fold_rows: list[dict] = []
    reconciliation_rows: list[dict] = []
    for dataset, dataset_config in DATASETS.items():
        destination = export_root / "results" / dataset / "softmax"
        destination.mkdir(parents=True, exist_ok=True)
        model_root = export_root / "models" / dataset
        id_to_name, name_to_id = parse_mapping(data_root / dataset / "mapping.txt")
        background = {index for index, label in id_to_name.items() if label == "background"}
        sample_rate = int(dataset_config["sample_rate"])
        dataset_fold_metrics: list[dict[str, float]] = []
        for fold in range(1, int(dataset_config["folds"]) + 1):
            source = study / "cells" / dataset / f"fold{fold}" / "export" / "softmax"
            if fold == 1:
                shutil.copy2(source / "mapping.txt", destination / "mapping.txt")
                shutil.copy2(source / "video_index_map.txt", destination / "video_index_map.txt")
            elif file_sha256(source / "mapping.txt") != file_sha256(destination / "mapping.txt") or file_sha256(source / "video_index_map.txt") != file_sha256(destination / "video_index_map.txt"):
                raise RuntimeError(f"Fold export metadata drift: {dataset}/fold{fold}")
            index = {
                case_id: int(value)
                for value, case_id in (
                    line.split("\t", maxsplit=1)
                    for line in read_nonempty_lines(source / "video_index_map.txt")
                )
            }
            cases: list[tuple[np.ndarray, np.ndarray]] = []
            for raw_case in read_nonempty_lines(data_root / dataset / "splits" / f"test.split{fold}.bundle"):
                case_id = normalize_case(raw_case)
                source_probability = source / f"{index[case_id]}.npy"
                target_probability = destination / source_probability.name
                if target_probability.exists():
                    raise RuntimeError(f"Duplicate Phase-B test export: {dataset}/{case_id}")
                shutil.copy2(source_probability, target_probability)
                probability = np.load(target_probability, allow_pickle=False)
                target = ground_truth(data_root, dataset, case_id, name_to_id, sample_rate)
                if probability.shape != (len(id_to_name), len(target)):
                    raise RuntimeError(f"Final frame alignment drift: {dataset}/{case_id}")
                prediction = probability.argmax(axis=0).astype(np.int64)
                metrics = per_video_metrics(prediction, target, background)
                per_video_rows.append(
                    {"backbone": "mstcn2", "dataset": dataset, "fold": fold, "case_id": case_id, "n_frames": len(target), **metrics}
                )
                cases.append((prediction, target))
            fold_metric = aggregate_standard(cases, background)
            dataset_fold_metrics.append(fold_metric)
            fold_rows.append({"backbone": "mstcn2", "dataset": dataset, "fold": fold, "n_cases": len(cases), **fold_metric})
            checkpoint = study / "cells" / dataset / f"fold{fold}" / "runtime" / "models" / dataset / f"split_{fold}" / "epoch-100.model"
            checkpoint_target = model_root / f"split_{fold}" / "epoch-100.model"
            checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
            os.link(checkpoint, checkpoint_target)
        observed = {key: float(np.mean([row[key] for row in dataset_fold_metrics])) for key in METRICS}
        published = PUBLISHED_METRICS[("mstcn2", dataset)]
        deltas = {key: observed[key] - float(published[key]) for key in METRICS}
        reconciliation_rows.append(
            {
                "backbone": "mstcn2",
                "dataset": dataset,
                "reconciliation_status": reconciliation_status(deltas),
                "mean_abs_delta_pp": float(np.mean([abs(value) for value in deltas.values()])),
                "max_abs_delta_pp": float(max(abs(value) for value in deltas.values())),
                **{f"observed_{key}": observed[key] for key in METRICS},
                **{f"published_{key}": float(published[key]) for key in METRICS},
                **{f"delta_{key}": deltas[key] for key in METRICS},
            }
        )

    result_dir = study / "results"
    pd.DataFrame(per_video_rows).sort_values(["dataset", "fold", "case_id"]).to_csv(result_dir / "mstcn2_per_video.csv", index=False)
    pd.DataFrame(fold_rows).sort_values(["dataset", "fold"]).to_csv(result_dir / "mstcn2_per_fold.csv", index=False)
    reconciliation = pd.DataFrame(reconciliation_rows).sort_values("dataset")
    reconciliation.to_csv(result_dir / "mstcn2_paper_reconciliation.csv", index=False)
    passed = bool(reconciliation.reconciliation_status.isin(["PASS", "PASS_WITH_NOTES"]).all())
    decision = {
        "phase_b_status": "PASS" if passed else "FAIL",
        "reconciled_dataset_count": len(reconciliation),
        "all_13_cells_exported": True,
        "phase_c_allowed": False,
        "phase_c_next_action": "Fable reviews Phase-B reconciliation before Phase C opens",
        "sealed_studies_opened": False,
    }
    decision["decision_digest"] = canonical_digest(decision)
    atomic_write_json(result_dir / "phase_b_decision.json", decision)
    lines = [
        "# MS-TCN++ Phase-B reconciliation",
        "",
        f"**Phase-B status: {decision['phase_b_status']}.** All 13 official folds were trained for 100 epochs with the authors' configuration and exported at the final-stage softmax.",
        "",
        "| Dataset | Status | Mean abs delta | Max abs delta | Acc | Edit | F1@10 | F1@25 | F1@50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in reconciliation.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['reconciliation_status']} | {row['mean_abs_delta_pp']:.2f} | {row['max_abs_delta_pp']:.2f} | "
            f"{row['observed_acc']:.2f} ({row['delta_acc']:+.2f}) | {row['observed_edit']:.2f} ({row['delta_edit']:+.2f}) | "
            f"{row['observed_f1@10']:.2f} ({row['delta_f1@10']:+.2f}) | {row['observed_f1@25']:.2f} ({row['delta_f1@25']:+.2f}) | "
            f"{row['observed_f1@50']:.2f} ({row['delta_f1@50']:+.2f}) |"
        )
    lines.extend(["", "Phase C remains closed until this reconciliation package is reviewed."])
    (result_dir / "findings.md").write_text("\n".join(lines) + "\n")
    outputs = [path for path in result_dir.iterdir() if path.is_file()]
    complete = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "decision_digest": decision["decision_digest"],
        "output_sha256": {path.name: file_sha256(path) for path in outputs},
    }
    atomic_write_json(result_dir / "phase_b_complete.json", complete)


def main() -> int:
    args = parse_args()
    execute(args.study_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
