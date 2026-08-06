#!/usr/bin/env python3
"""Finalize Branch-B selection or reconcile the frozen selected exports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common import PUBLISHED_METRICS, RECONCILIATION
from metrics import aggregate_standard, per_video_metrics
from phase_b_selection_common import EPOCH_GRID, atomic_write_json, canonical_digest, file_sha256, load_json, normalize_case, read_nonempty_lines, select_best_checkpoint, verify_manifest, verify_source
from phase_b_training_common import DATASETS, parse_mapping


METRICS = ("acc", "edit", "f1@10", "f1@25", "f1@50")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("select-retrained", "reconcile"), default="reconcile")
    return parser.parse_args()


def reconcile_status(deltas: dict[str, float]) -> str:
    values = np.asarray([abs(deltas[key]) for key in METRICS])
    if values.max() <= RECONCILIATION["pass_max_abs_delta_pp"] and values.mean() <= RECONCILIATION["pass_mean_abs_delta_pp"]:
        return "PASS"
    if values.max() <= RECONCILIATION["notes_max_abs_delta_pp"] and values.mean() <= RECONCILIATION["notes_mean_abs_delta_pp"]:
        return "PASS_WITH_NOTES"
    return "FAIL"


def select_retrained(study: Path, config: dict) -> None:
    decision = load_json(study / "selection" / "step0_decision.json")
    if decision["selected_branch"] != "B_RETRAIN_WITH_HELD_OUT_VALIDATION":
        raise RuntimeError("Retrained selection requested outside Branch B")
    records: list[dict] = []
    for dataset, dataset_config in DATASETS.items():
        for fold in range(1, int(dataset_config["folds"]) + 1):
            retrain = load_json(study / "status" / f"retrain_{dataset}_fold{fold}.json")
            status = load_json(study / "status" / f"retrained_validation_{dataset}_fold{fold}.json")
            path = study / "retrained_validation_metrics" / dataset / f"fold{fold}.csv"
            if retrain.get("test_data_used_for_training_or_selection") is not False or status.get("test_data_used_for_selection") is not False:
                raise RuntimeError("Branch-B train/validation provenance contamination")
            if file_sha256(path) != status["output_sha256"]:
                raise RuntimeError("Retrained validation metric hash drift")
            frame = pd.read_csv(path)
            if set(frame["scope"]) != {"retrained_validation"} or set(frame["epoch"].astype(int)) != set(EPOCH_GRID):
                raise RuntimeError("Retrained selection grid/scope drift")
            best = select_best_checkpoint(frame.to_dict("records"))
            epoch = int(best["epoch"])
            checkpoint = study / "retrain" / dataset / f"fold{fold}" / "runtime" / "models" / dataset / f"split_{fold}" / f"epoch-{epoch}.model"
            if file_sha256(checkpoint) != retrain["checkpoint_sha256"][str(epoch)]:
                raise RuntimeError("Selected retrained checkpoint hash drift")
            records.append(
                {
                    "dataset": dataset, "fold": fold, "selected_epoch": epoch,
                    "validation_composite": float(best["composite"]), "validation_acc": float(best["acc"]),
                    "validation_edit": float(best["edit"]), "validation_f1@10": float(best["f1@10"]),
                    "validation_f1@25": float(best["f1@25"]), "validation_f1@50": float(best["f1@50"]),
                    "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint),
                    "selection_source": "genuine_held_out_training_fold_validation_only",
                    "carve_sha256": file_sha256(study / "carves" / dataset / f"fold{fold}" / "carved_validation.bundle"),
                    "training_regime": "retrained_train_minus_carve", "seen_video_validation_caveat": False,
                    "test_data_used_for_selection": False,
                }
            )
    csv_path = study / "selection" / "selection_records.csv"
    pd.DataFrame(records).sort_values(["dataset", "fold"]).to_csv(csv_path, index=False)
    payload = {
        "branch": decision["selected_branch"], "record_count": len(records), "records": records,
        "csv_sha256": file_sha256(csv_path), "seen_video_validation_caveat": False,
        "test_data_used_for_selection": False,
    }
    payload["selection_digest"] = canonical_digest(payload)
    atomic_write_json(study / "selection" / "selection_records.json", payload)
    print(f"selected_retrained_cells={len(records)}")


def target_for(data_root: Path, dataset: str, case_id: str, name_to_id: dict[str, int], sample_rate: int) -> np.ndarray:
    labels = read_nonempty_lines(data_root / dataset / "groundTruth" / f"{case_id}.txt")
    return np.asarray([name_to_id[label] for label in labels], dtype=np.int64)[::sample_rate]


def reconcile(study: Path, config: dict) -> None:
    selection = load_json(study / "selection" / "selection_records.json")
    if selection["record_count"] != 13 or selection.get("test_data_used_for_selection") is not False:
        raise RuntimeError("Selected checkpoint record set is invalid")
    completions: dict[tuple[str, int], dict] = {}
    for record in selection["records"]:
        dataset, fold = record["dataset"], int(record["fold"])
        complete = load_json(study / "status" / f"selected_export_{dataset}_fold{fold}.json")
        if complete.get("test_used_for_selection") is not False or complete["selection_digest"] != selection["selection_digest"]:
            raise RuntimeError("Selected export provenance drift")
        for relative, expected in complete["output_sha256"].items():
            if file_sha256(study / relative) != expected:
                raise RuntimeError(f"Selected export digest drift: {relative}")
        completions[(dataset, fold)] = complete
    if len(completions) != 13:
        raise RuntimeError("Expected 13 selected export completions")

    data_root = Path(config["data_root"])
    per_video_rows: list[dict] = []
    fold_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    reconciliation_rows: list[dict] = []
    for dataset, dataset_config in DATASETS.items():
        id_to_name, name_to_id = parse_mapping(data_root / dataset / "mapping.txt")
        background = {index for index, label in id_to_name.items() if label == "background"}
        sample_rate = int(dataset_config["sample_rate"])
        dataset_fold_metrics: list[dict[str, float]] = []
        for fold in range(1, int(dataset_config["folds"]) + 1):
            cases = [normalize_case(value) for value in read_nonempty_lines(data_root / dataset / "splits" / f"test.split{fold}.bundle")]
            selected_pairs: list[tuple[np.ndarray, np.ndarray]] = []
            arms = ("selected", "epoch100", "epoch30") if dataset == "breakfast" else ("selected",)
            for arm in arms:
                pairs: list[tuple[np.ndarray, np.ndarray]] = []
                root = study / "exports" / "mstcn2" / arm / dataset / f"fold{fold}" / "softmax"
                for case_id in cases:
                    probability = np.load(root / f"{case_id}.npy", allow_pickle=False)
                    target = target_for(data_root, dataset, case_id, name_to_id, sample_rate)
                    if probability.shape != (len(id_to_name), len(target)):
                        raise RuntimeError(f"Selected reconciliation alignment drift: {dataset}/{case_id}")
                    prediction = probability.argmax(axis=0).astype(np.int64)
                    pairs.append((prediction, target))
                    if arm == "selected":
                        per_video_rows.append({"backbone": "mstcn2", "dataset": dataset, "fold": fold, "case_id": case_id, "n_frames": len(target), **per_video_metrics(prediction, target, background)})
                metrics = aggregate_standard(pairs, background)
                if arm == "selected":
                    selected_pairs = pairs
                    dataset_fold_metrics.append(metrics)
                    fold_rows.append({"backbone": "mstcn2", "dataset": dataset, "fold": fold, "n_cases": len(pairs), **metrics})
                if dataset == "breakfast":
                    sensitivity_rows.append({"backbone": "mstcn2", "dataset": dataset, "fold": fold, "checkpoint_choice": arm, "n_cases": len(pairs), **metrics})
            if not selected_pairs:
                raise RuntimeError("Selected reconciliation arm missing")
        observed = {key: float(np.mean([row[key] for row in dataset_fold_metrics])) for key in METRICS}
        published = PUBLISHED_METRICS[("mstcn2", dataset)]
        deltas = {key: observed[key] - float(published[key]) for key in METRICS}
        reconciliation_rows.append(
            {
                "backbone": "mstcn2", "dataset": dataset, "reconciliation_status": reconcile_status(deltas),
                "mean_abs_delta_pp": float(np.mean([abs(value) for value in deltas.values()])),
                "max_abs_delta_pp": float(max(abs(value) for value in deltas.values())),
                **{f"observed_{key}": observed[key] for key in METRICS},
                **{f"published_{key}": float(published[key]) for key in METRICS},
                **{f"delta_{key}": deltas[key] for key in METRICS},
            }
        )
    result = study / "reconciliation"
    pd.DataFrame(per_video_rows).sort_values(["dataset", "fold", "case_id"]).to_csv(result / "mstcn2_selected_per_video.csv", index=False)
    pd.DataFrame(fold_rows).sort_values(["dataset", "fold"]).to_csv(result / "mstcn2_selected_per_fold.csv", index=False)
    pd.DataFrame(sensitivity_rows).sort_values(["fold", "checkpoint_choice"]).to_csv(result / "breakfast_mstcn2_checkpoint_sensitivity.csv", index=False)
    reconciliation_frame = pd.DataFrame(reconciliation_rows).sort_values("dataset")
    reconciliation_frame.to_csv(result / "mstcn2_selected_paper_reconciliation.csv", index=False)
    decision = {
        "status": "complete", "selection_branch": selection["branch"],
        "published_numbers_remain_reference": True, "trajectory_is_deviation_explanation_only": True,
        "phase_c_allowed": False, "phase_c_next_action": "Fable digest/reconciliation review",
        "all_13_cells_exported": True, "test_data_used_for_selection": False,
    }
    decision["decision_digest"] = canonical_digest(decision)
    atomic_write_json(result / "selected_reconciliation_decision.json", decision)
    lines = [
        "# Phase-B val-selected reconciliation", "",
        f"Selection branch: **{selection['branch']}**. Published metrics remain the comparison reference.", "",
        "The per-epoch test trajectory is a descriptive deviation exhibit and was firewalled from checkpoint selection.", "",
        "| Dataset | Status | Mean abs delta | Max abs delta | Acc | Edit | F1@10 | F1@25 | F1@50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in reconciliation_frame.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['reconciliation_status']} | {row['mean_abs_delta_pp']:.2f} | {row['max_abs_delta_pp']:.2f} | "
            f"{row['observed_acc']:.2f} ({row['delta_acc']:+.2f}) | {row['observed_edit']:.2f} ({row['delta_edit']:+.2f}) | "
            f"{row['observed_f1@10']:.2f} ({row['delta_f1@10']:+.2f}) | {row['observed_f1@25']:.2f} ({row['delta_f1@25']:+.2f}) | "
            f"{row['observed_f1@50']:.2f} ({row['delta_f1@50']:+.2f}) |"
        )
    (result / "findings.md").write_text("\n".join(lines) + "\n")
    outputs = [path for path in result.iterdir() if path.is_file()]
    atomic_write_json(result / "selected_reconciliation_complete.json", {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(), "output_sha256": {path.name: file_sha256(path) for path in outputs}})


def main() -> int:
    args = parse_args()
    study = args.study_dir.resolve()
    metadata = load_json(study / "study_metadata.json")
    config = load_json(study / "study_config.json")
    verify_source(metadata["source_provenance"])
    verify_manifest(load_json(study / "input_manifest.json"), full_hash=False)
    if not config.get("fable_approval_digest") or config["phase_c_allowed"]:
        raise RuntimeError("Selected finalization is not approved or Phase C is already open")
    if args.mode == "select-retrained":
        select_retrained(study, config)
    else:
        reconcile(study, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
