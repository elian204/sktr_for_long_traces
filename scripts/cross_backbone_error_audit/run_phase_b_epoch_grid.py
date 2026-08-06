#!/usr/bin/env python3
"""Evaluate the pre-registered checkpoint grid on validation or descriptive test."""

from __future__ import annotations

import argparse
import importlib.util
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import aggregate_standard, segments
from phase_b_selection_common import (
    COMPOSITE_METRICS,
    EPOCH_GRID,
    atomic_write_json,
    canonical_digest,
    file_sha256,
    load_json,
    normalize_case,
    read_nonempty_lines,
    verify_manifest,
    verify_source,
)
from phase_b_training_common import DATASETS, OFFICIAL_CONFIG, parse_mapping


SCOPES = ("carved_validation", "retrained_validation", "test_trajectory")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=SCOPES, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", type=int, choices=(0, 1, 2, 3), required=True)
    return parser.parse_args()


def load_model_class(model_source: Path):
    spec = importlib.util.spec_from_file_location("step0_mstcn2_model", model_source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load hash-locked MS-TCN++ model source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MS_TCN2


def verify_parent_row(rows: dict[str, dict], role: str) -> Path:
    if role not in rows:
        raise RuntimeError(f"Parent Phase-B manifest lacks required role: {role}")
    row = rows[role]
    path = Path(row["path"])
    if not path.is_file() or int(path.stat().st_size) != int(row["size_bytes"]):
        raise RuntimeError(f"Parent Phase-B input missing/size drift: {role}")
    if file_sha256(path) != row["sha256"]:
        raise RuntimeError(f"Parent Phase-B input hash drift: {role}")
    return path


def execute(study: Path, scope: str, dataset: str, fold: int, physical_device: int) -> None:
    metadata = load_json(study / "study_metadata.json")
    config = load_json(study / "study_config.json")
    manifest = load_json(study / "input_manifest.json")
    verify_source(metadata["source_provenance"])
    paths = verify_manifest(manifest, full_hash=False)
    if not config.get("fable_approval_digest"):
        raise RuntimeError("Step-0 production execution lacks Fable approval digest")
    if scope in {"carved_validation", "retrained_validation"} and not config["step0_validation_inference_allowed"]:
        raise RuntimeError("Step-0 validation inference is review-blocked")
    if scope == "test_trajectory" and not config["test_trajectory_inference_allowed"]:
        raise RuntimeError("Test trajectory inference is review-blocked")
    if config["phase_c_allowed"] or config["selection_may_read_test_trajectory"]:
        raise RuntimeError("Step-0 selection/test firewall drift")
    folds = int(DATASETS[dataset]["folds"])
    if not 1 <= fold <= folds:
        raise ValueError(f"Invalid fold {fold} for {dataset}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    import torch
    import torch.nn.functional as torch_f

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Step-0 fail-closed GPU pinning failed")
    device = torch.device("cuda:0")
    parent = Path(config["parent_phase_b_study"])
    retrained_hashes: dict[str, str] = {}
    if scope == "retrained_validation":
        decision = load_json(study / "selection" / "step0_decision.json")
        if decision["selected_branch"] != "B_RETRAIN_WITH_HELD_OUT_VALIDATION":
            raise RuntimeError("Retrained validation requested outside Branch B")
        retrain_complete = load_json(study / "status" / f"retrain_{dataset}_fold{fold}.json")
        if retrain_complete.get("status") != "complete":
            raise RuntimeError("Branch-B retraining is incomplete")
        model_source = study / "retrain" / dataset / f"fold{fold}" / "runtime" / "model.py"
        if file_sha256(model_source) != retrain_complete["runtime_source_sha256"]["model.py"]:
            raise RuntimeError("Retrained runtime model source drift")
        retrained_hashes = dict(retrain_complete["checkpoint_sha256"])
    else:
        model_source = paths["runtime_source/model.py"]
    if file_sha256(model_source) != next(row["sha256"] for row in manifest["files"] if row["role"] == "runtime_source/model.py"):
        raise RuntimeError("Runtime model source hash drift")
    model_class = load_model_class(model_source)
    data_root = Path(config["data_root"])
    dataset_root = data_root / dataset
    id_to_name, name_to_id = parse_mapping(dataset_root / "mapping.txt")
    background = {index for index, label in id_to_name.items() if label == "background"}
    sample_rate = int(DATASETS[dataset]["sample_rate"])
    parent_manifest = load_json(paths["parent/input_manifest"])
    if parent_manifest["manifest_digest"] != config["parent_input_manifest_digest"]:
        raise RuntimeError("Nested parent input-manifest digest drift")
    parent_rows = {str(row["role"]): row for row in parent_manifest["files"]}
    if scope in {"carved_validation", "retrained_validation"}:
        bundle = study / "carves" / dataset / f"fold{fold}" / "carved_validation.bundle"
    else:
        bundle = dataset_root / "splits" / f"test.split{fold}.bundle"
    cases = [normalize_case(value) for value in read_nonempty_lines(bundle)]
    if not cases or len(cases) != len(set(cases)):
        raise RuntimeError(f"Invalid evaluation bundle: {bundle}")
    inputs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for case_id in cases:
        feature_path = verify_parent_row(parent_rows, f"data/{dataset}/feature/{case_id}.npy")
        gt_path = verify_parent_row(parent_rows, f"data/{dataset}/ground_truth/{case_id}.txt")
        feature = np.load(feature_path, allow_pickle=False)[:, ::sample_rate]
        target = np.asarray(
            [name_to_id[label] for label in read_nonempty_lines(gt_path)], dtype=np.int64
        )[::sample_rate]
        if feature.shape[0] != OFFICIAL_CONFIG["features_dim"] or feature.shape[1] != len(target):
            raise RuntimeError(f"Feature/GT alignment drift: {dataset}/{case_id}")
        inputs.append((case_id, feature.astype(np.float32, copy=False), target))

    rows: list[dict] = []
    checkpoint_hashes: dict[str, str] = {}
    for epoch in EPOCH_GRID:
        checkpoint_role = f"checkpoint/{dataset}/fold{fold}/epoch{epoch}"
        if scope == "retrained_validation":
            checkpoint = study / "retrain" / dataset / f"fold{fold}" / "runtime" / "models" / dataset / f"split_{fold}" / f"epoch-{epoch}.model"
            expected_checkpoint_hash = retrained_hashes[str(epoch)]
        else:
            checkpoint = paths[checkpoint_role]
            expected_checkpoint_hash = next(
                row["sha256"] for row in manifest["files"] if row["role"] == checkpoint_role
            )
        observed_checkpoint_hash = file_sha256(checkpoint)
        if observed_checkpoint_hash != expected_checkpoint_hash:
            raise RuntimeError(f"Checkpoint hash drift: {checkpoint_role}")
        model = model_class(
            OFFICIAL_CONFIG["num_layers_PG"], OFFICIAL_CONFIG["num_layers_R"],
            OFFICIAL_CONFIG["num_R"], OFFICIAL_CONFIG["num_f_maps"],
            OFFICIAL_CONFIG["features_dim"], len(id_to_name),
        )
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        cases_for_metric: list[tuple[np.ndarray, np.ndarray]] = []
        with torch.no_grad():
            for _, feature, target in inputs:
                tensor = torch.from_numpy(feature).unsqueeze(0).to(device)
                probability = torch_f.softmax(model(tensor)[-1], dim=1)[0]
                prediction = probability.argmax(dim=0).cpu().numpy().astype(np.int64)
                if prediction.shape != target.shape:
                    raise RuntimeError("Checkpoint evaluation frame mismatch")
                cases_for_metric.append((prediction, target))
        metrics = aggregate_standard(cases_for_metric, background)
        predicted_segments = sum(len(segments(prediction, background)) for prediction, _ in cases_for_metric)
        gt_segments = sum(len(segments(target, background)) for _, target in cases_for_metric)
        n_frames = sum(len(target) for _, target in cases_for_metric)
        n_errors = sum(int(np.sum(prediction != target)) for prediction, target in cases_for_metric)
        composite = sum(metrics[key] for key in COMPOSITE_METRICS) / len(COMPOSITE_METRICS)
        rows.append(
            {
                "scope": scope,
                "dataset": dataset,
                "fold": fold,
                "epoch": epoch,
                "n_cases": len(cases_for_metric),
                "n_frames": n_frames,
                "n_errors": n_errors,
                **metrics,
                "composite": composite,
                "predicted_segments": predicted_segments,
                "gt_segments": gt_segments,
                "predseg_gt_ratio": predicted_segments / max(gt_segments, 1),
            }
        )
        checkpoint_hashes[str(epoch)] = observed_checkpoint_hash
        del model, state
        torch.cuda.empty_cache()
        print(f"{scope} {dataset}/fold{fold} epoch={epoch} composite={composite:.3f}", flush=True)

    namespace = {
        "carved_validation": "validation_metrics",
        "retrained_validation": "retrained_validation_metrics",
        "test_trajectory": "test_trajectory",
    }[scope]
    output = study / namespace / dataset / f"fold{fold}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    complete = {
        "status": "complete",
        "scope": scope,
        "dataset": dataset,
        "fold": fold,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_path": str(bundle.resolve()),
        "bundle_sha256": file_sha256(bundle),
        "checkpoint_sha256": checkpoint_hashes,
        "output_path": str(output),
        "output_sha256": file_sha256(output),
        "test_data_used_for_selection": False,
    }
    complete["completion_digest"] = canonical_digest(complete)
    atomic_write_json(study / "status" / f"{scope}_{dataset}_fold{fold}.json", complete)


def main() -> int:
    args = parse_args()
    study = args.study_dir.resolve()
    failed = study / "status" / f"{args.scope}_{args.dataset}_fold{args.fold}_failed.json"
    try:
        execute(study, args.scope, args.dataset, args.fold, args.device)
    except Exception as error:
        atomic_write_json(
            failed,
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
