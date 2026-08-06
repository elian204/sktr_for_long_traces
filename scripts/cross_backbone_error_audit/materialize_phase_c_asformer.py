#!/usr/bin/env python3
"""Materialize missing official or descriptive ASFormer Phase-C exports."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from phase_c_common import (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--arm", choices=("official", "epoch30", "epoch100"), required=True)
    parser.add_argument("--device", type=int, choices=(0, 1, 2, 3), required=True)
    return parser.parse_args()


def verified_role(rows: Mapping[str, Mapping[str, Any]], role: str) -> Path:
    if role not in rows:
        raise RuntimeError(f"Phase-C manifest lacks materialization role: {role}")
    row = rows[role]
    path = Path(row["path"])
    if not path.is_file() or int(path.stat().st_size) != int(row["size_bytes"]):
        raise RuntimeError(f"Materialization input missing/size drift: {role}")
    if file_sha256(path) != row["sha256"]:
        raise RuntimeError(f"Materialization input hash drift: {role}")
    return path


def load_model_class(path: Path):
    source_dir = str(path.parent)
    sys.path.insert(0, source_dir)
    try:
        spec = importlib.util.spec_from_file_location("phase_c_asformer_model", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot import hash-locked ASFormer model source")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.MyTransformer
    finally:
        if sys.path[0] == source_dir:
            sys.path.pop(0)


def execute(study: Path, dataset: str, fold: int, arm: str, device_id: int) -> None:
    config = load_json(study / "study_config.json")
    metadata = load_json(study / "study_metadata.json")
    manifest = load_json(study / "input_manifest.json")
    verify_source(metadata["source_provenance"])
    verify_manifest(manifest, [Path(value) for value in config["allowed_input_roots"]], full_hash=False)
    if not config["phase_c_allowed"] or not config["asformer_materialization_allowed"] or not config.get("fable_approval_digest"):
        raise RuntimeError("Phase-C ASFormer materialization is review-blocked")
    if dataset in {"gtea", "50salads"} and arm != "official":
        raise RuntimeError("Only official ASFormer materialization is allowed outside Breakfast")
    if dataset == "breakfast" and arm not in {"epoch30", "epoch100"}:
        raise RuntimeError("Breakfast official ASFormer exports must come from immutable Option 0")
    if not 1 <= fold <= int(DATASETS[dataset]["folds"]):
        raise ValueError("Invalid Phase-C fold")
    rows = {str(row["role"]): row for row in manifest["files"]}
    if arm == "official":
        source_prefix = "asformer/official/source"
        checkpoint_role = f"asformer/official/{dataset}/fold{fold}/checkpoint"
        output_arm = "official"
    else:
        source_prefix = "asformer/local_sensitivity/source"
        epoch = int(arm.removeprefix("epoch"))
        checkpoint_role = f"asformer/local_sensitivity/breakfast/fold{fold}/epoch{epoch}/checkpoint"
        output_arm = arm
    source = verified_role(rows, f"{source_prefix}/model.py")
    eval_source = verified_role(rows, f"{source_prefix}/eval.py")
    if eval_source.parent != source.parent:
        raise RuntimeError("ASFormer sibling-source layout drift")
    checkpoint = verified_role(rows, checkpoint_role)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    import torch
    import torch.nn.functional as torch_f
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Phase-C ASFormer fail-closed GPU pinning failed")
    _, name_to_id = parse_mapping(verified_role(rows, f"data/{dataset}/mapping"))
    model_class = load_model_class(source)
    model = model_class(3, 10, 2, 2, 64, 2048, len(name_to_id), 0.3)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    model.eval().cuda()
    sample_rate = int(DATASETS[dataset]["sample_rate"])
    test_bundle = verified_role(rows, f"data/{dataset}/fold{fold}/test_bundle")
    cases = [normalize_case(value) for value in read_nonempty_lines(test_bundle)]
    nested_features: dict[str, Mapping[str, Any]] = {}
    if dataset == "breakfast":
        nested_path = verified_role(rows, "data/breakfast/nested_feature_manifest")
        nested_features = {
            str(row["case_id"]): row for row in load_json(nested_path)["features"]
        }
    output = study / "materialized" / "asformer" / output_arm / dataset / f"fold{fold}"
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for case_id in cases:
            if dataset == "breakfast":
                entry = nested_features[case_id]
                feature_path = Path(entry["path"])
                if int(feature_path.stat().st_size) != int(entry["bytes"]) or file_sha256(feature_path) != entry["sha256"]:
                    raise RuntimeError(f"Nested Breakfast feature hash drift: {case_id}")
            else:
                feature_path = verified_role(rows, f"data/{dataset}/feature/{case_id}")
            gt_path = verified_role(rows, f"data/{dataset}/ground_truth/{case_id}")
            feature = np.load(feature_path, allow_pickle=False)
            gt_names = read_nonempty_lines(gt_path)
            if feature.ndim != 2 or feature.shape[0] != 2048 or feature.shape[1] != len(gt_names):
                raise RuntimeError(f"ASFormer feature/GT alignment drift: {dataset}/{case_id}")
            feature = feature[:, ::sample_rate].astype(np.float32, copy=False)
            tensor = torch.from_numpy(feature).unsqueeze(0).cuda()
            probability = torch_f.softmax(model(tensor, torch.ones_like(tensor))[-1], dim=1)[0].cpu().numpy().astype(np.float32)
            if probability.shape != (len(name_to_id), feature.shape[1]) or not np.isfinite(probability).all():
                raise RuntimeError(f"ASFormer output validation failure: {dataset}/{case_id}")
            if float(np.max(np.abs(probability.sum(axis=0) - 1.0))) > 1e-5:
                raise RuntimeError(f"ASFormer probability normalization drift: {dataset}/{case_id}")
            path = output / f"{case_id}.npy"
            temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
            with temporary.open("wb") as handle:
                np.save(handle, probability, allow_pickle=False)
            os.replace(temporary, path)
            records.append({"dataset": dataset, "fold": fold, "arm": output_arm, "case_id": case_id, "frames": probability.shape[1], "path": str(path), "sha256": file_sha256(path)})
    manifest_path = output / "export_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    completion = {
        "status": "complete", "dataset": dataset, "fold": fold, "arm": output_arm,
        "device": device_id, "cases": len(records), "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "model_source_path": str(source), "model_source_sha256": file_sha256(source),
        "eval_source_path": str(eval_source), "eval_source_sha256": file_sha256(eval_source),
        "export_manifest_path": str(manifest_path), "export_manifest_sha256": file_sha256(manifest_path),
        "training_performed": False, "headline_eligible": arm == "official",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    completion["completion_digest"] = canonical_digest(completion)
    atomic_write_json(study / "status" / f"asformer_{output_arm}_{dataset}_fold{fold}.json", completion)


def main() -> int:
    args = parse_args()
    study = args.study_dir.resolve()
    failed = study / "status" / f"asformer_{args.arm}_{args.dataset}_fold{args.fold}_failed.json"
    try:
        execute(study, args.dataset, args.fold, args.arm, args.device)
    except Exception as error:
        atomic_write_json(failed, {"status": "failed", "failed_utc": datetime.now(timezone.utc).isoformat(), "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
