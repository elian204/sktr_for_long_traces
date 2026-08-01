#!/usr/bin/env python3
"""Export author-released ASFormer Breakfast probabilities for one fold."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from phase_b_option0_common import (
    ASFORMER_OFFICIAL_GIT_HEAD,
    OUTER_FOLDS,
    atomic_write_json,
    canonical_digest,
    feature_manifest_entries,
    file_sha256,
    git_head,
    load_json,
    normalize_case,
    parse_mapping,
    read_nonempty_lines,
    verify_flat_manifest,
    verify_source_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=OUTER_FOLDS, required=True)
    parser.add_argument("--device", type=int, required=True)
    return parser.parse_args()


def verify_nested_entry(entry: Mapping[str, Any]) -> Path:
    path = Path(entry["path"])
    if not path.is_file() or int(path.stat().st_size) != int(entry["bytes"]):
        raise RuntimeError(f"Nested input missing/size drift: {path}")
    if file_sha256(path) != entry["sha256"]:
        raise RuntimeError(f"Nested input hash drift: {path}")
    return path


def atomic_save_array(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def execute(study_dir: Path, fold: int, device_id: int) -> None:
    metadata = load_json(study_dir / "study_metadata.json")
    config = load_json(study_dir / "study_config.json")
    manifest = load_json(study_dir / "input_manifest.json")
    verify_source_provenance(metadata["source_provenance"])
    paths = verify_flat_manifest(manifest)
    if config["gpu_training_allowed"] or config["phase_c_allowed"]:
        raise RuntimeError("Option-0 study unexpectedly authorizes training or Phase C")
    if git_head(Path(config["asformer_source"])) != ASFORMER_OFFICIAL_GIT_HEAD:
        raise RuntimeError("Official ASFormer source HEAD drift")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    import torch
    import torch.nn.functional as torch_f

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Fail-closed GPU pinning failed")
    source = Path(config["asformer_source"])
    sys.path.insert(0, str(source))
    from model import MyTransformer

    nested = load_json(paths["breakfast/feature_manifest"])
    features, ground_truth = feature_manifest_entries(nested)
    mapping_path = paths["breakfast/mapping"]
    _, name_to_id = parse_mapping(mapping_path)
    if len(name_to_id) != 48:
        raise RuntimeError("Breakfast mapping must contain 48 classes")
    cases = [
        normalize_case(value)
        for value in read_nonempty_lines(paths[f"breakfast/fold{fold}/test_bundle"])
    ]
    if len(cases) != len(set(cases)):
        raise RuntimeError(f"Duplicate fold-{fold} test case")

    checkpoint = paths[f"official/asformer/breakfast/fold{fold}/checkpoint"]
    model = MyTransformer(3, 10, 2, 2, 64, 2048, 48, 0.3)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().cuda()

    output_dir = study_dir / "exports" / f"fold{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    with torch.no_grad():
        for index, case_id in enumerate(cases, start=1):
            feature_path = verify_nested_entry(features[case_id])
            gt_path = verify_nested_entry(ground_truth[case_id])
            feature = np.load(feature_path, allow_pickle=False)
            if feature.ndim != 2 or feature.shape[0] != 2048:
                raise ValueError(f"Unexpected feature shape for {case_id}: {feature.shape}")
            gt_names = read_nonempty_lines(gt_path)
            if feature.shape[1] != len(gt_names):
                raise ValueError(
                    f"Feature/GT length drift for {case_id}: {feature.shape[1]} vs {len(gt_names)}"
                )
            if any(label not in name_to_id for label in gt_names):
                raise ValueError(f"Unknown GT label for {case_id}")
            tensor = torch.from_numpy(np.asarray(feature, dtype=np.float32)).unsqueeze(0).cuda()
            logits = model(tensor, torch.ones_like(tensor))
            probability = torch_f.softmax(logits[-1], dim=1)[0].cpu().numpy().astype(np.float32)
            if probability.shape != (48, feature.shape[1]):
                raise RuntimeError(f"Probability-shape drift for {case_id}: {probability.shape}")
            if not np.isfinite(probability).all():
                raise RuntimeError(f"Non-finite probability for {case_id}")
            max_sum_error = float(np.max(np.abs(probability.sum(axis=0) - 1.0)))
            if max_sum_error > 1e-5:
                raise RuntimeError(f"Probability normalization drift for {case_id}: {max_sum_error}")
            output = output_dir / f"{case_id}.npy"
            atomic_save_array(output, probability)
            records.append(
                {
                    "fold": fold,
                    "case_id": case_id,
                    "frames": int(probability.shape[1]),
                    "probability_path": str(output),
                    "probability_sha256": file_sha256(output),
                    "max_probability_sum_error": max_sum_error,
                }
            )
            if index == 1 or index % 25 == 0 or index == len(cases):
                atomic_write_json(
                    study_dir / "status" / f"fold{fold}.json",
                    {
                        "status": "running",
                        "fold": fold,
                        "completed_cases": index,
                        "total_cases": len(cases),
                        "elapsed_seconds": time.monotonic() - started,
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )

    manifest_path = output_dir / "export_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    completion = {
        "status": "complete",
        "fold": fold,
        "device_physical": device_id,
        "cases": len(records),
        "frames": sum(int(row["frames"]) for row in records),
        "checkpoint_sha256": file_sha256(checkpoint),
        "source_head": ASFORMER_OFFICIAL_GIT_HEAD,
        "export_manifest_sha256": file_sha256(manifest_path),
        "elapsed_seconds": time.monotonic() - started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    completion["completion_digest"] = canonical_digest(completion)
    atomic_write_json(study_dir / "status" / f"fold{fold}.json", completion)


def main() -> int:
    args = parse_args()
    execute(args.study_dir.resolve(), args.fold, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
