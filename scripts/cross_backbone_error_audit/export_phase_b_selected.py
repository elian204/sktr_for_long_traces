#!/usr/bin/env python3
"""Export val-selected MS-TCN++ probabilities plus locked Breakfast sensitivity arms."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from phase_b_selection_common import atomic_write_json, canonical_digest, file_sha256, load_json, normalize_case, read_nonempty_lines, verify_manifest, verify_source
from phase_b_training_common import DATASETS, OFFICIAL_CONFIG, parse_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", type=int, choices=(0, 1, 2, 3), required=True)
    return parser.parse_args()


def load_model_class(model_source: Path):
    spec = importlib.util.spec_from_file_location("selected_mstcn2_model", model_source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load digest-locked MS-TCN++ model")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MS_TCN2


def execute(study: Path, dataset: str, fold: int, device_id: int) -> None:
    metadata = load_json(study / "study_metadata.json")
    config = load_json(study / "study_config.json")
    manifest = load_json(study / "input_manifest.json")
    verify_source(metadata["source_provenance"])
    paths = verify_manifest(manifest, full_hash=False)
    if not config.get("fable_approval_digest") or not config["selected_export_allowed"]:
        raise RuntimeError("Selected checkpoint export is not approved")
    if config["phase_c_allowed"] or config["selection_may_read_test_trajectory"]:
        raise RuntimeError("Phase-C/selection firewall drift")
    records_payload = load_json(study / "selection" / "selection_records.json")
    if records_payload.get("test_data_used_for_selection") is not False:
        raise RuntimeError("Selection provenance is contaminated")
    record = next((row for row in records_payload["records"] if row["dataset"] == dataset and int(row["fold"]) == fold), None)
    if record is None:
        raise RuntimeError(f"No selected checkpoint record: {dataset}/fold{fold}")

    selected = Path(record["checkpoint_path"])
    if file_sha256(selected) != record["checkpoint_sha256"]:
        raise RuntimeError("Selected checkpoint digest drift")
    if record["training_regime"] == "existing_full_train_checkpoint":
        checkpoint_root = Path(config["parent_phase_b_study"]) / "cells" / dataset / f"fold{fold}" / "runtime" / "models" / dataset / f"split_{fold}"
        model_source = paths["runtime_source/model.py"]
    elif record["training_regime"] == "retrained_train_minus_carve":
        runtime = study / "retrain" / dataset / f"fold{fold}" / "runtime"
        checkpoint_root = runtime / "models" / dataset / f"split_{fold}"
        model_source = runtime / "model.py"
    else:
        raise RuntimeError("Unknown selected checkpoint training regime")
    arms = {"selected": selected}
    if dataset == "breakfast":
        arms.update({"epoch100": checkpoint_root / "epoch-100.model", "epoch30": checkpoint_root / "epoch-30.model"})

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    import torch
    import torch.nn.functional as torch_f
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Selected export fail-closed GPU pinning failed")
    device = torch.device("cuda:0")
    model_class = load_model_class(model_source)
    data_root = Path(config["data_root"])
    dataset_root = data_root / dataset
    id_to_name, name_to_id = parse_mapping(dataset_root / "mapping.txt")
    sample_rate = int(DATASETS[dataset]["sample_rate"])
    test_bundle = paths[f"data/{dataset}/fold{fold}/test"]
    cases = [normalize_case(value) for value in read_nonempty_lines(test_bundle)]
    parent_manifest = load_json(paths["parent/input_manifest"])
    parent_rows = {str(row["role"]): row for row in parent_manifest["files"]}

    output_hashes: dict[str, str] = {}
    manifest_rows = {str(row["role"]): row for row in manifest["files"]}
    retrain_complete = (
        load_json(study / "status" / f"retrain_{dataset}_fold{fold}.json")
        if record["training_regime"] == "retrained_train_minus_carve" else None
    )
    for arm, checkpoint in arms.items():
        epoch = int(record["selected_epoch"]) if arm == "selected" else int(arm.removeprefix("epoch"))
        if arm == "selected":
            expected = record["checkpoint_sha256"]
        elif retrain_complete is not None:
            expected = retrain_complete["checkpoint_sha256"][str(epoch)]
        else:
            expected = manifest_rows[f"checkpoint/{dataset}/fold{fold}/epoch{epoch}"]["sha256"]
        observed = file_sha256(checkpoint)
        if observed != expected:
            raise RuntimeError(f"Checkpoint changed before {arm} export")
        model = model_class(
            OFFICIAL_CONFIG["num_layers_PG"], OFFICIAL_CONFIG["num_layers_R"], OFFICIAL_CONFIG["num_R"],
            OFFICIAL_CONFIG["num_f_maps"], OFFICIAL_CONFIG["features_dim"], len(id_to_name),
        )
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
        model.eval().to(device)
        output = study / "exports" / "mstcn2" / arm / dataset / f"fold{fold}" / "softmax"
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
        with torch.no_grad():
            for case_id in cases:
                feature_row = parent_rows[f"data/{dataset}/feature/{case_id}.npy"]
                gt_row = parent_rows[f"data/{dataset}/ground_truth/{case_id}.txt"]
                feature_path, gt_path = Path(feature_row["path"]), Path(gt_row["path"])
                if file_sha256(feature_path) != feature_row["sha256"] or file_sha256(gt_path) != gt_row["sha256"]:
                    raise RuntimeError(f"Selected export input hash drift: {dataset}/{case_id}")
                feature = np.load(feature_path, allow_pickle=False)[:, ::sample_rate].astype(np.float32, copy=False)
                target = np.asarray([name_to_id[label] for label in read_nonempty_lines(gt_path)], dtype=np.int64)[::sample_rate]
                probability = torch_f.softmax(model(torch.from_numpy(feature).unsqueeze(0).to(device))[-1], dim=1)[0].cpu().numpy().astype(np.float32)
                if probability.shape != (len(id_to_name), len(target)) or not np.isfinite(probability).all():
                    raise RuntimeError(f"Selected export alignment/numeric failure: {dataset}/{case_id}")
                path = output / f"{case_id}.npy"
                np.save(path, probability, allow_pickle=False)
                output_hashes[str(path.relative_to(study))] = file_sha256(path)
        shutil.copy2(dataset_root / "mapping.txt", output / "mapping.txt")
        (output / "cases.txt").write_text("".join(f"{case_id}\n" for case_id in cases))
        output_hashes[str((output / "mapping.txt").relative_to(study))] = file_sha256(output / "mapping.txt")
        output_hashes[str((output / "cases.txt").relative_to(study))] = file_sha256(output / "cases.txt")
        del model
        torch.cuda.empty_cache()
    complete = {
        "status": "complete", "dataset": dataset, "fold": fold, "device": device_id,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "selection_digest": records_payload["selection_digest"],
        "selected_epoch": int(record["selected_epoch"]), "arms": {key: {"checkpoint_path": str(value.resolve()), "checkpoint_sha256": file_sha256(value)} for key, value in arms.items()},
        "test_cases": len(cases), "test_used_for_selection": False, "output_sha256": output_hashes,
    }
    complete["completion_digest"] = canonical_digest(complete)
    atomic_write_json(study / "status" / f"selected_export_{dataset}_fold{fold}.json", complete)


def main() -> int:
    args = parse_args()
    study = args.study_dir.resolve()
    failed = study / "status" / f"selected_export_{args.dataset}_fold{args.fold}_failed.json"
    try:
        execute(study, args.dataset, args.fold, args.device)
    except Exception as error:
        atomic_write_json(failed, {"status": "failed", "failed_utc": datetime.now(timezone.utc).isoformat(), "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
