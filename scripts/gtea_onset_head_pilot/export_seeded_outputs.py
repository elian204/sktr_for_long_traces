#!/usr/bin/env python3
"""Export seeded activity probabilities, official predictions, and onset curves."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

from common import (
    INFERENCE_SEED_STRIDE,
    atomic_write_json,
    assert_variant_config,
    checkpoint_path,
    export_dir,
    file_sha256,
    load_json,
    load_task,
    read_bundle,
    verify_export,
    verify_source_digest,
)


def restore_matrix(
    matrix: Any,
    *,
    full_length: int,
    left_offset: int,
    right_offset: int,
    sample_rate: int,
    restore_full_sequence: Any,
    normalize_columns: bool,
) -> Any:
    import numpy as np

    matrix = np.asarray(matrix, dtype=np.float64)
    restored = np.stack(
        [
            restore_full_sequence(
                row, full_length, left_offset, right_offset, sample_rate
            )
            for row in matrix
        ],
        axis=0,
    )
    if normalize_columns:
        restored /= np.maximum(restored.sum(axis=0, keepdims=True), 1e-12)
    return restored.astype(np.float32, copy=False)


def official_prediction(
    canonical_sub: Any,
    *,
    full_length: int,
    left_offset: int,
    right_offset: int,
    sample_rate: int,
    postprocess: Any,
    restore_full_sequence: Any,
    get_labels_start_end_time: Any,
    mode_filter: Any,
) -> Any:
    import numpy as np

    prediction = restore_full_sequence(
        np.argmax(canonical_sub, axis=0),
        full_length,
        left_offset,
        right_offset,
        sample_rate,
    )
    if postprocess["type"] == "mode":
        prediction = mode_filter(prediction, postprocess["value"])
    elif postprocess["type"] == "purge":
        transitions, starts, ends = get_labels_start_end_time(prediction)
        for index in range(len(transitions)):
            duration = ends[index] - starts[index]
            if duration > postprocess["value"]:
                continue
            if index == 0:
                prediction[starts[index] : ends[index]] = transitions[index + 1]
            elif index == len(transitions) - 1:
                prediction[starts[index] : ends[index]] = transitions[index - 1]
            else:
                middle = starts[index] + duration // 2
                prediction[starts[index] : middle] = transitions[index - 1]
                prediction[middle : ends[index]] = transitions[index + 1]
    return prediction.astype("int32", copy=False)


def seeded_outputs(
    trainer: Any,
    video_index: int,
    test_dataset: Any,
    inference_seed: int,
    torch: Any,
    median_filter: Any,
    restore_full_sequence: Any,
    get_labels_start_end_time: Any,
    mode_filter: Any,
) -> Tuple[str, Any, Any, Any, Any]:
    import numpy as np

    feature, label, _, _, video = test_dataset[video_index]
    model_seed = int(inference_seed) * INFERENCE_SEED_STRIDE + int(video_index)
    with torch.no_grad():
        activity_outputs = [
            trainer.model.ddim_sample(value.to(trainer.device), model_seed).cpu()
            for value in feature
        ]
        onset_outputs = [
            trainer.model.predict_class_specific_onsets(
                value.to(trainer.device)
            ).cpu()
            for value in feature
        ]
    min_length = min(
        [value.shape[2] for value in activity_outputs]
        + [value.shape[2] for value in onset_outputs]
    )
    raw_sub = (
        torch.cat(
            [value[:, :, :min_length] for value in activity_outputs], dim=0
        )
        .mean(0)
        .numpy()
    )
    onset_sub = (
        torch.cat(
            [value[:, :, :min_length] for value in onset_outputs], dim=0
        )
        .mean(0)
        .numpy()
    )
    canonical_sub = np.array(raw_sub, copy=True)
    if trainer.postprocess["type"] == "median":
        canonical_sub = np.stack(
            [
                median_filter(row, size=trainer.postprocess["value"])
                for row in canonical_sub
            ]
        )
        canonical_sub /= np.maximum(
            canonical_sub.sum(axis=0, keepdims=True), 1e-12
        )
    left_offset = trainer.sample_rate // 2
    right_offset = (trainer.sample_rate - 1) // 2
    full_length = int(label.shape[-1])
    raw = restore_matrix(
        raw_sub,
        full_length=full_length,
        left_offset=left_offset,
        right_offset=right_offset,
        sample_rate=trainer.sample_rate,
        restore_full_sequence=restore_full_sequence,
        normalize_columns=True,
    )
    canonical = restore_matrix(
        canonical_sub,
        full_length=full_length,
        left_offset=left_offset,
        right_offset=right_offset,
        sample_rate=trainer.sample_rate,
        restore_full_sequence=restore_full_sequence,
        normalize_columns=True,
    )
    onset = restore_matrix(
        onset_sub,
        full_length=full_length,
        left_offset=left_offset,
        right_offset=right_offset,
        sample_rate=trainer.sample_rate,
        restore_full_sequence=restore_full_sequence,
        normalize_columns=False,
    )
    prediction = official_prediction(
        canonical_sub,
        full_length=full_length,
        left_offset=left_offset,
        right_offset=right_offset,
        sample_rate=trainer.sample_rate,
        postprocess=trainer.postprocess,
        restore_full_sequence=restore_full_sequence,
        get_labels_start_end_time=get_labels_start_end_time,
        mode_filter=mode_filter,
    )
    if raw.shape != onset.shape or raw.shape[1] != full_length:
        raise ValueError(f"{video}: activity/onset alignment failed")
    return video, raw, canonical, prediction, onset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--inference-seed", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()

    study_dir = args.study_dir.resolve()
    metadata = load_json(study_dir / "study_metadata.json")
    diffact_root = Path(metadata["diffact_root"])
    verify_source_digest(metadata, diffact_root)
    task = load_task(study_dir, args.task_id)
    if args.device != int(task["physical_gpu"]):
        raise ValueError("Exporter device differs from immutable task")
    if args.checkpoint_epoch not in task["checkpoint_epochs"]:
        raise ValueError("Checkpoint is outside the immutable grid")
    if args.inference_seed not in task["inference_seeds"]:
        raise ValueError("Inference seed is outside the immutable grid")
    config = load_json(Path(task["config_path"]))
    assert_variant_config(config, task, metadata)
    checkpoint = checkpoint_path(task, args.checkpoint_epoch)
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(checkpoint)
    checkpoint_sha256 = file_sha256(checkpoint)
    expected_cases = read_bundle(Path(task["test_manifest"]))
    output_dir = export_dir(task, args.checkpoint_epoch, args.inference_seed)
    complete_path = output_dir / "export_complete.json"
    if complete_path.is_file():
        summary = verify_export(output_dir, expected_cases)
        print(f"SKIP validated export: {output_dir} ({summary['frame_count']} frames)")
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    sys.path.insert(0, str(diffact_root))
    import numpy as np
    import torch
    from scipy.ndimage import median_filter
    from dataset import VideoFeatureDataset, get_data_dict, restore_full_sequence
    from main import Trainer
    from utils import get_labels_start_end_time, load_config_file, mode_filter

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable after device pinning")
    device = torch.device("cuda")
    params = load_config_file(str(Path(task["config_path"])))
    root_data_dir = Path(params["root_data_dir"])
    mapping_file = root_data_dir / params["dataset_name"] / "mapping.txt"
    mapping = np.loadtxt(mapping_file, dtype=str)
    event_list = [row[1] for row in mapping]
    data_dict = get_data_dict(
        feature_dir=str(root_data_dir / params["dataset_name"] / "features"),
        label_dir=str(root_data_dir / params["dataset_name"] / "groundTruth"),
        video_list=expected_cases,
        event_list=event_list,
        sample_rate=params["sample_rate"],
        temporal_aug=params["temporal_aug"],
        boundary_smooth=params["boundary_smooth"],
        onset_smooth=params["class_specific_onset_smooth"],
    )
    dataset = VideoFeatureDataset(data_dict, len(event_list), mode="test")
    trainer = Trainer(
        dict(params["encoder_params"]),
        dict(params["decoder_params"]),
        dict(params["diffusion_params"]),
        event_list,
        params["sample_rate"],
        params["temporal_aug"],
        params["set_sampling_seed"],
        params["postprocess"],
        device=device,
    )
    trainer.model.load_state_dict(torch.load(str(checkpoint), map_location=device))
    trainer.model.eval()
    trainer.model.to(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    onset_hashes = {}
    for index in range(len(dataset)):
        video, raw, canonical, prediction, onset = seeded_outputs(
            trainer,
            index,
            dataset,
            args.inference_seed,
            torch,
            median_filter,
            restore_full_sequence,
            get_labels_start_end_time,
            mode_filter,
        )
        if video != expected_cases[index]:
            raise ValueError(f"Dataset/map order mismatch: {video}")
        np.save(output_dir / f"{index}_raw.npy", raw.astype(np.float32))
        np.save(output_dir / f"{index}.npy", canonical.astype(np.float32))
        np.save(output_dir / f"{index}_pred.npy", prediction.astype(np.int32))
        np.save(output_dir / f"{index}_onset.npy", onset.astype(np.float32))
        onset_hashes[str(index)] = file_sha256(output_dir / f"{index}_onset.npy")
    align_dir = Path(task["align_dir"])
    shutil.copy2(mapping_file, output_dir / "mapping.txt")
    shutil.copy2(align_dir / "video_index_map.txt", output_dir / "video_index_map.txt")
    shutil.copy2(align_dir / "ground_truth.csv", output_dir / "ground_truth.csv")
    summary = verify_export(
        output_dir, expected_cases, verify_recorded_hashes=False
    )
    atomic_write_json(
        complete_path,
        {
            "schema_version": 1,
            "completed": True,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "task_id": args.task_id,
            "checkpoint_epoch": args.checkpoint_epoch,
            "checkpoint_sha256": checkpoint_sha256,
            "inference_seed": args.inference_seed,
            "model_seed_formula": (
                f"{args.inference_seed} * {INFERENCE_SEED_STRIDE} + video_index"
            ),
            "onset_is_deterministic_encoder_readout": True,
            "onset_artifact_sha256": onset_hashes,
            "physical_gpu": args.device,
            "source_digest": metadata["source_provenance"]["source_digest"],
            "case_count": summary["case_count"],
            "frame_count": summary["frame_count"],
            "class_count": summary["class_count"],
            "artifact_sha256": summary["artifact_sha256"],
        },
    )
    verify_export(output_dir, expected_cases)
    print(f"COMPLETE activity+onset export: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
