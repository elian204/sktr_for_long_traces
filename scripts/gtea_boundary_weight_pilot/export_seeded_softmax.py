#!/usr/bin/env python3
"""Export one checkpoint with an explicit, reproducible diffusion sampling seed."""

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
    PHYSICAL_GPU,
    atomic_write_json,
    assert_variant_config,
    checkpoint_path,
    export_dir,
    file_sha256,
    load_json,
    load_task,
    map_rows,
    read_bundle,
    verify_export,
    verify_source_digest,
)


def seeded_decoder_output(
    trainer: Any,
    video_index: int,
    test_dataset: Any,
    device: Any,
    inference_seed: int,
    base_export: Any,
    torch: Any,
) -> Tuple[str, Any, Any, Any]:
    feature, label, _, video = test_dataset[video_index]
    model_seed = int(inference_seed) * INFERENCE_SEED_STRIDE + int(video_index)
    with torch.no_grad():
        output = [
            trainer.model.ddim_sample(feature[offset].to(device), model_seed).cpu()
            for offset in range(len(feature))
        ]
        min_length = min(value.shape[2] for value in output)
        output_np = torch.cat([value[:, :, :min_length] for value in output], 0).mean(0).numpy()
        left_offset = trainer.sample_rate // 2
        right_offset = (trainer.sample_rate - 1) // 2
        full_length = int(label.shape[-1])
        raw, canonical, prediction = base_export.aligned_output_streams(
            output_np,
            full_length,
            left_offset,
            right_offset,
            trainer.sample_rate,
            trainer.postprocess,
        )
    return video, raw, canonical, prediction


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
    if args.device != PHYSICAL_GPU or args.device != int(task["physical_gpu"]):
        raise ValueError(
            f"Pilot is pinned to physical GPU {PHYSICAL_GPU}; got --device {args.device}"
        )
    if args.checkpoint_epoch not in [int(value) for value in task["checkpoint_epochs"]]:
        raise ValueError(f"Checkpoint is outside the immutable grid: {args.checkpoint_epoch}")
    if args.inference_seed not in [int(value) for value in task["inference_seeds"]]:
        raise ValueError(f"Inference seed is outside the immutable grid: {args.inference_seed}")

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
        complete = load_json(complete_path)
        summary = verify_export(output_dir, expected_cases)
        expected = {
            "task_id": args.task_id,
            "checkpoint_epoch": args.checkpoint_epoch,
            "checkpoint_sha256": checkpoint_sha256,
            "inference_seed": args.inference_seed,
            "physical_gpu": PHYSICAL_GPU,
        }
        mismatches = {
            key: {"expected": value, "actual": complete.get(key)}
            for key, value in expected.items()
            if complete.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Existing export completion metadata mismatch: {mismatches}")
        print(
            f"SKIP validated export task={args.task_id} epoch={args.checkpoint_epoch} "
            f"inference_seed={args.inference_seed} frames={summary['frame_count']}",
            flush=True,
        )
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    if str(diffact_root) not in sys.path:
        sys.path.insert(0, str(diffact_root))
    import numpy as np
    import torch
    from tqdm import tqdm

    import export_softmax as base_export
    from dataset import VideoFeatureDataset, get_data_dict
    from main import Trainer
    from utils import load_config_file

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable after pinning physical GPU 3")
    device = torch.device("cuda")
    params = load_config_file(str(Path(task["config_path"])))
    root_data_dir = Path(params["root_data_dir"])
    dataset_name = params["dataset_name"]
    mapping_file = root_data_dir / dataset_name / "mapping.txt"
    mapping = np.loadtxt(mapping_file, dtype=str)
    event_list = [row[1] for row in mapping]
    align_dir = Path(task["align_dir"])
    video_entries = base_export.parse_video_index_map(align_dir / "video_index_map.txt")
    if [case for _, case in video_entries] != expected_cases:
        raise ValueError("Alignment map order differs from immutable test manifest")
    data_dict = get_data_dict(
        feature_dir=str(root_data_dir / dataset_name / "features"),
        label_dir=str(root_data_dir / dataset_name / "groundTruth"),
        video_list=expected_cases,
        event_list=event_list,
        sample_rate=params["sample_rate"],
        temporal_aug=params["temporal_aug"],
        boundary_smooth=params["boundary_smooth"],
    )
    test_dataset = VideoFeatureDataset(data_dict, len(event_list), mode="test")
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
    state = torch.load(str(checkpoint), map_location=device)
    trainer.model.load_state_dict(state)
    trainer.model.eval()
    trainer.model.to(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    for index in tqdm(range(len(test_dataset)), desc="Seeded softmax export"):
        video, raw, canonical, prediction = seeded_decoder_output(
            trainer,
            index,
            test_dataset,
            device,
            args.inference_seed,
            base_export,
            torch,
        )
        if video != expected_cases[index]:
            raise ValueError(
                f"Dataset/map order mismatch at {index}: {video}/{expected_cases[index]}"
            )
        np.save(output_dir / f"{index}_raw.npy", raw.astype(np.float32, copy=False))
        np.save(output_dir / f"{index}.npy", canonical.astype(np.float32, copy=False))
        np.save(output_dir / f"{index}_pred.npy", prediction.astype(np.int32, copy=False))
    shutil.copy2(mapping_file, output_dir / "mapping.txt")
    shutil.copy2(align_dir / "video_index_map.txt", output_dir / "video_index_map.txt")
    shutil.copy2(align_dir / "ground_truth.csv", output_dir / "ground_truth.csv")

    summary = verify_export(output_dir, expected_cases, verify_recorded_hashes=False)
    payload = {
        "schema_version": 1,
        "completed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": args.task_id,
        "decoder_boundary_loss": task["decoder_boundary_loss"],
        "training_seed": task["training_seed"],
        "checkpoint_epoch": args.checkpoint_epoch,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "inference_seed": args.inference_seed,
        "model_seed_formula": (
            f"{args.inference_seed} * {INFERENCE_SEED_STRIDE} + video_index"
        ),
        "physical_gpu": PHYSICAL_GPU,
        "config": task["config_path"],
        "source_digest": metadata["source_provenance"]["source_digest"],
        "case_count": summary["case_count"],
        "frame_count": summary["frame_count"],
        "class_count": summary["class_count"],
        "artifact_sha256": summary["artifact_sha256"],
    }
    atomic_write_json(complete_path, payload)
    verify_export(output_dir, expected_cases)
    print(
        f"COMPLETE export task={args.task_id} epoch={args.checkpoint_epoch} "
        f"inference_seed={args.inference_seed} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

