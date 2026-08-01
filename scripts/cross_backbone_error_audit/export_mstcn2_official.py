#!/usr/bin/env python3
"""Export final-stage probabilities from an official MS-TCN++ checkpoint."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from phase_b_training_common import DATASETS, OFFICIAL_CONFIG, normalize_case, parse_mapping, read_nonempty_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    import torch
    import torch.nn.functional as torch_f

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("MS-TCN++ export fail-closed GPU pinning failed")
    runtime = args.runtime.resolve()
    sys.path.insert(0, str(runtime))
    from model import MS_TCN2

    dataset = args.dataset
    dataset_config = DATASETS[dataset]
    if args.fold < 1 or args.fold > int(dataset_config["folds"]):
        raise ValueError(f"Invalid fold for {dataset}: {args.fold}")
    data_root = args.data_root.resolve()
    dataset_root = data_root / dataset
    id_to_name, name_to_id = parse_mapping(dataset_root / "mapping.txt")
    model = MS_TCN2(
        OFFICIAL_CONFIG["num_layers_PG"],
        OFFICIAL_CONFIG["num_layers_R"],
        OFFICIAL_CONFIG["num_R"],
        OFFICIAL_CONFIG["num_f_maps"],
        OFFICIAL_CONFIG["features_dim"],
        len(id_to_name),
    )
    checkpoint = runtime / "models" / dataset / f"split_{args.fold}" / "epoch-100.model"
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().cuda()

    feature_paths = sorted((dataset_root / "features").glob("*.npy"))
    case_to_index = {path.stem: index for index, path in enumerate(feature_paths)}
    test_cases = [
        normalize_case(value)
        for value in read_nonempty_lines(dataset_root / "splits" / f"test.split{args.fold}.bundle")
    ]
    if len(test_cases) != len(set(test_cases)) or not set(test_cases) <= set(case_to_index):
        raise RuntimeError("Invalid MS-TCN++ test bundle or feature coverage")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sample_rate = int(dataset_config["sample_rate"])
    with torch.no_grad():
        for case_id in test_cases:
            feature = np.load(dataset_root / "features" / f"{case_id}.npy", allow_pickle=False)
            feature = feature[:, ::sample_rate]
            tensor = torch.from_numpy(feature.astype(np.float32, copy=False)).unsqueeze(0).cuda()
            logits = model(tensor)[-1]
            probability = torch_f.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float32)
            target_labels = read_nonempty_lines(dataset_root / "groundTruth" / f"{case_id}.txt")
            target = np.asarray([name_to_id[label] for label in target_labels], dtype=np.int64)[::sample_rate]
            if probability.shape != (len(id_to_name), len(target)):
                raise RuntimeError(
                    f"Frame alignment failure for {dataset}/fold{args.fold}/{case_id}: "
                    f"probability={probability.shape}, target={target.shape}"
                )
            if not np.isfinite(probability).all() or float(np.max(np.abs(probability.sum(axis=0) - 1))) > 1e-5:
                raise RuntimeError(f"Probability validation failure: {dataset}/{case_id}")
            np.save(output / f"{case_to_index[case_id]}.npy", probability, allow_pickle=False)
    shutil.copy2(dataset_root / "mapping.txt", output / "mapping.txt")
    (output / "video_index_map.txt").write_text(
        "".join(f"{index}\t{path.stem}\n" for index, path in enumerate(feature_paths))
    )
    print(f"exported={len(test_cases)} dataset={dataset} fold={args.fold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
