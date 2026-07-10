#!/usr/bin/env python3
"""Shared helpers for the low-data DiffAct + Petri-net ablation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_DIR = WORKSPACE_ROOT / "results" / "low_data_petri_diffact_ablation"
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path.home() / "data" / "data"))
DEFAULT_DIFFACT_ROOT = WORKSPACE_ROOT / "baselines" / "DiffAct"
FRACTIONS = (25, 50, 75, 100)
SEEDS = (0, 1, 2, 3, 4)
LOW_DATA_PROTOCOL_VERSION = "official-train-nested-v2"
EXPECTED_SUBSET_CASE_COUNTS = {
    "gtea": {25: 5, 50: 10, 75: 16, 100: 21},
    "50salads": {25: 10, 50: 20, 75: 30, 100: 40},
}


def normalize_case_id(value: str) -> str:
    """Return the video/case stem without the .txt suffix."""
    text = str(value).strip()
    return text[:-4] if text.endswith(".txt") else text


def bundle_name(case_id: str) -> str:
    return f"{normalize_case_id(case_id)}.txt"


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_lines(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(x) for x in rows) + "\n")


def read_bundle(path: Path) -> List[str]:
    return [normalize_case_id(x) for x in read_lines(path)]


def write_bundle(path: Path, case_ids: Sequence[str]) -> None:
    write_lines(path, [bundle_name(c) for c in case_ids])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mapping(mapping_path: Path) -> Tuple[Dict[str, int], List[str]]:
    label_to_idx: Dict[str, int] = {}
    idx_to_label: Dict[int, str] = {}
    with mapping_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            idx_s, label = parts
            idx = int(idx_s)
            label_to_idx[label] = idx
            idx_to_label[idx] = label
    labels = [idx_to_label[i] for i in sorted(idx_to_label)]
    return label_to_idx, labels


def load_ground_truth_labels(data_root: Path, dataset: str, case_id: str) -> List[str]:
    path = data_root / dataset / "groundTruth" / f"{normalize_case_id(case_id)}.txt"
    return read_lines(path)


def load_ground_truth_indices(
    data_root: Path,
    dataset: str,
    case_id: str,
    label_to_idx: Mapping[str, int],
) -> List[int]:
    labels = load_ground_truth_labels(data_root, dataset, case_id)
    missing = sorted({lab for lab in labels if lab not in label_to_idx})
    if missing:
        raise ValueError(f"{case_id}: labels missing from mapping.txt: {missing[:8]}")
    return [int(label_to_idx[lab]) for lab in labels]


def collapse_runs(seq: Sequence[Any]) -> Tuple[Any, ...]:
    out: List[Any] = []
    sentinel = object()
    prev: Any = sentinel
    for item in seq:
        if item != prev:
            out.append(item)
            prev = item
    return tuple(out)


def variant_key(labels: Sequence[Any]) -> str:
    collapsed = collapse_runs(labels)
    payload = json.dumps(list(collapsed), separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{digest}:{payload}"


def load_official_split(data_root: Path, dataset: str, fold: int) -> Tuple[List[str], List[str]]:
    split_dir = data_root / dataset / "splits"
    train_path = split_dir / f"train.split{fold}.bundle"
    test_path = split_dir / f"test.split{fold}.bundle"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"Official split files not found for {dataset} fold {fold}: "
            f"{train_path}, {test_path}"
        )
    return read_bundle(train_path), read_bundle(test_path)


def build_case_table(data_root: Path, dataset: str, case_ids: Sequence[str]) -> pd.DataFrame:
    label_to_idx, _ = load_mapping(data_root / dataset / "mapping.txt")
    rows: List[Dict[str, Any]] = []
    for case_id in case_ids:
        labels = load_ground_truth_labels(data_root, dataset, case_id)
        idxs = [label_to_idx[x] for x in labels]
        collapsed = collapse_runs(labels)
        rows.append(
            {
                "case_id": normalize_case_id(case_id),
                "n_frames": len(labels),
                "n_segments": len(collapsed),
                "variant": variant_key(labels),
                "activities": sorted({str(x) for x in idxs}, key=lambda x: int(x)),
            }
        )
    return pd.DataFrame(rows)


def stratified_order_by_variant(
    case_table: pd.DataFrame,
    candidate_cases: Sequence[str],
    seed: int,
) -> List[str]:
    """Return a deterministic variant-round-robin order for nested prefixes."""
    rng = random.Random(seed)
    candidate_set = {normalize_case_id(c) for c in candidate_cases}
    groups: Dict[str, List[str]] = defaultdict(list)

    for _, row in case_table.iterrows():
        case_id = normalize_case_id(row["case_id"])
        if case_id in candidate_set:
            groups[str(row["variant"])].append(case_id)

    for cases in groups.values():
        rng.shuffle(cases)

    variants = sorted(groups)
    rng.shuffle(variants)

    order: List[str] = []
    while len(order) < len(candidate_set):
        any_added = False
        for variant in variants:
            bucket = groups[variant]
            if bucket:
                order.append(bucket.pop(0))
                any_added = True
        if not any_added:
            break

    missing = [c for c in candidate_cases if normalize_case_id(c) not in set(order)]
    order.extend(normalize_case_id(c) for c in missing)
    return order


def fraction_size(n_cases: int, fraction: int) -> int:
    if fraction >= 100:
        return n_cases
    if n_cases <= 0:
        return 0
    return max(1, int(round(n_cases * float(fraction) / 100.0)))


def expected_fraction_size(dataset: str, n_cases: int, fraction: int) -> int:
    """Return the protocol's pre-specified case count for a data fraction."""
    expected_counts = EXPECTED_SUBSET_CASE_COUNTS.get(dataset)
    if expected_counts is None or fraction not in expected_counts:
        return fraction_size(n_cases, fraction)

    expected_total = expected_counts[100]
    if n_cases != expected_total:
        raise ValueError(
            f"{dataset} official training fold has {n_cases} cases; "
            f"the {LOW_DATA_PROTOCOL_VERSION} protocol requires {expected_total}"
        )
    return expected_counts[fraction]


def split_validation_from_train(
    case_table: pd.DataFrame,
    official_train_cases: Sequence[str],
    val_ratio: float,
    split_seed: int,
) -> Tuple[List[str], List[str]]:
    n_train = len(official_train_cases)
    if n_train < 3:
        raise ValueError("Need at least 3 official training cases to carve validation split")
    n_val = max(1, int(round(n_train * val_ratio)))
    n_val = min(n_val, n_train - 1)
    order = stratified_order_by_variant(case_table, official_train_cases, split_seed)
    val_cases = sorted(order[:n_val])
    val_set = set(val_cases)
    train_pool = [normalize_case_id(c) for c in official_train_cases if normalize_case_id(c) not in val_set]
    return train_pool, val_cases


def summarize_subset(
    case_table: pd.DataFrame,
    subset_cases: Sequence[str],
    train_pool_cases: Sequence[str],
    all_activities: Sequence[str],
) -> Dict[str, Any]:
    subset_set = {normalize_case_id(c) for c in subset_cases}
    pool_set = {normalize_case_id(c) for c in train_pool_cases}
    sub = case_table[case_table["case_id"].isin(subset_set)].copy()
    pool = case_table[case_table["case_id"].isin(pool_set)].copy()

    subset_variants = set(sub["variant"].tolist())
    pool_variants = set(pool["variant"].tolist())
    subset_activities = set()
    for acts in sub["activities"].tolist():
        subset_activities.update(str(x) for x in acts)
    all_activity_set = {str(x) for x in all_activities}
    missing = sorted(all_activity_set.difference(subset_activities), key=lambda x: int(x))

    return {
        "n_cases": int(len(sub)),
        "percentage_of_train_pool": float(100.0 * len(sub) / max(1, len(train_pool_cases))),
        "n_variants": int(len(subset_variants)),
        "variant_coverage": float(len(subset_variants) / max(1, len(pool_variants))),
        "n_activities": int(len(subset_activities)),
        "activity_coverage": float(len(subset_activities) / max(1, len(all_activity_set))),
        "mean_trace_length": float(sub["n_frames"].mean()) if not sub.empty else 0.0,
        "median_trace_length": float(sub["n_frames"].median()) if not sub.empty else 0.0,
        "missing_activities": ";".join(missing),
    }


def ensure_symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            return
        if dst.is_dir() and src.is_dir():
            return
        if dst.is_file() and src.is_file():
            return
        raise FileExistsError(f"Refusing to replace existing path: {dst}")
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def create_dataset_view(
    data_root: Path,
    dataset: str,
    view_root: Path,
    train_cases: Sequence[str],
    validation_cases: Optional[Sequence[str]] = None,
) -> None:
    """Create a DiffAct data view, optionally with a non-training evaluation split."""
    dataset_root = view_root / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    for name in ("features", "groundTruth"):
        ensure_symlink_or_copy(data_root / dataset / name, dataset_root / name)
    ensure_symlink_or_copy(data_root / dataset / "mapping.txt", dataset_root / "mapping.txt")
    splits = dataset_root / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    write_bundle(splits / "train.split1.bundle", train_cases)
    test_bundle = splits / "test.split1.bundle"
    if validation_cases is not None:
        write_bundle(test_bundle, validation_cases)
    elif test_bundle.exists() or test_bundle.is_symlink():
        test_bundle.unlink()


def write_alignment_dir(
    data_root: Path,
    dataset: str,
    out_dir: Path,
    split_name: str,
    case_ids: Sequence[str],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    map_rows: List[Dict[str, Any]] = []
    with (out_dir / "video_index_map.txt").open("w", encoding="utf-8") as f:
        for i, case_id in enumerate(case_ids):
            stem = normalize_case_id(case_id)
            f.write(f"{i}\t{stem}\n")
            map_rows.append(
                {
                    "split": split_name,
                    "original_case_id": stem,
                    "sequential_case_id": str(i),
                    "softmax_list_index": i,
                }
            )
    pd.DataFrame(map_rows).to_csv(out_dir / "case_id_map.csv", index=False)

    label_to_idx, _ = load_mapping(data_root / dataset / "mapping.txt")
    rows: List[Dict[str, Any]] = []
    for i, case_id in enumerate(case_ids):
        for idx in load_ground_truth_indices(data_root, dataset, case_id, label_to_idx):
            rows.append({"case:concept:name": str(i), "concept:name": str(idx)})
    pd.DataFrame(rows).to_csv(out_dir / "ground_truth.csv", index=False)
    return out_dir / "case_id_map.csv"


def one_hot_softmax(labels: Sequence[int], n_classes: int, eps: float = 1e-6) -> np.ndarray:
    mat = np.full((n_classes, len(labels)), eps / max(1, n_classes - 1), dtype=np.float32)
    for t, lab in enumerate(labels):
        mat[int(lab), t] = 1.0 - eps
    col_sum = mat.sum(axis=0, keepdims=True)
    return mat / np.maximum(col_sum, 1e-12)


def read_case_manifest(path: Path) -> List[str]:
    return [normalize_case_id(x) for x in read_lines(path)]


def write_csv_header(path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(columns))


def ensure_report_stub(experiment_dir: Path, payload: Mapping[str, Any]) -> None:
    report = experiment_dir / "REPORT.md"
    if report.exists():
        return
    report.write_text(
        "# Low-Data DiffAct + Petri-Net Ablation\n\n"
        "Status: setup/manifests generated. Training, inference, postprocessing, "
        "and aggregation are driven by the scripts in "
        "`scripts/low_data_petri_diffact/`.\n\n"
        "## Experiment Metadata\n\n"
        "```json\n"
        + json.dumps(dict(payload), indent=2, sort_keys=True)
        + "\n```\n\n"
        "## Current Conclusion\n\n"
        "No scientific conclusion is available until low-data DiffAct models are "
        "trained from scratch and final fixed-test postprocessing metrics are aggregated.\n",
        encoding="utf-8",
    )
