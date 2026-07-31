#!/usr/bin/env python3
"""Shared contracts and helpers for the Breakfast visual repair-head study."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
PROTOCOL_VERSION = "breakfast-visual-repair-head-v1"
OUTER_FOLDS = (1, 2, 3, 4)
FEATURE_DIMENSION = 2048
PRIMARY_BUDGET = 0.05
PRIMARY_THRESHOLD = 0.5
CLASSIFIER_MAX_ITER = 2000
SENSITIVITY_THRESHOLDS = tuple(round(value / 10, 1) for value in range(3, 10))
TOP_K = 5

DEFAULT_SELECTOR_STUDY = Path(
    "/data1/eli-bogdanov/sktr_runs/breakfast_selector_allfolds_seed0_v2"
)
DEFAULT_DATA_ROOT = Path.home() / "data" / "data" / "breakfast"
DEFAULT_FEATURE_ROOT = DEFAULT_DATA_ROOT / "features"
DEFAULT_STUDY_DIR = Path(
    "/data1/eli-bogdanov/sktr_runs/breakfast_visual_repair_head_allfolds_v1"
)

SOURCE_FILES = (
    "scripts/breakfast_visual_repair_head/README.md",
    "scripts/breakfast_visual_repair_head/common.py",
    "scripts/breakfast_visual_repair_head/prepare_study.py",
    "scripts/breakfast_visual_repair_head/run_repair_head.py",
    "src/evaluation.py",
)

PRIMARY_GATES = {
    "minimum_pooled_acc_gain_pp": 0.5,
    "minimum_positive_acc_folds": 3,
    "minimum_pooled_edit_gain_pp": 0.0,
    "minimum_pooled_f1_at_25_gain_pp": 0.0,
    "maximum_video_acc_drop_pp": 5.0,
    "maximum_single_video_gain_fraction": 0.5,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_lines(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [str(row) for row in rows]
    path.write_text("\n".join(values) + ("\n" if values else ""))


def git_provenance(repo: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        )
        return proc.stdout.strip()

    status = run("status", "--short")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return {
        "repo": str(repo.resolve()),
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def source_hashes() -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = WORKSPACE_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required source is missing: {path}")
        hashes[relative] = file_sha256(path)
    return hashes


def source_provenance() -> Dict[str, Any]:
    hashes = source_hashes()
    return {
        "git": git_provenance(WORKSPACE_ROOT),
        "file_sha256": hashes,
        "source_digest": canonical_digest(hashes),
    }


def verify_source_provenance(expected: Mapping[str, Any]) -> None:
    actual = source_hashes()
    if actual != dict(expected["file_sha256"]):
        expected_hashes = dict(expected["file_sha256"])
        changed = sorted(
            key
            for key in set(actual) | set(expected_hashes)
            if actual.get(key) != expected_hashes.get(key)
        )
        raise RuntimeError(f"Source provenance mismatch: {changed}")
    if canonical_digest(actual) != expected["source_digest"]:
        raise RuntimeError("Source digest does not match its file-hash table")


def stable_uint64(*parts: Any) -> np.uint64:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return np.uint64(
        int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), "little")
    )


def feature_orientation(shape: Sequence[int]) -> Tuple[str, int]:
    """Return canonical orientation and time length, keyed only on the 2048 axis."""
    dimensions = tuple(int(value) for value in shape)
    if len(dimensions) != 2:
        raise ValueError(f"I3D feature must be 2-D, got shape={dimensions}")
    axes = [index for index, value in enumerate(dimensions) if value == FEATURE_DIMENSION]
    if len(axes) != 1:
        raise ValueError(
            f"I3D orientation requires exactly one {FEATURE_DIMENSION} axis, "
            f"got shape={dimensions}"
        )
    if axes[0] == 0:
        return "features_by_time", dimensions[1]
    return "time_by_features", dimensions[0]


def canonical_feature_view(array: np.ndarray, expected_frames: int | None = None) -> np.ndarray:
    orientation, frames = feature_orientation(array.shape)
    if expected_frames is not None and frames != int(expected_frames):
        raise ValueError(
            f"I3D time length {frames} does not match expected {int(expected_frames)}"
        )
    return array if orientation == "features_by_time" else array.T


def pool_span_mean_std(
    feature_array: np.ndarray, start: int, end: int, expected_frames: int | None = None
) -> np.ndarray:
    canonical = canonical_feature_view(feature_array, expected_frames)
    start, end = int(start), int(end)
    if not 0 <= start < end <= canonical.shape[1]:
        raise ValueError(
            f"Invalid span [{start}, {end}) for feature time length {canonical.shape[1]}"
        )
    span = canonical[:, start:end]
    return np.concatenate(
        [span.mean(axis=1, dtype=np.float64), span.std(axis=1, dtype=np.float64)]
    ).astype(np.float32)


def candidate_columns(rank: int) -> Tuple[str, str, str]:
    if rank < 1 or rank > TOP_K:
        raise ValueError(rank)
    prefix = f"candidate_rank_{rank}"
    return (
        f"{prefix}_class_id",
        f"{prefix}_label",
        f"{prefix}_mean_probability",
    )


def evaluate_primary_gates(
    pooled_delta: Mapping[str, float],
    per_fold_acc_deltas: Sequence[float],
    per_video_acc_deltas: Sequence[float],
    per_video_net_correct_frames: Sequence[int],
    rules: Mapping[str, float] = PRIMARY_GATES,
) -> Dict[str, Any]:
    net = np.asarray(per_video_net_correct_frames, dtype=np.int64)
    total_net = int(net.sum())
    largest_positive = int(np.maximum(net, 0).max(initial=0))
    largest_fraction = (
        float(largest_positive / total_net) if total_net > 0 else float("inf")
    )
    checks = {
        "pooled_acc_gain": float(pooled_delta["acc"])
        >= float(rules["minimum_pooled_acc_gain_pp"]),
        "positive_acc_folds": int(np.sum(np.asarray(per_fold_acc_deltas) > 0))
        >= int(rules["minimum_positive_acc_folds"]),
        "pooled_edit_nonnegative": float(pooled_delta["edit"])
        >= float(rules["minimum_pooled_edit_gain_pp"]),
        "pooled_f1_at_25_nonnegative": float(pooled_delta["f1@25"])
        >= float(rules["minimum_pooled_f1_at_25_gain_pp"]),
        "no_video_acc_drop_over_limit": float(np.min(per_video_acc_deltas))
        >= -float(rules["maximum_video_acc_drop_pp"]),
        "single_video_contribution_bounded": largest_fraction
        <= float(rules["maximum_single_video_gain_fraction"]),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "positive_acc_fold_count": int(np.sum(np.asarray(per_fold_acc_deltas) > 0)),
        "worst_video_acc_delta_pp": float(np.min(per_video_acc_deltas)),
        "pooled_net_correct_frames": total_net,
        "largest_positive_video_net_correct_frames": largest_positive,
        "largest_single_video_gain_fraction": largest_fraction,
    }
