#!/usr/bin/env python3
"""Shared contracts and model for the nested-OOF temporal pairwise verifier."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as torch_f


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import _edit_score_asformer, _segmental_f1_counts_asformer

PROTOCOL_VERSION = "independent-acceptance-verifier-v1-screen-v1"
DEFAULT_V0_DIR = Path(
    "/data1/eli-bogdanov/sktr_runs/independent_acceptance_verifier_v0_review_v2"
)
DEFAULT_STUDY_DIR = Path(
    "/data1/eli-bogdanov/sktr_runs/independent_acceptance_verifier_v1_review_v1"
)
DEFAULT_FEATURE_MANIFEST = Path(
    "/data1/eli-bogdanov/sktr_runs/breakfast_visual_repair_head_oof_screen_v2/"
    "oof_input_manifest.json"
)

OUTER_FOLDS = (1, 2, 3, 4)
INNER_FOLDS = (1, 2, 3)
N_CLASSES = 48
FEATURE_DIM = 2048
PROJECTED_DIM = 256
CORE_BINS = 48
CONTEXT_BINS = 16
TOTAL_BINS = CORE_BINS + 2 * CONTEXT_BINS
LABEL_EMBED_DIM = 32
THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.50, 1.00, 0.05))

MODEL_CONFIG = {
    "architecture": "dilated_tcn",
    "projection_dim": PROJECTED_DIM,
    "tcn_channels": 256,
    "tcn_dilations": [1, 2, 4, 8],
    "kernel_size": 3,
    "dropout": 0.1,
    "label_embedding_dim": LABEL_EMBED_DIM,
    "core_bins": CORE_BINS,
    "context_bins_each_side": CONTEXT_BINS,
    "context_extent": "one_selected_span_length_per_side",
    "proposal_confidence_is_model_input": False,
}
TRAIN_CONFIG = {
    "epochs": 40,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "hard_negative_probability_threshold": 0.5,
    "hard_negative_loss_multiplier": 2.0,
    "length_weighting": "sqrt_selected_frames_normalized",
    "positive_class_weight": "weighted_negative_mass_over_positive_mass_capped_at_10",
    "threshold_grid": list(THRESHOLDS),
    "threshold_tie_break": "max_delta_acc_then_fixed_broken_ratio_then_higher_threshold",
    "seed": 20260801,
}
V1_GATES = {
    "pooled_delta_acc_min_pp": 0.75,
    "held_inner_rule": "all_three_positive_or_two_positive_and_none_below_minus_0.1",
    "worst_video_delta_acc_min_pp": -5.0,
    "fixed_to_broken_changed_frames_min": 3.0,
    "pooled_delta_edit_min_pp": 0.0,
    "pooled_delta_f1_at_25_min_pp": 0.0,
}

V1_SOURCE_FILES = (
    "scripts/independent_acceptance_verifier/README.md",
    "scripts/independent_acceptance_verifier/v1_common.py",
    "scripts/independent_acceptance_verifier/prepare_v1.py",
    "scripts/independent_acceptance_verifier/build_v1_features.py",
    "scripts/independent_acceptance_verifier/train_v1_rotation.py",
    "scripts/independent_acceptance_verifier/aggregate_v1.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in V1_SOURCE_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = file_sha256(path)
    return result


def source_provenance() -> dict[str, Any]:
    hashes = source_hashes()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "git_head": head,
        "git_dirty": bool(status),
        "git_status_short": status.splitlines(),
        "file_sha256": hashes,
        "source_digest": canonical_digest(hashes),
    }


def verify_source_provenance(expected: Mapping[str, Any]) -> None:
    actual = source_hashes()
    if actual != dict(expected["file_sha256"]):
        changed = sorted(
            key
            for key in set(actual) | set(expected["file_sha256"])
            if actual.get(key) != expected["file_sha256"].get(key)
        )
        raise RuntimeError(f"V1 source drift: {changed}")
    if canonical_digest(actual) != expected["source_digest"]:
        raise RuntimeError("V1 source digest drift")


def verify_flat_manifest(manifest: Mapping[str, Any]) -> dict[str, Path]:
    rows = list(manifest["files"])
    paths: dict[str, Path] = {}
    compact: list[dict[str, str]] = []
    for row in rows:
        role = str(row["role"])
        if role in paths:
            raise RuntimeError(f"Duplicate V1 input role: {role}")
        path = Path(row["path"])
        if not path.is_file() or int(path.stat().st_size) != int(row["size_bytes"]):
            raise RuntimeError(f"V1 input missing/size drift: {role}")
        observed = file_sha256(path)
        if observed != row["sha256"]:
            raise RuntimeError(f"V1 input hash drift: {role}")
        paths[role] = path
        compact.append({"role": role, "sha256": observed})
    if len(rows) != int(manifest["file_count"]):
        raise RuntimeError("V1 input count drift")
    if canonical_digest(compact) != manifest["manifest_digest"]:
        raise RuntimeError("V1 manifest digest drift")
    return paths


def verify_nested_manifest(manifest: Mapping[str, Any]) -> None:
    clean = dict(manifest)
    expected = clean.pop("manifest_digest")
    if canonical_digest(clean) != expected:
        raise RuntimeError("Nested feature manifest self-digest drift")


def stable_seed(*values: Any) -> int:
    payload = "|".join(str(value) for value in values).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resample_slice(feature: np.ndarray, start: int, end: int, bins: int) -> np.ndarray:
    if feature.ndim != 2 or feature.shape[0] != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM}xT feature matrix, got {feature.shape}")
    frames = int(feature.shape[1])
    start = max(0, min(int(start), frames))
    end = max(start, min(int(end), frames))
    if end <= start:
        anchor = min(max(start, 0), frames - 1)
        return np.repeat(feature[:, anchor : anchor + 1].T, bins, axis=0).astype(np.float32)
    positions = np.linspace(start, end - 1, bins, dtype=np.float64)
    lower = np.floor(positions).astype(int)
    upper = np.ceil(positions).astype(int)
    fraction = (positions - lower).astype(np.float32)
    sampled = feature[:, lower] * (1.0 - fraction) + feature[:, upper] * fraction
    return sampled.T.astype(np.float32)


def temporal_span_view(feature: np.ndarray, start: int, end: int) -> np.ndarray:
    if end <= start:
        raise ValueError("Selected span must be non-empty")
    length = end - start
    left = _resample_slice(feature, start - length, start, CONTEXT_BINS)
    core = _resample_slice(feature, start, end, CORE_BINS)
    right = _resample_slice(feature, end, end + length, CONTEXT_BINS)
    result = np.concatenate([left, core, right], axis=0)
    if result.shape != (TOTAL_BINS, FEATURE_DIM) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid temporal span view: {result.shape}")
    return result


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm = nn.GroupNorm(16, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.dropout(torch_f.gelu(self.norm(self.conv(value))))
        return value + update


class TemporalPairwiseVerifier(nn.Module):
    """Independent verifier: proposal confidence is deliberately absent."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(FEATURE_DIM, PROJECTED_DIM)
        self.blocks = nn.ModuleList(
            [ResidualTCNBlock(PROJECTED_DIM, dilation, 0.1) for dilation in (1, 2, 4, 8)]
        )
        self.incumbent_embedding = nn.Embedding(N_CLASSES, LABEL_EMBED_DIM)
        self.candidate_embedding = nn.Embedding(N_CLASSES, LABEL_EMBED_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(PROJECTED_DIM + 2 * LABEL_EMBED_DIM, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(
        self, temporal: torch.Tensor, incumbent: torch.Tensor, candidate: torch.Tensor
    ) -> torch.Tensor:
        value = self.projection(temporal).transpose(1, 2)
        for block in self.blocks:
            value = block(value)
        pooled = value.mean(dim=2)
        joined = torch.cat(
            [
                pooled,
                self.incumbent_embedding(incumbent),
                self.candidate_embedding(candidate),
            ],
            dim=1,
        )
        return self.classifier(joined).squeeze(1)


@dataclass(frozen=True)
class CaseData:
    outer_fold: int
    inner_fold: int
    case_id: str
    ground_truth: np.ndarray
    baseline: np.ndarray

    @property
    def key(self) -> tuple[int, int, str]:
        return (self.outer_fold, self.inner_fold, self.case_id)


def metric_sufficient(
    cases: Sequence[CaseData], predictions: Mapping[tuple[int, int, str], np.ndarray]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "frames": 0,
        "correct": 0,
        "cases": len(cases),
        "edit_sum": 0.0,
        "f1_counts": {"10": [0, 0, 0], "25": [0, 0, 0], "50": [0, 0, 0]},
    }
    for case in cases:
        prediction = np.asarray(predictions[case.key], dtype=str)
        target = np.asarray(case.ground_truth, dtype=str)
        result["frames"] += len(target)
        result["correct"] += int(np.sum(prediction == target))
        result["edit_sum"] += float(
            _edit_score_asformer(prediction.tolist(), target.tolist(), ["background"])
        )
        for overlap in (0.10, 0.25, 0.50):
            tp, fp, fn = _segmental_f1_counts_asformer(
                target.tolist(), prediction.tolist(), overlap, None
            )
            counts = result["f1_counts"][str(int(overlap * 100))]
            counts[0] += int(tp)
            counts[1] += int(fp)
            counts[2] += int(fn)
    return result


def metrics_from_sufficient(value: Mapping[str, Any]) -> dict[str, float]:
    frames = int(value["frames"])
    cases = int(value["cases"])
    result = {
        "acc": 100.0 * float(value["correct"]) / max(frames, 1),
        "edit": float(value["edit_sum"]) / max(cases, 1),
    }
    for suffix in (10, 25, 50):
        tp, fp, fn = value["f1_counts"][str(suffix)]
        result[f"f1@{suffix}"] = 100.0 * 2 * tp / max(2 * tp + fp + fn, 1)
    return result


def combine_sufficient(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    combined: dict[str, Any] = {
        "frames": 0,
        "correct": 0,
        "cases": 0,
        "edit_sum": 0.0,
        "f1_counts": {"10": [0, 0, 0], "25": [0, 0, 0], "50": [0, 0, 0]},
    }
    for value in values:
        for key in ("frames", "correct", "cases"):
            combined[key] += int(value[key])
        combined["edit_sum"] += float(value["edit_sum"])
        for suffix in ("10", "25", "50"):
            for index in range(3):
                combined["f1_counts"][suffix][index] += int(value["f1_counts"][suffix][index])
    return combined
