#!/usr/bin/env python3
"""Immutable contracts for the leakage-safe Breakfast repair-head v2 study."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.breakfast_visual_repair_head.common import (  # noqa: E402
    CLASSIFIER_MAX_ITER,
    FEATURE_DIMENSION,
    OUTER_FOLDS,
    PRIMARY_GATES,
    atomic_write_json,
    candidate_columns,
    canonical_digest,
    canonical_feature_view,
    feature_orientation,
    file_sha256,
    load_json,
    pool_span_mean_std,
    stable_uint64,
    write_lines,
)


PROTOCOL_VERSION = "breakfast-visual-repair-head-v2"
PRIMARY_BUDGET = 0.05
BASELINE_FORENSICS_THRESHOLD = 0.5
TAU_CANDIDATES = (0.3, 0.5)
SENSITIVITY_THRESHOLDS = tuple(round(value / 10, 1) for value in range(3, 10))
BUDGET_CANDIDATES = (0.05, 0.08, 0.10)
CAP_PERCENT_CANDIDATES = (1.0, 2.0, 3.0, 5.0)
MARGIN_CANDIDATES = (0.05, 0.10, 0.20, 0.30)
SPAN_SIZE_PERCENT_CANDIDATES = (2.0, 5.0, 10.0, 20.0)
LARGE_SPAN_THRESHOLD_INCREMENT = 0.2
MAXIMUM_HIGHER_CONFIDENCE = 0.9
MODEL_PROMOTION_GAIN_PP = 0.1
SECONDARY_BUDGET_GAIN_PP = 0.15
OOF_HARM_FLOOR_PP = -5.0
TOP_K = 5

DEFAULT_V1_REVIEW_STUDY = Path(
    "/home/dsi/eli-bogdanov/breakfast_visual_repair_head_review_v3"
)
DEFAULT_OOF_STUDY = Path(
    "/data1/eli-bogdanov/sktr_runs/breakfast_visual_repair_head_oof_screen_v2"
)
DEFAULT_OUTER_REVIEW_STUDY = Path(
    "/home/dsi/eli-bogdanov/breakfast_visual_repair_head_v2_outer_review_v1"
)

SOURCE_FILES = (
    "scripts/breakfast_visual_repair_head_v2/README.md",
    "scripts/breakfast_visual_repair_head_v2/common.py",
    "scripts/breakfast_visual_repair_head_v2/core.py",
    "scripts/breakfast_visual_repair_head_v2/prepare_study.py",
    "scripts/breakfast_visual_repair_head_v2/run_oof_screening.py",
    "scripts/breakfast_visual_repair_head_v2/prepare_outer_review.py",
    "scripts/breakfast_visual_repair_head_v2/record_outer_approval.py",
    "scripts/breakfast_visual_repair_head_v2/run_outer_evaluation.py",
    "scripts/breakfast_visual_repair_head/common.py",
    "src/evaluation.py",
)


def source_hashes() -> Dict[str, str]:
    result: Dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = WORKSPACE_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = file_sha256(path)
    return result


def git_provenance(repo: Path = WORKSPACE_ROOT) -> Dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()

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


def source_provenance() -> Dict[str, Any]:
    hashes = source_hashes()
    return {
        "git": git_provenance(),
        "file_sha256": hashes,
        "source_digest": canonical_digest(hashes),
    }


def verify_source_provenance(expected: Mapping[str, Any]) -> None:
    actual = source_hashes()
    if actual != dict(expected["file_sha256"]):
        old = dict(expected["file_sha256"])
        changed = sorted(
            key for key in set(actual) | set(old) if actual.get(key) != old.get(key)
        )
        raise RuntimeError(f"Source provenance mismatch: {changed}")
    if canonical_digest(actual) != expected["source_digest"]:
        raise RuntimeError("Source digest mismatch")


def self_digest(payload: Mapping[str, Any], digest_key: str) -> str:
    clean = dict(payload)
    clean.pop(digest_key, None)
    return canonical_digest(clean)


def verify_self_digest(payload: Mapping[str, Any], digest_key: str) -> None:
    if self_digest(payload, digest_key) != payload[digest_key]:
        raise RuntimeError(f"Invalid self digest: {digest_key}")
