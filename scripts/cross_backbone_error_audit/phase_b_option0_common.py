#!/usr/bin/env python3
"""Contracts shared by the Phase-B author-checkpoint Option-0 study."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROTOCOL_VERSION = "cross-backbone-phase-b-option0-v1"

DEFAULT_PHASE_A_DIR = Path(
    "/data1/eli-bogdanov/sktr_runs/cross_backbone_error_audit_phase_a_review_v4"
)
DEFAULT_STUDY_DIR = Path(
    "/data1/eli-bogdanov/sktr_runs/cross_backbone_phase_b_option0_review_v1"
)
DEFAULT_CACHE_DIR = Path(
    "/data1/eli-bogdanov/sktr_runs/cross_backbone_official_checkpoints_cache"
)
DEFAULT_ASFORMER_SOURCE = Path(
    "/home/dsi/eli-bogdanov/ASFormer_official_checkpoint_inference"
)
DEFAULT_DATA_ROOT = Path("/home/dsi/eli-bogdanov/data/data")
DEFAULT_FEATURE_MANIFEST = Path(
    "/data1/eli-bogdanov/sktr_runs/breakfast_visual_repair_head_oof_screen_v2/"
    "oof_input_manifest.json"
)

ASFORMER_ARCHIVE_URL = (
    "https://drive.google.com/file/d/1xNykN3vXMHCpHIYT0eb5ZHnKSu3Y2K8r/view"
)
ASFORMER_ARCHIVE_SHA256 = (
    "7b255d8cefb90012b192aedef6f10366474acc291e3988e759a0aae3dadf5909"
)
ASFORMER_OFFICIAL_GIT_HEAD = "e1bbe4f3ed083748f91467c51a63ac2a8b9277ad"
ASFORMER_BREAKFAST_PUBLISHED = {
    "acc": 73.5,
    "edit": 75.0,
    "f1@10": 76.0,
    "f1@25": 70.6,
    "f1@50": 57.4,
}
RECONCILIATION = {
    "pass_max_abs_delta_pp": 2.0,
    "pass_mean_abs_delta_pp": 1.0,
    "notes_max_abs_delta_pp": 3.0,
    "notes_mean_abs_delta_pp": 1.5,
}
OUTER_FOLDS = (1, 2, 3, 4)

RESIDUAL_MSTCN2_PLAN = {
    "cells": 13,
    "active_gpu_hours": 13.1,
    "four_gpu_ideal_wall_hours": 3.4,
    "buffered_wall_hours": "4-6",
    "historical_checkpoint_storage_gib": 10.2,
    "planning_only_refresh_before_launch": True,
}

OPTION0_SOURCE_FILES = (
    "scripts/cross_backbone_error_audit/README.md",
    "scripts/cross_backbone_error_audit/common.py",
    "scripts/cross_backbone_error_audit/metrics.py",
    "scripts/cross_backbone_error_audit/phase_b_option0_common.py",
    "scripts/cross_backbone_error_audit/prepare_phase_b_option0.py",
    "scripts/cross_backbone_error_audit/export_asformer_official.py",
    "scripts/cross_backbone_error_audit/finalize_phase_b_option0.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_nonempty_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def normalize_case(value: str) -> str:
    return Path(value).name.rsplit(".", 1)[0]


def parse_mapping(path: Path) -> tuple[dict[int, str], dict[str, int]]:
    id_to_name: dict[int, str] = {}
    for line in read_nonempty_lines(path):
        class_id, label = line.split(maxsplit=1)
        id_to_name[int(class_id)] = label
    if sorted(id_to_name) != list(range(len(id_to_name))):
        raise ValueError(f"Non-contiguous class mapping: {path}")
    return id_to_name, {label: class_id for class_id, label in id_to_name.items()}


def git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in OPTION0_SOURCE_FILES:
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
    return {
        "git_head": git_head(REPO_ROOT),
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
        raise RuntimeError(f"Option-0 source drift: {changed}")
    if canonical_digest(actual) != expected["source_digest"]:
        raise RuntimeError("Option-0 source digest drift")


def manifest_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    compact = [{"role": row["role"], "sha256": row["sha256"]} for row in rows]
    return canonical_digest(compact)


def verify_flat_manifest(manifest: Mapping[str, Any]) -> dict[str, Path]:
    rows = list(manifest["files"])
    if len(rows) != int(manifest["file_count"]):
        raise RuntimeError("Option-0 input count drift")
    paths: dict[str, Path] = {}
    compact: list[dict[str, str]] = []
    for row in rows:
        role = str(row["role"])
        if role in paths:
            raise RuntimeError(f"Duplicate input role: {role}")
        path = Path(row["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if int(path.stat().st_size) != int(row["size_bytes"]):
            raise RuntimeError(f"Input size drift: {role}")
        observed = file_sha256(path)
        if observed != row["sha256"]:
            raise RuntimeError(f"Input hash drift: {role}")
        paths[role] = path
        compact.append({"role": role, "sha256": observed})
    if canonical_digest(compact) != manifest["manifest_digest"]:
        raise RuntimeError("Option-0 input-manifest digest drift")
    return paths


def feature_manifest_entries(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    features = {str(row["case_id"]): row for row in manifest["features"]}
    ground_truth = {str(row["case_id"]): row for row in manifest["ground_truth"]}
    if len(features) != 1712 or set(features) != set(ground_truth):
        raise RuntimeError("Breakfast feature/GT manifest coverage drift")
    return features, ground_truth


def residual_plan(asformer_status: str) -> dict[str, Any]:
    if asformer_status in {"PASS", "PASS_WITH_NOTES"}:
        return {
            "training_pairs": [
                {"backbone": "mstcn2", "dataset": dataset}
                for dataset in ("gtea", "50salads", "breakfast")
            ],
            **RESIDUAL_MSTCN2_PLAN,
        }
    return {
        "training_pairs": [
            {"backbone": "asformer", "dataset": "breakfast"},
            *[
                {"backbone": "mstcn2", "dataset": dataset}
                for dataset in ("gtea", "50salads", "breakfast")
            ],
        ],
        "cells": 17,
        "active_gpu_hours": 408.2,
        "four_gpu_ideal_wall_hours": 113.1,
        "buffered_wall_hours": "120-144",
        "historical_checkpoint_storage_gib": 25.2,
        "planning_only_refresh_before_launch": True,
    }
