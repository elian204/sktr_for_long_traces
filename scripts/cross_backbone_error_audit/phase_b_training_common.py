#!/usr/bin/env python3
"""Contracts for the approved MS-TCN++ Phase-B retraining study."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROTOCOL_VERSION = "cross-backbone-phase-b-mstcn2-retraining-v1"
OFFICIAL_MSTCN2_HEAD = "f423a9e65f4ccb1cd7322eb9f94946a19e787993"
DATASETS = {
    "gtea": {"folds": 4, "sample_rate": 1},
    "50salads": {"folds": 5, "sample_rate": 2},
    "breakfast": {"folds": 4, "sample_rate": 1},
}
OFFICIAL_CONFIG = {
    "num_epochs": 100,
    "features_dim": 2048,
    "batch_size": 1,
    "learning_rate": 0.0005,
    "num_f_maps": 64,
    "num_layers_PG": 11,
    "num_layers_R": 10,
    "num_R": 3,
    "seed": 1538574472,
}
QUEUE_ASSIGNMENTS = {
    0: (("breakfast", 1), ("gtea", 1), ("50salads", 1)),
    1: (("breakfast", 2), ("gtea", 2), ("50salads", 2)),
    2: (("breakfast", 3), ("gtea", 3), ("50salads", 3)),
    3: (("breakfast", 4), ("gtea", 4), ("50salads", 4), ("50salads", 5)),
}
SOURCE_FILES = (
    "scripts/cross_backbone_error_audit/README.md",
    "scripts/cross_backbone_error_audit/common.py",
    "scripts/cross_backbone_error_audit/metrics.py",
    "scripts/cross_backbone_error_audit/phase_b_training_common.py",
    "scripts/cross_backbone_error_audit/prepare_phase_b_training.py",
    "scripts/cross_backbone_error_audit/verify_phase_b_training.py",
    "scripts/cross_backbone_error_audit/run_phase_b_training_task.py",
    "scripts/cross_backbone_error_audit/export_mstcn2_official.py",
    "scripts/cross_backbone_error_audit/finalize_phase_b_training.py",
)
OFFICIAL_SOURCE_FILES = ("main.py", "model.py", "batch_gen.py", "eval.py", "train.sh")


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
    return json.loads(path.read_text())


def git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def git_clean(repo: Path) -> bool:
    return not subprocess.run(
        ["git", "status", "--short"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def source_hashes() -> dict[str, str]:
    return {relative: file_sha256(REPO_ROOT / relative) for relative in SOURCE_FILES}


def source_provenance() -> dict[str, Any]:
    hashes = source_hashes()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "git_head": git_head(REPO_ROOT),
        "git_dirty": bool(status),
        "git_status_short": status.splitlines(),
        "file_sha256": hashes,
        "source_digest": canonical_digest(hashes),
    }


def verify_source(expected: Mapping[str, Any]) -> None:
    actual = source_hashes()
    if actual != dict(expected["file_sha256"]):
        changed = sorted(
            key
            for key in set(actual) | set(expected["file_sha256"])
            if actual.get(key) != expected["file_sha256"].get(key)
        )
        raise RuntimeError(f"Phase-B source drift: {changed}")
    if canonical_digest(actual) != expected["source_digest"]:
        raise RuntimeError("Phase-B source digest drift")


def verify_manifest(manifest: Mapping[str, Any], *, full_hash: bool) -> dict[str, Path]:
    rows = list(manifest["files"])
    if len(rows) != int(manifest["file_count"]):
        raise RuntimeError("Phase-B input count drift")
    result: dict[str, Path] = {}
    compact: list[dict[str, str]] = []
    for row in rows:
        role = str(row["role"])
        path = Path(row["path"])
        if role in result or not path.is_file():
            raise RuntimeError(f"Missing or duplicate Phase-B input: {role}")
        if int(path.stat().st_size) != int(row["size_bytes"]):
            raise RuntimeError(f"Phase-B input size drift: {role}")
        observed = file_sha256(path) if full_hash else str(row["sha256"])
        if observed != row["sha256"]:
            raise RuntimeError(f"Phase-B input hash drift: {role}")
        result[role] = path
        compact.append({"role": role, "sha256": observed})
    if canonical_digest(compact) != manifest["manifest_digest"]:
        raise RuntimeError("Phase-B input-manifest digest drift")
    return result


def normalize_case(value: str) -> str:
    return Path(value.strip()).name.rsplit(".", 1)[0]


def read_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def parse_mapping(path: Path) -> tuple[dict[int, str], dict[str, int]]:
    id_to_name: dict[int, str] = {}
    for line in read_nonempty_lines(path):
        index, label = line.split(maxsplit=1)
        id_to_name[int(index)] = label
    if sorted(id_to_name) != list(range(len(id_to_name))):
        raise RuntimeError(f"Non-contiguous mapping: {path}")
    return id_to_name, {label: index for index, label in id_to_name.items()}
