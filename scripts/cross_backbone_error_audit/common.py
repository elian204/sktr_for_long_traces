#!/usr/bin/env python3
"""Shared immutable-study contracts for the cross-backbone error audit."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROTOCOL_VERSION = "cross-backbone-error-audit-v1"

DEFAULT_DATA_ROOT = Path("/home/dsi/eli-bogdanov/data/data")
DEFAULT_MSTCN2_ROOT = Path("/home/dsi/eli-bogdanov/MS-TCN2")
DEFAULT_ASFORMER_ROOT = Path("/home/dsi/eli-bogdanov/ASFormer")
DEFAULT_STUDY_DIR = Path(
    "/data1/eli-bogdanov/sktr_runs/cross_backbone_error_audit_phase_a_review_v1"
)

BACKBONES = {
    "mstcn2": {
        "root": DEFAULT_MSTCN2_ROOT,
        "epochs": 50,
        "paper": "MS-TCN++ (TPAMI 2020)",
    },
    "asformer": {
        "root": DEFAULT_ASFORMER_ROOT,
        "epochs": 120,
        "paper": "ASFormer (BMVC 2021)",
    },
}

DATASETS = {
    "gtea": {"folds": 4, "sample_rate": 1},
    "50salads": {"folds": 5, "sample_rate": 2},
    "breakfast": {"folds": 4, "sample_rate": 1},
}

# Values are the rows for the exact named method in the original papers.
PUBLISHED_METRICS = {
    ("mstcn2", "50salads"): {
        "acc": 83.7, "edit": 74.3, "f1@10": 80.7, "f1@25": 78.5, "f1@50": 70.1,
    },
    ("mstcn2", "gtea"): {
        "acc": 80.1, "edit": 83.5, "f1@10": 88.8, "f1@25": 85.7, "f1@50": 76.0,
    },
    ("mstcn2", "breakfast"): {
        "acc": 67.6, "edit": 65.6, "f1@10": 64.1, "f1@25": 58.6, "f1@50": 45.9,
    },
    ("asformer", "50salads"): {
        "acc": 85.6, "edit": 79.6, "f1@10": 85.1, "f1@25": 83.4, "f1@50": 76.0,
    },
    ("asformer", "gtea"): {
        "acc": 79.7, "edit": 84.6, "f1@10": 90.1, "f1@25": 88.8, "f1@50": 79.2,
    },
    ("asformer", "breakfast"): {
        "acc": 73.5, "edit": 75.0, "f1@10": 76.0, "f1@25": 70.6, "f1@50": 57.4,
    },
}

RECONCILIATION = {
    "pass_max_abs_delta_pp": 2.0,
    "pass_mean_abs_delta_pp": 1.0,
    "notes_max_abs_delta_pp": 3.0,
    "notes_mean_abs_delta_pp": 1.5,
}

# Planning-only estimate from the median adjacent-checkpoint cadence in the
# historical repositories, observed on 2026-08-01.  It is not a scientific
# result and must be refreshed in the separately reviewed Phase-B study before
# launch.  The estimate covers only the 17 cells whose reconciliation failed.
PHASE_B_PLANNING_ESTIMATE = {
    "method": "historical_median_adjacent_checkpoint_cadence",
    "observed_date": "2026-08-01",
    "failed_cell_count": 17,
    "active_gpu_hours": 408.2,
    "four_gpu_ideal_wall_hours": 113.1,
    "buffered_wall_days": "5-6",
    "historical_checkpoint_storage_gib": 25.2,
    "required_artifact_root": "/data1/eli-bogdanov/sktr_runs",
    "planning_only_refresh_before_launch": True,
}

SOURCE_FILES = (
    "scripts/cross_backbone_error_audit/README.md",
    "scripts/cross_backbone_error_audit/common.py",
    "scripts/cross_backbone_error_audit/metrics.py",
    "scripts/cross_backbone_error_audit/prepare_study.py",
    "scripts/cross_backbone_error_audit/run_phase_a.py",
)


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


def write_lines(path: Path, rows: Iterable[str]) -> None:
    values = [str(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_provenance(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()

    status = run("status", "--short")
    tracked_diff = subprocess.run(
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
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
    }


def source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[relative] = file_sha256(path)
    return hashes


def source_provenance() -> dict[str, Any]:
    hashes = source_hashes()
    return {
        "git": git_provenance(REPO_ROOT),
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
        raise RuntimeError(f"Source provenance mismatch: {changed}")
    if canonical_digest(actual) != expected["source_digest"]:
        raise RuntimeError("Source digest mismatch")


def read_nonempty_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_mapping(path: Path) -> tuple[dict[int, str], dict[str, int]]:
    id_to_name: dict[int, str] = {}
    for line in read_nonempty_lines(path):
        class_id, label = line.split(maxsplit=1)
        class_id_int = int(class_id)
        if class_id_int in id_to_name:
            raise ValueError(f"Duplicate class id {class_id_int} in {path}")
        id_to_name[class_id_int] = label
    if sorted(id_to_name) != list(range(len(id_to_name))):
        raise ValueError(f"Non-contiguous class ids in {path}")
    return id_to_name, {label: class_id for class_id, label in id_to_name.items()}


def parse_video_index(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    seen_indices: set[int] = set()
    for line in read_nonempty_lines(path):
        index, name = line.split("\t", maxsplit=1)
        value = int(index)
        if name in mapping or value in seen_indices:
            raise ValueError(f"Duplicate video index mapping in {path}")
        mapping[name] = value
        seen_indices.add(value)
    return mapping


def normalize_bundle_case(value: str) -> str:
    return Path(value).name.rsplit(".", 1)[0]


def cell_paths(backbone: str, dataset: str, fold: int, roots: Mapping[str, Path]) -> dict[str, Path]:
    root = Path(roots[backbone])
    epoch = int(BACKBONES[backbone]["epochs"])
    return {
        "root": root,
        "softmax_dir": root / "results" / dataset / "softmax",
        "video_index_map": root / "results" / dataset / "softmax" / "video_index_map.txt",
        "stored_mapping": root / "results" / dataset / "softmax" / "mapping.txt",
        "checkpoint": root / "models" / dataset / f"split_{fold}" / f"epoch-{epoch}.model",
    }


def study_config(
    *, data_root: Path, mstcn2_root: Path, asformer_root: Path
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "phase": "A_inventory_and_reconciliation_only",
        "data_root": str(data_root.resolve()),
        "backbone_roots": {
            "mstcn2": str(mstcn2_root.resolve()),
            "asformer": str(asformer_root.resolve()),
        },
        "backbones": {
            key: {
                "epochs": int(value["epochs"]),
                "paper": str(value["paper"]),
            }
            for key, value in BACKBONES.items()
        },
        "datasets": DATASETS,
        "published_metrics": {
            f"{backbone}__{dataset}": metrics
            for (backbone, dataset), metrics in PUBLISHED_METRICS.items()
        },
        "reconciliation": RECONCILIATION,
        "phase_b_planning_estimate": PHASE_B_PLANNING_ESTIMATE,
        "phase_c_launch_allowed": False,
        "gpu_training_allowed": False,
        "sealed_studies_read": False,
    }
