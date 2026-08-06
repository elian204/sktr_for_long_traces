#!/usr/bin/env python3
"""Contracts for val-selected MS-TCN++ checkpoints and the Phase-C firewall."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from phase_b_training_common import DATASETS, canonical_digest, file_sha256


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROTOCOL_VERSION = "cross-backbone-phase-b-step0-selection-v1"
DEFAULT_PARENT_STUDY = Path(
    "/data1/eli-bogdanov/sktr_runs/cross_backbone_phase_b_mstcn2_v2"
)
DEFAULT_OPTION0_STUDY = Path(
    "/data1/eli-bogdanov/sktr_runs/cross_backbone_phase_b_option0_review_v1"
)
DEFAULT_DATA_ROOT = Path("/home/dsi/eli-bogdanov/data/data")
DEFAULT_REVIEW_STUDY = Path(
    "/data1/eli-bogdanov/sktr_runs/cross_backbone_phase_b_step0_review_v1"
)
STALE_CACHE_ROOT = Path("/home/dsi/eli-bogdanov/cross_backbone_pred_cache")
CARVE_NAMESPACE = "cross-backbone-step0-carve-v1"
CARVE_FRACTION = 0.15
EPOCH_GRID = tuple([*range(5, 101, 5), 96, 97, 98, 99])
EPOCH_GRID = tuple(sorted(set(EPOCH_GRID)))
COMPOSITE_METRICS = ("edit", "f1@10", "f1@25")
INFORMATIVE_MAX_PEAK_EPOCH = 85
INFORMATIVE_MIN_DECLINE_TO_EPOCH100 = 5.0
ANALYSIS_FPS = 15.0
PHASE_C_SOURCE_DISCLOSURE = {
    "asformer": "author-released checkpoints; Breakfast selected checkpoint is release epoch 120",
    "diffact": "author-released checkpoints/official release exports",
    "mstcn2": "our training with checkpoint selected only by carved training-fold validation",
}
PHASE_C_METRIC_ADDITIONS = {
    "analysis_fps": ANALYSIS_FPS,
    "absolute_error_rates": ["errors_per_gt_segment", "errors_per_minute"],
    "report_alongside": ["error_bucket_shares", "frame_weighted", "per_video_macro"],
    "breakfast_checkpoint_sensitivity": {
        "backbones": ["mstcn2", "asformer"],
        "checkpoints": ["selected", "epoch100", "epoch30"],
        "asformer_selected_definition": "author release epoch120",
        "descriptive_only": True,
    },
}
SOURCE_FILES = (
    "scripts/cross_backbone_error_audit/README.md",
    "scripts/cross_backbone_error_audit/common.py",
    "scripts/cross_backbone_error_audit/metrics.py",
    "scripts/cross_backbone_error_audit/phase_b_training_common.py",
    "scripts/cross_backbone_error_audit/phase_b_selection_common.py",
    "scripts/cross_backbone_error_audit/prepare_phase_b_step0.py",
    "scripts/cross_backbone_error_audit/run_phase_b_epoch_grid.py",
    "scripts/cross_backbone_error_audit/finalize_phase_b_step0.py",
    "scripts/cross_backbone_error_audit/finalize_phase_b_trajectory.py",
    "scripts/cross_backbone_error_audit/run_phase_b_retrain_cell.py",
    "scripts/cross_backbone_error_audit/export_phase_b_selected.py",
    "scripts/cross_backbone_error_audit/finalize_phase_b_selected.py",
    "scripts/cross_backbone_error_audit/phase_c_input_guard.py",
    "tests/test_cross_backbone_phase_b_step0.py",
)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def normalize_case(value: str) -> str:
    return Path(value.strip()).name.rsplit(".", 1)[0]


def source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = file_sha256(path)
    return result


def source_provenance() -> dict[str, Any]:
    hashes = source_hashes()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "git_head": head,
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
        raise RuntimeError(f"Phase-B Step-0 source drift: {changed}")
    if canonical_digest(actual) != expected["source_digest"]:
        raise RuntimeError("Phase-B Step-0 source digest drift")


def manifest_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_digest(
        [{"role": str(row["role"]), "sha256": str(row["sha256"])} for row in rows]
    )


def verify_manifest(manifest: Mapping[str, Any], *, full_hash: bool = True) -> dict[str, Path]:
    rows = list(manifest["files"])
    if len(rows) != int(manifest["file_count"]):
        raise RuntimeError("Step-0 manifest count drift")
    paths: dict[str, Path] = {}
    compact: list[dict[str, str]] = []
    for row in rows:
        role = str(row["role"])
        path = Path(row["path"])
        if role in paths or not path.is_file():
            raise RuntimeError(f"Missing or duplicate Step-0 input: {role}")
        if int(path.stat().st_size) != int(row["size_bytes"]):
            raise RuntimeError(f"Step-0 input size drift: {role}")
        observed = file_sha256(path) if full_hash else str(row["sha256"])
        if observed != row["sha256"]:
            raise RuntimeError(f"Step-0 input hash drift: {role}")
        paths[role] = path
        compact.append({"role": role, "sha256": observed})
    if canonical_digest(compact) != manifest["manifest_digest"]:
        raise RuntimeError("Step-0 manifest digest drift")
    return paths


def carve_hash(dataset: str, fold: int, case_id: str) -> str:
    payload = f"{CARVE_NAMESPACE}|{dataset}|{int(fold)}|{normalize_case(case_id)}".encode()
    return hashlib.sha256(payload).hexdigest()


def deterministic_carve(dataset: str, fold: int, cases: Sequence[str]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    normalized = [normalize_case(value) for value in cases]
    if len(normalized) != len(set(normalized)) or not normalized:
        raise ValueError(f"Invalid training bundle for {dataset}/fold{fold}")
    ranked = sorted(
        ((carve_hash(dataset, fold, case_id), case_id) for case_id in normalized),
        key=lambda row: (row[0], row[1]),
    )
    carve_count = max(1, int(math.ceil(CARVE_FRACTION * len(ranked))))
    validation_set = {case_id for _, case_id in ranked[:carve_count]}
    train = [case_id for case_id in normalized if case_id not in validation_set]
    validation = [case_id for case_id in normalized if case_id in validation_set]
    audit = [
        {
            "dataset": dataset,
            "fold": int(fold),
            "case_id": case_id,
            "hash_sha256": digest,
            "hash_rank": rank,
            "partition": "carved_validation" if case_id in validation_set else "train_remainder",
        }
        for rank, (digest, case_id) in enumerate(ranked, start=1)
    ]
    return train, validation, audit


def checkpoint_composite(row: Mapping[str, Any]) -> float:
    return sum(float(row[key]) for key in COMPOSITE_METRICS) / len(COMPOSITE_METRICS)


def select_best_checkpoint(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    if {int(row["epoch"]) for row in values} != set(EPOCH_GRID):
        raise RuntimeError("Checkpoint grid is incomplete")
    for row in values:
        row["composite"] = checkpoint_composite(row)
    return max(values, key=lambda row: (float(row["composite"]), -int(row["epoch"])))


def informative_breakfast_fold(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    best = select_best_checkpoint(values)
    epoch100 = next(row for row in values if int(row["epoch"]) == 100)
    epoch100_composite = checkpoint_composite(epoch100)
    decline = float(best["composite"] - epoch100_composite)
    informative = (
        int(best["epoch"]) <= INFORMATIVE_MAX_PEAK_EPOCH
        and decline >= INFORMATIVE_MIN_DECLINE_TO_EPOCH100
    )
    return {
        "best_epoch": int(best["epoch"]),
        "best_composite": float(best["composite"]),
        "epoch100_composite": float(epoch100_composite),
        "decline_to_epoch100_pp": decline,
        "informative": bool(informative),
    }


def choose_step0_branch(fold_results: Mapping[int, Mapping[str, Any]]) -> str:
    """Choose the declared branch without consulting any test-trajectory result."""
    expected = set(range(1, int(DATASETS["breakfast"]["folds"]) + 1))
    if set(int(fold) for fold in fold_results) != expected:
        raise RuntimeError("Breakfast validation evidence is incomplete")
    return (
        "A_REUSE_EXISTING_CHECKPOINTS"
        if all(bool(fold_results[fold]["informative"]) for fold in sorted(expected))
        else "B_RETRAIN_WITH_HELD_OUT_VALIDATION"
    )


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def assert_not_stale_cache(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    stale = STALE_CACHE_ROOT.expanduser().resolve(strict=False)
    if is_relative_to(resolved, stale):
        raise RuntimeError(f"Forbidden stale Phase-A cache path: {resolved}")
    return resolved


def validate_phase_c_inputs(paths: Iterable[Path], allowed_roots: Iterable[Path]) -> list[Path]:
    roots = [assert_not_stale_cache(root) for root in allowed_roots]
    result: list[Path] = []
    for path in paths:
        resolved = assert_not_stale_cache(path)
        if not any(is_relative_to(resolved, root) for root in roots):
            raise RuntimeError(f"Phase-C input is outside digest-approved roots: {resolved}")
        result.append(resolved)
    return result
