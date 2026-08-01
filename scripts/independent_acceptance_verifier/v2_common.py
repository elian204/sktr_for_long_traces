#!/usr/bin/env python3
"""Immutable-study contracts for V2 B0 sampler validity."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROTOCOL_VERSION = "independent-verifier-v2-b0-sampler-validity-v1"
DEFAULT_STUDY_DIR = Path(
    "/data1/eli-bogdanov/sktr_runs/independent_acceptance_verifier_v2_b0_review_v1"
)
DEFAULT_V0_DIR = Path(
    "/data1/eli-bogdanov/sktr_runs/independent_acceptance_verifier_v0_review_v2"
)
DEFAULT_SELECTOR_DIR = Path(
    "/data1/eli-bogdanov/sktr_runs/breakfast_selector_allfolds_seed0_v2"
)
DEFAULT_DIFFACT_ROOT = Path(
    "/home/dsi/eli-bogdanov/sktr_for_long_traces/baselines/DiffAct"
)
DEFAULT_FEATURE_MANIFEST = Path(
    "/data1/eli-bogdanov/sktr_runs/breakfast_visual_repair_head_oof_screen_v2/"
    "oof_input_manifest.json"
)
OUTER_FOLDS = (1, 2, 3, 4)
INNER_FOLDS = (1, 2, 3)
HALO_WIDTHS = (0, 8, 16)
RESTART_TIMES = (250, 500, 750, 999)
PURE_NOISE = "pure_noise"
MAX_SAMPLES = 15
EARLY_STOP_MIN_SAMPLES = 7
EARLY_STOP_PATIENCE = 5
B1_KILL_BARS = {
    "close_below_best_of_k_delta_acc_pp": 1.4,
    "join_v1_min_best_of_k_delta_acc_pp": 1.75,
    "join_v1_min_correct_candidate_wrong_mass_pct": 35.0,
    "join_v1_min_incremental_oracle_over_visual_pp": 0.5,
}

V2_SOURCE_FILES = (
    "scripts/independent_acceptance_verifier/V2_README.md",
    "scripts/independent_acceptance_verifier/masked_diffact_sampler.py",
    "scripts/independent_acceptance_verifier/v2_common.py",
    "scripts/independent_acceptance_verifier/prepare_v2_b0.py",
    "scripts/independent_acceptance_verifier/run_v2_b0_validity.py",
    "scripts/independent_acceptance_verifier/run_v2_b1_oracle.py",
    "scripts/independent_acceptance_verifier/finalize_v2_b1.py",
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


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in V2_SOURCE_FILES:
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
        raise RuntimeError("V2 B0 source drift")
    if canonical_digest(actual) != expected["source_digest"]:
        raise RuntimeError("V2 B0 source digest drift")


def verify_manifest(manifest: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    compact: list[dict[str, str]] = []
    for row in manifest["files"]:
        role = str(row["role"])
        path = Path(row["path"])
        if role in paths or not path.is_file() or int(path.stat().st_size) != int(row["size_bytes"]):
            raise RuntimeError(f"V2 B0 input missing/duplicate/size drift: {role}")
        observed = file_sha256(path)
        if observed != row["sha256"]:
            raise RuntimeError(f"V2 B0 input hash drift: {role}")
        paths[role] = path
        compact.append({"role": role, "sha256": observed})
    if len(paths) != int(manifest["file_count"]):
        raise RuntimeError("V2 B0 input count drift")
    if canonical_digest(compact) != manifest["manifest_digest"]:
        raise RuntimeError("V2 B0 manifest digest drift")
    return paths


def checkpoint_and_config(selector_dir: Path, outer: int, inner: int) -> tuple[Path, Path]:
    if outer == 1:
        source_run = Path(
            load_json(
                selector_dir
                / "imports"
                / "outer_fold_1"
                / f"inner_fold_{inner}"
                / "import_complete.json"
            )["source_run"]
        )
        config = source_run / "config.json"
    else:
        source_run = selector_dir / "runs" / f"outer_fold_{outer}" / f"inner_fold_{inner}"
        config = source_run / "config.json"
    naming = load_json(config)["naming"]
    checkpoint = source_run / "training" / naming / "epoch-1000.model"
    return checkpoint, config
