#!/usr/bin/env python3
"""Shared immutable-study contracts for the independent verifier campaign."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROTOCOL_VERSION = "independent-acceptance-verifier-v0"
OUTER_FOLDS = (1, 2, 3, 4)
INNER_FOLDS = (1, 2, 3)
PRIMARY_BUDGET = 0.05
N_CLASSES = 48
VISUAL_TOP_K = 3
DIFFACT_TOP_K = 5

DEFAULT_SELECTOR_STUDY = Path(
    "/data1/eli-bogdanov/sktr_runs/breakfast_selector_allfolds_seed0_v2"
)
DEFAULT_VISUAL_OOF_STUDY = Path(
    "/data1/eli-bogdanov/sktr_runs/breakfast_visual_repair_head_oof_screen_v2"
)
DEFAULT_STUDY_DIR = Path(
    "/data1/eli-bogdanov/sktr_runs/independent_acceptance_verifier_v0_review_v1"
)

SELECTOR_ANALYSIS_RELATIVE = Path("analysis/video_selector_allfolds_v2")
CORPUS_RELATIVE = SELECTOR_ANALYSIS_RELATIVE / "repair_training_corpus_segments.csv"
VISUAL_PROBABILITY_RELATIVE = Path("oof_results/crossfit_probabilities.npz")

SOURCE_FILES = (
    "scripts/independent_acceptance_verifier/README.md",
    "scripts/independent_acceptance_verifier/common.py",
    "scripts/independent_acceptance_verifier/prepare_v0.py",
    "scripts/independent_acceptance_verifier/run_v0.py",
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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_lines(path: Path, rows: Iterable[str]) -> None:
    values = [str(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def stable_uint64(*parts: Any) -> np.uint64:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    return np.uint64(int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "little"))


def git_provenance(repo: Path = REPO_ROOT) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()

    status = run("status", "--short")
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=repo, check=True, stdout=subprocess.PIPE
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
        "git": git_provenance(),
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


def validate_oof_corpus(frame: pd.DataFrame) -> None:
    required = {
        "segment_id", "outer_fold", "scope", "mode", "inner_fold", "case_id",
        "segment_index", "start", "end", "length", "predicted_label",
        "correct_label", "correct_label_fraction", "base_score",
    }
    for rank in range(1, DIFFACT_TOP_K + 1):
        required.update(
            {
                f"candidate_rank_{rank}_class_id",
                f"candidate_rank_{rank}_label",
                f"candidate_rank_{rank}_mean_probability",
            }
        )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"OOF corpus is missing columns: {missing}")
    if len(frame) != 36194:
        raise ValueError(f"Expected 36,194 OOF segments, found {len(frame)}")
    if frame["segment_id"].duplicated().any():
        raise ValueError("OOF segment IDs are not unique")
    if set(frame["scope"].astype(str)) != {"oof_validation"}:
        raise ValueError("V0 must not open outer-test segment rows")
    if set(frame["mode"].astype(str)) != {"official"}:
        raise ValueError("V0 accepts official DiffAct segments only")
    if set(frame["outer_fold"].astype(int)) != set(OUTER_FOLDS):
        raise ValueError("OOF corpus does not cover four outer folds")
    if frame["inner_fold"].isna().any() or not set(frame["inner_fold"].astype(int)).issubset(INNER_FOLDS):
        raise ValueError("OOF corpus has invalid inner-fold identity")
    if frame["base_score"].isna().any():
        raise ValueError("OOF corpus has missing frozen selector scores")
    if not (frame["end"].astype(int) - frame["start"].astype(int)).equals(frame["length"].astype(int)):
        raise ValueError("OOF segment lengths are inconsistent")


def select_budget_rows(segments: pd.DataFrame, total_frames: int, budget: float = PRIMARY_BUDGET) -> pd.DataFrame:
    requested = max(1, int(round(float(budget) * int(total_frames))))
    scoped = segments.copy()
    scoped["tie_key"] = [
        int(stable_uint64(row.scope, row.mode, row.inner_fold, row.case_id, row.segment_index))
        for row in scoped.itertuples()
    ]
    scoped = scoped.sort_values(
        ["base_score", "tie_key"], ascending=[False, True], kind="mergesort"
    )
    remaining = requested
    rows: list[dict[str, Any]] = []
    cutoffs = 0
    for row in scoped.itertuples():
        if remaining <= 0:
            break
        start, end = int(row.start), int(row.end)
        selected = min(end - start, remaining)
        selected_start = start
        if selected < end - start:
            selected_start = start + (end - start - selected) // 2
            cutoffs += 1
        rows.append(
            {
                "segment_id": int(row.segment_id),
                "outer_fold": int(row.outer_fold),
                "inner_fold": int(row.inner_fold),
                "scope": str(row.scope),
                "case_id": str(row.case_id),
                "segment_index": int(row.segment_index),
                "segment_start": start,
                "segment_end": end,
                "selected_start": selected_start,
                "selected_end": selected_start + selected,
                "selected_frames": selected,
                "is_partial_budget_cutoff": selected < end - start,
                "base_score": float(row.base_score),
                "tie_key": int(row.tie_key),
            }
        )
        remaining -= selected
    if remaining != 0 or cutoffs > 1:
        raise AssertionError(f"Exact-budget selection failed: remaining={remaining}, cutoffs={cutoffs}")
    result = pd.DataFrame(rows)
    if int(result.selected_frames.sum()) != requested:
        raise AssertionError("Selected frame count does not equal requested budget")
    return result
