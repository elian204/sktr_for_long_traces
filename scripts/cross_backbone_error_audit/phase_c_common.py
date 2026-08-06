#!/usr/bin/env python3
"""Immutable contracts for the cross-backbone Phase-C audit sweep."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROTOCOL_VERSION = "cross-backbone-phase-c-taxonomy-v1"
DEFAULT_REVIEW_STUDY = Path(
    "/data1/eli-bogdanov/sktr_runs/cross_backbone_phase_c_review_v1"
)
DEFAULT_STEP0_STUDY = Path(
    "/data1/eli-bogdanov/sktr_runs/cross_backbone_phase_b_step0_v1"
)
DEFAULT_OPTION0_STUDY = Path(
    "/data1/eli-bogdanov/sktr_runs/cross_backbone_phase_b_option0_review_v1"
)
DEFAULT_PARENT_MSTCN2_STUDY = Path(
    "/data1/eli-bogdanov/sktr_runs/cross_backbone_phase_b_mstcn2_v2"
)
DEFAULT_DIFFACT_ROOT = Path(
    "/data1/eli-bogdanov/sktr_runs/moved_from_home/sktr_for_long_traces/"
    "baselines/DiffAct/results"
)
DEFAULT_ASFORMER_CACHE = Path(
    "/data1/eli-bogdanov/sktr_runs/cross_backbone_official_checkpoints_cache/ASFormer"
)
DEFAULT_ASFORMER_SOURCE = Path(
    "/home/dsi/eli-bogdanov/ASFormer_official_checkpoint_inference"
)
DEFAULT_ASFORMER_LOCAL = Path("/home/dsi/eli-bogdanov/ASFormer")
DEFAULT_DATA_ROOT = Path("/home/dsi/eli-bogdanov/data/data")
STALE_PHASE_A_CACHE = Path("/home/dsi/eli-bogdanov/cross_backbone_pred_cache")
ANALYSIS_FPS = 15.0
BACKBONE_ORDER = ("mstcn2", "asformer", "diffact")
DATASETS = {
    "gtea": {"folds": 4, "sample_rate": 1},
    "50salads": {"folds": 5, "sample_rate": 2},
    "breakfast": {"folds": 4, "sample_rate": 1},
}
PRIMARY_BUCKETS = (
    "boundary_offset",
    "fragmentation",
    "illegal_order",
    "legal_substitution",
)
SOURCE_FILES = (
    "eval_diffact_sktr_fold1_paper.py",
    "scripts/cross_backbone_error_audit/README.md",
    "scripts/cross_backbone_error_audit/metrics.py",
    "scripts/cross_backbone_error_audit/phase_b_training_common.py",
    "scripts/cross_backbone_error_audit/phase_b_selection_common.py",
    "scripts/cross_backbone_error_audit/phase_c_input_guard.py",
    "scripts/cross_backbone_error_audit/phase_c_common.py",
    "scripts/cross_backbone_error_audit/phase_c_taxonomy.py",
    "scripts/cross_backbone_error_audit/prepare_phase_c.py",
    "scripts/cross_backbone_error_audit/materialize_phase_c_asformer.py",
    "scripts/cross_backbone_error_audit/run_phase_c_audit.py",
    "tests/test_cross_backbone_phase_c.py",
)

PHASE_C_SPEC = {
    "primary_matrix": {
        "backbones": list(BACKBONE_ORDER),
        "datasets": list(DATASETS),
        "checkpoint_sources": {
            "mstcn2": "validation-selected Phase-B checkpoints trained on train-minus-carve",
            "asformer": "author-released epoch-120 checkpoints",
            "diffact": "author-released checkpoints / official release exports",
        },
    },
    "exclusive_error_taxonomy": {
        "precedence": list(PRIMARY_BUCKETS),
        "boundary_offset": "wrong frame within +/-25 analysis frames of a GT transition, predicted as either adjacent GT label",
        "fragmentation": "remaining wrong frame in an internal <=25-frame predicted island bounded by the enclosing GT label",
        "illegal_order": "remaining wrong frame in the destination predicted segment of a fold-train DFG-illegal internal/start transition, or an illegal end segment",
        "legal_substitution": "all remaining wrong frames; split descriptively into present-nonadjacent and absent-label confusion",
        "sum_invariant": "the four exclusive frame counts equal total wrong frames for every case",
    },
    "orthogonal_readouts": {
        "boundary_width_sensitivity": [10, 25, 50],
        "wrong_label_span_mass": "error frames in predicted segments whose strict GT-majority label differs from the predicted label",
        "long_substitution": "contiguous error span >=100 frames with >=90% modal predicted label",
        "dfg": "fold-pure full official training split; GT downsampled at the model evaluation rate before collapse",
        "over_segmentation_ratio": "predicted non-background segments / GT non-background segments",
        "candidate_rank": "best rank of GT class in probability vector; top-2/3/5, median, and p90 over wrong frames",
    },
    "aggregation": {
        "frame_weighted": "sum numerators and denominators before division",
        "per_video_macro": "unweighted mean of per-video values; error-share means exclude zero-error videos and report their denominator",
        "absolute_rates": [
            "error_frames_per_gt_segment",
            "error_frames_per_minute",
            "error_spans_per_gt_segment",
            "error_spans_per_minute",
        ],
        "analysis_fps": ANALYSIS_FPS,
    },
    "model_generation_hypothesis": {
        "generation_order": list(BACKBONE_ORDER),
        "primary_endpoints": {
            "fragmentation_frames_per_minute": "decrease",
            "illegal_order_frames_per_minute": "decrease",
            "legal_substitution_share": "increase",
        },
        "test": "paired official-fold direction sign test for DiffAct minus MS-TCN++, one-sided exact binomial; Holm correction across three endpoints",
        "support_rule": "all three Holm-adjusted p<=0.05 with expected-direction pooled delta; descriptive per-dataset and monotonicity rows remain visible regardless",
        "caveat": "folds are the pre-registered unit; the three-generation table is not treated as a high-powered continuous trend",
    },
    "descriptive_sensitivity": {
        "breakfast": {
            "backbones": ["mstcn2", "asformer"],
            "arms": ["selected", "epoch100", "epoch30"],
            "asformer_selected": "author release epoch120",
            "claim_status": "descriptive_only",
        },
        "full_train_ep100_robustness": {
            "datasets": ["gtea", "50salads"],
            "backbone": "mstcn2",
            "purpose": "repeat share-direction conclusions using original full-train epoch100 exports",
            "claim_status": "pre_registered_secondary",
        },
    },
    "source_asymmetry_disclosure": {
        "asformer": "author-released checkpoints; missing official GTEA/50Salads exports materialized without training",
        "diffact": "official release exports from author-released checkpoints",
        "mstcn2": "our official-config retraining, selected on a genuine held-out carve",
        "mstcn2_ep100_secondary": "our original full-train official-config epoch100 reproduction",
        "asformer_epoch30_epoch100_sensitivity": "local reproduction checkpoints; descriptive only and excluded from headline claims",
    },
}


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


def read_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def normalize_case(value: str) -> str:
    return Path(value.strip()).name.rsplit(".", 1)[0]


def parse_mapping(path: Path) -> tuple[dict[int, str], dict[str, int]]:
    result: dict[int, str] = {}
    for line in read_nonempty_lines(path):
        index, label = line.split(maxsplit=1)
        result[int(index)] = label
    if sorted(result) != list(range(len(result))):
        raise RuntimeError(f"Non-contiguous mapping: {path}")
    return result, {label: index for index, label in result.items()}


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
            key for key in set(actual) | set(expected["file_sha256"])
            if actual.get(key) != expected["file_sha256"].get(key)
        )
        raise RuntimeError(f"Phase-C source drift: {changed}")
    if canonical_digest(actual) != expected["source_digest"]:
        raise RuntimeError("Phase-C source digest drift")


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def assert_provenance_path(path: Path, allowed_roots: Iterable[Path]) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    stale = STALE_PHASE_A_CACHE.resolve(strict=False)
    if is_relative_to(resolved, stale):
        raise RuntimeError(f"Forbidden stale Phase-A cache path: {resolved}")
    roots = [root.expanduser().resolve(strict=False) for root in allowed_roots]
    if not any(is_relative_to(resolved, root) for root in roots):
        raise RuntimeError(f"Phase-C input outside frozen provenance roots: {resolved}")
    return resolved


def manifest_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_digest(
        [{"role": str(row["role"]), "sha256": str(row["sha256"])} for row in rows]
    )


def verify_manifest(
    manifest: Mapping[str, Any], allowed_roots: Iterable[Path], *, full_hash: bool = True
) -> dict[str, Path]:
    rows = list(manifest["files"])
    if len(rows) != int(manifest["file_count"]):
        raise RuntimeError("Phase-C input count drift")
    paths: dict[str, Path] = {}
    compact: list[dict[str, str]] = []
    for row in rows:
        role = str(row["role"])
        path = assert_provenance_path(Path(row["path"]), allowed_roots)
        if role in paths or not path.is_file():
            raise RuntimeError(f"Missing or duplicate Phase-C input: {role}")
        if int(path.stat().st_size) != int(row["size_bytes"]):
            raise RuntimeError(f"Phase-C input size drift: {role}")
        observed = file_sha256(path) if full_hash else str(row["sha256"])
        if observed != row["sha256"]:
            raise RuntimeError(f"Phase-C input hash drift: {role}")
        paths[role] = path
        compact.append({"role": role, "sha256": observed})
    if canonical_digest(compact) != manifest["manifest_digest"]:
        raise RuntimeError("Phase-C input-manifest digest drift")
    return paths


def exact_one_sided_sign_p(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    return sum(
        __import__("math").comb(trials, value) for value in range(successes, trials + 1)
    ) / (2.0 ** trials)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted((float(value), key) for key, value in p_values.items())
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (value, key) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[key] = running
    return adjusted
