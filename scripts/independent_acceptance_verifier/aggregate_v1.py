#!/usr/bin/env python3
"""Aggregate the twelve nested-OOF verifier rotations and apply the V1 gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from v1_common import (
    INNER_FOLDS,
    OUTER_FOLDS,
    V1_GATES,
    atomic_write_json,
    canonical_digest,
    combine_sufficient,
    file_sha256,
    load_json,
    metrics_from_sufficient,
    verify_flat_manifest,
    verify_source_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    return parser.parse_args()


def summarize_rotations(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline_sufficient = combine_sufficient(
        [row["evaluation"]["baseline_sufficient"] for row in rows]
    )
    final_sufficient = combine_sufficient(
        [row["evaluation"]["final_sufficient"] for row in rows]
    )
    baseline = metrics_from_sufficient(baseline_sufficient)
    final = metrics_from_sufficient(final_sufficient)
    return {
        "baseline_sufficient": baseline_sufficient,
        "final_sufficient": final_sufficient,
        "baseline_metrics": baseline,
        "final_metrics": final,
        "delta_metrics": {metric: final[metric] - baseline[metric] for metric in baseline},
        "fixed_frames": sum(int(row["evaluation"]["fixed_frames"]) for row in rows),
        "broken_frames": sum(int(row["evaluation"]["broken_frames"]) for row in rows),
        "accepted_spans": sum(int(row["evaluation"]["accepted_spans"]) for row in rows),
    }


def held_inner_rule(deltas: Sequence[float]) -> bool:
    positives = sum(value > 0 for value in deltas)
    return positives == 3 or (positives >= 2 and min(deltas) >= -0.1)


def execute(study_dir: Path) -> None:
    metadata = load_json(study_dir / "study_metadata.json")
    config = load_json(study_dir / "study_config.json")
    manifest = load_json(study_dir / "input_manifest.json")
    verify_source_provenance(metadata["source_provenance"])
    verify_flat_manifest(manifest)
    if not config["gpu_training_allowed"] or not config.get("fable_approval_digest"):
        raise RuntimeError("V1 aggregation is review-blocked")
    if config["outer_test_open_allowed"] or config["v3_outer_evaluation_allowed"]:
        raise RuntimeError("V1 aggregation cannot authorize outer evaluation")

    rotations: list[dict[str, Any]] = []
    video_parts: list[pd.DataFrame] = []
    for outer in OUTER_FOLDS:
        for held in INNER_FOLDS:
            directory = study_dir / "rotations" / f"outer{outer}_held{held}"
            complete = load_json(directory / "rotation_complete.json")
            if (
                complete.get("status") != "complete"
                or int(complete["outer_fold"]) != outer
                or int(complete["held_inner"]) != held
            ):
                raise RuntimeError(f"Rotation completion drift: outer{outer}/held{held}")
            for name, expected in complete["output_sha256"].items():
                if file_sha256(directory / name) != expected:
                    raise RuntimeError(f"Rotation output hash drift: outer{outer}/held{held}/{name}")
            rotations.append(complete)
            video_parts.append(pd.read_csv(directory / "held_per_video.csv"))

    pooled = summarize_rotations(rotations)
    per_inner: dict[str, Any] = {}
    inner_delta_acc: list[float] = []
    for held in INNER_FOLDS:
        summary = summarize_rotations(
            [row for row in rotations if int(row["held_inner"]) == held]
        )
        per_inner[str(held)] = summary
        inner_delta_acc.append(float(summary["delta_metrics"]["acc"]))
    videos = pd.concat(video_parts, ignore_index=True)
    fixed = int(pooled["fixed_frames"])
    broken = int(pooled["broken_frames"])
    ratio = float(fixed / broken) if broken else (1e12 if fixed else 0.0)
    checks = {
        "pooled_delta_acc": float(pooled["delta_metrics"]["acc"])
        >= float(V1_GATES["pooled_delta_acc_min_pp"]),
        "held_inner_consistency": held_inner_rule(inner_delta_acc),
        "worst_video": float(videos.delta_acc.min())
        >= float(V1_GATES["worst_video_delta_acc_min_pp"]),
        "fixed_to_broken_ratio": ratio
        >= float(V1_GATES["fixed_to_broken_changed_frames_min"]),
        "pooled_edit": float(pooled["delta_metrics"]["edit"])
        >= float(V1_GATES["pooled_delta_edit_min_pp"]),
        "pooled_f1_at_25": float(pooled["delta_metrics"]["f1@25"])
        >= float(V1_GATES["pooled_delta_f1_at_25_min_pp"]),
    }
    passed = all(checks.values())
    decision = {
        "v1_status": "PASS" if passed else "FAIL",
        "checks": checks,
        "pooled": pooled,
        "per_held_inner": per_inner,
        "held_inner_delta_acc": inner_delta_acc,
        "worst_video_delta_acc": float(videos.delta_acc.min()),
        "fixed_to_broken_ratio": ratio,
        "outer_evaluation_authorized": False,
        "next_action": (
            "Fable review and frozen-policy construction; V3 remains separately gated"
            if passed
            else "Close campaign at V1; no outer evaluation"
        ),
    }
    decision["decision_digest"] = canonical_digest(decision)
    result_dir = study_dir / "results"
    videos.to_csv(result_dir / "v1_per_video.csv", index=False)
    atomic_write_json(result_dir / "v1_gate_decision.json", decision)
    lines = [
        "# Independent temporal verifier — V1 nested-OOF findings",
        "",
        f"**V1 gate: {decision['v1_status']}.** Pooled ΔAcc {pooled['delta_metrics']['acc']:+.3f} pp, ΔEdit {pooled['delta_metrics']['edit']:+.3f}, ΔF1@25 {pooled['delta_metrics']['f1@25']:+.3f}.",
        "",
        f"Held-inner ΔAcc values: {', '.join(f'{value:+.3f}' for value in inner_delta_acc)} pp. Worst video: {videos.delta_acc.min():+.3f} pp. Fixed:broken changed-frame ratio: {ratio:.3f}.",
        "",
        "This is nested OOF evidence only. Outer-test access remains disabled. A failure closes the campaign; a pass still requires Fable review and a separately frozen V3 policy before the single sealed attempt.",
    ]
    (result_dir / "findings.md").write_text("\n".join(lines) + "\n")
    complete = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "decision_digest": decision["decision_digest"],
        "outer_test_opened": False,
        "v3_authorized": False,
        "output_sha256": {
            "v1_per_video.csv": file_sha256(result_dir / "v1_per_video.csv"),
            "v1_gate_decision.json": file_sha256(result_dir / "v1_gate_decision.json"),
            "findings.md": file_sha256(result_dir / "findings.md"),
        },
    }
    atomic_write_json(result_dir / "v1_complete.json", complete)


def main() -> int:
    args = parse_args()
    execute(args.study_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
