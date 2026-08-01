#!/usr/bin/env python3
"""Aggregate the twelve OOF B1 candidate-oracle tasks and apply kill bars."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from v2_common import (
    B1_KILL_BARS,
    INNER_FOLDS,
    OUTER_FOLDS,
    atomic_write_json,
    canonical_digest,
    file_sha256,
    load_json,
    verify_manifest,
    verify_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    return parser.parse_args()


def classify_b1(best_delta: float, availability: float, incremental: float) -> str:
    if best_delta < float(B1_KILL_BARS["close_below_best_of_k_delta_acc_pp"]):
        return "CLOSE_V2"
    if (
        best_delta >= float(B1_KILL_BARS["join_v1_min_best_of_k_delta_acc_pp"])
        and availability >= float(B1_KILL_BARS["join_v1_min_correct_candidate_wrong_mass_pct"])
        and incremental >= float(B1_KILL_BARS["join_v1_min_incremental_oracle_over_visual_pp"])
    ):
        return "PASS_CANDIDATE_JOIN_REVIEW_REQUIRED"
    return "NO_JOIN_DIAGNOSTIC_ONLY"


def execute(study: Path) -> None:
    metadata = load_json(study / "study_metadata.json")
    config = load_json(study / "study_config.json")
    manifest = load_json(study / "input_manifest.json")
    verify_source(metadata["source_provenance"])
    verify_manifest(manifest)
    if not config["b1_oracle_allowed"] or config["v1_candidate_join_allowed"] or config["outer_test_open_allowed"]:
        raise RuntimeError("Invalid V2 B1 finalization authorization")
    summaries: list[dict] = []
    candidate_parts: list[pd.DataFrame] = []
    span_parts: list[pd.DataFrame] = []
    for outer in OUTER_FOLDS:
        for inner in INNER_FOLDS:
            directory = study / "results" / f"outer{outer}_inner{inner}_b1"
            summary_path = directory / "b1_summary.json"
            status = load_json(study / "status" / f"outer{outer}_inner{inner}_b1.json")
            if status["status"] != "complete" or file_sha256(summary_path) != status["summary_sha256"]:
                raise RuntimeError(f"B1 completion drift: outer{outer}/inner{inner}")
            summary = load_json(summary_path)
            for name, expected in summary["output_sha256"].items():
                if file_sha256(directory / name) != expected:
                    raise RuntimeError(f"B1 output drift: outer{outer}/inner{inner}/{name}")
            summaries.append(summary)
            candidate_parts.append(pd.read_csv(directory / "inpainting_candidates.csv"))
            spans = pd.read_csv(directory / "span_oracle.csv")
            spans.insert(0, "inner_fold", inner)
            spans.insert(0, "outer_fold", outer)
            span_parts.append(spans)
    total_frames = sum(int(row["total_oof_frames"]) for row in summaries)
    flagged_wrong = sum(int(row["flagged_wrong_frames"]) for row in summaries)
    available_wrong = sum(int(row["correct_candidate_available_wrong_frames"]) for row in summaries)
    best_net = sum(int(row["best_of_k_net_frames"]) for row in summaries)
    visual_net = sum(int(row["visual_oracle_net_frames"]) for row in summaries)
    union_net = sum(int(row["union_oracle_net_frames"]) for row in summaries)
    best_delta = 100.0 * best_net / total_frames
    visual_delta = 100.0 * visual_net / total_frames
    union_delta = 100.0 * union_net / total_frames
    incremental = union_delta - visual_delta
    availability = 100.0 * available_wrong / flagged_wrong
    status = classify_b1(best_delta, availability, incremental)
    close = status == "CLOSE_V2"
    join = status == "PASS_CANDIDATE_JOIN_REVIEW_REQUIRED"
    result_dir = study / "results" / "b1_aggregate"
    result_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.concat(candidate_parts, ignore_index=True)
    spans = pd.concat(span_parts, ignore_index=True)
    candidate_path = result_dir / "inpainting_candidates_all_oof.csv"
    span_path = result_dir / "span_oracle_all_oof.csv"
    candidates.to_csv(candidate_path, index=False)
    spans.to_csv(span_path, index=False)
    decision = {
        "v2_b1_status": status,
        "best_of_k_delta_acc_pp": best_delta,
        "visual_candidate_oracle_delta_acc_pp": visual_delta,
        "union_candidate_oracle_delta_acc_pp": union_delta,
        "incremental_oracle_over_visual_pp": incremental,
        "correct_candidate_available_wrong_mass_pct": availability,
        "total_oof_frames": total_frames,
        "flagged_wrong_frames": flagged_wrong,
        "candidate_rows": len(candidates),
        "kill_bars": B1_KILL_BARS,
        "current_v1_study_mutated": False,
        "outer_test_opened": False,
        "next_action": (
            "Fable reviews the packaged candidates before a separately versioned V1 extension"
            if join
            else "Close V2 and do not add inpainting candidates"
            if close
            else "Report diagnosis; candidates do not join V1"
        ),
    }
    decision["decision_digest"] = canonical_digest(decision)
    decision_path = result_dir / "v2_b1_decision.json"
    atomic_write_json(decision_path, decision)
    findings = [
        "# V2 masked-inpainting candidate oracle",
        "",
        f"**Decision: {status}.** Best-of-k oracle ΔAcc is {best_delta:+.3f} pp; correct candidates cover {availability:.2f}% of flagged wrong mass; incremental union oracle over the visual pool is {incremental:+.3f} pp.",
        "",
        f"Visual-only oracle: {visual_delta:+.3f} pp. Visual + inpainting union oracle: {union_delta:+.3f} pp.",
        "",
        "This is OOF candidate availability, not realized repair. The running V1 study was not modified, and no outer-test data was opened.",
    ]
    findings_path = result_dir / "findings.md"
    findings_path.write_text("\n".join(findings) + "\n")
    complete = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "decision_digest": decision["decision_digest"],
        "output_sha256": {
            candidate_path.name: file_sha256(candidate_path),
            span_path.name: file_sha256(span_path),
            decision_path.name: file_sha256(decision_path),
            findings_path.name: file_sha256(findings_path),
        },
    }
    atomic_write_json(result_dir / "v2_b1_complete.json", complete)


def main() -> int:
    args = parse_args()
    execute(args.study_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
