#!/usr/bin/env python3
"""Build the OOF-only independent-verifier candidate corpus."""

from __future__ import annotations

import argparse
import fcntl
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from common import (
    CORPUS_RELATIVE,
    DIFFACT_TOP_K,
    INNER_FOLDS,
    N_CLASSES,
    OUTER_FOLDS,
    PRIMARY_BUDGET,
    SELECTOR_ANALYSIS_RELATIVE,
    VISUAL_PROBABILITY_RELATIVE,
    VISUAL_TOP_K,
    atomic_write_json,
    canonical_digest,
    file_sha256,
    load_json,
    select_budget_rows,
    validate_oof_corpus,
    verify_source_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    return parser.parse_args()


def verify_manifest(manifest: Mapping[str, Any]) -> dict[str, Path]:
    if manifest.get("outer_test_roles") or manifest.get("sealed_outer_opened"):
        raise RuntimeError("V0 manifest unexpectedly includes an outer-test input")
    rows = list(manifest["files"])
    if len(rows) != int(manifest["file_count"]):
        raise RuntimeError("Input manifest file-count mismatch")
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
            raise RuntimeError(f"Input size changed: {role}")
        observed = file_sha256(path)
        if observed != row["sha256"]:
            raise RuntimeError(f"Input hash changed: {role}")
        compact.append({"role": role, "sha256": observed})
        paths[role] = path
    if canonical_digest(compact) != manifest["manifest_digest"]:
        raise RuntimeError("Input manifest digest mismatch")
    return paths


def read_video_index(path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        index, case_id = line.split("\t", maxsplit=1)
        value = int(index)
        if value in mapping:
            raise ValueError(f"Duplicate video index {value} in {path}")
        mapping[value] = case_id
    return mapping


def load_ground_truth(paths: Mapping[str, Path]) -> dict[tuple[int, int, str], np.ndarray]:
    result: dict[tuple[int, int, str], np.ndarray] = {}
    for outer in OUTER_FOLDS:
        for inner in INNER_FOLDS:
            video_map = read_video_index(paths[f"ground_truth/outer{outer}/inner{inner}/video_index"])
            frame = pd.read_csv(paths[f"ground_truth/outer{outer}/inner{inner}/rows"])
            required = {"case:concept:name", "concept:name"}
            if not required.issubset(frame.columns):
                raise ValueError(f"GT CSV is missing columns for outer{outer}/inner{inner}")
            frame["case:concept:name"] = frame["case:concept:name"].astype(int)
            for local_index, group in frame.groupby("case:concept:name", sort=False):
                if int(local_index) not in video_map:
                    raise ValueError(f"GT case index missing from map: outer{outer}/inner{inner}/{local_index}")
                values = group["concept:name"].to_numpy(dtype=np.int16)
                if len(values) == 0 or values.min() < 0 or values.max() >= N_CLASSES:
                    raise ValueError(f"Invalid GT labels: outer{outer}/inner{inner}/{local_index}")
                key = (outer, inner, str(video_map[int(local_index)]))
                if key in result:
                    raise ValueError(f"Duplicate GT key: {key}")
                result[key] = values
    return result


def validate_partitions(corpus: pd.DataFrame, ground_truth: Mapping[tuple[int, int, str], np.ndarray]) -> None:
    seen_cases: set[tuple[int, int, str]] = set()
    for keys, group in corpus.groupby(["outer_fold", "inner_fold", "case_id"], sort=True):
        outer, inner, case_id = int(keys[0]), int(keys[1]), str(keys[2])
        key = (outer, inner, case_id)
        if key not in ground_truth:
            raise KeyError(f"Missing GT for OOF case {key}")
        ordered = group.sort_values(["segment_index", "start"], kind="mergesort")
        starts = ordered.start.to_numpy(dtype=int)
        ends = ordered.end.to_numpy(dtype=int)
        gt = ground_truth[key]
        if starts[0] != 0 or ends[-1] != len(gt) or np.any(starts[1:] != ends[:-1]):
            raise ValueError(f"Predicted segments do not partition OOF case {key}")
        for row in ordered.itertuples():
            span_gt = gt[int(row.start):int(row.end)]
            counts = np.bincount(span_gt, minlength=N_CLASSES)
            target = int(row.correct_label)
            if counts[target] != counts.max():
                raise ValueError(f"Stored GT-majority label drift: segment {row.segment_id}")
            observed_fraction = float(counts[target] / len(span_gt))
            if abs(observed_fraction - float(row.correct_label_fraction)) > 1e-9:
                raise ValueError(f"Stored GT-majority fraction drift: segment {row.segment_id}")
        seen_cases.add(key)
    unused = set(ground_truth) - seen_cases
    if unused:
        raise ValueError(f"Ground-truth inputs contain unexpected cases: {list(sorted(unused))[:5]}")


def majority_label(labels: np.ndarray) -> tuple[int, float, bool]:
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Majority label requires a non-empty 1-D label array")
    counts = np.bincount(labels.astype(int), minlength=N_CLASSES)
    label = int(np.argmax(counts))
    maximum = int(counts[label])
    is_unique = int(np.sum(counts == maximum)) == 1
    return label, float(maximum / len(labels)), bool(is_unique and maximum > len(labels) / 2)


def candidate_net_effect(gt: np.ndarray, incumbent: int, candidate: int) -> tuple[int, int, int, str]:
    incumbent_correct = int(np.sum(gt == int(incumbent)))
    candidate_correct = int(np.sum(gt == int(candidate)))
    net = candidate_correct - incumbent_correct
    effect = "helpful" if net > 0 else "harmful" if net < 0 else "lateral"
    return incumbent_correct, candidate_correct, net, effect


def build_label_map(corpus: pd.DataFrame) -> dict[int, str]:
    candidates: dict[int, set[str]] = defaultdict(set)
    for class_column, name_column in [
        ("predicted_label", "predicted_label_name"),
        ("correct_label", "correct_label_name"),
    ]:
        for class_id, label in zip(corpus[class_column], corpus[name_column]):
            candidates[int(class_id)].add(str(label))
    for rank in range(1, DIFFACT_TOP_K + 1):
        for class_id, label in zip(
            corpus[f"candidate_rank_{rank}_class_id"],
            corpus[f"candidate_rank_{rank}_label"],
        ):
            candidates[int(class_id)].add(str(label))
    conflicts = {class_id: names for class_id, names in candidates.items() if len(names) != 1}
    if conflicts:
        raise ValueError(f"Conflicting class names: {conflicts}")
    if set(candidates) != set(range(N_CLASSES)):
        raise ValueError("Candidate corpus does not resolve all 48 class names")
    return {class_id: next(iter(names)) for class_id, names in candidates.items()}


def execute(study_dir: Path) -> None:
    metadata = load_json(study_dir / "study_metadata.json")
    config = load_json(study_dir / "study_config.json")
    manifest = load_json(study_dir / "input_manifest.json")
    verify_source_provenance(metadata["source_provenance"])
    paths = verify_manifest(manifest)
    forbidden = [
        "outer_test_open_allowed", "v1_training_allowed", "v2_sampling_allowed", "v3_outer_evaluation_allowed"
    ]
    if any(bool(config[field]) for field in forbidden):
        raise RuntimeError("V0 config authorizes a forbidden later-stage action")

    corpus = pd.read_csv(paths["selector/oof_segment_corpus"])
    validate_oof_corpus(corpus)
    ground_truth = load_ground_truth(paths)
    validate_partitions(corpus, ground_truth)
    id_to_name = build_label_map(corpus)

    archive = np.load(paths["visual_oof/probabilities"], allow_pickle=False)
    if set(archive.files) != {"segment_id", "plain_logistic", "isotonic_logistic", "logistic_mlp_average"}:
        raise ValueError(f"Unexpected visual OOF archive keys: {archive.files}")
    visual_ids = archive["segment_id"].astype(np.int64)
    visual_probability = archive["plain_logistic"].astype(np.float64)
    if visual_probability.shape != (len(corpus), N_CLASSES):
        raise ValueError(f"Unexpected visual probability shape: {visual_probability.shape}")
    if not np.isfinite(visual_probability).all() or np.max(np.abs(visual_probability.sum(axis=1) - 1.0)) > 1e-8:
        raise ValueError("Visual-head probabilities are not finite normalized distributions")
    if set(visual_ids.tolist()) != set(corpus.segment_id.astype(int).tolist()):
        raise ValueError("Visual-head segment IDs do not match the OOF corpus")
    visual_by_segment = {int(segment_id): visual_probability[index] for index, segment_id in enumerate(visual_ids)}

    selected_parts: list[pd.DataFrame] = []
    budget_rows: list[dict[str, Any]] = []
    for fold in OUTER_FOLDS:
        fold_corpus = corpus[corpus.outer_fold.astype(int) == fold].copy()
        total_frames = int(fold_corpus.length.astype(int).sum())
        selected = select_budget_rows(fold_corpus, total_frames, PRIMARY_BUDGET)
        selected_parts.append(selected)
        budget_rows.append(
            {
                "outer_fold": fold,
                "total_oof_frames": total_frames,
                "requested_budget": PRIMARY_BUDGET,
                "requested_frames": int(round(PRIMARY_BUDGET * total_frames)),
                "selected_frames": int(selected.selected_frames.sum()),
                "selected_spans": int(len(selected)),
                "partial_cutoff_spans": int(selected.is_partial_budget_cutoff.sum()),
            }
        )
    selected = pd.concat(selected_parts, ignore_index=True)
    corpus_by_segment = corpus.set_index("segment_id", drop=False)
    span_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for span_id, selection in enumerate(selected.itertuples(index=False)):
        row = corpus_by_segment.loc[int(selection.segment_id)]
        key = (int(selection.outer_fold), int(selection.inner_fold), str(selection.case_id))
        gt = ground_truth[key][int(selection.selected_start):int(selection.selected_end)]
        gt_majority, gt_fraction, gt_majority_strict = majority_label(gt)
        incumbent = int(row.predicted_label)
        visual = visual_by_segment[int(selection.segment_id)]
        visual_order = np.argsort(-visual, kind="stable")[:VISUAL_TOP_K]

        source_records: list[dict[str, Any]] = [
            {
                "class_id": incumbent,
                "source": "incumbent",
                "source_rank": 1,
                "source_probability": float(row.pred_probability_mean),
            }
        ]
        for rank, class_id in enumerate(visual_order, start=1):
            source_records.append(
                {
                    "class_id": int(class_id),
                    "source": "visual_head_plain_logistic",
                    "source_rank": rank,
                    "source_probability": float(visual[int(class_id)]),
                }
            )
        for rank in range(1, DIFFACT_TOP_K + 1):
            source_records.append(
                {
                    "class_id": int(row[f"candidate_rank_{rank}_class_id"]),
                    "source": "diffact_segment_mean_softmax",
                    "source_rank": rank,
                    "source_probability": float(row[f"candidate_rank_{rank}_mean_probability"]),
                }
            )

        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for source in source_records:
            grouped[int(source["class_id"])].append(source)
        for class_id in sorted(grouped):
            sources = grouped[class_id]
            incumbent_correct, candidate_correct, net_effect, effect = candidate_net_effect(
                gt, incumbent, class_id
            )
            ranks = {source["source"]: int(source["source_rank"]) for source in sources}
            probabilities = {
                source["source"]: float(source["source_probability"]) for source in sources
            }
            candidate_rows.append(
                {
                    "v0_span_id": span_id,
                    "segment_id": int(selection.segment_id),
                    "outer_fold": int(selection.outer_fold),
                    "inner_fold": int(selection.inner_fold),
                    "case_id": str(selection.case_id),
                    "selected_start": int(selection.selected_start),
                    "selected_end": int(selection.selected_end),
                    "selected_frames": int(selection.selected_frames),
                    "incumbent_class_id": incumbent,
                    "incumbent_label": id_to_name[incumbent],
                    "candidate_class_id": class_id,
                    "candidate_label": id_to_name[class_id],
                    "candidate_sources": "|".join(sorted(ranks)),
                    "is_incumbent": class_id == incumbent,
                    "visual_head_rank": ranks.get("visual_head_plain_logistic"),
                    "visual_head_probability": probabilities.get("visual_head_plain_logistic"),
                    "diffact_rank": ranks.get("diffact_segment_mean_softmax"),
                    "diffact_mean_probability": probabilities.get("diffact_segment_mean_softmax"),
                    "gt_majority_class_id": gt_majority,
                    "gt_majority_label": id_to_name[gt_majority],
                    "gt_majority_fraction": gt_fraction,
                    "gt_majority_is_strict": gt_majority_strict,
                    "candidate_matches_gt_majority": class_id == gt_majority,
                    "incumbent_correct_frames": incumbent_correct,
                    "candidate_correct_frames": candidate_correct,
                    "net_frame_effect": net_effect,
                    "candidate_effect": effect,
                    "v2_inpainting_candidate": False,
                    "v2_inpainting_cluster_id": None,
                    "v2_inpainting_medoid_rank": None,
                }
            )

        wrong_frames = int(np.sum(gt != incumbent))
        visual_labels = {int(value) for value in visual_order}
        diffact_labels = {
            int(row[f"candidate_rank_{rank}_class_id"]) for rank in range(1, DIFFACT_TOP_K + 1)
        }
        union_labels = set(grouped)
        span_rows.append(
            {
                "v0_span_id": span_id,
                **selection._asdict(),
                "incumbent_class_id": incumbent,
                "incumbent_label": id_to_name[incumbent],
                "gt_majority_class_id": gt_majority,
                "gt_majority_label": id_to_name[gt_majority],
                "gt_majority_fraction": gt_fraction,
                "gt_majority_is_strict": gt_majority_strict,
                "selected_wrong_frames": wrong_frames,
                "selected_error_fraction": float(wrong_frames / len(gt)),
                "visual_head_gt_majority_available": gt_majority in visual_labels,
                "diffact_gt_majority_available": gt_majority in diffact_labels,
                "union_gt_majority_available": gt_majority in union_labels,
                "candidate_union_size": len(union_labels),
                "visual_head_unique_candidates": len(visual_labels),
                "diffact_unique_candidates": len(diffact_labels),
            }
        )

    spans = pd.DataFrame(span_rows)
    candidates = pd.DataFrame(candidate_rows)
    budgets = pd.DataFrame(budget_rows)
    if candidates.duplicated(["v0_span_id", "candidate_class_id"]).any():
        raise AssertionError("Candidate union contains duplicate labels within a span")
    if not set(candidates.v0_span_id.astype(int)) == set(spans.v0_span_id.astype(int)):
        raise AssertionError("Every flagged span must have candidates")

    source_summary_rows: list[dict[str, Any]] = []
    for source, rank_column in [
        ("visual_head_plain_logistic", "visual_head_rank"),
        ("diffact_segment_mean_softmax", "diffact_rank"),
        ("union", None),
    ]:
        if rank_column is None:
            available = spans.union_gt_majority_available.astype(bool)
        else:
            available_ids = set(
                candidates.loc[candidates[rank_column].notna() & candidates.candidate_matches_gt_majority, "v0_span_id"].astype(int)
            )
            available = spans.v0_span_id.astype(int).isin(available_ids)
        source_summary_rows.append(
            {
                "candidate_source": source,
                "flagged_spans": int(len(spans)),
                "flagged_frames": int(spans.selected_frames.sum()),
                "wrong_frames": int(spans.selected_wrong_frames.sum()),
                "gt_majority_available_spans": int(available.sum()),
                "gt_majority_available_span_pct": 100.0 * float(available.mean()),
                "wrong_frames_in_available_spans": int(spans.loc[available, "selected_wrong_frames"].sum()),
                "wrong_frame_mass_coverage_pct": 100.0
                * float(spans.loc[available, "selected_wrong_frames"].sum())
                / max(int(spans.selected_wrong_frames.sum()), 1),
            }
        )
    source_summary = pd.DataFrame(source_summary_rows)

    result_dir = study_dir / "results"
    budgets.to_csv(result_dir / "frozen_5pct_budget_audit.csv", index=False)
    spans.to_csv(result_dir / "flagged_oof_spans.csv", index=False)
    candidates.to_csv(result_dir / "candidate_corpus.csv", index=False)
    source_summary.to_csv(result_dir / "candidate_availability_summary.csv", index=False)
    schema = {
        "protocol_version": config["protocol_version"],
        "row_grain": "one deduplicated candidate label for one selector-flagged OOF span",
        "selection": "exact 5% frames independently per outer fold using frozen base_score and centered partial cutoff",
        "candidate_sources": {
            "incumbent": "current official DiffAct segment label",
            "visual_head_plain_logistic": "top 3 from the existing nested-OOF plain visual logistic probability matrix",
            "diffact_segment_mean_softmax": "top 5 saved in the OOF repair corpus",
        },
        "label_contract": {
            "candidate_matches_gt_majority": "candidate equals deterministic GT modal class on the selected span",
            "gt_majority_is_strict": "modal class is unique and occupies >50% of selected frames",
            "net_frame_effect": "candidate-correct frames minus incumbent-correct frames on selected span",
            "candidate_effect": "helpful if net>0, harmful if net<0, lateral otherwise",
        },
        "v2_reserved_columns": [
            "v2_inpainting_candidate", "v2_inpainting_cluster_id", "v2_inpainting_medoid_rank"
        ],
        "outer_test_opened": False,
    }
    schema["schema_digest"] = canonical_digest(schema)
    atomic_write_json(result_dir / "candidate_corpus_schema.json", schema)

    total_frames = int(spans.selected_frames.sum())
    wrong_frames = int(spans.selected_wrong_frames.sum())
    lines = [
        "# Independent-acceptance verifier — V0 findings",
        "",
        f"V0 assembled {len(candidates):,} unique candidate rows for {len(spans):,} selector-flagged OOF spans ({total_frames:,} frames; {wrong_frames:,} currently wrong).",
        "",
        "| Source | GT-majority available spans | Wrong-frame mass covered |",
        "|---|---:|---:|",
    ]
    for row in source_summary.itertuples(index=False):
        lines.append(
            f"| {row.candidate_source} | {row.gt_majority_available_span_pct:.1f}% | {row.wrong_frame_mass_coverage_pct:.1f}% |"
        )
    lines.extend(
        [
            "",
            "This is candidate availability, not realized repair performance. Candidate correctness and net-frame effect use OOF GT solely to train/evaluate the future verifier; no outer-test row or sealed outer input was opened.",
            "",
            "V1 and V2 remain blocked for Fable review of the schema, exact-budget masks, source hashes, and candidate-label contract.",
        ]
    )
    (result_dir / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    paths_after = verify_manifest(manifest)
    if paths_after != paths:
        raise RuntimeError("Input path mapping changed during V0")
    outputs = sorted(path for path in result_dir.iterdir() if path.is_file())
    complete = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest_digest": manifest["manifest_digest"],
        "source_digest": metadata["source_provenance"]["source_digest"],
        "schema_digest": schema["schema_digest"],
        "outer_test_opened": False,
        "gpu_used": False,
        "v1_v2_v3_authorized": False,
        "output_sha256": {path.name: file_sha256(path) for path in outputs},
    }
    atomic_write_json(result_dir / "v0_complete.json", complete)


def main() -> int:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    lock_path = study_dir / ".v0.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("V0 is already running") from error
        try:
            execute(study_dir)
        except Exception as error:
            atomic_write_json(
                study_dir / "results" / "v0_failed.json",
                {
                    "status": "failed",
                    "failed_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
