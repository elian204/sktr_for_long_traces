#!/usr/bin/env python3
"""Generate OOF inpainting candidates and measure the pre-registered B1 oracle."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v2_common import (
    HALO_WIDTHS,
    INNER_FOLDS,
    OUTER_FOLDS,
    RESTART_TIMES,
    atomic_write_json,
    canonical_digest,
    file_sha256,
    load_json,
    stable_seed,
    verify_manifest,
    verify_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, choices=OUTER_FOLDS, required=True)
    parser.add_argument("--inner-fold", type=int, choices=INNER_FOLDS, required=True)
    parser.add_argument("--device", type=int, choices=(0, 1, 2, 3), required=True)
    return parser.parse_args()


def parse_mapping(path: Path) -> list[str]:
    rows = [line.split(maxsplit=1) for line in path.read_text().splitlines() if line.strip()]
    mapping = {int(index): label for index, label in rows}
    if sorted(mapping) != list(range(48)):
        raise RuntimeError("Breakfast mapping drift")
    return [mapping[index] for index in range(48)]


def read_video_index(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            index, case_id = line.split("\t", maxsplit=1)
            result[int(index)] = case_id
    return result


def load_targets(paths: dict[str, Path], outer: int, inner: int) -> dict[str, np.ndarray]:
    prefix = f"v0_nested/ground_truth/outer{outer}/inner{inner}"
    index = read_video_index(paths[f"{prefix}/video_index"])
    rows = pd.read_csv(paths[f"{prefix}/rows"])
    return {
        str(index[int(case_index)]): group["concept:name"].to_numpy(dtype=np.int16)
        for case_index, group in rows.groupby("case:concept:name", sort=False)
    }


def reconstruct_incumbent(corpus: pd.DataFrame, outer: int, inner: int, case_id: str, length: int) -> np.ndarray:
    rows = corpus[
        (corpus.outer_fold.astype(int) == outer)
        & (corpus.inner_fold.astype(int) == inner)
        & (corpus.case_id.astype(str) == case_id)
    ].sort_values(["segment_index", "start"], kind="mergesort")
    starts = rows.start.to_numpy(dtype=int)
    ends = rows.end.to_numpy(dtype=int)
    if len(rows) == 0 or starts[0] != 0 or ends[-1] != length or np.any(starts[1:] != ends[:-1]):
        raise RuntimeError(f"OOF incumbent partition drift: {outer}/{inner}/{case_id}")
    result = np.empty(length, dtype=np.int16)
    for row in rows.itertuples(index=False):
        result[int(row.start) : int(row.end)] = int(row.predicted_label)
    return result


def modal_label(labels: np.ndarray) -> int:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=48)
    return int(np.flatnonzero(counts == counts.max())[0])


def execute(study: Path, outer: int, inner: int, physical_device: int) -> None:
    metadata = load_json(study / "study_metadata.json")
    config = load_json(study / "study_config.json")
    manifest = load_json(study / "input_manifest.json")
    verify_source(metadata["source_provenance"])
    paths = verify_manifest(manifest)
    if not config["b1_oracle_allowed"] or not config.get("fable_approval_digest"):
        raise RuntimeError("V2 B1 oracle is not approved")
    if config["v1_candidate_join_allowed"] or config["outer_test_open_allowed"]:
        raise RuntimeError("B1 may neither mutate V1 nor open outer-test data")
    b0_statuses = sorted((study / "status").glob("outer*_inner*_b0.json"))
    if len(b0_statuses) != 12 or any(load_json(path)["status"] != "PASS" for path in b0_statuses):
        raise RuntimeError("All twelve B0 validity checks must pass before B1")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("V2 B1 fail-closed GPU pinning failed")
    device = torch.device("cuda:0")
    diffact_root = Path(config["diffact_root"])
    sys.path.insert(0, str(diffact_root))
    from main import Trainer

    from masked_diffact_sampler import cluster_medoid_indices, sample_candidate_medoids

    task = next(
        row
        for row in config["tasks"]
        if int(row["outer_fold"]) == outer and int(row["inner_fold"]) == inner
    )
    task_config = load_json(Path(task["config"]))
    event_list = parse_mapping(paths["breakfast/mapping"])
    trainer = Trainer(
        copy.deepcopy(task_config["encoder_params"]),
        copy.deepcopy(task_config["decoder_params"]),
        copy.deepcopy(task_config["diffusion_params"]),
        event_list,
        task_config["sample_rate"],
        task_config["temporal_aug"],
        task_config["set_sampling_seed"],
        task_config["postprocess"],
        device,
    )
    state = torch.load(Path(task["checkpoint"]), map_location="cpu")
    trainer.model.load_state_dict(state, strict=True)
    trainer.model.eval().to(device)

    nested = load_json(paths["features/nested_manifest"])
    clean_nested = dict(nested)
    expected_digest = clean_nested.pop("manifest_digest")
    if canonical_digest(clean_nested) != expected_digest:
        raise RuntimeError("Feature manifest self-digest drift")
    feature_entries = {str(row["case_id"]): row for row in nested["features"]}
    spans = pd.read_csv(paths["v0/results/flagged_oof_spans.csv"])
    spans = spans[
        (spans.outer_fold.astype(int) == outer) & (spans.inner_fold.astype(int) == inner)
    ].sort_values(["case_id", "selected_start", "v0_span_id"], kind="mergesort")
    baseline_candidates = pd.read_csv(paths["v0/results/candidate_corpus.csv"])
    baseline_candidates = baseline_candidates[
        (baseline_candidates.outer_fold.astype(int) == outer)
        & (baseline_candidates.inner_fold.astype(int) == inner)
    ]
    corpus = pd.read_csv(paths["v0/oof_segment_corpus"])
    targets = load_targets(paths, outer, inner)
    total_frames = int(sum(len(target) for target in targets.values()))
    candidate_rows: list[dict[str, Any]] = []
    span_rows: list[dict[str, Any]] = []
    setting_runs = 0
    samples_generated = 0
    early_stops = 0

    grouped = list(spans.groupby("case_id", sort=True))
    for case_number, (case_id_value, case_spans) in enumerate(grouped, start=1):
        case_id = str(case_id_value)
        target = targets[case_id]
        incumbent = reconstruct_incumbent(corpus, outer, inner, case_id, len(target))
        feature_entry = feature_entries[case_id]
        feature_path = Path(feature_entry["path"])
        if file_sha256(feature_path) != feature_entry["sha256"]:
            raise RuntimeError(f"Feature hash drift: {case_id}")
        feature = np.load(feature_path, allow_pickle=False).astype(np.float32)
        if feature.shape != (2048, len(target)):
            raise RuntimeError(f"B1 feature orientation/alignment drift: {case_id}/{feature.shape}")
        video_feature = torch.from_numpy(feature).unsqueeze(0).to(device)
        with torch.no_grad():
            if trainer.model.use_instance_norm:
                video_feature = trainer.model.ins_norm(video_feature)
            _, backbone = trainer.model.encoder(video_feature, get_features=True)
        intervals = [
            (int(row.selected_start), int(row.selected_end))
            for row in case_spans.itertuples(index=False)
        ]
        sequences: list[np.ndarray] = []
        sequence_meta: list[dict[str, Any]] = []
        seen_exact: set[bytes] = set()
        for halo in HALO_WIDTHS:
            for restart in RESTART_TIMES:
                pure_noise = int(restart) == 999
                batch = sample_candidate_medoids(
                    trainer.model,
                    backbone,
                    incumbent,
                    intervals,
                    halo_width=int(halo),
                    t_start=int(restart),
                    pure_noise=pure_noise,
                    base_seed=stable_seed("b1", outer, inner, case_id, halo, restart),
                    postprocess=task_config["postprocess"],
                )
                setting_runs += 1
                samples_generated += len(batch.samples)
                early_stops += int(batch.stopped_early)
                for medoid_rank, sample_index in enumerate(batch.medoid_indices):
                    labels = batch.samples[sample_index].labels
                    key = labels.tobytes()
                    if key in seen_exact:
                        continue
                    seen_exact.add(key)
                    sequences.append(labels.copy())
                    sequence_meta.append(
                        {
                            "halo_width": int(halo),
                            "restart_time": int(restart),
                            "pure_noise": bool(pure_noise),
                            "setting_medoid_rank": int(medoid_rank),
                        }
                    )
        global_clusters = cluster_medoid_indices(sequences, threshold=0.25)
        medoids = [(cluster_id, medoid, sequences[medoid], sequence_meta[medoid]) for cluster_id, (_, medoid) in enumerate(global_clusters)]
        for span in case_spans.itertuples(index=False):
            start, end = int(span.selected_start), int(span.selected_end)
            incumbent_correct = int(np.sum(incumbent[start:end] == target[start:end]))
            by_label: dict[int, tuple[int, np.ndarray, dict[str, Any]]] = {}
            for cluster_id, medoid_index, sequence, medoid_meta in medoids:
                label = modal_label(sequence[start:end])
                if label == int(span.incumbent_class_id) or label in by_label:
                    continue
                by_label[label] = (cluster_id, sequence, medoid_meta)
            best_inpainting_net = 0
            has_gt_majority = False
            for rank, (label, (cluster_id, sequence, medoid_meta)) in enumerate(sorted(by_label.items()), start=1):
                candidate_correct = int(np.sum(target[start:end] == label))
                net = candidate_correct - incumbent_correct
                best_inpainting_net = max(best_inpainting_net, net)
                has_gt_majority |= label == int(span.gt_majority_class_id)
                candidate_rows.append(
                    {
                        "v0_span_id": int(span.v0_span_id),
                        "outer_fold": outer,
                        "inner_fold": inner,
                        "case_id": case_id,
                        "selected_start": start,
                        "selected_end": end,
                        "incumbent_class_id": int(span.incumbent_class_id),
                        "candidate_class_id": label,
                        "candidate_correct_frames": candidate_correct,
                        "incumbent_correct_frames": incumbent_correct,
                        "net_frame_effect": net,
                        "global_cluster_id": int(cluster_id),
                        "global_medoid_rank": rank,
                        "collapsed_medoid_trace": json.dumps([int(value) for value in __import__("masked_diffact_sampler").collapse_labels(sequence)]),
                        **medoid_meta,
                    }
                )
            visual = baseline_candidates[
                (baseline_candidates.v0_span_id.astype(int) == int(span.v0_span_id))
                & pd.to_numeric(baseline_candidates.visual_head_rank, errors="coerce").notna()
            ]
            best_visual_net = max([0, *visual.net_frame_effect.astype(int).tolist()])
            span_rows.append(
                {
                    "v0_span_id": int(span.v0_span_id),
                    "selected_frames": int(span.selected_frames),
                    "selected_wrong_frames": int(span.selected_wrong_frames),
                    "inpainting_candidate_count": len(by_label),
                    "gt_majority_available": bool(has_gt_majority),
                    "best_inpainting_net_frames": int(best_inpainting_net),
                    "best_visual_net_frames": int(best_visual_net),
                    "best_union_net_frames": int(max(best_inpainting_net, best_visual_net)),
                }
            )
        print(
            f"outer={outer} inner={inner} cases={case_number}/{len(grouped)} case={case_id} medoids={len(medoids)}",
            flush=True,
        )
        del backbone, video_feature
        torch.cuda.empty_cache()

    candidates_frame = pd.DataFrame(
        candidate_rows,
        columns=[
            "v0_span_id", "outer_fold", "inner_fold", "case_id", "selected_start",
            "selected_end", "incumbent_class_id", "candidate_class_id",
            "candidate_correct_frames", "incumbent_correct_frames", "net_frame_effect",
            "global_cluster_id", "global_medoid_rank", "collapsed_medoid_trace",
            "halo_width", "restart_time", "pure_noise", "setting_medoid_rank",
        ],
    )
    span_frame = pd.DataFrame(
        span_rows,
        columns=[
            "v0_span_id", "selected_frames", "selected_wrong_frames",
            "inpainting_candidate_count", "gt_majority_available",
            "best_inpainting_net_frames", "best_visual_net_frames",
            "best_union_net_frames",
        ],
    )
    result_dir = study / "results" / f"outer{outer}_inner{inner}_b1"
    result_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = result_dir / "inpainting_candidates.csv"
    span_path = result_dir / "span_oracle.csv"
    candidates_frame.to_csv(candidate_path, index=False)
    span_frame.to_csv(span_path, index=False)
    inpainting_net = int(span_frame.best_inpainting_net_frames.sum())
    visual_net = int(span_frame.best_visual_net_frames.sum())
    union_net = int(span_frame.best_union_net_frames.sum())
    wrong_mass = int(span_frame.selected_wrong_frames.sum())
    available_wrong_mass = int(span_frame.loc[span_frame.gt_majority_available, "selected_wrong_frames"].sum())
    summary = {
        "status": "complete",
        "outer_fold": outer,
        "inner_fold": inner,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "total_oof_frames": total_frames,
        "flagged_spans": len(span_frame),
        "candidate_rows": len(candidates_frame),
        "setting_runs": setting_runs,
        "samples_generated": samples_generated,
        "early_stops": early_stops,
        "best_of_k_net_frames": inpainting_net,
        "visual_oracle_net_frames": visual_net,
        "union_oracle_net_frames": union_net,
        "incremental_over_visual_net_frames": union_net - visual_net,
        "flagged_wrong_frames": wrong_mass,
        "correct_candidate_available_wrong_frames": available_wrong_mass,
        "output_sha256": {
            candidate_path.name: file_sha256(candidate_path),
            span_path.name: file_sha256(span_path),
        },
        "outer_test_opened": False,
    }
    summary["summary_digest"] = canonical_digest(summary)
    summary_path = result_dir / "b1_summary.json"
    atomic_write_json(summary_path, summary)
    atomic_write_json(
        study / "status" / f"outer{outer}_inner{inner}_b1.json",
        {"status": "complete", "summary_sha256": file_sha256(summary_path)},
    )


def main() -> int:
    args = parse_args()
    study = args.study_dir.resolve()
    try:
        execute(study, args.outer_fold, args.inner_fold, args.device)
    except Exception as error:
        atomic_write_json(
            study / "status" / f"outer{args.outer_fold}_inner{args.inner_fold}_b1_failed.json",
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
