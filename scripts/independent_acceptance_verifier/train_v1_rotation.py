#!/usr/bin/env python3
"""Train one nested-OOF temporal-verifier rotation and score its held inner fold."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from v1_common import (
    INNER_FOLDS,
    N_CLASSES,
    OUTER_FOLDS,
    THRESHOLDS,
    TRAIN_CONFIG,
    CaseData,
    TemporalPairwiseVerifier,
    atomic_write_json,
    canonical_digest,
    file_sha256,
    load_json,
    metric_sufficient,
    metrics_from_sufficient,
    set_determinism,
    stable_seed,
    verify_flat_manifest,
    verify_source_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, choices=OUTER_FOLDS, required=True)
    parser.add_argument("--held-inner", type=int, choices=INNER_FOLDS, required=True)
    parser.add_argument("--device", type=int, required=True)
    return parser.parse_args()


def verify_v0_manifest(manifest: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    compact: list[dict[str, str]] = []
    for row in manifest["files"]:
        role = str(row["role"])
        path = Path(row["path"])
        if not path.is_file() or int(path.stat().st_size) != int(row["size_bytes"]):
            raise RuntimeError(f"V0 nested input missing/size drift: {role}")
        observed = file_sha256(path)
        if observed != row["sha256"]:
            raise RuntimeError(f"V0 nested input hash drift: {role}")
        paths[role] = path
        compact.append({"role": role, "sha256": observed})
    if canonical_digest(compact) != manifest["manifest_digest"]:
        raise RuntimeError("V0 nested manifest digest drift")
    return paths


def read_video_index(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            index, case_id = line.split("\t", maxsplit=1)
            result[int(index)] = case_id
    return result


def load_cases(
    corpus: pd.DataFrame,
    v0_paths: Mapping[str, Path],
    outer_fold: int,
    inner_folds: Sequence[int],
) -> list[CaseData]:
    result: list[CaseData] = []
    for inner in inner_folds:
        video_index = read_video_index(
            v0_paths[f"ground_truth/outer{outer_fold}/inner{inner}/video_index"]
        )
        gt_frame = pd.read_csv(v0_paths[f"ground_truth/outer{outer_fold}/inner{inner}/rows"])
        ground_truth = {
            str(video_index[int(case_index)]): group["concept:name"].to_numpy(dtype=np.int16)
            for case_index, group in gt_frame.groupby("case:concept:name", sort=False)
        }
        scoped = corpus[
            (corpus.outer_fold.astype(int) == outer_fold)
            & (corpus.inner_fold.astype(int) == inner)
        ]
        for case_id, group in scoped.groupby("case_id", sort=True):
            case_id = str(case_id)
            ordered = group.sort_values(["segment_index", "start"], kind="mergesort")
            target = ground_truth[case_id]
            starts = ordered.start.to_numpy(dtype=int)
            ends = ordered.end.to_numpy(dtype=int)
            if starts[0] != 0 or ends[-1] != len(target) or np.any(starts[1:] != ends[:-1]):
                raise RuntimeError(f"OOF baseline is not a partition: {outer_fold}/{inner}/{case_id}")
            baseline = np.empty(len(target), dtype=np.int16)
            for row in ordered.itertuples(index=False):
                baseline[int(row.start) : int(row.end)] = int(row.predicted_label)
            result.append(CaseData(outer_fold, inner, case_id, target, baseline))
    return result


class CandidateDataset:
    def __init__(self, frame: pd.DataFrame, cache: np.ndarray) -> None:
        import torch

        self.frame = frame.reset_index(drop=True)
        self.cache = cache
        lengths = np.sqrt(self.frame.selected_frames.to_numpy(dtype=np.float64))
        lengths /= max(float(lengths.mean()), 1e-12)
        targets = (self.frame.net_frame_effect.to_numpy(dtype=int) > 0).astype(np.float32)
        positive_mass = float(lengths[targets == 1].sum())
        negative_mass = float(lengths[targets == 0].sum())
        positive_multiplier = min(negative_mass / max(positive_mass, 1e-12), 10.0)
        weights = lengths * np.where(targets == 1, positive_multiplier, 1.0)
        probability = np.nanmax(
            np.column_stack(
                [
                    pd.to_numeric(self.frame.visual_head_probability, errors="coerce").to_numpy(),
                    pd.to_numeric(self.frame.diffact_mean_probability, errors="coerce").to_numpy(),
                ]
            ),
            axis=1,
        )
        hard_negative = (
            (self.frame.candidate_effect.astype(str).to_numpy() == "harmful")
            & (np.nan_to_num(probability, nan=-np.inf) >= 0.5)
        )
        weights[hard_negative] *= float(TRAIN_CONFIG["hard_negative_loss_multiplier"])
        self.targets = torch.from_numpy(targets)
        self.weights = torch.from_numpy(weights.astype(np.float32))

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        import torch

        row = self.frame.iloc[index]
        temporal = torch.from_numpy(
            np.asarray(self.cache[int(row.v0_span_id)], dtype=np.float32)
        )
        return (
            temporal,
            torch.tensor(int(row.incumbent_class_id), dtype=torch.long),
            torch.tensor(int(row.candidate_class_id), dtype=torch.long),
            self.targets[index],
            self.weights[index],
        )


def candidate_frame(candidates: pd.DataFrame, outer: int, inners: Sequence[int]) -> pd.DataFrame:
    frame = candidates[
        (candidates.outer_fold.astype(int) == outer)
        & candidates.inner_fold.astype(int).isin([int(value) for value in inners])
        & ~candidates.is_incumbent.astype(bool)
    ].copy()
    if frame.empty or frame.candidate_class_id.astype(int).eq(frame.incumbent_class_id.astype(int)).any():
        raise RuntimeError("Invalid non-incumbent candidate training frame")
    return frame.sort_values(["v0_span_id", "candidate_class_id"], kind="mergesort")


def train_model(
    frame: pd.DataFrame, cache: np.ndarray, device: Any, seed: int
) -> tuple[TemporalPairwiseVerifier, list[float]]:
    import torch
    from torch.utils.data import DataLoader

    set_determinism(seed)
    dataset = CandidateDataset(frame, cache)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(TRAIN_CONFIG["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    model = TemporalPairwiseVerifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(TRAIN_CONFIG["learning_rate"]),
        weight_decay=float(TRAIN_CONFIG["weight_decay"]),
    )
    history: list[float] = []
    for _ in range(int(TRAIN_CONFIG["epochs"])):
        model.train()
        numerator = 0.0
        denominator = 0.0
        for temporal, incumbent, candidate, target, weight in loader:
            temporal = temporal.to(device)
            incumbent = incumbent.to(device)
            candidate = candidate.to(device)
            target = target.to(device)
            weight = weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            logit = model(temporal, incumbent, candidate)
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, target, reduction="none"
            )
            loss = (losses * weight).sum() / weight.sum().clamp_min(1e-12)
            loss.backward()
            optimizer.step()
            numerator += float((losses.detach() * weight).sum().cpu())
            denominator += float(weight.sum().cpu())
        history.append(numerator / max(denominator, 1e-12))
    return model, history


def predict_candidates(
    model: TemporalPairwiseVerifier,
    frame: pd.DataFrame,
    cache: np.ndarray,
    device: Any,
) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

    dataset = CandidateDataset(frame, cache)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for temporal, incumbent, candidate, _, _ in loader:
            logits = model(temporal.to(device), incumbent.to(device), candidate.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    result = frame.copy().reset_index(drop=True)
    result["verifier_probability"] = np.concatenate(probabilities)
    return result


def evaluate_policy(
    cases: Sequence[CaseData],
    spans: pd.DataFrame,
    scores: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    case_by_key = {case.key: case for case in cases}
    predictions = {case.key: case.baseline.copy() for case in cases}
    scoped_spans = spans[
        spans.apply(
            lambda row: (int(row.outer_fold), int(row.inner_fold), str(row.case_id))
            in case_by_key,
            axis=1,
        )
    ].copy()
    score_groups = {
        int(span_id): group.sort_values(
            ["verifier_probability", "candidate_class_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        for span_id, group in scores.groupby("v0_span_id", sort=False)
    }
    decisions: list[dict[str, Any]] = []
    fixed = 0
    broken = 0
    accepted = 0
    for span in scoped_spans.itertuples(index=False):
        group = score_groups[int(span.v0_span_id)]
        best = group.iloc[0]
        accept = float(best.verifier_probability) >= float(threshold)
        if accept:
            key = (int(span.outer_fold), int(span.inner_fold), str(span.case_id))
            predictions[key][int(span.selected_start) : int(span.selected_end)] = int(
                best.candidate_class_id
            )
            fixed += int(best.candidate_correct_frames)
            broken += int(best.incumbent_correct_frames)
            accepted += 1
        decisions.append(
            {
                "v0_span_id": int(span.v0_span_id),
                "outer_fold": int(span.outer_fold),
                "inner_fold": int(span.inner_fold),
                "case_id": str(span.case_id),
                "candidate_class_id": int(best.candidate_class_id),
                "verifier_probability": float(best.verifier_probability),
                "threshold": float(threshold),
                "accepted": bool(accept),
                "candidate_effect": str(best.candidate_effect),
                "fixed_frames": int(best.candidate_correct_frames) if accept else 0,
                "broken_frames": int(best.incumbent_correct_frames) if accept else 0,
            }
        )
    baseline_predictions = {case.key: case.baseline for case in cases}
    baseline_sufficient = metric_sufficient(cases, baseline_predictions)
    final_sufficient = metric_sufficient(cases, predictions)
    baseline_metrics = metrics_from_sufficient(baseline_sufficient)
    final_metrics = metrics_from_sufficient(final_sufficient)
    delta = {metric: final_metrics[metric] - baseline_metrics[metric] for metric in baseline_metrics}
    video_rows: list[dict[str, Any]] = []
    for case in cases:
        before = metrics_from_sufficient(metric_sufficient([case], {case.key: case.baseline}))
        after = metrics_from_sufficient(metric_sufficient([case], {case.key: predictions[case.key]}))
        video_rows.append(
            {
                "outer_fold": case.outer_fold,
                "inner_fold": case.inner_fold,
                "case_id": case.case_id,
                "frames": len(case.ground_truth),
                **{f"baseline_{key}": value for key, value in before.items()},
                **{f"final_{key}": value for key, value in after.items()},
                **{f"delta_{key}": after[key] - before[key] for key in before},
            }
        )
    ratio = float(fixed / broken) if broken else (1e12 if fixed else 0.0)
    return {
        "threshold": float(threshold),
        "baseline_sufficient": baseline_sufficient,
        "final_sufficient": final_sufficient,
        "baseline_metrics": baseline_metrics,
        "final_metrics": final_metrics,
        "delta_metrics": delta,
        "accepted_spans": accepted,
        "fixed_frames": fixed,
        "broken_frames": broken,
        "fixed_to_broken_ratio": ratio,
        "worst_video_delta_acc": min(row["delta_acc"] for row in video_rows),
        "video_rows": video_rows,
        "decisions": decisions,
    }


def choose_threshold(evaluations: Sequence[Mapping[str, Any]]) -> tuple[float, bool]:
    feasible = [
        row
        for row in evaluations
        if float(row["delta_metrics"]["edit"]) >= 0
        and float(row["delta_metrics"]["f1@25"]) >= 0
        and float(row["worst_video_delta_acc"]) >= -5.0
    ]
    candidates = feasible if feasible else [row for row in evaluations if row["threshold"] == max(THRESHOLDS)]
    best = max(
        candidates,
        key=lambda row: (
            float(row["delta_metrics"]["acc"]),
            float(row["fixed_to_broken_ratio"]),
            float(row["threshold"]),
        ),
    )
    return float(best["threshold"]), bool(feasible)


def execute(study_dir: Path, outer: int, held: int, physical_device: int) -> None:
    metadata = load_json(study_dir / "study_metadata.json")
    config = load_json(study_dir / "study_config.json")
    manifest = load_json(study_dir / "input_manifest.json")
    verify_source_provenance(metadata["source_provenance"])
    paths = verify_flat_manifest(manifest)
    if not config["gpu_training_allowed"] or not config.get("fable_approval_digest"):
        raise RuntimeError("V1 training is review-blocked")
    if config["outer_test_open_allowed"] or config["v3_outer_evaluation_allowed"]:
        raise RuntimeError("V1 study unexpectedly authorizes outer-test access")
    feature_complete = load_json(study_dir / "cache" / "feature_cache_complete.json")
    cache_path = study_dir / "cache" / "temporal_span_features.npy"
    index_path = study_dir / "cache" / "temporal_span_index.csv"
    if file_sha256(cache_path) != feature_complete["cache_sha256"]:
        raise RuntimeError("V1 feature-cache hash drift")
    if file_sha256(index_path) != feature_complete["index_sha256"]:
        raise RuntimeError("V1 feature-index hash drift")
    cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)

    v0_manifest = load_json(paths["v0/input_manifest"])
    v0_paths = verify_v0_manifest(v0_manifest)
    corpus = pd.read_csv(v0_paths["selector/oof_segment_corpus"])
    spans = pd.read_csv(paths["v0/results/flagged_oof_spans.csv"])
    candidates = pd.read_csv(paths["v0/results/candidate_corpus.csv"])
    tuning = tuple(value for value in INNER_FOLDS if value != held)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("V1 fail-closed GPU pinning failed")
    device = torch.device("cuda:0")

    crossfit_scores: list[pd.DataFrame] = []
    tuning_histories: dict[str, list[float]] = {}
    for train_inner, score_inner in ((tuning[0], tuning[1]), (tuning[1], tuning[0])):
        train = candidate_frame(candidates, outer, [train_inner])
        score = candidate_frame(candidates, outer, [score_inner])
        seed = stable_seed(TRAIN_CONFIG["seed"], outer, held, "threshold", train_inner)
        model, history = train_model(train, cache, device, seed)
        crossfit_scores.append(predict_candidates(model, score, cache, device))
        tuning_histories[f"train_inner{train_inner}_score_inner{score_inner}"] = history
        del model
        torch.cuda.empty_cache()
    crossfit = pd.concat(crossfit_scores, ignore_index=True)
    tuning_cases = load_cases(corpus, v0_paths, outer, tuning)
    tuning_spans = spans[
        (spans.outer_fold.astype(int) == outer) & spans.inner_fold.astype(int).isin(tuning)
    ]
    threshold_evaluations = [
        evaluate_policy(tuning_cases, tuning_spans, crossfit, threshold)
        for threshold in THRESHOLDS
    ]
    threshold, threshold_feasible = choose_threshold(threshold_evaluations)

    final_train = candidate_frame(candidates, outer, tuning)
    held_frame = candidate_frame(candidates, outer, [held])
    final_seed = stable_seed(TRAIN_CONFIG["seed"], outer, held, "final")
    model, final_history = train_model(final_train, cache, device, final_seed)
    held_scores = predict_candidates(model, held_frame, cache, device)
    held_cases = load_cases(corpus, v0_paths, outer, [held])
    held_spans = spans[
        (spans.outer_fold.astype(int) == outer) & (spans.inner_fold.astype(int) == held)
    ]
    evaluation = evaluate_policy(held_cases, held_spans, held_scores, threshold)

    output_dir = study_dir / "rotations" / f"outer{outer}_held{held}"
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "held_candidate_scores.csv"
    decision_path = output_dir / "held_span_decisions.csv"
    videos_path = output_dir / "held_per_video.csv"
    checkpoint_path = output_dir / "verifier.model"
    held_scores.to_csv(score_path, index=False)
    pd.DataFrame(evaluation.pop("decisions")).to_csv(decision_path, index=False)
    pd.DataFrame(evaluation.pop("video_rows")).to_csv(videos_path, index=False)
    torch.save(model.state_dict(), checkpoint_path)
    threshold_summary = [
        {
            "threshold": row["threshold"],
            "delta_metrics": row["delta_metrics"],
            "worst_video_delta_acc": row["worst_video_delta_acc"],
            "fixed_to_broken_ratio": row["fixed_to_broken_ratio"],
            "accepted_spans": row["accepted_spans"],
        }
        for row in threshold_evaluations
    ]
    complete = {
        "status": "complete",
        "outer_fold": outer,
        "held_inner": held,
        "tuning_inners": list(tuning),
        "threshold": threshold,
        "threshold_guardrails_feasible": threshold_feasible,
        "threshold_crossfit": threshold_summary,
        "evaluation": evaluation,
        "training": {
            "tuning_histories": tuning_histories,
            "final_history": final_history,
            "final_seed": final_seed,
        },
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "output_sha256": {
            score_path.name: file_sha256(score_path),
            decision_path.name: file_sha256(decision_path),
            videos_path.name: file_sha256(videos_path),
            checkpoint_path.name: file_sha256(checkpoint_path),
        },
    }
    complete["completion_digest"] = canonical_digest(complete)
    atomic_write_json(output_dir / "rotation_complete.json", complete)


def main() -> int:
    args = parse_args()
    execute(args.study_dir.resolve(), args.outer_fold, args.held_inner, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
