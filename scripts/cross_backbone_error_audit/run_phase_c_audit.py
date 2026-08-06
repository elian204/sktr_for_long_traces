#!/usr/bin/env python3
"""Run the provenance-gated Phase-C cross-backbone taxonomy sweep."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from phase_c_common import (
    BACKBONE_ORDER,
    DATASETS,
    PHASE_C_SPEC,
    PRIMARY_BUCKETS,
    atomic_write_json,
    canonical_digest,
    file_sha256,
    load_json,
    normalize_case,
    parse_mapping,
    read_nonempty_lines,
    verify_manifest,
    verify_source,
)
from phase_c_input_guard import verify_phase_c_study_records
from phase_c_taxonomy import DFG, aggregate_cases, analyze_case, generation_hypothesis_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    return parser.parse_args()


def source_disclosure(backbone: str, arm: str, analysis_role: str) -> str:
    disclosure = PHASE_C_SPEC["source_asymmetry_disclosure"]
    if backbone == "mstcn2" and analysis_role == "secondary_full_train_epoch100":
        return str(disclosure["mstcn2_ep100_secondary"])
    if backbone == "asformer" and arm in {"epoch30", "epoch100"}:
        return str(disclosure["asformer_epoch30_epoch100_sensitivity"])
    return str(disclosure[backbone])


def align_analysis_timeline(
    probability: np.ndarray,
    prediction: np.ndarray,
    *,
    target_frames: int,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Normalize native-rate exports to the common full 15-fps GT timeline."""
    if probability.shape[1] != len(prediction):
        raise RuntimeError("Probability/prediction timeline drift")
    if len(prediction) == target_frames:
        return probability, prediction, 1
    expected_native_frames = len(range(0, target_frames, sample_rate))
    if sample_rate <= 1 or len(prediction) != expected_native_frames:
        raise RuntimeError(
            f"Unsupported native timeline: export={len(prediction)} "
            f"target={target_frames} sample_rate={sample_rate}"
        )
    expanded_probability = np.repeat(probability, sample_rate, axis=1)[:, :target_frames]
    expanded_prediction = np.repeat(prediction, sample_rate)[:target_frames]
    if expanded_probability.shape[1] != target_frames or len(expanded_prediction) != target_frames:
        raise AssertionError("Full-timeline expansion failed")
    return expanded_probability, expanded_prediction, sample_rate


def materialized_exports(study: Path, config: Mapping[str, Any]) -> dict[tuple[str, str, int, str], Path]:
    result: dict[tuple[str, str, int, str], Path] = {}
    required = []
    for dataset in ("gtea", "50salads"):
        required.extend(("official", dataset, fold) for fold in range(1, int(DATASETS[dataset]["folds"]) + 1))
    required.extend((arm, "breakfast", fold) for fold in range(1, 5) for arm in ("epoch30", "epoch100"))
    for arm, dataset, fold in required:
        status_path = study / "status" / f"asformer_{arm}_{dataset}_fold{fold}.json"
        status = load_json(status_path)
        manifest_path = Path(status["export_manifest_path"])
        if status.get("status") != "complete" or file_sha256(manifest_path) != status["export_manifest_sha256"]:
            raise RuntimeError(f"Incomplete/drifted ASFormer materialization: {arm}/{dataset}/fold{fold}")
        frame = pd.read_csv(manifest_path)
        if len(frame) != int(status["cases"]):
            raise RuntimeError("ASFormer materialized case-count drift")
        for row in frame.itertuples(index=False):
            path = Path(row.path)
            if file_sha256(path) != row.sha256:
                raise RuntimeError(f"ASFormer materialized output drift: {path}")
            result[(arm, dataset, fold, str(row.case_id))] = path
    return result


def dfg_for(study: Path, config: Mapping[str, Any], dataset: str, fold: int) -> DFG:
    path = study / "dfg" / dataset / f"fold{fold}.json"
    if file_sha256(path) != config["dfg_sha256"][f"{dataset}/fold{fold}"]:
        raise RuntimeError("Fold-pure DFG digest drift")
    payload = load_json(path)
    if payload.get("test_gt_used") is not False or payload["discovery_source"] != "official_full_training_fold_ground_truth_only":
        raise RuntimeError("Fold-pure DFG discovery-source drift")
    return DFG(
        frozenset(int(value) for value in payload["starts"]),
        frozenset(int(value) for value in payload["ends"]),
        frozenset((int(left), int(right)) for left, right in payload["edges"]),
    )


def load_probability_and_prediction(
    *,
    study: Path,
    paths: Mapping[str, Path],
    materialized: Mapping[tuple[str, str, int, str], Path],
    backbone: str,
    dataset: str,
    fold: int,
    case_id: str,
    arm: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    if backbone == "mstcn2":
        if arm == "selected":
            role = f"mstcn2/selected/{dataset}/fold{fold}/probability/{case_id}"
        elif arm == "full_train_epoch100":
            role = f"mstcn2/full_train_epoch100/{dataset}/fold{fold}/probability/{case_id}"
        else:
            role = f"mstcn2/{arm}/{dataset}/fold{fold}/probability/{case_id}"
        probability = np.load(paths[role], allow_pickle=False)
        prediction = probability.argmax(axis=0).astype(np.int64)
        disagreement = 0
    elif backbone == "asformer":
        if arm == "selected" and dataset == "breakfast":
            probability = np.load(paths[f"asformer/official/breakfast/fold{fold}/probability/{case_id}"], allow_pickle=False)
        else:
            output_arm = "official" if arm == "selected" else arm
            probability = np.load(materialized[(output_arm, dataset, fold, case_id)], allow_pickle=False)
        prediction = probability.argmax(axis=0).astype(np.int64)
        disagreement = 0
    elif backbone == "diffact":
        probability = np.load(paths[f"diffact/official/{dataset}/fold{fold}/probability/{case_id}"], allow_pickle=False)
        prediction = np.load(paths[f"diffact/official/{dataset}/fold{fold}/prediction/{case_id}"], allow_pickle=False).astype(np.int64)
        disagreement = int(np.sum(prediction != probability.argmax(axis=0)))
    else:
        raise ValueError(backbone)
    if probability.ndim != 2 or prediction.ndim != 1 or probability.shape[1] != len(prediction):
        raise RuntimeError(f"Prediction/probability shape drift: {backbone}/{dataset}/{case_id}")
    if len(prediction) and (int(prediction.min()) < 0 or int(prediction.max()) >= probability.shape[0]):
        raise RuntimeError(f"Prediction label range drift: {backbone}/{dataset}/{case_id}")
    if not np.isfinite(probability).all() or float(np.max(np.abs(probability.sum(axis=0) - 1.0))) > 1e-4:
        raise RuntimeError(f"Probability validity drift: {backbone}/{dataset}/{case_id}")
    return probability, prediction, disagreement


def direction_rows(per_dataset: pd.DataFrame) -> pd.DataFrame:
    selected = per_dataset[
        (per_dataset["analysis_role"] == "primary") & (per_dataset["aggregation"] == "frame_weighted")
    ]
    endpoints = {
        "fragmentation_frames_per_minute": "decrease",
        "illegal_order_frames_per_minute": "decrease",
        "legal_substitution_share": "increase",
        "boundary_offset_frames_per_minute": "descriptive",
    }
    rows = []
    for dataset in DATASETS:
        subset = selected[selected["dataset"] == dataset].set_index("backbone")
        for endpoint, expected in endpoints.items():
            values = {backbone: float(subset.loc[backbone, endpoint]) for backbone in BACKBONE_ORDER}
            if expected == "decrease":
                monotonic = values["mstcn2"] >= values["asformer"] >= values["diffact"]
                end_direction = values["diffact"] < values["mstcn2"]
            elif expected == "increase":
                monotonic = values["mstcn2"] <= values["asformer"] <= values["diffact"]
                end_direction = values["diffact"] > values["mstcn2"]
            else:
                monotonic = None
                end_direction = None
            rows.append(
                {
                    "dataset": dataset, "endpoint": endpoint, "expected_direction": expected,
                    **values, "asformer_minus_mstcn2": values["asformer"] - values["mstcn2"],
                    "diffact_minus_asformer": values["diffact"] - values["asformer"],
                    "diffact_minus_mstcn2": values["diffact"] - values["mstcn2"],
                    "monotonic_generation_order": monotonic,
                    "end_to_end_expected_direction": end_direction,
                }
            )
    return pd.DataFrame(rows)


def robustness_rows(per_dataset: pd.DataFrame) -> pd.DataFrame:
    weighted = per_dataset[per_dataset["aggregation"] == "frame_weighted"]
    rows = []
    for dataset in ("gtea", "50salads"):
        primary = weighted[(weighted["dataset"] == dataset) & (weighted["analysis_role"] == "primary")].set_index("backbone")
        secondary = weighted[(weighted["dataset"] == dataset) & (weighted["analysis_role"] == "secondary_full_train_epoch100")].iloc[0]
        for bucket in PRIMARY_BUCKETS:
            endpoint = f"{bucket}_share"
            selected_value = float(primary.loc["mstcn2", endpoint])
            ep100_value = float(secondary[endpoint])
            diffact_value = float(primary.loc["diffact", endpoint])
            selected_sign = int(np.sign(diffact_value - selected_value))
            ep100_sign = int(np.sign(diffact_value - ep100_value))
            rows.append(
                {
                    "dataset": dataset, "bucket": bucket,
                    "mstcn2_selected_share": selected_value,
                    "mstcn2_full_train_epoch100_share": ep100_value,
                    "asformer_official_share": float(primary.loc["asformer", endpoint]),
                    "diffact_official_share": diffact_value,
                    "selected_to_diffact_sign": selected_sign,
                    "ep100_to_diffact_sign": ep100_sign,
                    "share_direction_robust": selected_sign == ep100_sign,
                    "comparison_role": "pre_registered_secondary",
                }
            )
    return pd.DataFrame(rows)


def sensitivity_rows(per_dataset: pd.DataFrame) -> pd.DataFrame:
    subset = per_dataset[
        (per_dataset["dataset"] == "breakfast")
        & (per_dataset["aggregation"].isin(["frame_weighted", "per_video_macro"]))
        & (per_dataset["backbone"].isin(["mstcn2", "asformer"]))
    ].copy()
    subset = subset[
        ((subset["analysis_role"] == "primary") & (subset["arm"] == "selected"))
        | (subset["analysis_role"] == "sensitivity")
    ]
    columns = [
        "dataset", "backbone", "arm", "analysis_role", "aggregation", "n_cases",
        "acc", "edit", "f1@10", "f1@25", "f1@50", "over_segmentation_ratio",
        *[f"{bucket}_share" for bucket in PRIMARY_BUCKETS],
        *[f"{bucket}_frames_per_minute" for bucket in PRIMARY_BUCKETS],
    ]
    result = subset[columns].copy()
    result["claim_status"] = "descriptive_only"
    result["selected_checkpoint_definition"] = np.where(
        result["backbone"] == "asformer", "author_release_epoch120", "heldout_validation_selected"
    )
    return result.sort_values(["backbone", "aggregation", "arm"])


def write_findings(results: Path, per_dataset: pd.DataFrame, direction: pd.DataFrame) -> None:
    weighted = per_dataset[
        (per_dataset["analysis_role"] == "primary") & (per_dataset["aggregation"] == "frame_weighted")
    ]
    output_dir = results / "per_dataset_findings"
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        subset = weighted[weighted["dataset"] == dataset].set_index("backbone")
        lines = [
            f"# {dataset} Phase-C findings", "",
            "Headline sources: validation-selected MS-TCN++, author-release ASFormer, and official-release DiffAct.", "",
            "| Backbone | Error frames/min | Fragmentation share | Boundary share | Illegal share | Legal-substitution share | Pred/GT segments |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for backbone in BACKBONE_ORDER:
            row = subset.loc[backbone]
            lines.append(
                f"| {backbone} | {row['n_errors']/row['duration_minutes']:.2f} | {100*row['fragmentation_share']:.2f}% | "
                f"{100*row['boundary_offset_share']:.2f}% | {100*row['illegal_order_share']:.2f}% | "
                f"{100*row['legal_substitution_share']:.2f}% | {row['over_segmentation_ratio']:.3f} |"
            )
        lines.extend(["", "Generation-direction checks:", ""])
        for row in direction[direction["dataset"] == dataset].itertuples(index=False):
            lines.append(
                f"- `{row.endpoint}`: MS-TCN++={row.mstcn2:.4f}, ASFormer={row.asformer:.4f}, DiffAct={row.diffact:.4f}; "
                f"monotonic={row.monotonic_generation_order}."
            )
        (output_dir / f"{dataset}.md").write_text("\n".join(lines) + "\n")


def execute(study: Path) -> None:
    config = load_json(study / "study_config.json")
    metadata = load_json(study / "study_metadata.json")
    manifest = load_json(study / "input_manifest.json")
    verify_source(metadata["source_provenance"])
    if not config["phase_c_allowed"] or not config["audit_execution_allowed"] or not config.get("fable_approval_digest"):
        raise RuntimeError("Phase-C audit is review-blocked")
    if config["phase_c_spec"] != PHASE_C_SPEC:
        raise RuntimeError("Phase-C pre-registered specification drift")
    guarded = verify_phase_c_study_records(study, manifest["files"])
    paths = verify_manifest(
        manifest, [Path(value) for value in config["allowed_input_roots"]], full_hash=False
    )
    if guarded != [paths[str(row["role"])] for row in manifest["files"]]:
        raise RuntimeError("Phase-C provenance-guard ordering drift")
    if file_sha256(study / "case_index.csv") != config["case_index_sha256"]:
        raise RuntimeError("Phase-C case index drift")
    cases = pd.read_csv(study / "case_index.csv")
    if len(cases) != 1790 or cases.duplicated(["dataset", "case_id"]).any():
        raise RuntimeError("Phase-C official test-case coverage drift")
    materialized = materialized_exports(study, config)
    arms = [*config["analysis_arms"]["primary"], *config["analysis_arms"]["secondary"], *config["analysis_arms"]["sensitivity"]]
    rows: list[dict[str, Any]] = []
    mapping_cache: dict[str, tuple[dict[int, str], dict[str, int], set[int]]] = {}
    dfg_cache: dict[tuple[str, int], DFG] = {}
    for arm_record in arms:
        backbone, dataset, arm = arm_record["backbone"], arm_record["dataset"], arm_record["arm"]
        dataset_cases = cases[cases["dataset"] == dataset]
        if dataset not in mapping_cache:
            id_to_name, name_to_id = parse_mapping(paths[f"data/{dataset}/mapping"])
            background = {index for index, label in id_to_name.items() if label == "background"}
            mapping_cache[dataset] = (id_to_name, name_to_id, background)
        id_to_name, name_to_id, background = mapping_cache[dataset]
        for case in dataset_cases.itertuples(index=False):
            fold, case_id, sample_rate = int(case.fold), str(case.case_id), int(case.sample_rate)
            dfg_cache.setdefault((dataset, fold), dfg_for(study, config, dataset, fold))
            gt_names = read_nonempty_lines(paths[f"data/{dataset}/ground_truth/{case_id}"])
            gt = np.asarray([name_to_id[label] for label in gt_names], dtype=np.int64)
            probability, prediction, disagreement = load_probability_and_prediction(
                study=study, paths=paths, materialized=materialized, backbone=backbone,
                dataset=dataset, fold=fold, case_id=case_id, arm=arm,
            )
            source_frames = len(prediction)
            probability, prediction, expansion_factor = align_analysis_timeline(
                probability,
                prediction,
                target_frames=len(gt_names),
                sample_rate=sample_rate,
            )
            disagreement = int(np.sum(prediction != probability.argmax(axis=0)))
            if probability.shape != (len(id_to_name), len(gt)) or prediction.shape != gt.shape:
                raise RuntimeError(f"Phase-C frame/class alignment drift: {backbone}/{dataset}/{case_id}")
            row = analyze_case(gt=gt, prediction=prediction, probability=probability, dfg=dfg_cache[(dataset, fold)], background=background)
            rows.append(
                {
                    "backbone": backbone, "generation_index": BACKBONE_ORDER.index(backbone),
                    "dataset": dataset, "fold": fold, "case_id": case_id, "arm": arm,
                    "analysis_role": arm_record["analysis_role"],
                    "source_asymmetry": source_disclosure(backbone, arm, arm_record["analysis_role"]),
                    "native_export_frames": source_frames,
                    "analysis_timeline_frames": len(prediction),
                    "timeline_expansion_factor": expansion_factor,
                    "official_prediction_vs_probability_argmax_disagreement_frames": disagreement,
                    **row,
                }
            )
    per_case = pd.DataFrame(rows).sort_values(["analysis_role", "dataset", "backbone", "arm", "fold", "case_id"])
    if not all(
        int(row.n_errors) == sum(int(getattr(row, f"{bucket}_frames")) for bucket in PRIMARY_BUCKETS)
        for row in per_case.itertuples(index=False)
    ):
        raise AssertionError("Phase-C per-case taxonomy sum invariant failed")
    per_fold = aggregate_cases(per_case, ["analysis_role", "dataset", "backbone", "arm", "fold"])
    per_dataset = aggregate_cases(per_case, ["analysis_role", "dataset", "backbone", "arm"])
    hypothesis = generation_hypothesis_test(per_fold)
    direction = direction_rows(per_dataset)
    robustness = robustness_rows(per_dataset)
    sensitivity = sensitivity_rows(per_dataset)
    primary_table = per_dataset[
        (per_dataset["analysis_role"] == "primary") & (per_dataset["aggregation"] == "frame_weighted")
    ].sort_values(["dataset", "backbone"])
    candidate = per_dataset[
        (per_dataset["analysis_role"] == "primary")
        & (per_dataset["aggregation"].isin(["frame_weighted", "per_video_macro"]))
    ][[
        "dataset", "backbone", "aggregation", "candidate_rank_observations",
        "candidate_gt_top2_coverage", "candidate_gt_top3_coverage", "candidate_gt_top5_coverage",
        "candidate_gt_rank_median", "candidate_gt_rank_p90",
    ]]
    results = study / "results"
    per_case.to_csv(results / "taxonomy_per_case.csv", index=False)
    per_fold.to_csv(results / "taxonomy_per_fold.csv", index=False)
    per_dataset.to_csv(results / "taxonomy_per_dataset.csv", index=False)
    primary_table.to_csv(results / "model_generation_table.csv", index=False)
    hypothesis.to_csv(results / "model_generation_hypothesis_test.csv", index=False)
    direction.to_csv(results / "model_generation_direction_by_dataset.csv", index=False)
    robustness.to_csv(results / "mstcn2_full_train_ep100_share_robustness.csv", index=False)
    sensitivity.to_csv(results / "breakfast_checkpoint_sensitivity.csv", index=False)
    candidate.to_csv(results / "candidate_rank_summary.csv", index=False)
    write_findings(results, per_dataset, direction)
    overall_support = bool(hypothesis["overall_pre_registered_support"].iloc[0])
    summary = {
        "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": config["protocol_version"], "input_manifest_digest": manifest["manifest_digest"],
        "case_rows": len(per_case), "official_test_cases": len(cases),
        "taxonomy_partition_ok": True, "model_generation_pre_registered_support": overall_support,
        "phase_c_spec": config["phase_c_spec"], "phase_a_stale_cache_used": False,
        "sealed_studies_modified": False,
    }
    summary["summary_digest"] = canonical_digest(summary)
    atomic_write_json(results / "summary.json", summary)
    lines = [
        "# Cross-backbone Phase-C audit", "",
        f"**Pre-registered model-generation support: {'YES' if overall_support else 'NO'}.**", "",
        "Headline comparison uses validation-selected MS-TCN++, author-release ASFormer, and official-release DiffAct.",
        "All descriptive sensitivity and full-train epoch-100 robustness rows are excluded from that headline test.", "",
        "## Source asymmetry", "",
    ]
    for key, value in PHASE_C_SPEC["source_asymmetry_disclosure"].items():
        lines.append(f"- **{key}:** {value}")
    lines.extend(["", "## Hypothesis endpoints", ""])
    for row in hypothesis.itertuples(index=False):
        lines.append(
            f"- `{row.endpoint}`: expected-direction folds {row.expected_direction_folds}/{row.non_tie_folds}; "
            f"delta={row.diffact_minus_mstcn2_mean_fold_delta:+.4f}; Holm p={row.holm_p:.4g}; support={row.endpoint_support}."
        )
    lines.extend(["", "Per-dataset findings and all exact tables are included beside this report."])
    (results / "report.md").write_text("\n".join(lines) + "\n")
    outputs = sorted(path for path in results.rglob("*") if path.is_file())
    output_manifest = {
        "protocol_version": config["protocol_version"],
        "files": [{"path": str(path.relative_to(study)), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in outputs],
    }
    output_manifest["output_digest"] = canonical_digest(output_manifest["files"])
    atomic_write_json(results / "output_manifest.json", output_manifest)
    atomic_write_json(study / "release" / "results_manifest.json", output_manifest)


def main() -> int:
    args = parse_args()
    execute(args.study_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
