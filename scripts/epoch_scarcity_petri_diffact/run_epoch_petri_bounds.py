#!/usr/bin/env python3
"""Discover one Petri context and decode the complete epoch checkpoint grid."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from epoch_scarcity_common import (
    CANONICAL_DECODER,
    CONDITIONS,
    DISCOVERY_FULL_TRAIN,
    DISCOVERY_OFFICIAL_TEST,
    ORACLE_CONDITION,
    PRIMARY_CONDITION,
    atomic_json,
    checkpoint_metadata_path,
    dataset_spec,
    file_sha256,
    read_json,
    read_lines,
    softmax_dir,
    validate_condition_source,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
LOW_DATA_DIR = Path(__file__).resolve().parents[1] / "low_data_petri_diffact"
for path in (WORKSPACE_ROOT, LOW_DATA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_petri_postprocessing_low_data import (  # noqa: E402
    DATASET_HP_DEFAULTS,
    HP_STRATEGIES,
    allowed_transition_set,
    attach_diffact_comparators,
    build_combined_inputs,
    build_metric_rows,
    build_per_case_metric_rows,
    build_runtime_tables,
    preflight_required_diffact_comparators,
)
from src.incremental_softmax_recovery import (  # noqa: E402
    build_recovery_context,
    incremental_softmax_recovery,
    setup_logging,
)
from src.utils import linear_prob_combiner  # noqa: E402
from export_epoch_checkpoint import validate_completed_export  # noqa: E402


APPROVED_SCORE_METHODS = {"raw_argmax", "official_diffact", "petri_conformance"}
CANONICAL_RECOVERY_KEYS = (
    "chunk_size",
    "conditioning_state_mode",
    "conditioning_top_m",
    "candidate_top_k",
    "restrict_log_moves",
    "restrict_model_moves_to_tau",
    "max_consecutive_tau_moves",
)


def output_dir_for(
    experiment_dir: Path,
    condition: str,
    seed: int,
    epoch: int,
) -> Path:
    source_dir = "petri_source_test" if condition == ORACLE_CONDITION else "petri_source_train"
    return (
        experiment_dir
        / "petri"
        / condition
        / f"seed_{seed}"
        / f"epoch_{epoch:05d}"
        / source_dir
        / "petri_conformance"
        / "test"
    )


def recovery_config(dataset: str, seed: int, workers: int, out_dir: Path) -> Dict[str, Any]:
    hp = DATASET_HP_DEFAULTS[(dataset, "diffact")]
    return {
        "n_train_traces": None,
        "n_test_traces": None,
        "ensure_train_variant_diversity": False,
        "ensure_test_variant_diversity": False,
        "use_same_traces_for_train_test": False,
        "cost_function": "linear",
        "model_move_cost": 1.0,
        "log_move_cost": 1.0,
        "tau_move_cost": 1e-6,
        "prob_threshold": 1e-6,
        "max_hist_len": 3,
        "use_calibration": False,
        "n_indices": 10**9,
        "n_per_run": None,
        "sequential_sampling": False,
        "independent_sampling": True,
        "round_precision": 2,
        "random_seed": seed,
        "save_model": False,
        "non_sync_penalty": 1.0,
        "chunk_size": CANONICAL_DECODER["chunk_size"],
        "conformance_switch_penalty_weight": 1.0,
        "conditioning_alpha": hp["alpha"],
        "conditioning_combine_fn": linear_prob_combiner,
        "conditioning_n_prev_labels": 3,
        "conditioning_interpolation_weights": HP_STRATEGIES[hp["strategy"]],
        "conditioning_state_mode": CANONICAL_DECODER["conditioning_state_mode"],
        "conditioning_top_m": CANONICAL_DECODER["conditioning_top_m"],
        "candidate_top_p": 1.0,
        "candidate_top_k": CANONICAL_DECODER["candidate_top_k"],
        "candidate_min_k": 1,
        "candidate_source": "conditioned",
        "candidate_apply_to_sync": True,
        "restrict_log_moves": CANONICAL_DECODER["restrict_log_moves"],
        "restrict_model_moves_to_tau": CANONICAL_DECODER["restrict_model_moves_to_tau"],
        "max_consecutive_tau_moves": CANONICAL_DECODER["max_consecutive_tau_moves"],
        "progress_log_interval_chunks": 20,
        "adaptive_chunk_sizing": False,
        "use_state_caching": True,
        "enabled_cache_size": 100000,
        "parallel_processing": False,
        "dataset_parallelization": True,
        "dataset_parallelization_context": None,
        "max_workers": workers,
        "merge_mismatched_boundaries": False,
        "case_output_dir": str(out_dir / "case_outputs"),
        "resume_case_outputs": True,
        "compute_marking_transition_map": False,
        "precompute_marking_transition_map": False,
        "verbose": True,
        "log_level": logging.INFO,
    }


def validate_canonical_recovery_config(config: Dict[str, Any]) -> None:
    mismatches = {
        key: {"expected": CANONICAL_DECODER[key], "actual": config.get(key)}
        for key in CANONICAL_RECOVERY_KEYS
        if config.get(key) != CANONICAL_DECODER[key]
    }
    if mismatches:
        raise ValueError(f"Recovery configuration drifted from CANONICAL_DECODER: {mismatches}")


def serializable_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: (value.__name__ if callable(value) and hasattr(value, "__name__") else str(value) if callable(value) else value)
        for key, value in config.items()
        if key != "recovery_context"
    }


def retain_approved_score_methods(frame: pd.DataFrame) -> pd.DataFrame:
    methods = set(frame["method"].astype(str))
    missing = APPROVED_SCORE_METHODS.difference(methods)
    if missing:
        raise ValueError(f"Required pre-specified score methods are missing: {sorted(missing)}")
    return frame[frame["method"].isin(APPROVED_SCORE_METHODS)].copy()


def validate_existing_output(
    out_dir: Path,
    *,
    epoch: int,
    condition: str,
    discovery_source: str,
    is_oracle: bool,
) -> None:
    required_outputs = (
        "results.csv",
        "metrics.csv",
        "per_case_metrics.csv",
        "runtime.csv",
        "per_case_runtime.csv",
        "accuracy_dict.json",
        "postprocess_config.json",
    )
    missing = [name for name in required_outputs if not (out_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing completed decode outputs under {out_dir}: {missing}")
    config = read_json(out_dir / "postprocess_config.json")
    expected = {
        "condition": condition,
        "petri_discovery_source": discovery_source,
        "oracle_upper_bound": is_oracle,
        "petri_discovery_uses_test_gt": is_oracle,
        "is_oracle_condition": is_oracle,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if int(config.get("checkpoint", {}).get("epoch_index", -1)) != epoch:
        mismatches["checkpoint_epoch_index"] = {
            "expected": epoch,
            "actual": config.get("checkpoint", {}).get("epoch_index"),
        }
    if config.get("canonical_decoder") != CANONICAL_DECODER:
        mismatches["canonical_decoder"] = {
            "expected": CANONICAL_DECODER,
            "actual": config.get("canonical_decoder"),
        }
    if not bool(config.get("context_reused_across_checkpoint_grid", False)):
        mismatches["context_reused_across_checkpoint_grid"] = {
            "expected": True,
            "actual": config.get("context_reused_across_checkpoint_grid"),
        }
    try:
        validate_canonical_recovery_config(config.get("recovery_config", {}))
    except ValueError as exc:
        mismatches["recovery_config"] = str(exc)
    if mismatches:
        raise ValueError(
            f"Existing epoch output is incompatible; rerun with --force: {out_dir}: "
            f"{mismatches}"
        )
    for name in ("metrics.csv", "per_case_metrics.csv"):
        frame = pd.read_csv(out_dir / name)
        methods = set(frame["method"].astype(str))
        if methods != APPROVED_SCORE_METHODS:
            raise ValueError(
                f"Existing {name} has methods={sorted(methods)}; "
                f"expected={sorted(APPROVED_SCORE_METHODS)}; rerun with --force"
            )
        epochs = set(frame["checkpoint_epoch_index"].astype(int))
        if epochs != {epoch}:
            raise ValueError(f"Existing {name} has checkpoint epochs {sorted(epochs)}")
        frame_expected = {
            key: value
            for key, value in expected.items()
            if key != "petri_discovery_uses_test_gt"
        }
        for key, expected_value in frame_expected.items():
            if key not in frame.columns:
                raise ValueError(f"Existing {name} is missing required column {key!r}")
            observed = set(frame[key].map(lambda value: str(value).strip().lower()))
            expected_normalized = str(expected_value).strip().lower()
            if observed != {expected_normalized}:
                raise ValueError(
                    f"Existing {name} has {key}={sorted(observed)}; "
                    f"expected={expected_normalized!r}"
                )


def validate_completed_decode_grid(
    experiment_dir: Path,
    *,
    seed: int,
    condition: str,
    discovery_source: str,
) -> Dict[str, Any]:
    """Revalidate a completed condition grid and every epoch output."""

    experiment_dir = experiment_dir.resolve()
    validate_condition_source(condition, discovery_source)
    metadata = read_json(experiment_dir / "experiment_metadata.json")
    epochs = list(map(int, dataset_spec(str(metadata["dataset"]))["checkpoint_epochs"]))
    marker_path = (
        experiment_dir / "petri" / condition / f"seed_{seed}" / "decode_grid_complete.json"
    )
    marker = read_json(marker_path)
    is_oracle = condition == ORACLE_CONDITION
    manifest = (
        experiment_dir / "manifests" / "test_cases.txt"
        if discovery_source == DISCOVERY_OFFICIAL_TEST
        else experiment_dir / "manifests" / "train_pool_cases.txt"
    )
    expected = {
        "completed": True,
        "condition": condition,
        "petri_discovery_source": discovery_source,
        "is_oracle_condition": is_oracle,
        "checkpoint_epoch_indices": epochs,
        "petri_discovery_manifest": str(manifest),
        "petri_discovery_manifest_sha256": file_sha256(manifest),
    }
    mismatches = {
        key: {"expected": value, "actual": marker.get(key)}
        for key, value in expected.items()
        if marker.get(key) != value
    }
    discovery_count = marker.get("petri_context_discovery_count_this_invocation")
    if discovery_count not in {0, 1}:
        mismatches["petri_context_discovery_count_this_invocation"] = {
            "expected": "0 or 1",
            "actual": discovery_count,
        }
    for epoch in epochs:
        validate_completed_export(experiment_dir, seed=seed, checkpoint_epoch=epoch)
        validate_existing_output(
            output_dir_for(experiment_dir, condition, seed, epoch),
            epoch=epoch,
            condition=condition,
            discovery_source=discovery_source,
            is_oracle=is_oracle,
        )
    if mismatches:
        raise ValueError(
            f"Completed decode grid failed immutable validation: {marker_path}: {mismatches}"
        )
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument(
        "--petri-discovery-source",
        choices=[DISCOVERY_FULL_TRAIN, DISCOVERY_OFFICIAL_TEST],
        required=True,
    )
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    validate_condition_source(args.condition, args.petri_discovery_source)

    experiment_dir = args.experiment_dir.resolve()
    metadata = read_json(experiment_dir / "experiment_metadata.json")
    dataset = str(metadata["dataset"])
    spec = dataset_spec(dataset)
    epochs = list(spec["checkpoint_epochs"])
    for epoch in epochs:
        validate_completed_export(
            experiment_dir,
            seed=args.seed,
            checkpoint_epoch=epoch,
        )
    if not args.execute:
        print(
            f"PREPARED condition={args.condition} source={args.petri_discovery_source} "
            f"epochs={epochs}; one Petri context will be reused across the grid"
        )
        return

    setup_logging()
    manifests = experiment_dir / "manifests"
    train_manifest = (
        manifests / "test_cases.txt"
        if args.petri_discovery_source == DISCOVERY_OFFICIAL_TEST
        else manifests / "train_pool_cases.txt"
    )
    discovery_cases = read_lines(train_manifest)
    eval_cases = read_lines(manifests / "test_cases.txt")
    is_oracle = args.condition == ORACLE_CONDITION

    context = None
    discovery_count = 0
    completed_epochs: List[int] = []
    for epoch in epochs:
        checkpoint_meta = read_json(checkpoint_metadata_path(experiment_dir, args.seed, epoch))
        out_dir = output_dir_for(experiment_dir, args.condition, args.seed, epoch)
        required_outputs = [
            out_dir / "results.csv",
            out_dir / "metrics.csv",
            out_dir / "per_case_metrics.csv",
            out_dir / "runtime.csv",
            out_dir / "per_case_runtime.csv",
            out_dir / "accuracy_dict.json",
            out_dir / "postprocess_config.json",
        ]
        if all(path.is_file() for path in required_outputs) and not args.force:
            validate_existing_output(
                out_dir,
                epoch=epoch,
                condition=args.condition,
                discovery_source=args.petri_discovery_source,
                is_oracle=is_oracle,
            )
            print(f"SKIP validated existing epoch output: {out_dir}")
            completed_epochs.append(epoch)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        combined_df, softmax_list, train_seq_ids, eval_seq_ids, map_path = build_combined_inputs(
            data_root=Path(metadata["data_root"]),
            dataset=dataset,
            train_cases=discovery_cases,
            eval_cases=eval_cases,
            eval_softmax_dir=softmax_dir(experiment_dir, args.seed, epoch),
            out_dir=out_dir,
            temperature=None,
            petri_discovery_representation="frame_events",
        )
        preflight_required_diffact_comparators(
            combined_map_path=map_path,
            softmax_dir=softmax_dir(experiment_dir, args.seed, epoch),
        )
        train_df = combined_df[
            combined_df["case:concept:name"].astype(str).isin(set(map(str, train_seq_ids)))
        ]
        config = recovery_config(dataset, args.seed, args.workers, out_dir)
        validate_canonical_recovery_config(config)
        config["resume_case_outputs"] = not args.force
        if context is None:
            context = build_recovery_context(
                train_df,
                max_hist_len=int(config["max_hist_len"]),
                enabled_cache_size=int(config["enabled_cache_size"]),
                compute_marking_transition_map=bool(config["compute_marking_transition_map"]),
                precompute_marking_transition_map=bool(config["precompute_marking_transition_map"]),
                conformance_switch_penalty_weight=float(config["conformance_switch_penalty_weight"]),
                conditioning_alpha=float(config["conditioning_alpha"]),
                save_model=False,
            )
            discovery_count += 1
        config.update(
            {
                "train_cases": train_seq_ids,
                "test_cases": eval_seq_ids,
                "allow_train_cases_in_test": is_oracle,
                "recovery_context": context,
            }
        )
        started = time.perf_counter()
        results_df, accuracy_dict, _ = incremental_softmax_recovery(
            df=combined_df,
            softmax_lst=softmax_list,
            **config,
        )
        total_seconds = time.perf_counter() - started
        results_df, comparator_provenance = attach_diffact_comparators(
            results_df=results_df,
            combined_map_path=map_path,
            softmax_dir=softmax_dir(experiment_dir, args.seed, epoch),
            require_comparators=True,
        )
        results_df["checkpoint_epoch_index"] = epoch
        results_df["completed_epochs"] = checkpoint_meta["completed_epochs"]
        results_df["schedule_fraction_pct"] = checkpoint_meta["schedule_fraction_pct"]
        results_df["condition"] = args.condition
        results_df["petri_discovery_source"] = args.petri_discovery_source
        results_df["is_oracle_condition"] = is_oracle
        results_df["oracle_upper_bound"] = is_oracle
        results_df.to_csv(out_dir / "results.csv", index=False)

        allowed = allowed_transition_set(combined_df, train_seq_ids)
        metrics = build_metric_rows(
            dataset=dataset,
            seed=args.seed,
            diffact_fraction=100,
            petri_fraction=None,
            condition=args.condition,
            split="test",
            post_method="petri_conformance",
            results_df=results_df,
            mapping_path=Path(metadata["data_root"]) / dataset / "mapping.txt",
            allowed_transitions=allowed,
            total_seconds=total_seconds,
            calibrated=False,
        )
        per_case = build_per_case_metric_rows(
            dataset=dataset,
            seed=args.seed,
            diffact_fraction=100,
            petri_fraction=None,
            condition=args.condition,
            split="test",
            post_method="petri_conformance",
            results_df=results_df,
            combined_map_path=map_path,
            mapping_path=Path(metadata["data_root"]) / dataset / "mapping.txt",
            allowed_transitions=allowed,
            calibrated=False,
        )
        metrics = retain_approved_score_methods(metrics)
        per_case = retain_approved_score_methods(per_case)
        runtime, per_case_runtime = build_runtime_tables(
            seed=args.seed,
            diffact_fraction=100,
            petri_fraction=None,
            condition=args.condition,
            split="test",
            method="petri_conformance",
            results_df=results_df,
            combined_map_path=map_path,
            total_seconds=total_seconds,
        )
        for frame in (metrics, per_case, runtime, per_case_runtime):
            frame["official_fold"] = metadata["fold"]
            frame["trajectory_seed"] = args.seed
            frame["checkpoint_epoch_index"] = epoch
            frame["completed_epochs"] = checkpoint_meta["completed_epochs"]
            frame["schedule_fraction_pct"] = checkpoint_meta["schedule_fraction_pct"]
            frame["training_item_presentations"] = checkpoint_meta["training_item_presentations"]
            frame["trainer_step"] = checkpoint_meta["trainer_step"]
            frame["optimizer_updates"] = checkpoint_meta["optimizer_updates"]
            frame["petri_discovery_source"] = args.petri_discovery_source
            frame["is_oracle_condition"] = is_oracle
            frame["oracle_upper_bound"] = is_oracle
            frame["artifact_scope"] = "final"
        metrics.to_csv(out_dir / "metrics.csv", index=False)
        per_case.to_csv(out_dir / "per_case_metrics.csv", index=False)
        runtime.to_csv(out_dir / "runtime.csv", index=False)
        per_case_runtime.to_csv(out_dir / "per_case_runtime.csv", index=False)
        atomic_json(out_dir / "accuracy_dict.json", accuracy_dict)
        atomic_json(
            out_dir / "postprocess_config.json",
            {
                "schema_version": 1,
                "study_type": metadata["study_type"],
                "dataset": dataset,
                "official_fold": metadata["fold"],
                "trajectory_seed": args.seed,
                "checkpoint": checkpoint_meta,
                "condition": args.condition,
                "petri_discovery_source": args.petri_discovery_source,
                "oracle_upper_bound": is_oracle,
                "petri_discovery_uses_test_gt": is_oracle,
                "is_oracle_condition": is_oracle,
                "petri_discovery_manifest": str(train_manifest),
                "petri_train_original_cases": discovery_cases,
                "eval_original_cases": eval_cases,
                "context_reused_across_checkpoint_grid": True,
                "context_discovery_ordinal": discovery_count,
                "context_train_case_ids": list(context.train_case_ids),
                "n_model_places": len(context.model.places),
                "n_model_transitions": len(context.model.transitions),
                "canonical_decoder": CANONICAL_DECODER,
                "recovery_config": serializable_config(config),
                "diffact_comparator_provenance": comparator_provenance,
                "total_seconds": total_seconds,
            },
        )
        completed_epochs.append(epoch)
        print(f"Completed {args.condition} epoch={epoch}: {out_dir / 'metrics.csv'}")

    if context is None and len(completed_epochs) != len(epochs):
        raise RuntimeError("No Petri recovery context was built")
    if discovery_count > 1:
        raise AssertionError(f"Petri context was discovered {discovery_count} times")
    atomic_json(
        experiment_dir
        / "petri"
        / args.condition
        / f"seed_{args.seed}"
        / "decode_grid_complete.json",
        {
            "schema_version": 1,
            "completed": sorted(completed_epochs) == sorted(epochs),
            "condition": args.condition,
            "petri_discovery_source": args.petri_discovery_source,
            "is_oracle_condition": is_oracle,
            "checkpoint_epoch_indices": completed_epochs,
            "petri_context_discovery_count_this_invocation": discovery_count,
            "petri_discovery_manifest": str(train_manifest),
            "petri_discovery_manifest_sha256": file_sha256(train_manifest),
        },
    )


if __name__ == "__main__":
    main()
