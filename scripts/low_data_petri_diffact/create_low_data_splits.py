#!/usr/bin/env python3
"""Create fixed splits and nested low-data training manifests."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from low_data_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DIFFACT_ROOT,
    DEFAULT_EXPERIMENT_DIR,
    EXPECTED_SUBSET_CASE_COUNTS,
    FRACTIONS,
    LOW_DATA_PROTOCOL_VERSION,
    SEEDS,
    WORKSPACE_ROOT,
    build_case_table,
    create_dataset_view,
    ensure_report_stub,
    expected_fraction_size,
    file_sha256,
    load_mapping,
    load_official_split,
    stratified_order_by_variant,
    summarize_subset,
    write_alignment_dir,
    write_lines,
)


def parse_ints(values: List[str]) -> List[int]:
    return [int(v) for v in values]


def baseline_config_path(diffact_root: Path, dataset: str, fold: int) -> Path:
    prefix = {
        "50salads": "50salads",
        "gtea": "GTEA",
        "breakfast": "Breakfast",
    }[dataset]
    return diffact_root / "configs" / f"{prefix}-Trained-S{fold}.json"


def checkpoint_path(diffact_root: Path, dataset: str, fold: int) -> Path:
    prefix = {
        "50salads": "50salads",
        "gtea": "GTEA",
        "breakfast": "Breakfast",
    }[dataset]
    return diffact_root / "trained_models" / f"{prefix}-Trained-S{fold}" / "release.model"


def find_existing_softmax_dirs(diffact_root: Path) -> List[str]:
    roots = []
    for path in sorted((diffact_root / "results").glob("**/video_index_map.txt")):
        roots.append(str(path.parent))
    return roots


def write_repro_commands(
    experiment_dir: Path,
    dataset: str,
    fold: int,
    data_root: Path,
    diffact_root: Path,
) -> None:
    script_dir = WORKSPACE_ROOT / "scripts" / "low_data_petri_diffact"
    cmd_path = experiment_dir / "commands_to_reproduce.sh"
    cmd_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        f"ROOT={WORKSPACE_ROOT}\n"
        f"EXP={experiment_dir}\n"
        f"DATA_ROOT={data_root}\n"
        f"DIFFACT_ROOT={diffact_root}\n\n"
        "cd \"$ROOT\"\n\n"
        f"python {script_dir / 'create_low_data_splits.py'} "
        f"--dataset {dataset} --fold {fold} --data-root \"$DATA_ROOT\" "
        "--experiment-dir \"$EXP\"\n"
        f"python {script_dir / 'verify_low_data_manifests.py'} --experiment-dir \"$EXP\"\n\n"
        "# Full training is intentionally explicit because it is expensive. The pipeline\n"
        "# waits for a preferred GPU if requested and skips already completed artifacts.\n"
        f"python {script_dir / 'run_low_data_pipeline.py'} "
        "--experiment-dir \"$EXP\" --data-root \"$DATA_ROOT\" --diffact-root \"$DIFFACT_ROOT\" "
        "--device auto --wait-for-gpu "
        "--petri-method petri_transition_viterbi --transition-illegal-penalty 2.0 "
        "--execute\n\n"
        "# Pilot example: one seed/fraction with an epoch override, clearly not final results.\n"
        f"# python {script_dir / 'run_low_data_pipeline.py'} "
        "--experiment-dir \"$EXP\" --data-root \"$DATA_ROOT\" --diffact-root \"$DIFFACT_ROOT\" "
        "--seeds 0 --fractions 25 --num-epochs 101 --device auto --wait-for-gpu "
        "--petri-method petri_transition_viterbi --transition-illegal-penalty 2.0 "
        "--execute\n",
        encoding="utf-8",
    )
    cmd_path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["50salads", "gtea", "breakfast"], default="50salads")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--diffact-root", type=Path, default=DEFAULT_DIFFACT_ROOT)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--seeds", nargs="+", default=[str(x) for x in SEEDS])
    parser.add_argument("--fractions", nargs="+", default=[str(x) for x in FRACTIONS])
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    manifests_dir = experiment_dir / "manifests"
    data_root = args.data_root.resolve()
    diffact_root = args.diffact_root.resolve()
    seeds = parse_ints(args.seeds)
    fractions = parse_ints(args.fractions)
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Seeds must be unique: {seeds}")
    if len(fractions) != len(set(fractions)) or any(f not in FRACTIONS for f in fractions):
        raise ValueError(f"Fractions must be unique values drawn from {list(FRACTIONS)}: {fractions}")

    experiment_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    official_train_path = data_root / args.dataset / "splits" / f"train.split{args.fold}.bundle"
    official_test_path = data_root / args.dataset / "splits" / f"test.split{args.fold}.bundle"
    official_train_sha256 = file_sha256(official_train_path)
    official_test_sha256 = file_sha256(official_test_path)
    official_train, official_test = load_official_split(data_root, args.dataset, args.fold)
    all_split_cases = list(dict.fromkeys(official_train + official_test))
    case_table = build_case_table(data_root, args.dataset, all_split_cases)
    train_pool = list(official_train)
    val_cases: List[str] = []

    overlap = {
        "train_test": sorted(set(train_pool).intersection(official_test)),
    }
    if any(overlap.values()):
        raise ValueError(f"Split overlap detected: {overlap}")

    split_rows: List[Dict[str, Any]] = []
    for case_id in train_pool:
        split_rows.append({"case_id": case_id, "split": "train_pool"})
    for case_id in official_test:
        split_rows.append({"case_id": case_id, "split": "test"})
    pd.DataFrame(split_rows).to_csv(manifests_dir / "split_manifest.csv", index=False)
    write_lines(manifests_dir / "official_train_cases.txt", official_train)
    write_lines(manifests_dir / "train_pool_cases.txt", train_pool)
    write_lines(manifests_dir / "val_cases.txt", val_cases)
    write_lines(manifests_dir / "test_cases.txt", official_test)

    label_to_idx, labels = load_mapping(data_root / args.dataset / "mapping.txt")
    all_activities = [str(i) for i in range(len(labels))]
    subset_summary_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        seed_dir = manifests_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        order = stratified_order_by_variant(case_table, train_pool, seed)
        write_lines(seed_dir / "train_pool_stratified_order.txt", order)

        for fraction in fractions:
            n = expected_fraction_size(args.dataset, len(train_pool), fraction)
            subset = order[:n]
            write_lines(seed_dir / f"train_cases_frac_{fraction}.txt", subset)

            summary = summarize_subset(
                case_table=case_table,
                subset_cases=subset,
                train_pool_cases=train_pool,
                all_activities=all_activities,
            )
            row = {"seed": seed, "fraction": fraction, **summary}
            subset_summary_rows.append(row)
            pd.DataFrame([row]).to_csv(
                seed_dir / f"subset_summary_frac_{fraction}.csv",
                index=False,
            )

            view_root = experiment_dir / "diffact_dataset_views" / f"seed_{seed}" / f"frac_{fraction}"
            create_dataset_view(
                data_root=data_root,
                dataset=args.dataset,
                view_root=view_root,
                train_cases=subset,
            )

    pd.DataFrame(subset_summary_rows).to_csv(
        experiment_dir / "subset_coverage_summary.csv",
        index=False,
    )

    align_root = experiment_dir / "align"
    legacy_val_alignment = align_root / "val"
    if legacy_val_alignment.is_symlink() or legacy_val_alignment.is_file():
        legacy_val_alignment.unlink()
    elif legacy_val_alignment.is_dir():
        shutil.rmtree(legacy_val_alignment)
    write_alignment_dir(data_root, args.dataset, align_root / "test", "test", official_test)

    if file_sha256(official_train_path) != official_train_sha256:
        raise RuntimeError(f"Official training bundle changed during generation: {official_train_path}")
    if file_sha256(official_test_path) != official_test_sha256:
        raise RuntimeError(f"Official test bundle changed during generation: {official_test_path}")

    expected_counts = {
        str(fraction): expected_fraction_size(args.dataset, len(train_pool), fraction)
        for fraction in fractions
    }
    metadata = {
        "protocol_version": LOW_DATA_PROTOCOL_VERSION,
        "goal": "Low-data DiffAct training-case ablation for Petri-net/SKTR postprocessing.",
        "dataset": args.dataset,
        "fold": args.fold,
        "workspace_root": str(WORKSPACE_ROOT),
        "postprocessing_code_path": str(WORKSPACE_ROOT / "src"),
        "diffact_repo_path": str(diffact_root),
        "data_root": str(data_root),
        "baseline_config_path": str(baseline_config_path(diffact_root, args.dataset, args.fold)),
        "previous_checkpoint_path": str(checkpoint_path(diffact_root, args.dataset, args.fold)),
        "previous_checkpoint_usage": "Reference only; low-data runs must initialize randomly.",
        "previous_result_path": "",
        "previous_softmax_output_dirs": find_existing_softmax_dirs(diffact_root),
        "official_train_split_path": str(official_train_path),
        "official_test_split_path": str(official_test_path),
        "official_train_split_sha256": official_train_sha256,
        "official_test_split_sha256": official_test_sha256,
        "generated_split_manifest_path": str(manifests_dir / "split_manifest.csv"),
        "generated_train_pool_cases_path": str(manifests_dir / "train_pool_cases.txt"),
        "generated_val_cases_path": str(manifests_dir / "val_cases.txt"),
        "generated_test_cases_path": str(manifests_dir / "test_cases.txt"),
        "validation_alignment_dir": None,
        "test_alignment_dir": str(align_root / "test"),
        "validation_policy": (
            "No validation cases are carved out. Every fraction is a nested prefix of a "
            "variant-stratified ordering of the complete official training fold."
        ),
        "periodic_evaluation_policy": (
            "Disabled during low-data training; official test inference occurs only after training."
        ),
        "checkpoint_policy": (
            "Fresh random initialization per seed/fraction, except resuming that run's own "
            "latest.pt; use the pre-specified final epoch checkpoint without test-based selection."
        ),
        "subset_source": "complete_official_training_fold",
        "n_official_train": len(official_train),
        "n_train_pool": len(train_pool),
        "n_val": len(val_cases),
        "n_test": len(official_test),
        "fractions": fractions,
        "seeds": seeds,
        "expected_subset_case_counts": expected_counts,
        "canonical_expected_subset_case_counts": EXPECTED_SUBSET_CASE_COUNTS.get(args.dataset),
        "methodological_rule": "Fractions sample whole training cases/videos/traces; traces are not shortened.",
        "previous_train_command_pattern": "cd baselines/DiffAct && python -u main.py --config configs/<dataset>-Trained-S<fold>.json --device <gpu>",
        "previous_infer_command_pattern": "cd baselines/DiffAct && python -u export_softmax.py --config configs/<dataset>-Trained-S<fold>.json --output_dir results/<dataset>/softmax_fold<fold> --root_data_dir <DATA_ROOT> --device <gpu>",
    }
    (experiment_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ensure_report_stub(experiment_dir, metadata)
    write_repro_commands(experiment_dir, args.dataset, args.fold, data_root, diffact_root)

    print(f"Output directory: {experiment_dir}")
    print(f"Dataset: {args.dataset}, fold: {args.fold}")
    print(f"Official train cases: {len(official_train)}")
    print(f"Training pool cases (complete official train): {len(train_pool)}")
    print("Validation cases: 0 (no-validation protocol)")
    print(f"Fixed test cases: {len(official_test)}")
    print(f"Subset coverage summary: {experiment_dir / 'subset_coverage_summary.csv'}")
    print(f"Reproduction commands: {experiment_dir / 'commands_to_reproduce.sh'}")


if __name__ == "__main__":
    main()
