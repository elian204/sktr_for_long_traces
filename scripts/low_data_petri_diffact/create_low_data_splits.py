#!/usr/bin/env python3
"""Create fixed splits and nested low-data training manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from low_data_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DIFFACT_ROOT,
    DEFAULT_EXPERIMENT_DIR,
    FRACTIONS,
    SEEDS,
    WORKSPACE_ROOT,
    build_case_table,
    create_dataset_view,
    ensure_report_stub,
    fraction_size,
    load_mapping,
    load_official_split,
    read_lines,
    split_validation_from_train,
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
        "--experiment-dir \"$EXP\" --val-ratio 0.2\n"
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
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=1729)
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    manifests_dir = experiment_dir / "manifests"
    data_root = args.data_root.resolve()
    diffact_root = args.diffact_root.resolve()
    seeds = parse_ints(args.seeds)
    fractions = parse_ints(args.fractions)

    experiment_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    official_train, official_test = load_official_split(data_root, args.dataset, args.fold)
    all_split_cases = list(dict.fromkeys(official_train + official_test))
    case_table = build_case_table(data_root, args.dataset, all_split_cases)
    train_pool, val_cases = split_validation_from_train(
        case_table=case_table,
        official_train_cases=official_train,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
    )

    overlap = {
        "train_val": sorted(set(train_pool).intersection(val_cases)),
        "train_test": sorted(set(train_pool).intersection(official_test)),
        "val_test": sorted(set(val_cases).intersection(official_test)),
    }
    if any(overlap.values()):
        raise ValueError(f"Split overlap detected: {overlap}")

    split_rows: List[Dict[str, Any]] = []
    for case_id in train_pool:
        split_rows.append({"case_id": case_id, "split": "train_pool"})
    for case_id in val_cases:
        split_rows.append({"case_id": case_id, "split": "val"})
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
            n = fraction_size(len(train_pool), fraction)
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
                validation_cases=val_cases,
            )

    pd.DataFrame(subset_summary_rows).to_csv(
        experiment_dir / "subset_coverage_summary.csv",
        index=False,
    )

    align_root = experiment_dir / "align"
    write_alignment_dir(data_root, args.dataset, align_root / "val", "val", val_cases)
    write_alignment_dir(data_root, args.dataset, align_root / "test", "test", official_test)

    previous_result = WORKSPACE_ROOT / "results" / "paper_diffact_50salads_w7_topm1_topk3"
    metadata = {
        "goal": "Low-data DiffAct training-case ablation for Petri-net/SKTR postprocessing.",
        "dataset": args.dataset,
        "fold": args.fold,
        "workspace_root": str(WORKSPACE_ROOT),
        "postprocessing_code_path": str(WORKSPACE_ROOT / "src"),
        "diffact_repo_path": str(diffact_root),
        "data_root": str(data_root),
        "baseline_config_path": str(baseline_config_path(diffact_root, args.dataset, args.fold)),
        "previous_checkpoint_path": str(checkpoint_path(diffact_root, args.dataset, args.fold)),
        "previous_result_path": str(previous_result if previous_result.exists() else ""),
        "previous_softmax_output_dirs": find_existing_softmax_dirs(diffact_root),
        "official_train_split_path": str(data_root / args.dataset / "splits" / f"train.split{args.fold}.bundle"),
        "official_test_split_path": str(data_root / args.dataset / "splits" / f"test.split{args.fold}.bundle"),
        "generated_split_manifest_path": str(manifests_dir / "split_manifest.csv"),
        "generated_train_pool_cases_path": str(manifests_dir / "train_pool_cases.txt"),
        "generated_val_cases_path": str(manifests_dir / "val_cases.txt"),
        "generated_test_cases_path": str(manifests_dir / "test_cases.txt"),
        "validation_alignment_dir": str(align_root / "val"),
        "test_alignment_dir": str(align_root / "test"),
        "validation_policy": (
            "No official validation split was found; validation was carved once from the "
            "official training split at case/video level using run-collapsed variants."
        ),
        "split_seed": args.split_seed,
        "val_ratio": args.val_ratio,
        "n_official_train": len(official_train),
        "n_train_pool": len(train_pool),
        "n_val": len(val_cases),
        "n_test": len(official_test),
        "fractions": fractions,
        "seeds": seeds,
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
    print(f"Fixed train pool cases: {len(train_pool)}")
    print(f"Fixed validation cases: {len(val_cases)}")
    print(f"Fixed test cases: {len(official_test)}")
    print(f"Subset coverage summary: {experiment_dir / 'subset_coverage_summary.csv'}")
    print(f"Reproduction commands: {experiment_dir / 'commands_to_reproduce.sh'}")


if __name__ == "__main__":
    main()
