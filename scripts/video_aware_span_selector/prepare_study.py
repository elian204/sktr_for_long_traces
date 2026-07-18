#!/usr/bin/env python3
"""Generate an immutable, leakage-safe Breakfast OOF selector study."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from common import (
    DATASET,
    DEFAULT_DATA_ROOT,
    DEFAULT_DIFFACT_ROOT,
    DEFAULT_STUDY_DIR,
    FINAL_EPOCH,
    INNER_FOLDS,
    OUTER_FOLD,
    PROTOCOL_VERSION,
    SCRIPT_DIR,
    SEED,
    atomic_write_json,
    create_alignment_dir,
    create_dataset_view,
    file_sha256,
    fold_summary,
    load_case_infos,
    make_subject_disjoint_inner_folds,
    official_splits,
    read_bundle,
    source_provenance,
    write_bundle,
    write_lines,
)


DEFAULT_OUTER_EXPORT_DIR = Path(
    "/data1/eli-bogdanov/sktr_runs/breakfast_diffact_raw_exports_v1/fold_1"
)
GPU_ASSIGNMENT = (0, 1, 2)


def write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def shell_command(parts: List[str]) -> str:
    return shlex.join(parts)


def validate_outer_export(export_dir: Path, outer_test: List[str]) -> Dict[str, Any]:
    map_path = export_dir / "video_index_map.txt"
    if not map_path.is_file():
        raise FileNotFoundError(f"Missing outer DiffAct export map: {map_path}")
    rows = [line.split(maxsplit=1) for line in map_path.read_text().splitlines() if line.strip()]
    if any(len(row) != 2 for row in rows):
        raise ValueError(f"Malformed export map: {map_path}")
    mapped = {row[1]: int(row[0]) for row in rows}
    missing = sorted(set(outer_test) - set(mapped))
    if missing:
        raise ValueError(f"Outer release export misses {len(missing)} test videos: {missing[:5]}")
    sample_index = mapped[outer_test[0]]
    required = (
        export_dir / f"{sample_index}_raw.npy",
        export_dir / f"{sample_index}.npy",
        export_dir / f"{sample_index}_pred.npy",
        export_dir / "mapping.txt",
        export_dir / "ground_truth.csv",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Outer release export is incomplete: {path}")
    return {
        "path": str(export_dir.resolve()),
        "video_count": len(rows),
        "outer_test_video_count": len(outer_test),
        "video_index_map_sha256": file_sha256(map_path),
        "stream_semantics": {
            "raw": "pre-postprocessing decoder probabilities ({case_id}_raw.npy)",
            "canonical": "median-smoothed normalized probabilities ({case_id}.npy)",
            "official": "DiffAct final discrete predictions ({case_id}_pred.npy)",
        },
    }


def build_study(args: argparse.Namespace) -> None:
    study_dir = args.study_dir.resolve()
    data_root = args.data_root.resolve()
    diffact_root = args.diffact_root.resolve()
    outer_export_dir = args.outer_export_dir.resolve()
    if study_dir.exists():
        raise FileExistsError(
            f"Study directory already exists: {study_dir}. Study metadata are immutable; "
            "choose a new directory/version."
        )
    baseline_config_path = diffact_root / "configs" / "Breakfast-Trained-S1.json"
    if not baseline_config_path.is_file():
        raise FileNotFoundError(baseline_config_path)
    baseline = json.loads(baseline_config_path.read_text(encoding="utf-8"))
    if int(baseline["num_epochs"]) != FINAL_EPOCH + 1:
        raise ValueError(
            f"Breakfast schedule must save epoch {FINAL_EPOCH}; config has "
            f"num_epochs={baseline['num_epochs']}"
        )
    if FINAL_EPOCH % int(baseline["log_freq"]):
        raise ValueError("The pre-specified final epoch is not a checkpoint epoch")

    outer_train, outer_test = official_splits(data_root, OUTER_FOLD)
    if len(outer_train) != 1460 or len(outer_test) != 252:
        raise ValueError(
            f"Unexpected Breakfast fold-1 cardinality: train={len(outer_train)}, "
            f"test={len(outer_test)}"
        )
    infos = load_case_infos(data_root, outer_train)
    inner_folds = make_subject_disjoint_inner_folds(infos, INNER_FOLDS, SEED)
    outer_export = validate_outer_export(outer_export_dir, outer_test)

    study_dir.mkdir(parents=True)
    manifests = study_dir / "manifests"
    write_bundle(manifests / "outer_train_cases.txt", outer_train)
    write_bundle(manifests / "outer_test_cases.txt", outer_test)

    tasks: List[Dict[str, Any]] = []
    inner_metadata: Dict[str, Dict[str, Any]] = {}
    all_train_set = set(outer_train)
    for inner_fold in range(1, INNER_FOLDS + 1):
        heldout_cases = inner_folds[inner_fold]
        heldout_set = set(heldout_cases)
        train_cases = sorted(all_train_set - heldout_set)
        train_people = {info.participant for info in infos if info.case_id in train_cases}
        heldout_people = {info.participant for info in infos if info.case_id in heldout_set}
        if train_people.intersection(heldout_people):
            raise ValueError(f"Inner fold {inner_fold} leaks participant groups")
        if len(train_people) != 26 or len(heldout_people) != 13:
            raise ValueError(
                f"Inner fold {inner_fold}: expected 26/13 train/held people; got "
                f"{len(train_people)}/{len(heldout_people)}"
            )

        inner_manifest = manifests / f"inner_fold_{inner_fold}"
        train_manifest = inner_manifest / "train_cases.txt"
        heldout_manifest = inner_manifest / "heldout_cases.txt"
        write_bundle(train_manifest, train_cases)
        write_bundle(heldout_manifest, heldout_cases)
        view_root = study_dir / "diffact_dataset_views" / f"inner_fold_{inner_fold}"
        create_dataset_view(view_root, data_root, train_cases, heldout_cases)
        align_dir = study_dir / "align" / f"inner_fold_{inner_fold}"
        create_alignment_dir(align_dir, data_root, heldout_cases)

        run_dir = study_dir / "runs" / f"inner_fold_{inner_fold}"
        naming = f"breakfast_outer1_inner{inner_fold}_seed{SEED}_oof"
        config = dict(baseline)
        config.update(
            {
                "naming": naming,
                "root_data_dir": str(view_root),
                "split_id": 1,
                "result_dir": str(run_dir / "training"),
                "random_seed": SEED,
                "initialization_seed": SEED,
                "evaluate_during_training": False,
                "log_train_results": False,
                "selector_protocol_version": PROTOCOL_VERSION,
                "outer_fold": OUTER_FOLD,
                "inner_fold": inner_fold,
                "training_subset_manifest": str(train_manifest),
                "heldout_oof_manifest": str(heldout_manifest),
                "checkpoint_selection": "pre_specified_final_epoch_no_heldout_selection",
                "pre_specified_final_epoch": FINAL_EPOCH,
            }
        )
        config_path = run_dir / "config.json"
        atomic_write_json(config_path, config)
        final_checkpoint = run_dir / "training" / naming / f"epoch-{FINAL_EPOCH}.model"
        output_dir = run_dir / "softmax_heldout"
        task_id = f"breakfast_outer1_inner{inner_fold}_seed{SEED}"
        gpu = GPU_ASSIGNMENT[inner_fold - 1]
        task = {
            "task_id": task_id,
            "dataset": DATASET,
            "outer_fold": OUTER_FOLD,
            "inner_fold": inner_fold,
            "seed": SEED,
            "gpu": gpu,
            "config_path": str(config_path),
            "train_manifest": str(train_manifest),
            "heldout_manifest": str(heldout_manifest),
            "align_dir": str(align_dir),
            "run_dir": str(run_dir),
            "naming": naming,
            "final_checkpoint": str(final_checkpoint),
            "softmax_output_dir": str(output_dir),
            "train_summary": fold_summary(infos, train_cases),
            "heldout_summary": fold_summary(infos, heldout_cases),
        }
        tasks.append(task)
        inner_metadata[str(inner_fold)] = {
            "train_manifest": str(train_manifest),
            "heldout_manifest": str(heldout_manifest),
            "train_summary": task["train_summary"],
            "heldout_summary": task["heldout_summary"],
            "train_manifest_sha256": file_sha256(train_manifest),
            "heldout_manifest_sha256": file_sha256(heldout_manifest),
        }

    provenance = source_provenance(diffact_root)
    metadata: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study_dir),
        "data_root": str(data_root),
        "diffact_root": str(diffact_root),
        "dataset": DATASET,
        "outer_fold": OUTER_FOLD,
        "seed": SEED,
        "inner_fold_count": INNER_FOLDS,
        "subject_disjoint_inner_folds": True,
        "outer_train_manifest": str(manifests / "outer_train_cases.txt"),
        "outer_test_manifest": str(manifests / "outer_test_cases.txt"),
        "outer_train_manifest_sha256": file_sha256(manifests / "outer_train_cases.txt"),
        "outer_test_manifest_sha256": file_sha256(manifests / "outer_test_cases.txt"),
        "outer_train_case_count": len(outer_train),
        "outer_test_case_count": len(outer_test),
        "inner_folds": inner_metadata,
        "outer_release_export": outer_export,
        "baseline_config": str(baseline_config_path),
        "baseline_config_sha256": file_sha256(baseline_config_path),
        "pre_specified_final_epoch": FINAL_EPOCH,
        "checkpoint_selection": "pre_specified_final_epoch_no_test_selection",
        "feature_provenance_rule": (
            "Each OOF video's learned priors/process features use only that inner fold's "
            "training manifest; outer-test features use only outer_train_cases.txt."
        ),
        "oracle_outputs_are_diagnostic_only": True,
        "source_provenance": provenance,
    }
    atomic_write_json(study_dir / "study_metadata.json", metadata)
    atomic_write_json(study_dir / "tasks.json", {"tasks": tasks})

    python = sys.executable
    for task in tasks:
        runner = shell_command(
            [
                python,
                "-u",
                str(SCRIPT_DIR / "run_oof_task.py"),
                "--study-dir",
                str(study_dir),
                "--task-id",
                task["task_id"],
            ]
        )
        write_executable(
            study_dir / "queues" / f"gpu_{task['gpu']}.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\n" + runner + "\n",
        )

    tmux_lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for task in tasks:
        session = f"bfast_sel_o1_i{task['inner_fold']}_g{task['gpu']}"
        queue = study_dir / "queues" / f"gpu_{task['gpu']}.sh"
        log = study_dir / "logs" / f"{task['task_id']}.tmux.log"
        tmux_command = f"{shlex.quote(str(queue))} 2>&1 | tee -a {shlex.quote(str(log))}"
        tmux_lines.extend(
            [
                f"if tmux has-session -t {shlex.quote(session)} 2>/dev/null; then",
                f"  echo 'Session already exists: {session}'",
                "else",
                f"  tmux new-session -d -s {shlex.quote(session)} {shlex.quote(tmux_command)}",
                f"  echo 'Started {session} on GPU {task['gpu']}'",
                "fi",
            ]
        )
    write_executable(study_dir / "launch_tmux.sh", "\n".join(tmux_lines) + "\n")

    status_command = shell_command(
        [python, str(SCRIPT_DIR / "study_status.py"), "--study-dir", str(study_dir)]
    )
    write_executable(
        study_dir / "status.sh", "#!/usr/bin/env bash\nset -euo pipefail\n" + status_command + "\n"
    )
    analyze_command = shell_command(
        [python, "-u", str(SCRIPT_DIR / "analyze_selector.py"), "--study-dir", str(study_dir)]
    )
    write_executable(
        study_dir / "analyze.sh", "#!/usr/bin/env bash\nset -euo pipefail\n" + analyze_command + "\n"
    )
    write_lines(
        study_dir / "DO_NOT_EDIT.txt",
        [
            "Immutable generated study metadata.",
            f"Protocol: {PROTOCOL_VERSION}",
            "If source/config/manifests change, generate a new versioned study directory.",
            "launch_tmux.sh is intentionally not run by prepare_study.py.",
        ],
    )
    print(f"Prepared immutable study: {study_dir}")
    for task in tasks:
        print(
            f"  inner={task['inner_fold']} gpu={task['gpu']} "
            f"train={task['train_summary']['case_count']} "
            f"heldout={task['heldout_summary']['case_count']} "
            f"participants={task['heldout_summary']['participant_count']}"
        )
    print(f"Source digest: {provenance['source_digest']}")
    print("Nothing was launched. Run launch_tmux.sh only after the assigned GPUs are free.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--diffact-root", type=Path, default=DEFAULT_DIFFACT_ROOT)
    parser.add_argument("--outer-export-dir", type=Path, default=DEFAULT_OUTER_EXPORT_DIR)
    args = parser.parse_args()
    build_study(args)


if __name__ == "__main__":
    main()

