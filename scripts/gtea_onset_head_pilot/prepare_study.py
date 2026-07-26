#!/usr/bin/env python3
"""Generate the immutable, non-launched GTEA onset-head review study."""

from __future__ import annotations

import argparse
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from common import (
    BASELINE_ONSET_LOSS_WEIGHT,
    CHECKPOINT_EPOCHS,
    DATASET,
    DECODER_BOUNDARY_WEIGHT,
    DEFAULT_DATA_ROOT,
    DEFAULT_DIFFACT_ROOT,
    DEFAULT_LOCKED_REFERENCE_CONFIG,
    DEFAULT_LOCKED_TRAIN_BUNDLE,
    DEFAULT_LOCKED_TRAIN_MANIFEST,
    DEFAULT_MULTIFOLD_DECISION,
    DEFAULT_STUDY_DIR,
    EXPECTED_NUM_EPOCHS,
    EXPECTED_TEST_CASES,
    EXPECTED_TRAIN_CASES,
    FINAL_EPOCH,
    FOLD,
    INFERENCE_SEEDS,
    INFERENCE_SEED_STRIDE,
    ONSET_BUMP_POSITIVE_WEIGHT,
    ONSET_LOSS_WEIGHTS,
    ONSET_SMOOTH,
    PHYSICAL_GPU,
    PRIMARY_ONSET_LOSS_WEIGHT,
    PROTOCOL_VERSION,
    SCRIPT_DIR,
    TRAINING_SEED,
    atomic_write_json,
    build_variant_config,
    canonical_digest,
    checkpoint_path,
    create_alignment_dir,
    create_dataset_view,
    file_sha256,
    load_json,
    read_bundle,
    source_provenance,
    training_invariant_digest,
    training_invariant_payload,
    variant_id,
    write_bundle,
    write_lines,
)


def write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def shell_command(parts: Sequence[Any]) -> str:
    return shlex.join([str(part) for part in parts])


def validate_reference_config(config: Mapping[str, Any]) -> None:
    expectations = {
        "dataset_name": DATASET,
        "boundary_smooth": ONSET_SMOOTH,
        "soft_label": 1.4,
        "num_epochs": EXPECTED_NUM_EPOCHS,
        "log_freq": 100,
        "sample_rate": 1,
        "random_seed": 0,
        "initialization_seed": 0,
        "evaluate_during_training": False,
        "log_train_results": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in expectations.items()
        if config.get(key) != expected
    }
    if config.get("postprocess") != {"type": "purge", "value": 3}:
        mismatches["postprocess"] = config.get("postprocess")
    if float(config["loss_weights"]["encoder_boundary_loss"]) != 0.0:
        mismatches["loss_weights.encoder_boundary_loss"] = config["loss_weights"][
            "encoder_boundary_loss"
        ]
    expected_conditions = ["full", "zero", "boundary03-", "segment=1", "segment=1"]
    if config["diffusion_params"]["cond_types"] != expected_conditions:
        mismatches["diffusion_params.cond_types"] = config["diffusion_params"][
            "cond_types"
        ]
    if any(epoch % int(config["log_freq"]) for epoch in CHECKPOINT_EPOCHS):
        mismatches["checkpoint_grid"] = list(CHECKPOINT_EPOCHS)
    if mismatches:
        raise ValueError(f"Locked reference violates onset-pilot assumptions: {mismatches}")


def build_study(args: argparse.Namespace) -> None:
    study_dir = args.study_dir.resolve()
    data_root = args.data_root.resolve()
    diffact_root = args.diffact_root.resolve()
    reference_path = args.locked_reference_config.resolve()
    locked_manifest = args.locked_train_manifest.resolve()
    locked_bundle = args.locked_train_bundle.resolve()
    multifold_decision = args.multifold_decision_path.resolve()
    if study_dir.exists():
        raise FileExistsError(
            f"Study directory exists: {study_dir}; choose a new immutable version"
        )
    for path in (
        reference_path,
        locked_manifest,
        locked_bundle,
        diffact_root / "main.py",
        diffact_root / "model.py",
        diffact_root / "dataset.py",
        diffact_root / "utils.py",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = load_json(reference_path)
    validate_reference_config(reference)

    split_root = data_root / DATASET / "splits"
    train_split = split_root / "train.split1.bundle"
    test_split = split_root / "test.split1.bundle"
    official_train_cases = read_bundle(train_split)
    train_cases = read_bundle(locked_manifest)
    test_cases = read_bundle(test_split)
    if train_cases != read_bundle(locked_bundle):
        raise ValueError("Locked manifest and actual DiffAct bundle order differ")
    if len(train_cases) != EXPECTED_TRAIN_CASES or len(test_cases) != EXPECTED_TEST_CASES:
        raise ValueError(
            f"Unexpected fold-1 counts: {len(train_cases)}/{len(test_cases)}"
        )
    if set(train_cases) != set(official_train_cases):
        raise ValueError("Locked train cases are not the official fold-1 train set")
    if set(train_cases).intersection(test_cases):
        raise ValueError("Fold-1 train/test overlap")

    study_dir.mkdir(parents=True)
    (study_dir / "logs").mkdir()
    train_manifest = study_dir / "manifests" / "train_cases_frac_100.txt"
    test_manifest = study_dir / "manifests" / "official_test_cases.txt"
    train_manifest.parent.mkdir(parents=True)
    train_manifest.write_bytes(locked_manifest.read_bytes())
    write_bundle(test_manifest, test_cases)
    view_root = create_dataset_view(
        study_dir, data_root, train_cases, test_cases, locked_bundle
    )
    generated_bundle = (
        view_root / DATASET / "splits" / "train.split1.bundle"
    )
    if generated_bundle.read_bytes() != locked_bundle.read_bytes():
        raise ValueError("Generated training bundle is not byte-identical")
    align_dir = create_alignment_dir(study_dir, data_root, test_cases)

    configs = {
        weight: build_variant_config(
            reference,
            study_dir=study_dir,
            view_root=view_root,
            train_manifest=train_manifest,
            test_manifest=test_manifest,
            onset_weight=weight,
        )
        for weight in ONSET_LOSS_WEIGHTS
    }
    invariant_digests = {
        weight: training_invariant_digest(config)
        for weight, config in configs.items()
    }
    if len(set(invariant_digests.values())) != 1:
        raise ValueError(f"Configs differ beyond the onset loss knob: {invariant_digests}")
    invariant_digest = next(iter(invariant_digests.values()))

    tasks: List[Dict[str, Any]] = []
    for weight in ONSET_LOSS_WEIGHTS:
        task_id = variant_id(weight)
        run_dir = study_dir / "runs" / task_id
        config_path = run_dir / "config.json"
        atomic_write_json(config_path, configs[weight])
        task: Dict[str, Any] = {
            "task_id": task_id,
            "task_type": "train_joint_activity_and_class_specific_onset",
            "execution_mode": "train",
            "dataset": DATASET,
            "official_fold": FOLD,
            "training_seed": TRAINING_SEED,
            "decoder_boundary_loss": DECODER_BOUNDARY_WEIGHT,
            "class_specific_onset_loss_weight": weight,
            "role": (
                "paired_onset_disabled_baseline"
                if weight == BASELINE_ONSET_LOSS_WEIGHT
                else "pre_registered_primary"
                if weight == PRIMARY_ONSET_LOSS_WEIGHT
                else "exploratory_onset_weight"
            ),
            "physical_gpu": PHYSICAL_GPU,
            "config_path": str(config_path),
            "run_dir": str(run_dir),
            "model_dir": str(run_dir / "training" / task_id),
            "final_checkpoint": str(
                run_dir / "training" / task_id / f"epoch-{FINAL_EPOCH}.model"
            ),
            "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
            "inference_seeds": list(INFERENCE_SEEDS),
            "train_manifest": str(train_manifest),
            "test_manifest": str(test_manifest),
            "align_dir": str(align_dir),
            "training_invariant_digest": invariant_digest,
            "state_path": str(study_dir / "state" / f"{task_id}.json"),
        }
        task["expected_checkpoints"] = {
            str(epoch): str(checkpoint_path(task, epoch))
            for epoch in CHECKPOINT_EPOCHS
        }
        tasks.append(task)

    provenance = source_provenance(diffact_root)
    metadata: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "immutable": True,
        "launch_gated": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study_dir),
        "workspace_root": str(SCRIPT_DIR.parents[1]),
        "data_root": str(data_root),
        "diffact_root": str(diffact_root),
        "dataset": DATASET,
        "official_fold": FOLD,
        "training_seed": TRAINING_SEED,
        "study_role": "screening_only",
        "screening_scope": "GTEA fold 1, training seed 0",
        "physical_gpu": PHYSICAL_GPU,
        "decoder_boundary_loss": DECODER_BOUNDARY_WEIGHT,
        "onset_loss_weights": list(ONSET_LOSS_WEIGHTS),
        "baseline_onset_loss_weight": BASELINE_ONSET_LOSS_WEIGHT,
        "pre_registered_primary_onset_loss_weight": PRIMARY_ONSET_LOSS_WEIGHT,
        "class_specific_onset_smooth": ONSET_SMOOTH,
        "onset_target": {
            "shape": "class x time",
            "centers": "every GT segment start, including frame 0",
            "class_identity": "starting activity only; preceding class is never marked",
            "smoothing": "Gaussian, sigma equals boundary_smooth",
        },
        "class_imbalance_option": {
            "name": "onset_bump_positive_weight",
            "value": ONSET_BUMP_POSITIVE_WEIGHT,
            "off_by_default": True,
            "implementation": (
                "element weight = 1 + (value - 1) * Gaussian onset target"
            ),
            "frozen_not_experimental": True,
        },
        "activity_supervision_contract": (
            "Original activity labels, encoder/decoder activity heads, and every "
            "pre-existing activity loss remain unchanged. The auxiliary 1x1 onset "
            "head is additive and backpropagates into encoder features."
        ),
        "single_new_knob": "loss_weights.class_specific_onset_loss",
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "inference_seeds": list(INFERENCE_SEEDS),
        "inference_seed_formula": (
            f"model_seed = inference_seed * {INFERENCE_SEED_STRIDE} + video_index"
        ),
        "training_invariant_payload": training_invariant_payload(
            configs[BASELINE_ONSET_LOSS_WEIGHT]
        ),
        "training_invariant_digest": invariant_digest,
        "frozen_fields_statement": (
            "Runtime configs vary only class_specific_onset_loss and output naming/path."
        ),
        "generated_train_manifest": str(train_manifest),
        "generated_train_manifest_sha256": file_sha256(train_manifest),
        "generated_train_bundle": str(generated_bundle),
        "generated_train_bundle_sha256": file_sha256(generated_bundle),
        "locked_train_manifest": str(locked_manifest),
        "locked_train_manifest_sha256": file_sha256(locked_manifest),
        "locked_train_bundle": str(locked_bundle),
        "locked_train_bundle_sha256": file_sha256(locked_bundle),
        "official_train_split": str(train_split),
        "official_train_split_sha256": file_sha256(train_split),
        "official_test_split": str(test_split),
        "official_test_split_sha256": file_sha256(test_split),
        "training_order_contract": {
            "generated_bundle_byte_identical_to_locked_bundle": True,
            "locked_manifest_order_equals_locked_bundle": True,
            "set_equals_official_fold1_train": True,
        },
        "multifold_launch_dependency": {
            "decision_path": str(multifold_decision),
            "required_field": "onset_head_launch_gate_passes",
            "required_value": True,
            "not_required_to_exist_for_review_staging": True,
        },
        "diagnostic_contract": {
            "activity_streams": {
                "pre_purge": "argmax of raw activity probability matrix",
                "post_purge": "official DiffAct purge-3 prediction",
            },
            "standard_metrics": ["acc", "edit", "f1@10", "f1@25", "f1@50"],
            "v2_boundary_metrics": True,
            "onset_exports": "{video_index}_onset.npy, shape class x full frames",
            "onset_quality": [
                "correct-class local-peak signed displacement",
                "mean/median/p90 absolute peak displacement",
                "class-specific peak F1 at +/-5 and +/-10 frames",
            ],
        },
        "decision_rule": {
            "primary_onset_weight": PRIMARY_ONSET_LOSS_WEIGHT,
            "baseline_onset_weight": BASELINE_ONSET_LOSS_WEIGHT,
            "epoch": FINAL_EPOCH,
            "stream": "post_purge",
            "activity_gate": (
                "delta_acc>=0, delta_edit>0, delta_f1@25>0, "
                "delta_segment_count_ratio<=0.02"
            ),
            "mechanistic_gate": (
                "correct-class onset mean absolute peak displacement decreases"
            ),
            "weights_0.1_and_0.5": "exploratory",
            "positive_screen_definition": (
                "The pre-registered 0.3 primary must pass every activity and "
                "mechanistic check."
            ),
            "best_weight_for_required_replication": {
                "eligible_weights": (
                    "Only screened weights passing the same locked activity and "
                    "mechanistic checks"
                ),
                "ranking": [
                    "highest mean delta_f1@25",
                    "highest mean delta_edit",
                    "smaller onset loss weight",
                ],
            },
            "required_positive_screen_replication": {
                "onset_loss_weights": [0.0, "best_screened_onset_weight"],
                "training_seed": 1,
                "new_immutable_study": True,
                "required_before_any_claim_advances": True,
            },
            "launch_policy": (
                "This rule is analysis-only; no task launches unless Task 1 first "
                "records onset_head_launch_gate_passes=true and Fable approves."
            ),
        },
        "source_provenance": provenance,
        "tasks": [task["task_id"] for task in tasks],
    }
    metadata["spec_sha256"] = canonical_digest(
        {key: value for key, value in metadata.items() if key != "spec_sha256"}
    )
    atomic_write_json(study_dir / "study_metadata.json", metadata)
    atomic_write_json(study_dir / "tasks.json", {"tasks": tasks})

    python = sys.executable
    queue_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"export CUDA_VISIBLE_DEVICES={PHYSICAL_GPU}",
    ]
    for task in tasks:
        queue_lines.append(
            shell_command(
                [
                    python,
                    "-u",
                    SCRIPT_DIR / "run_variant.py",
                    "--study-dir",
                    study_dir,
                    "--task-id",
                    task["task_id"],
                ]
            )
        )
    queue_lines.append(
        shell_command([python, "-u", SCRIPT_DIR / "analyze.py", "--study-dir", study_dir])
    )
    queue_path = study_dir / "queues" / "gpu_3.sh"
    write_executable(queue_path, "\n".join(queue_lines) + "\n")

    waiter = study_dir / "waiters" / "gpu_3_wait_then_run.sh"
    waiter_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"{shell_command([python, SCRIPT_DIR / 'verify_launch_gate.py', '--study-dir', study_dir])}",
        "free_checks=0",
        "interval=${FREE_CHECK_INTERVAL_SECONDS:-60}",
        "while (( free_checks < 2 )); do",
        (
            f"  if ! busy=$(nvidia-smi -i {PHYSICAL_GPU} "
            "--query-compute-apps=pid,used_memory,process_name "
            "--format=csv,noheader,nounits 2>&1); then"
        ),
        "    echo 'nvidia-smi failed; onset study refuses to infer availability' >&2",
        '    echo "$busy" >&2',
        "    exit 3",
        "  fi",
        "  busy=$(printf '%s\\n' \"$busy\" | sed '/^[[:space:]]*$/d')",
        '  if [[ -z "$busy" ]]; then',
        "    ((free_checks+=1))",
        '    echo "GPU 3 free check $free_checks/2"',
        "  else",
        "    free_checks=0",
        '    echo "GPU 3 busy; no process will be preempted: $busy"',
        "  fi",
        "  if (( free_checks < 2 )); then sleep \"$interval\"; fi",
        "done",
        f"exec {shlex.quote(str(queue_path))}",
    ]
    write_executable(waiter, "\n".join(waiter_lines) + "\n")

    session = "gtea_onset_head_f1_v1_g3"
    tmux_log = study_dir / "logs" / "gpu_3_queue.tmux.log"
    tmux_command = (
        f"{shlex.quote(str(waiter))} 2>&1 | tee -a {shlex.quote(str(tmux_log))}"
    )
    launch_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(str(study_dir / 'logs'))}",
        f"session={shlex.quote(session)}",
        'if tmux has-session -t "$session" 2>/dev/null; then',
        '  echo "Session already exists: $session" >&2',
        "  exit 4",
        "fi",
        f"tmux new-session -d -s \"$session\" {shlex.quote(tmux_command)}",
        f"echo 'Armed {session}; Task-1 gate and two GPU-free checks are mandatory'",
    ]
    write_executable(study_dir / "launch_tmux.sh", "\n".join(launch_lines) + "\n")
    write_executable(
        study_dir / "status.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + shell_command([python, SCRIPT_DIR / "study_status.py", "--study-dir", study_dir])
        + "\n",
    )
    write_executable(
        study_dir / "analyze.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + shell_command([python, "-u", SCRIPT_DIR / "analyze.py", "--study-dir", study_dir])
        + "\n",
    )
    write_lines(
        study_dir / "DO_NOT_EDIT.txt",
        [
            "Immutable staged GTEA class-specific onset-head pilot.",
            f"Protocol: {PROTOCOL_VERSION}",
            "Task 1 and Fable approval gate all launches.",
            "prepare_study.py launched nothing.",
        ],
    )
    print(f"Prepared onset-head review study: {study_dir}")
    print(f"Grid: onset weights={list(ONSET_LOSS_WEIGHTS)}; baseline boundary weight=1.0")
    print(f"Training invariant digest: {invariant_digest}")
    print(f"Source digest: {provenance['source_digest']}")
    print("Nothing was launched; Task 1 verdict and Fable approval remain mandatory.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--diffact-root", type=Path, default=DEFAULT_DIFFACT_ROOT)
    parser.add_argument(
        "--locked-reference-config",
        type=Path,
        default=DEFAULT_LOCKED_REFERENCE_CONFIG,
    )
    parser.add_argument(
        "--locked-train-manifest",
        type=Path,
        default=DEFAULT_LOCKED_TRAIN_MANIFEST,
    )
    parser.add_argument(
        "--locked-train-bundle", type=Path, default=DEFAULT_LOCKED_TRAIN_BUNDLE
    )
    parser.add_argument(
        "--multifold-decision-path",
        type=Path,
        default=DEFAULT_MULTIFOLD_DECISION,
    )
    build_study(parser.parse_args())


if __name__ == "__main__":
    main()
