#!/usr/bin/env python3
"""Generate the immutable, non-launched GTEA boundary-weight pilot."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from common import (
    BOUNDARY_MATCH_MAX_DISTANCE,
    BOUNDARY_WEIGHTS,
    CHECKPOINT_EPOCHS,
    DATASET,
    DEFAULT_DATA_ROOT,
    DEFAULT_DIFFACT_ROOT,
    DEFAULT_LOCKED_REFERENCE_CONFIG,
    DEFAULT_LOCKED_REFERENCE_EXPORT,
    DEFAULT_LOCKED_REFERENCE_TRAIN_BUNDLE,
    DEFAULT_LOCKED_REFERENCE_TRAIN_MANIFEST,
    DEFAULT_STUDY_DIR,
    EXPECTED_NUM_EPOCHS,
    EXPECTED_TEST_CASES,
    EXPECTED_TRAIN_CASES,
    FINAL_EPOCH,
    FOLD,
    INFERENCE_SEEDS,
    INFERENCE_SEED_STRIDE,
    PHYSICAL_GPU,
    PRIMARY_BOUNDARY_WEIGHT,
    PROTOCOL_VERSION,
    RECONCILIATION_FRAME_DISAGREEMENT_TOLERANCE,
    RECONCILIATION_METRIC_TOLERANCE,
    SCRIPT_DIR,
    SHORT_SEGMENT_MAX_LENGTH,
    TRAINING_SEED,
    atomic_write_json,
    build_variant_config,
    canonical_digest,
    checkpoint_path,
    create_alignment_dir,
    create_dataset_view,
    file_sha256,
    read_bundle,
    source_provenance,
    training_invariant_digest,
    training_invariant_payload,
    variant_id,
    verify_export,
    write_bundle,
    write_lines,
)


def write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def shell_command(parts: Sequence[Any]) -> str:
    return shlex.join([str(part) for part in parts])


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_reference_config(config: Dict[str, Any]) -> None:
    expectations = {
        "dataset_name": DATASET,
        "boundary_smooth": 1,
        "soft_label": 1.4,
        "num_epochs": EXPECTED_NUM_EPOCHS,
        "log_freq": 100,
        "sample_rate": 1,
        "random_seed": TRAINING_SEED,
        "evaluate_during_training": False,
        "log_train_results": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in expectations.items()
        if config.get(key) != expected
    }
    if config.get("postprocess") != {"type": "purge", "value": 3}:
        mismatches["postprocess"] = {
            "expected": {"type": "purge", "value": 3},
            "actual": config.get("postprocess"),
        }
    if float(config["loss_weights"]["decoder_boundary_loss"]) != 0.1:
        mismatches["loss_weights.decoder_boundary_loss"] = {
            "expected": 0.1,
            "actual": config["loss_weights"]["decoder_boundary_loss"],
        }
    if float(config["loss_weights"]["encoder_boundary_loss"]) != 0.0:
        mismatches["loss_weights.encoder_boundary_loss"] = {
            "expected": 0.0,
            "actual": config["loss_weights"]["encoder_boundary_loss"],
        }
    expected_conditions = ["full", "zero", "boundary03-", "segment=1", "segment=1"]
    if config["diffusion_params"]["cond_types"] != expected_conditions:
        mismatches["diffusion_params.cond_types"] = {
            "expected": expected_conditions,
            "actual": config["diffusion_params"]["cond_types"],
        }
    if FINAL_EPOCH % int(config["log_freq"]):
        mismatches["checkpoint_cadence"] = {
            "expected": f"epoch {FINAL_EPOCH} checkpointed",
            "actual": config["log_freq"],
        }
    if any(epoch % int(config["log_freq"]) for epoch in CHECKPOINT_EPOCHS):
        mismatches["checkpoint_grid"] = {
            "expected": list(CHECKPOINT_EPOCHS),
            "actual_log_freq": config["log_freq"],
        }
    if mismatches:
        raise ValueError(f"Locked D100 reference config violates pilot assumptions: {mismatches}")


def build_study(args: argparse.Namespace) -> None:
    study_dir = args.study_dir.resolve()
    data_root = args.data_root.resolve()
    diffact_root = args.diffact_root.resolve()
    locked_config_path = args.locked_reference_config.resolve()
    locked_export = args.locked_reference_export.resolve()
    locked_train_manifest = args.locked_reference_train_manifest.resolve()
    locked_train_bundle = args.locked_reference_train_bundle.resolve()
    if study_dir.exists():
        raise FileExistsError(
            f"Study directory already exists: {study_dir}. Metadata are immutable; "
            "choose a new versioned directory."
        )
    baseline_config_path = diffact_root / "configs" / "GTEA-Trained-S1.json"
    if not baseline_config_path.is_file():
        raise FileNotFoundError(baseline_config_path)
    if not locked_config_path.is_file():
        raise FileNotFoundError(locked_config_path)
    if not locked_train_manifest.is_file():
        raise FileNotFoundError(locked_train_manifest)
    if not locked_train_bundle.is_file():
        raise FileNotFoundError(locked_train_bundle)
    reference_config = load_json(locked_config_path)
    validate_reference_config(reference_config)
    source_training_config_path = Path(
        reference_config["epoch_scarcity_source_config"]
    ).resolve()
    if not source_training_config_path.is_file():
        raise FileNotFoundError(source_training_config_path)
    source_training_config = load_json(source_training_config_path)
    recorded_training_manifest = Path(
        source_training_config["training_subset_manifest"]
    ).resolve()
    if recorded_training_manifest != locked_train_manifest:
        raise ValueError(
            "Locked trajectory config points to a different training manifest: "
            f"{recorded_training_manifest} != {locked_train_manifest}"
        )
    recorded_train_bundle = (
        Path(source_training_config["root_data_dir"])
        / DATASET
        / "splits"
        / "train.split1.bundle"
    ).resolve()
    if not recorded_train_bundle.is_file():
        raise FileNotFoundError(recorded_train_bundle)
    if recorded_train_bundle != locked_train_bundle:
        raise ValueError(
            "Locked D100 source training config points to a different train bundle: "
            f"{recorded_train_bundle} != {locked_train_bundle}"
        )
    if recorded_train_bundle.read_bytes() != locked_train_bundle.read_bytes():
        raise ValueError(
            "Locked D100 source training bundle bytes differ from the explicit reference"
        )

    split_root = data_root / DATASET / "splits"
    train_split = split_root / f"train.split{FOLD}.bundle"
    test_split = split_root / f"test.split{FOLD}.bundle"
    official_train_cases = read_bundle(train_split)
    test_cases = read_bundle(test_split)
    locked_train_cases = read_bundle(locked_train_manifest)
    locked_bundle_cases = read_bundle(locked_train_bundle)
    if locked_train_cases != locked_bundle_cases:
        raise ValueError(
            "Locked frac-100 manifest order differs from the actual locked DiffAct train bundle"
        )
    train_cases = locked_train_cases
    if len(train_cases) != EXPECTED_TRAIN_CASES or len(test_cases) != EXPECTED_TEST_CASES:
        raise ValueError(
            f"Unexpected GTEA fold-1 counts: train={len(train_cases)}, test={len(test_cases)}"
        )
    if len(train_cases) != len(set(train_cases)) or len(test_cases) != len(set(test_cases)):
        raise ValueError("Official split contains duplicate cases")
    overlap = sorted(set(train_cases).intersection(test_cases))
    if overlap:
        raise ValueError(f"Official train/test overlap: {overlap}")
    if set(train_cases) != set(official_train_cases):
        missing = sorted(set(official_train_cases) - set(train_cases))
        extra = sorted(set(train_cases) - set(official_train_cases))
        raise ValueError(
            "Locked frac-100 manifest is not set-equal to official fold-1 train: "
            f"missing={missing}, extra={extra}"
        )
    locked_export_summary = verify_export(locked_export, test_cases)

    study_dir.mkdir(parents=True)
    manifests = study_dir / "manifests"
    train_manifest = manifests / "train_cases_frac_100.txt"
    test_manifest = manifests / "official_test_cases.txt"
    train_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(locked_train_manifest, train_manifest)
    write_bundle(test_manifest, test_cases)
    view_root = create_dataset_view(
        study_dir,
        data_root,
        train_cases,
        test_cases,
        locked_train_bundle=locked_train_bundle,
    )
    generated_train_bundle = view_root / DATASET / "splits" / "train.split1.bundle"
    if train_manifest.read_bytes() != locked_train_manifest.read_bytes():
        raise ValueError("Generated training manifest is not byte-identical to locked manifest")
    if generated_train_bundle.read_bytes() != locked_train_bundle.read_bytes():
        raise ValueError("Generated DiffAct train bundle is not byte-identical to locked bundle")
    align_dir = create_alignment_dir(study_dir, data_root, test_cases)

    variant_configs: Dict[str, Dict[str, Any]] = {}
    for weight in BOUNDARY_WEIGHTS:
        task_id = variant_id(weight)
        config = build_variant_config(
            reference_config,
            study_dir=study_dir,
            view_root=view_root,
            train_manifest=train_manifest,
            test_manifest=test_manifest,
            weight=weight,
        )
        variant_configs[task_id] = config
    invariant_digests = {
        task_id: training_invariant_digest(config)
        for task_id, config in variant_configs.items()
    }
    if len(set(invariant_digests.values())) != 1:
        raise ValueError(
            "Generated variants differ in frozen training fields: "
            f"{invariant_digests}"
        )
    invariant_digest = next(iter(invariant_digests.values()))

    tasks: List[Dict[str, Any]] = []
    for weight in BOUNDARY_WEIGHTS:
        task_id = variant_id(weight)
        run_dir = study_dir / "runs" / task_id
        config_path = run_dir / "config.json"
        atomic_write_json(config_path, variant_configs[task_id])
        task = {
            "task_id": task_id,
            "task_type": "train_and_export_variant",
            "dataset": DATASET,
            "official_fold": FOLD,
            "training_seed": TRAINING_SEED,
            "decoder_boundary_loss": weight,
            "role": (
                "in_harness_reproduction_baseline"
                if weight == 0.1
                else "pre_registered_primary"
                if weight == PRIMARY_BOUNDARY_WEIGHT
                else "exploratory_treatment"
            ),
            "physical_gpu": PHYSICAL_GPU,
            "config_path": str(config_path),
            "run_dir": str(run_dir),
            "model_dir": str(run_dir / "training" / task_id),
            "final_checkpoint": str(run_dir / "training" / task_id / f"epoch-{FINAL_EPOCH}.model"),
            "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
            "inference_seeds": list(INFERENCE_SEEDS),
            "train_manifest": str(train_manifest),
            "test_manifest": str(test_manifest),
            "align_dir": str(align_dir),
            "training_invariant_digest": invariant_digest,
            "state_path": str(study_dir / "state" / f"{task_id}.json"),
            "log_path": str(study_dir / "logs" / f"{task_id}.log"),
        }
        task["expected_checkpoints"] = {
            str(epoch): str(checkpoint_path(task, epoch)) for epoch in CHECKPOINT_EPOCHS
        }
        tasks.append(task)

    provenance = source_provenance(diffact_root)
    metadata: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "immutable": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study_dir),
        "workspace_root": str(SCRIPT_DIR.parents[1]),
        "data_root": str(data_root),
        "diffact_root": str(diffact_root),
        "dataset": DATASET,
        "official_fold": FOLD,
        "training_seed": TRAINING_SEED,
        "physical_gpu": PHYSICAL_GPU,
        "gpu_policy": {
            "physical_gpu_ids": [PHYSICAL_GPU],
            "fail_if_busy": True,
            "automatic_fallback": False,
            "serial_queue": True,
            "cuda_visible_devices": str(PHYSICAL_GPU),
        },
        "boundary_weights": list(BOUNDARY_WEIGHTS),
        "baseline_boundary_weight": 0.1,
        "pre_registered_primary_boundary_weight": PRIMARY_BOUNDARY_WEIGHT,
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "inference_seeds": list(INFERENCE_SEEDS),
        "inference_seed_formula": (
            f"model_seed = inference_seed * {INFERENCE_SEED_STRIDE} + video_index"
        ),
        "official_train_manifest": str(train_manifest),
        "official_test_manifest": str(test_manifest),
        "official_train_manifest_sha256": file_sha256(train_manifest),
        "official_test_manifest_sha256": file_sha256(test_manifest),
        "official_train_case_count": len(train_cases),
        "official_test_case_count": len(test_cases),
        "official_train_split": str(train_split),
        "official_test_split": str(test_split),
        "official_train_split_sha256": file_sha256(train_split),
        "official_test_split_sha256": file_sha256(test_split),
        "baseline_config": str(baseline_config_path),
        "baseline_config_sha256": file_sha256(baseline_config_path),
        "locked_reference_config": str(locked_config_path),
        "locked_reference_config_sha256": file_sha256(locked_config_path),
        "locked_source_training_config": str(source_training_config_path),
        "locked_source_training_config_sha256": file_sha256(source_training_config_path),
        "locked_reference_export": str(locked_export),
        "locked_reference_export_summary": locked_export_summary,
        "locked_reference_export_complete_sha256": file_sha256(
            locked_export / "export_complete.json"
        ),
        "locked_reference_train_manifest": str(locked_train_manifest),
        "locked_reference_train_manifest_sha256": file_sha256(locked_train_manifest),
        "locked_reference_train_bundle": str(locked_train_bundle),
        "locked_reference_train_bundle_sha256": file_sha256(locked_train_bundle),
        "generated_train_manifest": str(train_manifest),
        "generated_train_manifest_sha256": file_sha256(train_manifest),
        "generated_diffact_train_bundle": str(generated_train_bundle),
        "generated_diffact_train_bundle_sha256": file_sha256(generated_train_bundle),
        "training_order_contract": {
            "source": "locked low-data frac_100 manifest and its actual DiffAct train bundle",
            "case_order": train_cases,
            "set_equals_official_fold1_train": True,
            "generated_manifest_byte_identical_to_locked_manifest": True,
            "generated_bundle_byte_identical_to_locked_bundle": True,
            "official_split_order_intentionally_not_used": True,
        },
        "training_invariant_payload": training_invariant_payload(
            variant_configs[variant_id(0.1)]
        ),
        "training_invariant_digest": invariant_digest,
        "experimental_field": "loss_weights.decoder_boundary_loss",
        "frozen_fields_statement": (
            "The four generated runtime configs have the same training-invariant digest; "
            "only loss_weights.decoder_boundary_loss is experimental. Output naming/path "
            "fields identify separate trajectories."
        ),
        "checkpoint_selection": "pre_specified_grid_no_test_selection",
        "test_selection_policy": (
            "Weight 0.5 is pre-registered primary. Official fold-1 test outputs are not "
            "used to select a weight or checkpoint."
        ),
        "baseline_reconciliation_gate": {
            "locked_export": str(locked_export),
            "checkpoint_epoch": FINAL_EPOCH,
            "inference_seed": 0,
            "streams": ["pre_purge", "post_purge"],
            "metric_absolute_tolerance_points": RECONCILIATION_METRIC_TOLERANCE,
            "frame_disagreement_tolerance_fraction": (
                RECONCILIATION_FRAME_DISAGREEMENT_TOLERANCE
            ),
            "on_failure": "stop_serial_queue_before_all_treatments",
        },
        "diagnostic_contract": {
            "streams": {
                "pre_purge": "argmax of raw decoder probability matrix",
                "post_purge": "official DiffAct purge-3 discrete prediction",
            },
            "boundary_f1_tolerances_frames": [5, 10],
            "boundary_match_max_distance_frames": BOUNDARY_MATCH_MAX_DISTANCE,
            "boundary_matching": ["class_agnostic", "transition_aware"],
            "short_false_segment_max_length": SHORT_SEGMENT_MAX_LENGTH,
            "standard_metrics": ["acc", "edit", "f1@10", "f1@25", "f1@50"],
        },
        "decision_rule": {
            "variant": PRIMARY_BOUNDARY_WEIGHT,
            "baseline": 0.1,
            "epoch": FINAL_EPOCH,
            "stream": "post_purge",
            "aggregation": "mean_over_inference_seeds",
            "requirements": {
                "delta_boundary_f1_class_agnostic@10": ">0",
                "delta_edit": ">0",
                "delta_f1@25": ">0",
                "delta_acc": ">=0",
                "delta_segment_count_ratio": "<=0.02",
                "delta_boundary_offset_class_agnostic_median_absolute": "<0",
            },
        },
        "second_training_seed_policy": (
            "Not part of v1. A seed-1 {0.1,0.5} follow-up requires a new immutable study."
        ),
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
    for index, task in enumerate(tasks):
        if index == 1:
            queue_lines.append(
                shell_command(
                    [
                        python,
                        "-u",
                        SCRIPT_DIR / "reconcile_baseline.py",
                        "--study-dir",
                        study_dir,
                    ]
                )
            )
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
    queue_path = study_dir / "queues" / f"gpu_{PHYSICAL_GPU}.sh"
    write_executable(queue_path, "\n".join(queue_lines) + "\n")

    session = f"gtea_bweight_f{FOLD}_s{TRAINING_SEED}_g{PHYSICAL_GPU}"
    tmux_log = study_dir / "logs" / "gpu3_serial_queue.tmux.log"
    tmux_command = f"{shlex.quote(str(queue_path))} 2>&1 | tee -a {shlex.quote(str(tmux_log))}"
    launch_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"session={shlex.quote(session)}",
        'if tmux has-session -t "$session" 2>/dev/null; then',
        '  echo "Session already exists: $session" >&2',
        "  exit 2",
        "fi",
        (
            f"busy=$(nvidia-smi -i {PHYSICAL_GPU} --query-compute-apps=pid,used_memory,"
            "process_name --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
        ),
        'if [[ -n "$busy" ]]; then',
        f"  echo 'Physical GPU {PHYSICAL_GPU} is occupied; refusing to launch:' >&2",
        '  echo "$busy" >&2',
        "  exit 3",
        "fi",
        (
            f"tmux new-session -d -s \"$session\" "
            f"{shlex.quote(tmux_command)}"
        ),
        f"echo 'Started {session} on physical GPU {PHYSICAL_GPU}'",
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
            "Immutable generated GTEA boundary-weight pilot.",
            f"Protocol: {PROTOCOL_VERSION}",
            "Do not edit configs, manifests, tasks, or source after generation.",
            "Generate a new versioned study for any change.",
            "prepare_study.py did not launch training.",
        ],
    )
    print(f"Prepared immutable study: {study_dir}")
    print(f"Physical GPU: {PHYSICAL_GPU} only (fail closed if busy)")
    print(f"Variants: {list(BOUNDARY_WEIGHTS)}; primary={PRIMARY_BOUNDARY_WEIGHT}")
    print(f"Checkpoints: {list(CHECKPOINT_EPOCHS)}")
    print(f"Inference seeds: {list(INFERENCE_SEEDS)}")
    print(f"Training invariant digest: {invariant_digest}")
    print(f"Source digest: {provenance['source_digest']}")
    print("Nothing was launched. Review generated metadata/configs before launch_tmux.sh.")


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
        "--locked-reference-export",
        type=Path,
        default=DEFAULT_LOCKED_REFERENCE_EXPORT,
    )
    parser.add_argument(
        "--locked-reference-train-manifest",
        type=Path,
        default=DEFAULT_LOCKED_REFERENCE_TRAIN_MANIFEST,
    )
    parser.add_argument(
        "--locked-reference-train-bundle",
        type=Path,
        default=DEFAULT_LOCKED_REFERENCE_TRAIN_BUNDLE,
    )
    args = parser.parse_args()
    build_study(args)


if __name__ == "__main__":
    main()
