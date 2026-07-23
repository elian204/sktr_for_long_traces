#!/usr/bin/env python3
"""Generate the immutable, non-launched GTEA boundary-weight replication v2."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from common import (
    BASELINE_BOUNDARY_WEIGHT,
    BOUNDARY_MATCH_MAX_DISTANCE,
    BOUNDARY_WEIGHTS,
    CHECKPOINT_EPOCHS,
    DATASET,
    DEFAULT_DATA_ROOT,
    DEFAULT_DIFFACT_ROOT,
    DEFAULT_LOCKED_REFERENCE_CONFIG,
    DEFAULT_LOCKED_REFERENCE_TRAIN_BUNDLE,
    DEFAULT_LOCKED_REFERENCE_TRAIN_MANIFEST,
    DEFAULT_STUDY_DIR,
    DEFAULT_V1_STUDY_DIR,
    EXPECTED_NUM_EPOCHS,
    EXPECTED_TEST_CASES,
    EXPECTED_TRAIN_CASES,
    FINAL_EPOCH,
    FOLD,
    IMPORTED_CURVE_WEIGHTS,
    INFERENCE_SEEDS,
    INFERENCE_SEED_STRIDE,
    PHYSICAL_GPU,
    PRIMARY_BOUNDARY_WEIGHT,
    PROTOCOL_VERSION,
    SCRIPT_DIR,
    SHORT_SEGMENT_MAX_LENGTH,
    TRAINING_SEEDS,
    V1_IMPORTED_KEYS,
    atomic_write_json,
    build_variant_config,
    canonical_digest,
    checkpoint_path,
    create_alignment_dir,
    create_dataset_view,
    export_dir,
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


def validate_reference_config(config: Mapping[str, Any]) -> None:
    expectations = {
        "dataset_name": DATASET,
        "boundary_smooth": 1,
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
        mismatches["postprocess"] = {
            "expected": {"type": "purge", "value": 3},
            "actual": config.get("postprocess"),
        }
    if float(config["loss_weights"]["decoder_boundary_loss"]) != BASELINE_BOUNDARY_WEIGHT:
        mismatches["loss_weights.decoder_boundary_loss"] = {
            "expected": BASELINE_BOUNDARY_WEIGHT,
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
    if any(epoch % int(config["log_freq"]) for epoch in CHECKPOINT_EPOCHS):
        mismatches["checkpoint_grid"] = {
            "expected": list(CHECKPOINT_EPOCHS),
            "actual_log_freq": config["log_freq"],
        }
    if mismatches:
        raise ValueError(f"Locked D100 reference config violates v2 assumptions: {mismatches}")


def source_v1_run(v1_study_dir: Path, training_seed: int, weight: float) -> Path:
    if (training_seed, weight) not in V1_IMPORTED_KEYS:
        raise ValueError(f"Variant is not approved for v1 import: {(training_seed, weight)}")
    return v1_study_dir / "runs" / variant_id(weight, training_seed)


def record_import(
    *,
    study_dir: Path,
    task: Mapping[str, Any],
    v1_study_dir: Path,
    expected_cases: Sequence[str],
    expected_invariant_digest: str,
) -> Dict[str, Any]:
    source_run = Path(task["artifact_run_dir"])
    source_config = source_run / "config.json"
    source_complete = source_run / "task_complete.json"
    if not source_config.is_file() or not source_complete.is_file():
        raise FileNotFoundError(
            f"Completed v1 import source is missing config/completion: {source_run}"
        )
    source_config_payload = load_json(source_config)
    if training_invariant_digest(source_config_payload) != expected_invariant_digest:
        raise ValueError(f"Imported v1 config violates the v2 frozen protocol: {source_config}")
    if int(source_config_payload["random_seed"]) != int(task["training_seed"]):
        raise ValueError(f"Imported training seed mismatch: {source_config}")
    if float(source_config_payload["loss_weights"]["decoder_boundary_loss"]) != float(
        task["decoder_boundary_loss"]
    ):
        raise ValueError(f"Imported boundary weight mismatch: {source_config}")
    source_complete_payload = load_json(source_complete)
    if source_complete_payload.get("status") != "complete":
        raise ValueError(f"Imported v1 task is not complete: {source_complete}")

    checkpoint_hashes: Dict[str, str] = {}
    exports: List[Dict[str, Any]] = []
    for epoch in task["checkpoint_epochs"]:
        checkpoint = checkpoint_path(task, int(epoch))
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise FileNotFoundError(checkpoint)
        checkpoint_hashes[str(epoch)] = file_sha256(checkpoint)
        for inference_seed in task["inference_seeds"]:
            output_dir = export_dir(task, int(epoch), int(inference_seed))
            summary = verify_export(output_dir, expected_cases)
            exports.append(
                {
                    "checkpoint_epoch": int(epoch),
                    "inference_seed": int(inference_seed),
                    "output_dir": str(output_dir),
                    "export_complete_sha256": file_sha256(
                        output_dir / "export_complete.json"
                    ),
                    "artifact_digest": canonical_digest(summary["artifact_sha256"]),
                    "case_count": summary["case_count"],
                    "frame_count": summary["frame_count"],
                }
            )
    recorded_checkpoint_hashes = source_complete_payload.get("checkpoint_sha256")
    if recorded_checkpoint_hashes != checkpoint_hashes:
        raise ValueError(
            f"Imported checkpoint hashes differ from v1 completion marker: {source_run}"
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "imported_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": task["task_id"],
        "training_seed": task["training_seed"],
        "decoder_boundary_loss": task["decoder_boundary_loss"],
        "source_study_dir": str(v1_study_dir),
        "source_study_metadata_sha256": file_sha256(
            v1_study_dir / "study_metadata.json"
        ),
        "source_run_dir": str(source_run),
        "source_config_sha256": file_sha256(source_config),
        "source_task_complete_sha256": file_sha256(source_complete),
        "checkpoint_sha256": checkpoint_hashes,
        "exports": exports,
        "hash_verification": "all checkpoints and all exported inference artifacts",
    }
    import_path = Path(task["import_manifest_path"])
    atomic_write_json(import_path, payload)
    return payload


def build_study(args: argparse.Namespace) -> None:
    study_dir = args.study_dir.resolve()
    data_root = args.data_root.resolve()
    diffact_root = args.diffact_root.resolve()
    v1_study_dir = args.v1_study_dir.resolve()
    locked_config_path = args.locked_reference_config.resolve()
    locked_train_manifest = args.locked_reference_train_manifest.resolve()
    locked_train_bundle = args.locked_reference_train_bundle.resolve()
    if study_dir.exists():
        raise FileExistsError(
            f"Study directory already exists: {study_dir}. Choose a new immutable version."
        )
    for path in (
        diffact_root / "configs" / "GTEA-Trained-S1.json",
        locked_config_path,
        locked_train_manifest,
        locked_train_bundle,
        v1_study_dir / "study_metadata.json",
        v1_study_dir / "protocol_amendment_1.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    reference_config = load_json(locked_config_path)
    validate_reference_config(reference_config)
    source_training_config_path = Path(
        reference_config["epoch_scarcity_source_config"]
    ).resolve()
    source_training_config = load_json(source_training_config_path)
    recorded_training_manifest = Path(
        source_training_config["training_subset_manifest"]
    ).resolve()
    recorded_train_bundle = (
        Path(source_training_config["root_data_dir"])
        / DATASET
        / "splits"
        / "train.split1.bundle"
    ).resolve()
    if recorded_training_manifest != locked_train_manifest:
        raise ValueError("Locked D100 source config points to another training manifest")
    if recorded_train_bundle != locked_train_bundle:
        raise ValueError("Locked D100 source config points to another train bundle")
    if recorded_train_bundle.read_bytes() != locked_train_bundle.read_bytes():
        raise ValueError("Explicit and source-config D100 train bundles differ")

    v1_metadata = load_json(v1_study_dir / "study_metadata.json")
    if v1_metadata.get("protocol_version") != "gtea-boundary-weight-pilot-v1":
        raise ValueError("v1 import source has an unexpected protocol")
    amendment = load_json(v1_study_dir / "protocol_amendment_1.json")
    if amendment.get("decision", "").startswith("proceed with treatment") is False:
        raise ValueError("v1 protocol amendment does not authorize treatment artifacts")

    split_root = data_root / DATASET / "splits"
    train_split = split_root / f"train.split{FOLD}.bundle"
    test_split = split_root / f"test.split{FOLD}.bundle"
    official_train_cases = read_bundle(train_split)
    test_cases = read_bundle(test_split)
    train_cases = read_bundle(locked_train_manifest)
    if train_cases != read_bundle(locked_train_bundle):
        raise ValueError("Locked frac-100 manifest and train bundle order differ")
    if len(train_cases) != EXPECTED_TRAIN_CASES or len(test_cases) != EXPECTED_TEST_CASES:
        raise ValueError(
            f"Unexpected fold-1 counts: train={len(train_cases)}, test={len(test_cases)}"
        )
    if set(train_cases) != set(official_train_cases):
        raise ValueError("Locked train manifest is not set-equal to official fold-1 train")
    if set(train_cases).intersection(test_cases):
        raise ValueError("Official train/test overlap detected")

    study_dir.mkdir(parents=True)
    (study_dir / "logs").mkdir()
    train_manifest = study_dir / "manifests" / "train_cases_frac_100.txt"
    test_manifest = study_dir / "manifests" / "official_test_cases.txt"
    train_manifest.parent.mkdir(parents=True)
    shutil.copy2(locked_train_manifest, train_manifest)
    write_bundle(test_manifest, test_cases)
    view_root = create_dataset_view(
        study_dir,
        data_root,
        train_cases,
        test_cases,
        locked_train_bundle=locked_train_bundle,
    )
    generated_train_bundle = (
        view_root / DATASET / "splits" / f"train.split{FOLD}.bundle"
    )
    if train_manifest.read_bytes() != locked_train_manifest.read_bytes():
        raise ValueError("Generated train manifest is not byte-identical to D100")
    if generated_train_bundle.read_bytes() != locked_train_bundle.read_bytes():
        raise ValueError("Generated train bundle is not byte-identical to D100")
    align_dir = create_alignment_dir(study_dir, data_root, test_cases)

    specs = [
        (training_seed, weight, weight in BOUNDARY_WEIGHTS)
        for training_seed in TRAINING_SEEDS
        for weight in BOUNDARY_WEIGHTS
    ]
    specs.extend((0, weight, False) for weight in IMPORTED_CURVE_WEIGHTS)
    variant_configs: Dict[str, Dict[str, Any]] = {}
    for training_seed, weight, _ in specs:
        task_id = variant_id(weight, training_seed)
        variant_configs[task_id] = build_variant_config(
            reference_config,
            study_dir=study_dir,
            view_root=view_root,
            train_manifest=train_manifest,
            test_manifest=test_manifest,
            weight=weight,
            training_seed=training_seed,
        )
    invariant_digests = {
        task_id: training_invariant_digest(config)
        for task_id, config in variant_configs.items()
    }
    if len(set(invariant_digests.values())) != 1:
        raise ValueError(f"Variants differ beyond weight/seed/path: {invariant_digests}")
    invariant_digest = next(iter(invariant_digests.values()))

    tasks: List[Dict[str, Any]] = []
    for training_seed, weight, included_in_primary_grid in specs:
        task_id = variant_id(weight, training_seed)
        imported = (training_seed, weight) in V1_IMPORTED_KEYS
        run_dir = study_dir / "runs" / task_id
        config_path = run_dir / "config.json"
        atomic_write_json(config_path, variant_configs[task_id])
        source_run = (
            source_v1_run(v1_study_dir, training_seed, weight)
            if imported
            else run_dir
        )
        task: Dict[str, Any] = {
            "task_id": task_id,
            "task_type": (
                "hash_verified_v1_import" if imported else "train_and_export_variant"
            ),
            "execution_mode": "imported" if imported else "train",
            "dataset": DATASET,
            "official_fold": FOLD,
            "training_seed": training_seed,
            "decoder_boundary_loss": weight,
            "included_in_primary_grid": included_in_primary_grid,
            "role": (
                "paired_baseline"
                if weight == BASELINE_BOUNDARY_WEIGHT
                else "pre_registered_primary"
                if weight == PRIMARY_BOUNDARY_WEIGHT
                else "seed0_curve_import"
                if weight in IMPORTED_CURVE_WEIGHTS
                else "exploratory_dose_response"
            ),
            "physical_gpu": PHYSICAL_GPU,
            "config_path": str(config_path),
            "run_dir": str(run_dir),
            "model_dir": str(run_dir / "training" / task_id),
            "artifact_run_dir": str(source_run),
            "artifact_model_dir": str(source_run / "training" / task_id),
            "final_checkpoint": str(
                source_run / "training" / task_id / f"epoch-{FINAL_EPOCH}.model"
            ),
            "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
            "inference_seeds": list(INFERENCE_SEEDS),
            "train_manifest": str(train_manifest),
            "test_manifest": str(test_manifest),
            "align_dir": str(align_dir),
            "training_invariant_digest": invariant_digest,
            "state_path": str(study_dir / "state" / f"{task_id}.json"),
            "log_path": str(study_dir / "logs" / f"{task_id}.log"),
            "import_manifest_path": (
                str(study_dir / "imports" / task_id / "import_complete.json")
                if imported
                else None
            ),
        }
        task["expected_checkpoints"] = {
            str(epoch): str(checkpoint_path(task, epoch)) for epoch in CHECKPOINT_EPOCHS
        }
        tasks.append(task)

    imports: List[Dict[str, Any]] = []
    for task in tasks:
        if task["execution_mode"] == "imported":
            imports.append(
                record_import(
                    study_dir=study_dir,
                    task=task,
                    v1_study_dir=v1_study_dir,
                    expected_cases=test_cases,
                    expected_invariant_digest=invariant_digest,
                )
            )

    provenance = source_provenance(diffact_root)
    training_tasks = [task for task in tasks if task["execution_mode"] == "train"]
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
        "training_seeds": list(TRAINING_SEEDS),
        "physical_gpu": PHYSICAL_GPU,
        "gpu_policy": {
            "physical_gpu_ids": [PHYSICAL_GPU],
            "fail_if_busy": True,
            "automatic_fallback": False,
            "serial_queue": True,
            "cuda_visible_devices": str(PHYSICAL_GPU),
        },
        "boundary_weights": list(BOUNDARY_WEIGHTS),
        "imported_curve_weights": list(IMPORTED_CURVE_WEIGHTS),
        "baseline_boundary_weight": BASELINE_BOUNDARY_WEIGHT,
        "pre_registered_primary_boundary_weight": PRIMARY_BOUNDARY_WEIGHT,
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "inference_seeds": list(INFERENCE_SEEDS),
        "inference_seed_formula": (
            f"model_seed = inference_seed * {INFERENCE_SEED_STRIDE} + video_index"
        ),
        "new_training_count": len(training_tasks),
        "import_count": len(imports),
        "v1_import_study": str(v1_study_dir),
        "v1_import_study_metadata_sha256": file_sha256(
            v1_study_dir / "study_metadata.json"
        ),
        "v1_protocol_amendment": str(v1_study_dir / "protocol_amendment_1.json"),
        "v1_protocol_amendment_sha256": file_sha256(
            v1_study_dir / "protocol_amendment_1.json"
        ),
        "import_contract": {
            "keys": [
                {"training_seed": seed, "decoder_boundary_loss": weight}
                for seed, weight in V1_IMPORTED_KEYS
            ],
            "verification": "all checkpoint and export artifact hashes",
            "retrain_imported_variants": False,
        },
        "generated_train_manifest": str(train_manifest),
        "generated_train_manifest_sha256": file_sha256(train_manifest),
        "generated_diffact_train_bundle": str(generated_train_bundle),
        "generated_diffact_train_bundle_sha256": file_sha256(generated_train_bundle),
        "official_test_manifest": str(test_manifest),
        "official_test_manifest_sha256": file_sha256(test_manifest),
        "official_train_split": str(train_split),
        "official_test_split": str(test_split),
        "official_train_split_sha256": file_sha256(train_split),
        "official_test_split_sha256": file_sha256(test_split),
        "locked_reference_config": str(locked_config_path),
        "locked_reference_config_sha256": file_sha256(locked_config_path),
        "locked_source_training_config": str(source_training_config_path),
        "locked_source_training_config_sha256": file_sha256(
            source_training_config_path
        ),
        "locked_reference_train_manifest": str(locked_train_manifest),
        "locked_reference_train_manifest_sha256": file_sha256(
            locked_train_manifest
        ),
        "locked_reference_train_bundle": str(locked_train_bundle),
        "locked_reference_train_bundle_sha256": file_sha256(locked_train_bundle),
        "training_order_contract": {
            "source": "locked low-data frac_100 manifest and actual DiffAct train bundle",
            "case_order": train_cases,
            "set_equals_official_fold1_train": True,
            "generated_manifest_byte_identical_to_locked_manifest": True,
            "generated_bundle_byte_identical_to_locked_bundle": True,
            "official_split_order_intentionally_not_used": True,
        },
        "training_invariant_payload": training_invariant_payload(
            variant_configs[variant_id(BASELINE_BOUNDARY_WEIGHT, 0)]
        ),
        "training_invariant_digest": invariant_digest,
        "experimental_fields": [
            "loss_weights.decoder_boundary_loss",
            "random_seed",
            "initialization_seed",
        ],
        "frozen_fields_statement": (
            "All runtime configs share the same training-invariant digest. Only boundary "
            "weight, declared training seed, and output naming/path fields may differ."
        ),
        "checkpoint_selection": "pre_specified_grid_no_test_selection",
        "test_selection_policy": (
            "Weight 1.0 versus 0.1 is pre-registered primary across both training seeds. "
            "Weights 0.75 and 1.5 are exploratory; test outputs select nothing."
        ),
        "baseline_reconciliation_gate": None,
        "reconciliation_policy": (
            "No bitwise reconciliation gate. v1 protocol_amendment_1 established benign "
            "run-to-run nondeterminism; v2 reports cross-seed noise per weight."
        ),
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
            "cross_seed_noise": [
                "metric_delta_seed1_minus_seed0",
                "frame_disagreement_fraction",
            ],
        },
        "decision_rule": {
            "variant": PRIMARY_BOUNDARY_WEIGHT,
            "baseline": BASELINE_BOUNDARY_WEIGHT,
            "epoch": FINAL_EPOCH,
            "stream": "post_purge",
            "aggregation": "mean_over_2_training_seeds_x_3_inference_seeds",
            "requirements": {
                "delta_boundary_f1_class_agnostic@10": ">0",
                "delta_edit": ">0",
                "delta_f1@25": ">0",
                "delta_acc": ">=0",
                "delta_segment_count_ratio": "<=0.02",
                "delta_boundary_offset_class_agnostic_mean_absolute": "<0",
            },
            "offset_substitution": (
                "v1 median-absolute-offset was tie-degenerate at exactly 0.0; v2 "
                "pre-registers mean absolute offset before launch."
            ),
            "on_pass": "implement_class_specific_gaussian_onset_head",
            "on_fail": (
                "close_config_only_rung; decide onset-head go/no-go from combined evidence"
            ),
        },
        "source_provenance": provenance,
        "tasks": [task["task_id"] for task in tasks],
        "training_tasks": [task["task_id"] for task in training_tasks],
        "imported_tasks": [
            task["task_id"] for task in tasks if task["execution_mode"] == "imported"
        ],
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
    for task in training_tasks:
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

    session = f"gtea_bweight_f{FOLD}_v2_g{PHYSICAL_GPU}"
    tmux_log = study_dir / "logs" / "gpu3_serial_queue.tmux.log"
    tmux_command = f"{shlex.quote(str(queue_path))} 2>&1 | tee -a {shlex.quote(str(tmux_log))}"
    launch_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"session={shlex.quote(session)}",
        f"mkdir -p {shlex.quote(str(study_dir / 'logs'))}",
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
        f"tmux new-session -d -s \"$session\" {shlex.quote(tmux_command)}",
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
            "Immutable generated GTEA boundary-weight replication.",
            f"Protocol: {PROTOCOL_VERSION}",
            "Do not edit configs, manifests, tasks, imports, or source after generation.",
            "Generate a new versioned study for any change.",
            "prepare_study.py did not launch training.",
        ],
    )
    print(f"Prepared immutable study: {study_dir}")
    print(f"Physical GPU: {PHYSICAL_GPU} only (fail closed if busy)")
    print(
        f"Grid: weights={list(BOUNDARY_WEIGHTS)} x training_seeds={list(TRAINING_SEEDS)}"
    )
    print(f"Imports: {len(imports)}; net new trainings: {len(training_tasks)}")
    print(f"Primary: {PRIMARY_BOUNDARY_WEIGHT} versus {BASELINE_BOUNDARY_WEIGHT}")
    print(f"Training invariant digest: {invariant_digest}")
    print(f"Source digest: {provenance['source_digest']}")
    print("Nothing was launched. Review metadata/configs/import hashes first.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--diffact-root", type=Path, default=DEFAULT_DIFFACT_ROOT)
    parser.add_argument("--v1-study-dir", type=Path, default=DEFAULT_V1_STUDY_DIR)
    parser.add_argument(
        "--locked-reference-config",
        type=Path,
        default=DEFAULT_LOCKED_REFERENCE_CONFIG,
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
