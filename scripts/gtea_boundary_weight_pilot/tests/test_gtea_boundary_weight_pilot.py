from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from common import (
    BASELINE_BOUNDARY_WEIGHT,
    BOUNDARY_WEIGHTS,
    CHECKPOINT_EPOCHS,
    DEFAULT_DATA_ROOT,
    DEFAULT_DIFFACT_ROOT,
    DEFAULT_FOLD1_STUDY_DIR,
    DEFAULT_LOCKED_MANIFEST_ROOT,
    DEFAULT_LOCKED_REFERENCE_CONFIG,
    FOLDS,
    GPU_IDS,
    INFERENCE_SEEDS,
    NEW_TRAINING_FOLDS,
    PRIMARY_BOUNDARY_WEIGHT,
    TRAINING_SEEDS,
    build_variant_config,
    training_invariant_digest,
    variant_id,
)
from metrics import (
    Boundary,
    evaluate_stream,
    fixed_broke_ledger,
    ground_truth_indices,
    match_boundaries,
    short_false_segments,
)
from common import load_mapping, read_bundle
from prepare_study import build_study, validate_reference_config


def test_locked_reference_matches_pre_registered_contract():
    reference = json.loads(DEFAULT_LOCKED_REFERENCE_CONFIG.read_text())
    validate_reference_config(reference)
    assert reference["boundary_smooth"] == 1
    assert reference["soft_label"] == 1.4
    assert reference["postprocess"] == {"type": "purge", "value": 3}
    assert reference["loss_weights"]["decoder_boundary_loss"] == 0.1
    assert reference["loss_weights"]["encoder_boundary_loss"] == 0.0


def test_variants_share_training_invariant_and_change_only_boundary_weight(tmp_path: Path):
    reference = json.loads(DEFAULT_LOCKED_REFERENCE_CONFIG.read_text())
    configs = []
    for training_seed in TRAINING_SEEDS:
        for weight in BOUNDARY_WEIGHTS:
            config = build_variant_config(
                reference,
                study_dir=tmp_path,
                view_root=tmp_path / "view",
                train_manifest=tmp_path / "train.txt",
                test_manifest=tmp_path / "test.txt",
                fold=2,
                weight=weight,
                training_seed=training_seed,
            )
            configs.append(config)
            assert config["loss_weights"]["decoder_boundary_loss"] == weight
            assert config["random_seed"] == training_seed
            assert config["initialization_seed"] == training_seed
    assert len({training_invariant_digest(config) for config in configs}) == 1

    normalized = []
    for config in configs:
        value = deepcopy(config)
        value["naming"] = "<variant>"
        value["result_dir"] = "<variant>"
        value["pre_specified_final_checkpoint"] = "<variant>"
        value["gtea_boundary_weight_variant"] = "<experimental>"
        value["gtea_boundary_weight_training_seed"] = "<experimental>"
        value["gtea_boundary_weight_official_fold"] = "<experimental>"
        value["random_seed"] = "<experimental>"
        value["initialization_seed"] = "<experimental>"
        value["loss_weights"]["decoder_boundary_loss"] = "<experimental>"
        normalized.append(value)
    assert all(value == normalized[0] for value in normalized[1:])


def test_boundary_matching_is_one_to_one_and_transition_aware():
    ground_truth = [Boundary(10, 1, 2), Boundary(30, 2, 3)]
    predicted = [Boundary(12, 1, 2), Boundary(13, 9, 2), Boundary(27, 2, 3)]
    matches, true_positive, false_positive, false_negative = match_boundaries(
        ground_truth,
        predicted,
        max_distance=5,
        transition_aware=True,
    )
    assert [offset for _, _, offset in matches] == [2, -3]
    assert (true_positive, false_positive, false_negative) == (2, 1, 0)


def test_short_false_segments_and_fixed_broke_ledger():
    ground_truth = np.asarray([1, 1, 1, 1, 2, 2, 2, 2])
    prediction = np.asarray([1, 1, 3, 1, 2, 2, 2, 2])
    short_false, predicted_count = short_false_segments(ground_truth, prediction, None)
    assert short_false == 1
    assert predicted_count == 4

    variant = np.asarray([1, 1, 1, 1, 2, 2, 3, 2])
    ledger = fixed_broke_ledger(
        {"case": ground_truth}, {"case": prediction}, {"case": variant}
    )
    assert ledger["fixed_frames"] == 1
    assert ledger["broken_frames"] == 1
    assert ledger["net_fixed_minus_broken"] == 0


def test_perfect_actual_gtea_prediction_has_perfect_metrics_and_zero_offsets():
    cases = read_bundle(DEFAULT_DATA_ROOT / "gtea" / "splits" / "test.split1.bundle")
    label_to_index, _ = load_mapping(DEFAULT_DATA_ROOT / "gtea" / "mapping.txt")
    predictions = {
        case: ground_truth_indices(DEFAULT_DATA_ROOT, case, label_to_index) for case in cases
    }
    summary, per_case, matches, _ = evaluate_stream(
        data_root=DEFAULT_DATA_ROOT,
        case_ids=cases,
        predictions=predictions,
        stream="perfect",
    )
    assert summary["acc"] == 100.0
    assert summary["edit"] == 100.0
    assert summary["f1@25"] == 100.0
    assert summary["boundary_f1_class_agnostic@10"] == 100.0
    assert summary["boundary_offset_class_agnostic_median_absolute"] == 0.0
    assert summary["segment_count_ratio"] == 1.0
    assert len(per_case) == 7
    assert matches


def test_generated_study_is_non_launched_multifold_and_waiters_fail_closed(
    tmp_path: Path,
):
    study_dir = tmp_path / "study"
    args = argparse.Namespace(
        study_dir=study_dir,
        data_root=DEFAULT_DATA_ROOT,
        diffact_root=DEFAULT_DIFFACT_ROOT,
        fold1_study_dir=DEFAULT_FOLD1_STUDY_DIR,
        locked_reference_config=DEFAULT_LOCKED_REFERENCE_CONFIG,
        locked_manifest_root=DEFAULT_LOCKED_MANIFEST_ROOT,
    )
    build_study(args)
    metadata = json.loads((study_dir / "study_metadata.json").read_text())
    tasks = json.loads((study_dir / "tasks.json").read_text())["tasks"]
    assert metadata["pre_registered_primary_boundary_weight"] == PRIMARY_BOUNDARY_WEIGHT
    assert metadata["official_folds"] == list(FOLDS)
    assert metadata["new_training_folds"] == list(NEW_TRAINING_FOLDS)
    assert metadata["checkpoint_epochs"] == list(CHECKPOINT_EPOCHS)
    assert metadata["inference_seeds"] == list(INFERENCE_SEEDS)
    assert metadata["training_seeds"] == list(TRAINING_SEEDS)
    assert metadata["new_training_count"] == 12
    assert metadata["import_count"] == 9
    assert metadata["baseline_reconciliation_gate"] is None
    assert (
        metadata["decision_rule"]["six_requirements"][
            "delta_boundary_offset_class_agnostic_mean_absolute"
        ]
        == "<0"
    )
    assert metadata["decision_rule"]["primary_claim_additional_requirements"][
        "positive_delta_edit_folds"
    ] == ">=3 of 4"
    assert all(
        contract["generated_bundle_byte_identical_to_locked_bundle"]
        for contract in metadata["per_fold_training_order_contract"].values()
    )
    grid = [
        (task["official_fold"], task["training_seed"], task["decoder_boundary_loss"])
        for task in tasks
        if task["included_in_primary_grid"]
    ]
    assert set(grid) == {
        (fold, seed, weight)
        for fold in FOLDS
        for seed in TRAINING_SEEDS
        for weight in BOUNDARY_WEIGHTS
    }
    assert sum(task["execution_mode"] == "train" for task in tasks) == 12
    assert sum(task["execution_mode"] == "imported" for task in tasks) == 9
    assert set(task["physical_gpu"] for task in tasks if task["execution_mode"] == "train") == set(
        GPU_IDS
    )
    assert not (
        study_dir / "state" / f"{variant_id(2, BASELINE_BOUNDARY_WEIGHT, 1)}.json"
    ).exists()

    queued_task_ids = []
    for gpu in GPU_IDS:
        queue = (study_dir / "queues" / f"gpu_{gpu}.sh").read_text()
        assert f"CUDA_VISIBLE_DEVICES={gpu}" in queue
        assert "reconcile_baseline.py" not in queue
        for task in tasks:
            if task["execution_mode"] == "train" and task["task_id"] in queue:
                queued_task_ids.append(task["task_id"])
        waiter = (study_dir / "waiters" / f"gpu_{gpu}_wait_then_run.sh").read_text()
        assert "free_checks < 2" in waiter
        assert "nvidia-smi" in waiter
        assert "refusing to infer availability" in waiter
    assert sorted(queued_task_ids) == sorted(metadata["training_tasks"])

    launcher = (study_dir / "launch_tmux.sh").read_text()
    assert "mkdir -p" in launcher
    for gpu in GPU_IDS:
        assert f"gtea_bweight_multifold_g{gpu}" in launcher

    configs = [json.loads(Path(task["config_path"]).read_text()) for task in tasks]
    assert len({training_invariant_digest(config) for config in configs}) == 1
    for task, config in zip(tasks, configs):
        assert Path(config["result_dir"]) / task["task_id"] == Path(
            task["model_dir"]
        )
        assert Path(config["pre_specified_final_checkpoint"]) == Path(
            task["model_dir"]
        ) / f"epoch-{CHECKPOINT_EPOCHS[-1]}.model"
        if task["execution_mode"] == "imported":
            import_payload = json.loads(Path(task["import_manifest_path"]).read_text())
            assert len(import_payload["checkpoint_sha256"]) == len(CHECKPOINT_EPOCHS)
            assert len(import_payload["exports"]) == len(CHECKPOINT_EPOCHS) * len(
                INFERENCE_SEEDS
            )
    for fold in FOLDS:
        locked_manifest = (
            DEFAULT_LOCKED_MANIFEST_ROOT
            / f"fold_{fold}"
            / "manifests"
            / "seed_0"
            / "train_cases_frac_100.txt"
        )
        locked_bundle = (
            DEFAULT_LOCKED_MANIFEST_ROOT
            / f"fold_{fold}"
            / "diffact_dataset_views"
            / "seed_0"
            / "frac_100"
            / "gtea"
            / "splits"
            / "train.split1.bundle"
        )
        assert (
            study_dir / "manifests" / f"fold_{fold}" / "train_cases_frac_100.txt"
        ).read_bytes() == locked_manifest.read_bytes()
        assert (
            study_dir
            / "diffact_dataset_views"
            / f"fold_{fold}"
            / "gtea"
            / "splits"
            / f"train.split{fold}.bundle"
        ).read_bytes() == locked_bundle.read_bytes()
