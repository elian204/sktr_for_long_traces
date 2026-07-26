from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from common import (
    BASELINE_ONSET_LOSS_WEIGHT,
    DEFAULT_DATA_ROOT,
    DEFAULT_DIFFACT_ROOT,
    DEFAULT_LOCKED_REFERENCE_CONFIG,
    DEFAULT_LOCKED_TRAIN_BUNDLE,
    DEFAULT_LOCKED_TRAIN_MANIFEST,
    ONSET_LOSS_WEIGHTS,
    PRIMARY_ONSET_LOSS_WEIGHT,
    build_variant_config,
    training_invariant_digest,
)
from prepare_study import build_study, validate_reference_config
from verify_launch_gate import main as verify_launch_gate_main


def test_class_specific_target_marks_starting_class_only():
    sys.path.insert(0, str(DEFAULT_DIFFACT_ROOT))
    from dataset import get_class_specific_onset_seq

    labels = np.asarray([2, 2, 5, 5, 2])
    target = get_class_specific_onset_seq(labels, 7, onset_smooth=1)
    assert target.shape == (7, 5)
    assert target[2, 0] == pytest.approx(1.0)
    assert target[5, 2] == pytest.approx(1.0)
    assert target[2, 4] == pytest.approx(1.0)
    assert np.argmax(target[5]) == 2
    assert np.all(target[[0, 1, 3, 4, 6]] == 0)


def test_onset_head_is_additive_and_class_specific():
    source = (DEFAULT_DIFFACT_ROOT / "model.py").read_text()
    assert "self.class_specific_onset_head = nn.Conv1d" in source
    assert "decoder_params['input_dim'], num_classes" in source
    assert "binary_cross_entropy_with_logits" in source
    assert "onset_element_weight = 1.0 +" in source
    assert "predict_class_specific_onsets" in source


def test_configs_change_only_the_onset_loss_knob(tmp_path: Path):
    reference = json.loads(DEFAULT_LOCKED_REFERENCE_CONFIG.read_text())
    validate_reference_config(reference)
    configs = [
        build_variant_config(
            reference,
            study_dir=tmp_path,
            view_root=tmp_path / "view",
            train_manifest=tmp_path / "train.txt",
            test_manifest=tmp_path / "test.txt",
            onset_weight=weight,
        )
        for weight in ONSET_LOSS_WEIGHTS
    ]
    assert len({training_invariant_digest(config) for config in configs}) == 1
    normalized = []
    for config in configs:
        value = deepcopy(config)
        value["naming"] = "<path>"
        value["result_dir"] = "<path>"
        value["pre_specified_final_checkpoint"] = "<path>"
        value["loss_weights"]["class_specific_onset_loss"] = "<knob>"
        normalized.append(value)
    assert all(value == normalized[0] for value in normalized[1:])
    assert all(
        config["loss_weights"]["decoder_boundary_loss"] == 1.0
        for config in configs
    )
    assert all(config["onset_bump_positive_weight"] == 1.0 for config in configs)


def test_generated_review_study_is_nonlaunched_and_hard_gated(tmp_path: Path):
    study_dir = tmp_path / "study"
    missing_decision = tmp_path / "missing_task1_decision.json"
    build_study(
        argparse.Namespace(
            study_dir=study_dir,
            data_root=DEFAULT_DATA_ROOT,
            diffact_root=DEFAULT_DIFFACT_ROOT,
            locked_reference_config=DEFAULT_LOCKED_REFERENCE_CONFIG,
            locked_train_manifest=DEFAULT_LOCKED_TRAIN_MANIFEST,
            locked_train_bundle=DEFAULT_LOCKED_TRAIN_BUNDLE,
            multifold_decision_path=missing_decision,
        )
    )
    metadata = json.loads((study_dir / "study_metadata.json").read_text())
    tasks = json.loads((study_dir / "tasks.json").read_text())["tasks"]
    assert metadata["launch_gated"]
    assert metadata["study_role"] == "screening_only"
    assert metadata["single_new_knob"] == "loss_weights.class_specific_onset_loss"
    assert metadata["pre_registered_primary_onset_loss_weight"] == (
        PRIMARY_ONSET_LOSS_WEIGHT
    )
    assert metadata["baseline_onset_loss_weight"] == BASELINE_ONSET_LOSS_WEIGHT
    assert metadata["class_imbalance_option"]["off_by_default"]
    replication = metadata["decision_rule"][
        "required_positive_screen_replication"
    ]
    assert replication["onset_loss_weights"] == [
        0.0,
        "best_screened_onset_weight",
    ]
    assert replication["training_seed"] == 1
    assert replication["required_before_any_claim_advances"]
    assert len(tasks) == len(ONSET_LOSS_WEIGHTS) == 4
    assert not (study_dir / "state").exists()
    launcher = (study_dir / "launch_tmux.sh").read_text()
    waiter = (study_dir / "waiters" / "gpu_3_wait_then_run.sh").read_text()
    assert "gtea_onset_head_f1_v1_g3" in launcher
    assert "verify_launch_gate.py" in waiter
    assert "free_checks < 2" in waiter
    assert "nvidia-smi" in waiter
    assert (
        study_dir / "diffact_dataset_view" / "gtea" / "splits" / "train.split1.bundle"
    ).read_bytes() == DEFAULT_LOCKED_TRAIN_BUNDLE.read_bytes()
    for task in tasks:
        config = json.loads(Path(task["config_path"]).read_text())
        assert config["loss_weights"]["class_specific_onset_loss"] == task[
            "class_specific_onset_loss_weight"
        ]

    old_argv = sys.argv
    try:
        sys.argv = ["verify_launch_gate.py", "--study-dir", str(study_dir)]
        with pytest.raises(FileNotFoundError):
            verify_launch_gate_main()
    finally:
        sys.argv = old_argv
