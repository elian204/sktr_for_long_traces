from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts" / "independent_acceptance_verifier"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_v1 import held_inner_rule  # noqa: E402
from train_v1_rotation import choose_threshold  # noqa: E402
from v1_common import (  # noqa: E402
    FEATURE_DIM,
    TOTAL_BINS,
    TemporalPairwiseVerifier,
    combine_sufficient,
    metrics_from_sufficient,
    temporal_span_view,
)


def test_temporal_span_view_shape_and_edge_padding() -> None:
    feature = np.tile(np.arange(10, dtype=np.float32), (FEATURE_DIM, 1))
    view = temporal_span_view(feature, 0, 2)
    assert view.shape == (TOTAL_BINS, FEATURE_DIM)
    assert np.isfinite(view).all()
    assert np.all(view[:16] == 0)


def test_model_has_no_confidence_input_and_returns_one_logit() -> None:
    model = TemporalPairwiseVerifier()
    value = model(
        torch.zeros(2, TOTAL_BINS, FEATURE_DIM),
        torch.tensor([1, 2]),
        torch.tensor([3, 4]),
    )
    assert value.shape == (2,)


def test_sufficient_statistics_combine_exactly() -> None:
    one = {
        "frames": 10,
        "correct": 8,
        "cases": 1,
        "edit_sum": 90.0,
        "f1_counts": {"10": [2, 1, 1], "25": [2, 1, 1], "50": [1, 2, 2]},
    }
    combined = combine_sufficient([one, one])
    metrics = metrics_from_sufficient(combined)
    assert combined["frames"] == 20
    assert metrics["acc"] == 80.0
    assert metrics["edit"] == 90.0


def test_held_inner_consistency_rule() -> None:
    assert held_inner_rule([0.2, 0.1, 0.3])
    assert held_inner_rule([0.2, 0.1, -0.05])
    assert not held_inner_rule([0.2, 0.1, -0.2])
    assert not held_inner_rule([0.2, -0.01, -0.02])


def test_threshold_choice_uses_guardrails_then_accuracy() -> None:
    rows = [
        {
            "threshold": 0.5,
            "delta_metrics": {"acc": 2.0, "edit": -1.0, "f1@25": 1.0},
            "worst_video_delta_acc": -2.0,
            "fixed_to_broken_ratio": 5.0,
        },
        {
            "threshold": 0.55,
            "delta_metrics": {"acc": 1.0, "edit": 0.1, "f1@25": 0.1},
            "worst_video_delta_acc": -2.0,
            "fixed_to_broken_ratio": 4.0,
        },
    ]
    threshold, feasible = choose_threshold(rows)
    assert feasible
    assert threshold == 0.55
