from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from common import OOF_HARM_FLOOR_PP, PRIMARY_BUDGET, canonical_digest
from core import (
    CaseData,
    _fit_logistic,
    _fit_mlp,
    apply_configuration,
    apply_isotonic,
    evaluate_configuration,
    primary_gate_decision,
    select_budget_rows,
    selection_sort_key,
)
from prepare_study import OOF_ROLES, SEALED_OUTER_ROLES, _derived_manifest
from run_outer_evaluation import _anonymize_ledger


def _segments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segment_id": 1,
                "outer_fold": 1,
                "inner_fold": 1,
                "scope": "oof_validation",
                "mode": "official",
                "case_id": "video",
                "segment_index": 0,
                "start": 0,
                "end": 4,
                "length": 4,
                "predicted_label": 0,
                "correct_label": 1,
                "correct_label_fraction": 1.0,
                "base_score": 0.9,
            },
            {
                "segment_id": 2,
                "outer_fold": 1,
                "inner_fold": 1,
                "scope": "oof_validation",
                "mode": "official",
                "case_id": "video",
                "segment_index": 1,
                "start": 4,
                "end": 10,
                "length": 6,
                "predicted_label": 2,
                "correct_label": 2,
                "correct_label_fraction": 1.0,
                "base_score": 0.8,
            },
        ]
    )


def _case() -> CaseData:
    return CaseData(
        1,
        1,
        "video",
        np.asarray([1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=np.int16),
        np.asarray([0, 0, 0, 0, 2, 2, 2, 2, 2, 2], dtype=np.int16),
    )


def _selection() -> pd.DataFrame:
    rows = select_budget_rows(_segments(), total_frames=10, budget=0.8)
    assert rows.selected_frames.sum() == 8
    return rows


def _probabilities() -> np.ndarray:
    values = np.zeros((2, 48), dtype=float)
    values[0, 0] = 0.05
    values[0, 1] = 0.90
    values[0, 2] = 0.05
    values[1, 0] = 0.05
    values[1, 1] = 0.20
    values[1, 2] = 0.75
    return values


def test_oof_manifest_excludes_every_outer_score_artifact() -> None:
    assert "selector_analysis/segment_scores.csv" not in OOF_ROLES
    assert "selector_analysis/baseline_metrics.csv" not in OOF_ROLES
    assert "selector_analysis/selector_budget_metrics.csv" not in OOF_ROLES
    assert "selector_analysis/segment_scores.csv" in SEALED_OUTER_ROLES


def test_derived_oof_manifest_contains_only_allowlisted_fixed_roles() -> None:
    roles = set(OOF_ROLES) | set(SEALED_OUTER_ROLES)
    upstream = {
        "manifest_digest": "upstream",
        "fixed_artifacts": [
            {"role": role, "path": f"/{index}", "bytes": 1, "sha256": "x"}
            for index, role in enumerate(sorted(roles))
        ],
        "ground_truth": [],
        "features": [],
    }
    derived = _derived_manifest(upstream, OOF_ROLES, phase="oof_screening")
    assert {row["role"] for row in derived["fixed_artifacts"]} == set(OOF_ROLES)
    clean = dict(derived)
    digest = clean.pop("manifest_digest")
    assert canonical_digest(clean) == digest


def test_budget_is_exact_and_has_at_most_one_centered_cutoff() -> None:
    selected = select_budget_rows(_segments(), total_frames=100, budget=PRIMARY_BUDGET)
    assert selected.selected_frames.sum() == 5
    assert selected.is_partial_budget_cutoff.sum() == 1
    cutoff = selected[selected.is_partial_budget_cutoff].iloc[0]
    assert (cutoff.selected_start, cutoff.selected_end) == (6, 7)


def test_incumbent_margin_rejects_insufficient_advantage() -> None:
    probabilities = _probabilities()
    probabilities[0, 0] = 0.40
    probabilities[0, 1] = 0.55
    predictions, ledger, _ = apply_configuration(
        [_case()],
        _segments(),
        probabilities,
        _selection(),
        threshold=0.5,
        rule_name="incumbent_margin",
        rule_parameter=0.2,
    )
    first = ledger[ledger.segment_id == 1].iloc[0]
    assert not first.accepted
    assert first.decision_reason == "below_incumbent_margin"
    np.testing.assert_array_equal(predictions[_case().key], _case().baseline)


def test_video_cap_prioritizes_confidence_and_center_clips() -> None:
    predictions, ledger, _ = apply_configuration(
        [_case()],
        _segments(),
        _probabilities(),
        _selection(),
        threshold=0.5,
        rule_name="video_cap",
        rule_parameter=20.0,
    )
    accepted = ledger[ledger.accepted]
    assert len(accepted) == 1
    row = accepted.iloc[0]
    assert row.segment_id == 1
    assert (row.repair_start, row.repair_end) == (1, 3)
    assert row.relabelled_frames == 2
    assert predictions[_case().key].tolist() == [0, 1, 1, 0, 2, 2, 2, 2, 2, 2]


def test_large_span_guard_raises_tau_only_for_large_spans() -> None:
    probabilities = _probabilities()
    probabilities[0, 1] = 0.60
    probabilities[0, 0] = 0.35
    probabilities[0, 2] = 0.05
    _, ledger, _ = apply_configuration(
        [_case()],
        _segments(),
        probabilities,
        _selection(),
        threshold=0.5,
        rule_name="large_span_guard",
        rule_parameter=20.0,
    )
    first = ledger[ledger.segment_id == 1].iloc[0]
    assert not first.accepted
    assert first.decision_reason == "large_span_below_higher_tau"


def test_logistic_and_mlp_convergence_warnings_are_hard_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def warning_fit(self, x, y, *args, **kwargs):
        del x, y, args, kwargs
        warnings.warn("no convergence", ConvergenceWarning)
        return self

    monkeypatch.setattr(LogisticRegression, "fit", warning_fit)
    with pytest.raises(RuntimeError, match="did not converge"):
        _fit_logistic(np.asarray([[0.0], [1.0]]), np.asarray([0, 1]), np.ones(2))
    monkeypatch.setattr(MLPClassifier, "fit", warning_fit)
    with pytest.raises(RuntimeError, match="did not converge"):
        _fit_mlp(np.asarray([[0.0], [1.0]]), np.asarray([0, 1]))


def test_harm_constraint_uses_strictly_below_minus_five() -> None:
    result, _, _, videos = evaluate_configuration(
        [_case()],
        _segments(),
        _probabilities(),
        _selection(),
        threshold=0.5,
        rule_name="video_cap",
        rule_parameter=20.0,
    )
    assert result["worst_video_delta_acc"] > OOF_HARM_FLOOR_PP
    assert result["harm_constraint_pass"]
    videos.loc[0, "delta_acc"] = -5.0
    assert (videos.delta_acc >= OOF_HARM_FLOOR_PP).all()


def test_anonymized_ledger_drops_raw_video_and_segment_keys() -> None:
    _, ledger, _ = apply_configuration(
        [_case()],
        _segments(),
        _probabilities(),
        _selection(),
        threshold=0.5,
        rule_name="none",
        rule_parameter=None,
    )
    result = _anonymize_ledger(ledger)
    assert "case_id" not in result
    assert "case_key" not in result
    assert "segment_id" not in result
    assert result.segment_key_sha256.str.len().eq(64).all()


def test_primary_gate_preserves_worst_video_and_concentration_guards() -> None:
    videos = pd.DataFrame(
        {
            "delta_acc": [-5.0, 1.0, 1.0, 1.0],
            "net_correct_frames": [2, 2, 2, 2],
        }
    )
    result = primary_gate_decision(
        {"acc": 0.6, "edit": 0.1, "f1@25": 0.1},
        [0.1, 0.2, 0.3, -0.1],
        videos,
    )
    assert result["pass"]
    videos.loc[0, "delta_acc"] = -5.01
    failed = primary_gate_decision(
        {"acc": 0.6, "edit": 0.1, "f1@25": 0.1},
        [0.1, 0.2, 0.3, -0.1],
        videos,
    )
    assert not failed["pass"]
    assert not failed["checks"]["no_video_acc_drop_over_limit"]


def test_selection_tie_order_is_deterministic() -> None:
    a = {
        "delta_acc": 1.0,
        "delta_f1@25": 1.0,
        "delta_edit": 0.0,
        "worst_video_delta_acc": -1.0,
        "rule_name": "video_cap",
        "rule_parameter": 1.0,
    }
    b = {**a, "rule_name": "incumbent_margin"}
    order = {"video_cap": 0, "incumbent_margin": 1}
    assert selection_sort_key(a, order) > selection_sort_key(b, order)
