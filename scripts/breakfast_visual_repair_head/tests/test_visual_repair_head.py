from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from common import (
    CLASSIFIER_MAX_ITER,
    PRIMARY_GATES,
    canonical_digest,
    canonical_feature_view,
    evaluate_primary_gates,
    feature_orientation,
    file_sha256,
    pool_span_mean_std,
)
from run_repair_head import (
    fit_length_weighted_classifier,
    proposal_for_segment,
    select_budget_rows,
    validate_and_build_cases,
    verify_inputs,
)


def test_i3d_orientation_keys_only_on_unique_2048_axis() -> None:
    features_by_time = np.arange(2048 * 3, dtype=np.float32).reshape(2048, 3)
    time_by_features = features_by_time.T
    assert feature_orientation(features_by_time.shape) == ("features_by_time", 3)
    assert feature_orientation(time_by_features.shape) == ("time_by_features", 3)
    np.testing.assert_array_equal(
        canonical_feature_view(features_by_time, 3),
        canonical_feature_view(time_by_features, 3),
    )
    with pytest.raises(ValueError, match="exactly one 2048 axis"):
        feature_orientation((2048, 2048))
    with pytest.raises(ValueError, match="exactly one 2048 axis"):
        feature_orientation((1024, 10))
    with pytest.raises(ValueError, match="does not match"):
        canonical_feature_view(features_by_time, 4)


def test_mean_std_pooling_uses_full_span_and_population_std() -> None:
    array = np.zeros((2048, 4), dtype=np.float32)
    array[0] = [1, 3, 5, 7]
    pooled = pool_span_mean_std(array, 1, 3, 4)
    assert pooled.shape == (4096,)
    assert pooled[0] == pytest.approx(4.0)
    assert pooled[2048] == pytest.approx(1.0)
    assert np.count_nonzero(pooled) == 2


def _segment_row(
    segment_id: int,
    start: int,
    end: int,
    score: float,
    case_id: str = "case",
) -> dict:
    return {
        "segment_id": segment_id,
        "outer_fold": 1,
        "scope": "outer_test",
        "mode": "official",
        "inner_fold": np.nan,
        "case_id": case_id,
        "segment_index": segment_id,
        "start": start,
        "end": end,
        "length": end - start,
        "base_score": score,
    }


def test_budget_selection_is_exact_and_uses_one_centered_cutoff(monkeypatch) -> None:
    # Make 5% of 200 frames equal ten selected frames. The top span has length 6;
    # the second contributes a centered four-frame cutoff from a length-8 span.
    rows = pd.DataFrame(
        [
            _segment_row(1, 0, 6, 0.9),
            _segment_row(2, 6, 14, 0.8),
            _segment_row(3, 14, 200, 0.1),
        ]
    )
    selected = select_budget_rows(rows, 200)
    assert selected["selected_frames"].sum() == 10
    assert selected["is_partial_cutoff"].sum() == 1
    cutoff = selected[selected["is_partial_cutoff"]].iloc[0]
    assert (cutoff.selected_start, cutoff.selected_end) == (8, 12)


def test_primary_gate_requires_all_six_checks() -> None:
    passed = evaluate_primary_gates(
        {"acc": 0.7, "edit": 0.1, "f1@25": 0.2},
        [0.1, 0.2, 0.3, -0.1],
        [-5.0, 1.0, 2.0],
        [-5, 10, 10],
        PRIMARY_GATES,
    )
    assert passed["positive_acc_fold_count"] == 3
    assert passed["largest_single_video_gain_fraction"] == pytest.approx(2 / 3)
    # The contribution limit should actually make the constructed example fail.
    assert not passed["checks"]["single_video_contribution_bounded"]
    assert not passed["pass"]


def test_primary_gate_passes_balanced_gain() -> None:
    result = evaluate_primary_gates(
        {"acc": 0.7, "edit": 0.1, "f1@25": 0.2},
        [0.1, 0.2, 0.3, -0.1],
        [-5.0, 1.0, 2.0, 0.5],
        [2, 2, 2, 2],
        PRIMARY_GATES,
    )
    assert result["pass"]
    assert all(result["checks"].values())


def test_proposal_variants_use_unrenormalized_head_probabilities() -> None:
    classifier = LogisticRegression().fit(
        np.asarray([[0.0], [1.0], [2.0], [3.0]]), np.asarray([0, 1, 2, 2])
    )
    probabilities = np.asarray([0.1, 0.2, 0.7])
    segment = {
        "candidate_rank_1_class_id": 0,
        "candidate_rank_2_class_id": 1,
        "candidate_rank_3_class_id": 7,
        "candidate_rank_4_class_id": 8,
        "candidate_rank_5_class_id": 9,
    }
    free_label, free_p, _ = proposal_for_segment(
        segment, classifier, probabilities, "free_choice"
    )
    restricted_label, restricted_p, _ = proposal_for_segment(
        segment, classifier, probabilities, "top5_restricted"
    )
    assert (free_label, free_p) == (2, pytest.approx(0.7))
    assert (restricted_label, restricted_p) == (1, pytest.approx(0.2))


def test_classifier_matches_prototype_iteration_contract() -> None:
    classifier = fit_length_weighted_classifier(
        np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        np.asarray([0, 0, 1, 1]),
        np.asarray([1.0, 2.0, 3.0, 4.0]),
    )
    assert classifier.C == 1.0
    assert classifier.max_iter == CLASSIFIER_MAX_ITER == 2000
    assert classifier.solver == "lbfgs"


def test_convergence_warning_aborts_instead_of_becoming_an_audit_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def warning_fit(self, features, targets, sample_weight=None):
        del features, targets, sample_weight
        import warnings

        warnings.warn("synthetic non-convergence", ConvergenceWarning)
        return self

    monkeypatch.setattr(LogisticRegression, "fit", warning_fit)
    with pytest.raises(RuntimeError, match="ConvergenceWarning"):
        fit_length_weighted_classifier(
            np.asarray([[0.0], [1.0]]),
            np.asarray([0, 1]),
            np.asarray([1.0, 1.0]),
        )


def test_outer_partition_reconstructs_official_prediction_and_validates_majority() -> None:
    rows = pd.DataFrame(
        [
            {
                "outer_fold": 1,
                "case_id": "x",
                "segment_index": 0,
                "start": 0,
                "end": 2,
                "predicted_label": 1,
                "correct_label": 0,
            },
            {
                "outer_fold": 1,
                "case_id": "x",
                "segment_index": 1,
                "start": 2,
                "end": 5,
                "predicted_label": 2,
                "correct_label": 2,
            },
        ]
    )
    cases = validate_and_build_cases(rows, {"x": np.asarray([0, 0, 2, 2, 1])})
    np.testing.assert_array_equal(cases[0].baseline, [1, 1, 2, 2, 2])
    broken = rows.copy()
    broken.loc[1, "correct_label"] = 0
    with pytest.raises(ValueError, match="Frozen majority target is invalid"):
        validate_and_build_cases(broken, {"x": np.asarray([0, 0, 2, 2, 1])})


def test_input_verification_hashes_every_declared_file(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed.txt"
    gt = tmp_path / "gt.txt"
    feature = tmp_path / "feature.npy"
    fixed.write_text("fixed\n")
    gt.write_text("SIL\n")
    np.save(feature, np.zeros((2048, 1), dtype=np.float32))

    def entry(path: Path, **extra):
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            **extra,
        }

    payload = {
        "protocol_version": "test",
        "fixed_artifacts": [entry(fixed)],
        "ground_truth": [entry(gt, case_id="x", time_frames=1)],
        "features": [
            entry(
                feature,
                case_id="x",
                shape=[2048, 1],
                dtype="float32",
                orientation="features_by_time",
                time_frames=1,
            )
        ],
        "counts": {},
    }
    payload["manifest_digest"] = canonical_digest(payload)
    result = verify_inputs(payload, workers=2)
    assert result["files_verified"] == 3
    fixed.write_text("tampered\n")
    with pytest.raises(RuntimeError, match="Input hash mismatch"):
        verify_inputs(payload, workers=2)
