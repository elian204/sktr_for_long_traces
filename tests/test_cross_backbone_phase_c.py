from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cross_backbone_error_audit"
sys.path.insert(0, str(SCRIPT))

from phase_c_common import (  # noqa: E402
    PHASE_C_SPEC,
    PRIMARY_BUCKETS,
    SOURCE_FILES,
    STALE_PHASE_A_CACHE,
    assert_provenance_path,
    exact_one_sided_sign_p,
    holm_adjust,
)
from run_phase_c_audit import align_analysis_timeline, source_disclosure  # noqa: E402
from materialize_phase_c_asformer import load_model_class  # noqa: E402
from phase_c_taxonomy import (  # noqa: E402
    DFG,
    aggregate_cases,
    analyze_case,
    boundary_mask,
    candidate_ranks,
    discover_dfg,
    fragmentation_mask,
    generation_hypothesis_test,
    histogram_quantile,
    illegal_order_mask,
    long_substitution_mask,
    wrong_label_span_mask,
)


def probability_for(prediction: np.ndarray, classes: int = 5) -> np.ndarray:
    result = np.full((classes, len(prediction)), 0.01, dtype=np.float32)
    result[prediction, np.arange(len(prediction))] = 0.96
    result /= result.sum(axis=0, keepdims=True)
    return result


def test_taxonomy_is_exclusive_and_exhaustive() -> None:
    gt = np.asarray([0] * 40 + [1] * 40 + [2] * 40)
    prediction = gt.copy()
    prediction[10:15] = 3  # internal fragmentation
    prediction[36:44] = np.asarray([0] * 4 + [1] * 4)  # mostly correct boundary context
    prediction[80:100] = 4  # illegal destination segment
    dfg = DFG(frozenset({0}), frozenset({2}), frozenset({(0, 1), (1, 2)}))
    row = analyze_case(gt=gt, prediction=prediction, probability=probability_for(prediction), dfg=dfg, background=set())
    assert row["n_errors"] == sum(row[f"{bucket}_frames"] for bucket in PRIMARY_BUCKETS)
    assert row["fragmentation_frames"] == 5
    assert row["illegal_order_frames"] > 0


def test_boundary_offset_requires_adjacent_gt_label() -> None:
    gt = np.asarray([0] * 20 + [1] * 20)
    adjacent = gt.copy()
    adjacent[17:20] = 1
    unrelated = gt.copy()
    unrelated[17:20] = 2
    assert boundary_mask(gt, adjacent, 5).sum() == 3
    assert boundary_mask(gt, unrelated, 5).sum() == 0


def test_fragmentation_is_internal_short_island_only() -> None:
    gt = np.asarray([1] * 50)
    prediction = gt.copy()
    prediction[20:25] = 2
    assert fragmentation_mask(gt, prediction).sum() == 5
    prediction[10:40] = 2
    assert fragmentation_mask(gt, prediction).sum() == 0


def test_illegal_order_marks_destination_and_reports_start_end_separately() -> None:
    prediction = np.asarray([0] * 5 + [2] * 7)
    dfg = DFG(frozenset({0}), frozenset({1}), frozenset({(0, 1)}))
    mask, counts = illegal_order_mask(prediction, dfg)
    assert not mask[:5].any()
    assert mask[5:].all()
    assert counts["illegal_internal_transitions"] == 1
    assert counts["illegal_end_events"] == 1


def test_dfg_discovery_is_trace_collapsed_and_fold_pure_by_construction() -> None:
    dfg = discover_dfg([[0, 0, 1, 1, 2], [0, 2, 2]])
    assert dfg.starts == frozenset({0})
    assert dfg.ends == frozenset({2})
    assert dfg.edges == frozenset({(0, 1), (1, 2), (0, 2)})


def test_candidate_rank_uses_best_rank_under_ties() -> None:
    probability = np.asarray([[0.5, 0.2], [0.5, 0.7], [0.0, 0.1]])
    gt = np.asarray([0, 2])
    assert candidate_ranks(probability, gt).tolist() == [1, 3]


def test_wrong_label_span_requires_strict_gt_majority() -> None:
    gt = np.asarray([1, 1, 1, 2, 2, 2])
    prediction = np.asarray([3, 3, 3, 3, 3, 3])
    assert wrong_label_span_mask(gt, prediction).sum() == 0
    gt = np.asarray([1, 1, 1, 1, 2, 2])
    assert wrong_label_span_mask(gt, prediction).sum() == 6


def test_long_substitution_thresholds_are_locked() -> None:
    gt = np.zeros(120, dtype=int)
    prediction = np.ones(120, dtype=int)
    assert long_substitution_mask(gt, prediction).all()
    prediction[:20] = 2
    assert not long_substitution_mask(gt, prediction).any()


def test_histogram_quantile_matches_numpy_linear_quantile() -> None:
    values = np.asarray([1, 1, 2, 4, 4, 4, 8])
    histogram = {1: 2, 2: 1, 4: 3, 8: 1}
    assert histogram_quantile(histogram, 0.5) == pytest.approx(float(np.quantile(values, 0.5)))
    assert histogram_quantile(histogram, 0.9) == pytest.approx(float(np.quantile(values, 0.9)))


def test_aggregate_reports_frame_weighted_and_video_macro_side_by_side() -> None:
    gt_a = np.asarray([0] * 20 + [1] * 20)
    pred_a = gt_a.copy()
    pred_a[:10] = 2
    gt_b = np.asarray([0] * 10 + [1] * 10)
    pred_b = gt_b.copy()
    dfg = DFG(frozenset({0}), frozenset({1}), frozenset({(0, 1)}))
    rows = []
    for case_id, gt, prediction in (("a", gt_a, pred_a), ("b", gt_b, pred_b)):
        rows.append({"dataset": "x", "case_id": case_id, **analyze_case(gt=gt, prediction=prediction, probability=probability_for(prediction), dfg=dfg, background=set())})
    aggregate = aggregate_cases(pd.DataFrame(rows), ["dataset"])
    assert set(aggregate["aggregation"]) == {"frame_weighted", "per_video_macro"}
    weighted = aggregate[aggregate["aggregation"] == "frame_weighted"].iloc[0]
    assert weighted["n_errors"] == 10
    assert weighted["error_share_case_count"] == 1


def test_sign_test_and_holm_are_exact() -> None:
    assert exact_one_sided_sign_p(13, 13) == pytest.approx(1 / 8192)
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
    assert adjusted == {"a": pytest.approx(0.03), "b": pytest.approx(0.06), "c": pytest.approx(0.20)}


def test_generation_hypothesis_uses_thirteen_paired_fold_rows() -> None:
    rows = []
    folds = [("gtea", fold) for fold in range(1, 5)] + [("50salads", fold) for fold in range(1, 6)] + [("breakfast", fold) for fold in range(1, 5)]
    for dataset, fold in folds:
        for backbone, value in (("mstcn2", 3.0), ("asformer", 2.0), ("diffact", 1.0)):
            rows.append({"analysis_role": "primary", "aggregation": "frame_weighted", "dataset": dataset, "fold": fold, "backbone": backbone, "fragmentation_frames_per_minute": value, "illegal_order_frames_per_minute": value, "legal_substitution_share": 4.0 - value})
    result = generation_hypothesis_test(pd.DataFrame(rows))
    assert set(result["non_tie_folds"]) == {13}
    assert result["overall_pre_registered_support"].all()


def test_stale_phase_a_cache_is_forbidden_even_via_symlink(tmp_path: Path) -> None:
    link = tmp_path / "legacy"
    link.symlink_to(STALE_PHASE_A_CACHE, target_is_directory=True)
    with pytest.raises(RuntimeError, match="Forbidden stale"):
        assert_provenance_path(link / "x.npy", [tmp_path, STALE_PHASE_A_CACHE.parent])


def test_phase_c_spec_freezes_sources_aggregations_and_sensitivities() -> None:
    assert PHASE_C_SPEC["aggregation"]["analysis_fps"] == 15.0
    assert PHASE_C_SPEC["exclusive_error_taxonomy"]["precedence"] == list(PRIMARY_BUCKETS)
    assert PHASE_C_SPEC["descriptive_sensitivity"]["breakfast"]["arms"] == ["selected", "epoch100", "epoch30"]
    assert PHASE_C_SPEC["descriptive_sensitivity"]["full_train_ep100_robustness"]["datasets"] == ["gtea", "50salads"]


def test_phase_c_source_contract_is_complete() -> None:
    assert all((ROOT / relative).is_file() for relative in SOURCE_FILES)
    assert "scripts/cross_backbone_error_audit/phase_c_input_guard.py" in SOURCE_FILES


def test_source_disclosure_distinguishes_primary_and_descriptive_arms() -> None:
    assert "held-out" in source_disclosure("mstcn2", "selected", "primary")
    assert "full-train" in source_disclosure(
        "mstcn2", "full_train_epoch100", "secondary_full_train_epoch100"
    )
    assert "descriptive only" in source_disclosure("asformer", "epoch30", "sensitivity")


def test_asformer_loader_resolves_hash_locked_sibling_import(tmp_path: Path) -> None:
    (tmp_path / "eval.py").write_text("SENTINEL = 7\n")
    (tmp_path / "model.py").write_text(
        "from eval import SENTINEL\n"
        "class MyTransformer:\n"
        "    sibling_value = SENTINEL\n"
    )
    loaded = load_model_class(tmp_path / "model.py")
    assert loaded.sibling_value == 7


def test_native_sample_rate_export_expands_to_common_full_timeline() -> None:
    probability = np.asarray([[0.8, 0.2, 0.7], [0.2, 0.8, 0.3]])
    prediction = np.asarray([0, 1, 0])
    expanded_probability, expanded_prediction, factor = align_analysis_timeline(
        probability, prediction, target_frames=5, sample_rate=2
    )
    assert factor == 2
    assert expanded_probability.shape == (2, 5)
    assert expanded_prediction.tolist() == [0, 0, 1, 1, 0]
    assert np.allclose(expanded_probability[:, 0], expanded_probability[:, 1])


def test_full_resolution_export_is_not_resampled() -> None:
    probability = np.asarray([[0.8, 0.2], [0.2, 0.8]])
    prediction = np.asarray([0, 1])
    output_probability, output_prediction, factor = align_analysis_timeline(
        probability, prediction, target_frames=2, sample_rate=2
    )
    assert factor == 1
    assert output_probability is probability
    assert output_prediction is prediction


def test_unrecognized_export_timeline_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="Unsupported native timeline"):
        align_analysis_timeline(
            np.ones((2, 4)) / 2, np.zeros(4, dtype=int), target_frames=10, sample_rate=2
        )
