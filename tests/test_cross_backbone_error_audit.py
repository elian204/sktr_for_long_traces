from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "cross_backbone_error_audit"
sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    PHASE_B_PLANNING_ESTIMATE,
    PUBLISHED_METRICS,
    normalize_bundle_case,
    parse_mapping,
)
from metrics import aggregate_standard, edit_score, f_counts, per_video_metrics, segments  # noqa: E402
from run_phase_a import reconciliation_status  # noqa: E402


def test_expected_paper_matrix_has_six_rows_and_five_metrics() -> None:
    assert len(PUBLISHED_METRICS) == 6
    assert all(set(row) == {"acc", "edit", "f1@10", "f1@25", "f1@50"} for row in PUBLISHED_METRICS.values())


def test_bundle_normalization() -> None:
    assert normalize_bundle_case("path/to/video.txt") == "video"
    assert normalize_bundle_case("video") == "video"


def test_parse_mapping_requires_contiguous_ids(tmp_path: Path) -> None:
    path = tmp_path / "mapping.txt"
    path.write_text("0 background\n1 cut\n", encoding="utf-8")
    id_to_name, name_to_id = parse_mapping(path)
    assert id_to_name == {0: "background", 1: "cut"}
    assert name_to_id == {"background": 0, "cut": 1}


def test_segments_excludes_only_explicit_background() -> None:
    assert segments([0, 0, 1, 1, 0], {0}) == [(2, 4, 1)]
    assert segments([0, 0, 1], set()) == [(0, 2, 0), (2, 3, 1)]


def test_perfect_metrics_are_100() -> None:
    truth = np.asarray([0, 0, 1, 1, 2, 2])
    row = per_video_metrics(truth, truth)
    assert all(value == 100.0 for value in row.values())
    aggregate = aggregate_standard([(truth, truth)])
    assert all(value == 100.0 for value in aggregate.values())


def test_edit_and_overlap_detect_fragmentation() -> None:
    truth = [1, 1, 1, 2, 2, 2]
    prediction = [1, 1, 3, 1, 2, 2]
    assert edit_score(prediction, truth) < 100.0
    tp, fp, fn = f_counts(prediction, truth, 0.5)
    assert fp > 0 or fn > 0


def test_reconciliation_bands_are_pre_registered() -> None:
    assert reconciliation_status({metric: 0.5 for metric in ("acc", "edit", "f1@10", "f1@25", "f1@50")}) == "PASS"
    assert reconciliation_status({"acc": 2.5, "edit": 0.5, "f1@10": 0.5, "f1@25": 0.5, "f1@50": 0.5}) == "PASS_WITH_NOTES"
    assert reconciliation_status({metric: 4.0 for metric in ("acc", "edit", "f1@10", "f1@25", "f1@50")}) == "FAIL"


def test_fold_mean_differs_from_pooled_when_fold_sizes_differ() -> None:
    fold_one = np.asarray([1, 1])
    fold_two = np.asarray([0] * 8)
    mean_fold_acc = np.mean(
        [
            aggregate_standard([(fold_one, fold_one)])["acc"],
            aggregate_standard([(fold_two, np.ones_like(fold_two))])["acc"],
        ]
    )
    pooled_acc = aggregate_standard(
        [(fold_one, fold_one), (fold_two, np.ones_like(fold_two))]
    )["acc"]
    assert mean_fold_acc == 50.0
    assert pooled_acc == 20.0


def test_phase_b_plan_is_explicitly_non_scientific_and_uses_data1() -> None:
    assert PHASE_B_PLANNING_ESTIMATE["failed_cell_count"] == 17
    assert PHASE_B_PLANNING_ESTIMATE["planning_only_refresh_before_launch"] is True
    assert PHASE_B_PLANNING_ESTIMATE["required_artifact_root"].startswith("/data1/")
