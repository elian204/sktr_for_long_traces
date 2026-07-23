from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analyze_selector import (
    CaseData,
    build_route_petri,
    exact_prefix_costs,
    long_substitution_mask,
    make_scale_decision,
    make_shape_decision,
    segment_candidate_pool,
    segment_probability_shape_features,
    select_spans_at_budget,
    validated_feature_sets,
    validated_repair_corpus_schema,
    variant_features,
)
from common import (
    BASE_NUMERIC_FEATURES,
    CaseInfo,
    FORBIDDEN_PRIMARY_FEATURE_PREFIXES,
    REPAIR_CANDIDATE_FIELDS,
    REPAIR_CANDIDATE_TOP_K,
    SHAPE_FEATURES,
    atomic_write_json,
    create_alignment_dir,
    file_sha256,
    fold_summary,
    load_case_infos,
    make_subject_disjoint_inner_folds,
    official_splits,
    parse_case,
    repair_candidate_columns,
)


DATA_ROOT = Path("/home/dsi/eli-bogdanov/data/data")


def test_all_official_breakfast_folds_partition_the_dataset_by_subject():
    expected_sizes = {
        1: (1460, 252),
        2: (1261, 451),
        3: (1279, 433),
        4: (1136, 576),
    }
    outer_tests = []
    for outer_fold, expected in expected_sizes.items():
        train, test = official_splits(DATA_ROOT, outer_fold)
        assert (len(train), len(test)) == expected
        assert not set(train) & set(test)
        assert len(set(train) | set(test)) == 1712
        train_people = {parse_case(case, 1).participant for case in train}
        test_people = {parse_case(case, 1).participant for case in test}
        assert len(train_people) == 39
        assert len(test_people) == 13
        assert not train_people & test_people
        outer_tests.extend(test)
    assert len(outer_tests) == len(set(outer_tests)) == 1712


def test_actual_breakfast_inner_split_is_subject_disjoint_balanced_and_complete():
    train, _ = official_splits(DATA_ROOT, 1)
    infos = load_case_infos(DATA_ROOT, train)
    folds = make_subject_disjoint_inner_folds(infos)
    observed = [case for cases in folds.values() for case in cases]
    assert len(observed) == len(set(observed)) == len(train) == 1460
    assert set(observed) == set(train)
    summaries = [fold_summary(infos, folds[fold]) for fold in sorted(folds)]
    people = [set(summary["participants"]) for summary in summaries]
    assert all(len(group) == 13 for group in people)
    assert not people[0] & people[1]
    assert not people[0] & people[2]
    assert not people[1] & people[2]
    frames = [summary["frame_count"] for summary in summaries]
    cases = [summary["case_count"] for summary in summaries]
    assert max(frames) / min(frames) < 1.01
    assert max(cases) - min(cases) <= 4
    for task in summaries[0]["task_case_counts"]:
        counts = [summary["task_case_counts"][task] for summary in summaries]
        assert max(counts) - min(counts) <= 2


def test_small_subject_split_is_deterministic():
    infos = [
        CaseInfo(f"P{person:02d}_cam01_P{person:02d}_tea", f"P{person:02d}", "tea", "cam", 10 + person)
        for person in range(1, 7)
    ]
    first = make_subject_disjoint_inner_folds(infos, n_folds=3, seed=4)
    second = make_subject_disjoint_inner_folds(infos, n_folds=3, seed=4)
    assert first == second


def test_case_parser_keeps_task_and_normalizes_camera():
    info = parse_case("P03_webcam02_P03_friedegg.txt", 123)
    assert info.participant == "P03"
    assert info.camera == "webcam"
    assert info.task == "friedegg"
    assert info.n_frames == 123


def test_alignment_dir_uses_exporter_contract(tmp_path: Path):
    cases = ["P03_cam01_P03_cereals", "P03_cam01_P03_coffee"]
    align = tmp_path / "align"
    create_alignment_dir(align, DATA_ROOT, cases)
    assert (align / "video_index_map.txt").read_text().splitlines() == [
        "0\tP03_cam01_P03_cereals",
        "1\tP03_cam01_P03_coffee",
    ]
    rows = (align / "ground_truth.csv").read_text().splitlines()
    assert rows[0] == "case:concept:name,concept:name"
    assert rows[1].split(",")[0] == "0"
    assert rows[-1].split(",")[0] == "1"


def test_exact_prefix_costs_accept_valid_prefix_and_charge_illegal_event(tmp_path: Path):
    runtime = build_route_petri("toy", [["a", "b"]], tmp_path / "toy.pnml")
    valid, valid_states = exact_prefix_costs(["a", "b"], runtime)
    assert valid.tolist() == [0.0, 0.0, 0.0]
    assert valid_states > 0
    invalid, _ = exact_prefix_costs(["a", "x", "b"], runtime)
    assert invalid.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_long_substitution_definition():
    gt = np.asarray(["a"] * 120 + ["b"] * 20)
    pred = np.asarray(["c"] * 110 + ["d"] * 10 + ["b"] * 20)
    mask = long_substitution_mask(gt, pred, min_len=100, homogeneity=0.90)
    assert mask[:120].all()
    assert not mask[120:].any()


def test_span_budget_uses_one_centered_cutoff_without_ground_truth():
    length = 20
    case = CaseData(
        outer_fold=1,
        scope="outer_test",
        mode="official",
        inner_fold=None,
        case_id="case",
        source_case_id="0",
        participant="P01",
        task="tea",
        gt=np.asarray(["a"] * length),
        pred=np.asarray(["b"] * length),
        long_substitution=np.zeros(length, dtype=bool),
        segment_ids=np.asarray([0] * 12 + [1] * 8),
    )
    segments = pd.DataFrame(
        [
            {"segment_id": 0, "scope": "outer_test", "mode": "official", "inner_fold": None, "case_id": "case", "segment_index": 0, "start": 0, "end": 12, "base_score": 0.9},
            {"segment_id": 1, "scope": "outer_test", "mode": "official", "inner_fold": None, "case_id": "case", "segment_index": 1, "start": 12, "end": 20, "base_score": 0.1},
        ]
    )
    masks = select_spans_at_budget([case], segments, "base_score", 0.25)
    selected = np.flatnonzero(masks[case.key])
    assert selected.tolist() == [3, 4, 5, 6, 7]


def test_variant_schema_keeps_process_features_out_of_base():
    features = variant_features(["task__tea", "camera__webcam"])
    assert set(features) == {"base", "base_plus_metadata", "base_plus_shape"}
    assert features["base"] == list(BASE_NUMERIC_FEATURES)
    assert not any(
        column.startswith(FORBIDDEN_PRIMARY_FEATURE_PREFIXES)
        for column in features["base"]
    )
    assert features["base_plus_metadata"] == [
        *BASE_NUMERIC_FEATURES,
        "task__tea",
        "camera__webcam",
    ]
    assert features["base_plus_shape"] == [
        *BASE_NUMERIC_FEATURES,
        *SHAPE_FEATURES,
    ]


def test_candidate_pool_and_probability_shape_features_are_deterministic():
    probabilities = np.asarray(
        [
            [0.90, 0.80, 0.70, 0.60, 0.50],
            [0.08, 0.10, 0.20, 0.30, 0.40],
            [0.02, 0.10, 0.10, 0.10, 0.10],
        ],
        dtype=float,
    )
    raw_argmax = probabilities.argmax(axis=0)
    candidates, order = segment_candidate_pool(
        probabilities, 0, 5, {0: "a", 1: "b", 2: "c"}, top_k=3
    )
    assert order.tolist() == [0, 1, 2]
    assert candidates["candidate_rank_1_label"] == "a"
    assert candidates["candidate_rank_2_label"] == "b"
    assert candidates["candidate_rank_3_label"] == "c"
    assert candidates["candidate_rank_1_mean_probability"] == pytest.approx(0.70)

    shape = segment_probability_shape_features(
        probabilities, raw_argmax, 0, 5
    )
    assert list(shape) == list(SHAPE_FEATURES)
    assert shape["confidence_slope"] == pytest.approx(-0.40)
    assert shape["edge_vs_core_margin"] == pytest.approx(-0.04)
    assert shape["flicker_rate"] == 0.0
    assert shape["runner_up_gap"] == pytest.approx(0.484)
    assert shape["runner_up_consistency"] == 1.0


def test_runtime_feature_assertion_fails_closed_on_metadata_in_primary(tmp_path: Path):
    config_path = tmp_path / "selector_config.json"
    config = {
        "variants": {
            "base": {"feature_columns": list(BASE_NUMERIC_FEATURES)},
            "base_plus_metadata": {},
            "base_plus_shape": {
                "base_feature_columns": list(BASE_NUMERIC_FEATURES),
                "additional_feature_columns": list(SHAPE_FEATURES),
            },
        },
        "forbidden_variants": ["base_plus_dfg", "base_plus_prefix_petri"],
    }
    atomic_write_json(config_path, config)
    metadata = {
        "selector_config": str(config_path),
        "selector_config_sha256": file_sha256(config_path),
    }
    features = validated_feature_sets(
        tmp_path, metadata, ["task__tea", "camera__webcam"]
    )
    assert features["base"] == list(BASE_NUMERIC_FEATURES)

    config["variants"]["base"]["feature_columns"].append("task__tea")
    atomic_write_json(config_path, config)
    metadata["selector_config_sha256"] = file_sha256(config_path)
    with pytest.raises(ValueError, match="BASE_NUMERIC_FEATURES exactly"):
        validated_feature_sets(
            tmp_path, metadata, ["task__tea", "camera__webcam"]
        )


def test_repair_corpus_schema_is_hash_locked_and_exact(tmp_path: Path):
    path = tmp_path / "repair_corpus_schema.json"
    schema = {
        "candidate_pool": {
            "top_k": REPAIR_CANDIDATE_TOP_K,
            "fields_per_rank": list(REPAIR_CANDIDATE_FIELDS),
            "columns": list(repair_candidate_columns()),
        },
        "primary_feature_columns": list(BASE_NUMERIC_FEATURES),
        "exploratory_shape_feature_columns": list(SHAPE_FEATURES),
    }
    atomic_write_json(path, schema)
    metadata = {
        "repair_corpus_schema": str(path),
        "repair_corpus_schema_sha256": file_sha256(path),
    }
    assert validated_repair_corpus_schema(tmp_path, metadata) == schema

    schema["candidate_pool"]["columns"].pop()
    atomic_write_json(path, schema)
    metadata["repair_corpus_schema_sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="candidate columns"):
        validated_repair_corpus_schema(tmp_path, metadata)


def test_scale_decision_uses_the_locked_four_fold_and_pooled_thresholds():
    rows = []
    for fold, recall in enumerate((18.0, 19.0, 20.0, 21.0), start=1):
        rows.append(
            {
                "scope": "official_outer_test",
                "mode": "official",
                "fold": f"outer_{fold}",
                "variant": "base",
                "requested_budget": 0.05,
                "error_recall_pct": recall,
                "error_precision_pct": 90.0,
                "long_substitution_recall_pct": 10.0,
                "oracle_acc_gain_pp": 4.0,
                "oracle_f1_at_25_gain_pp": 3.0,
            }
        )
    rows.append(
        {
            "scope": "pooled_outer_test",
            "mode": "official",
            "fold": "pooled",
            "variant": "base",
            "requested_budget": 0.05,
            "error_recall_pct": 19.5,
            "error_precision_pct": 90.0,
            "long_substitution_recall_pct": 10.0,
            "oracle_acc_gain_pp": 4.0,
            "oracle_f1_at_25_gain_pp": 3.0,
        }
    )
    metadata = {
        "success_rule": {
            "all_outer_folds": {
                "minimum_error_precision_pct": 85.0,
                "minimum_error_recall_pct": 15.0,
                "maximum_recall_range_across_folds_pp": 8.0,
            },
            "pooled": {
                "minimum_error_precision_pct": 85.0,
                "minimum_error_recall_pct": 18.0,
            },
        }
    }
    decision = make_scale_decision(pd.DataFrame(rows), metadata)
    assert decision["green_light_repair_head"]
    assert decision["recall_range_across_folds_pp"] == 3.0

    failing = pd.DataFrame(rows)
    failing.loc[failing["fold"] == "outer_4", "error_recall_pct"] = 13.0
    decision = make_scale_decision(failing, metadata)
    assert not decision["green_light_repair_head"]
    assert not decision["checks"]["all_folds_recall_at_least_15pct"]


def test_shape_decision_uses_locked_pooled_and_cross_fold_rule():
    rows = []
    for fold in [*(f"outer_{index}" for index in range(1, 5)), "pooled"]:
        base_recall = 20.0
        shape_gain = -0.1 if fold == "outer_4" else 0.6
        for variant, gain in (("base", 0.0), ("base_plus_shape", shape_gain)):
            rows.append(
                {
                    "scope": (
                        "pooled_outer_test"
                        if fold == "pooled"
                        else "official_outer_test"
                    ),
                    "mode": "official",
                    "fold": fold,
                    "variant": variant,
                    "requested_budget": 0.05,
                    "error_recall_pct": base_recall + gain,
                    "error_precision_pct": 90.0 + gain,
                    "long_substitution_recall_pct": 10.0 + gain,
                    "oracle_acc_gain_pp": 4.0 + gain,
                    "oracle_f1_at_25_gain_pp": 3.0 + gain,
                }
            )
    metadata = {
        "shape_variant_comparison_rule": {
            "minimum_pooled_error_recall_gain_pp": 0.5,
            "minimum_outer_folds_with_positive_error_recall_gain": 3,
            "minimum_pooled_oracle_acc_gain_difference_pp": 0.0,
            "minimum_pooled_oracle_f1_at_25_gain_difference_pp": 0.0,
        }
    }
    decision = make_shape_decision(pd.DataFrame(rows), metadata)
    assert decision["retain_shape_features"]
    assert decision["positive_outer_fold_count"] == 3
    assert decision["deltas"]["pooled"]["error_recall_gain_pp"] == pytest.approx(
        0.6
    )

    failing = pd.DataFrame(rows)
    failing.loc[
        (failing["fold"] == "pooled")
        & (failing["variant"] == "base_plus_shape"),
        "error_recall_pct",
    ] = 20.2
    decision = make_shape_decision(failing, metadata)
    assert not decision["retain_shape_features"]
