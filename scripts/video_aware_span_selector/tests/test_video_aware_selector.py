from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analyze_selector import (
    CaseData,
    build_route_petri,
    exact_prefix_costs,
    long_substitution_mask,
    select_spans_at_budget,
    variant_features,
)
from common import (
    CaseInfo,
    create_alignment_dir,
    fold_summary,
    load_case_infos,
    make_subject_disjoint_inner_folds,
    official_splits,
    parse_case,
)


DATA_ROOT = Path("/home/dsi/eli-bogdanov/data/data")


def test_actual_breakfast_inner_split_is_subject_disjoint_balanced_and_complete():
    train, _ = official_splits(DATA_ROOT)
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
    features = variant_features(["task__tea"])
    assert "dfg_segment_severity" not in features["base"]
    assert "petri_prefix_cost_increment" not in features["base"]
    assert "dfg_segment_severity" in features["base_plus_dfg"]
    assert "petri_prefix_cost_increment" in features["base_plus_prefix_petri"]

