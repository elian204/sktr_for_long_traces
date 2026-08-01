from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cross_backbone_error_audit"
sys.path.insert(0, str(SCRIPT))

from finalize_phase_b_training import reconciliation_status
from phase_b_training_common import (
    DATASETS,
    OFFICIAL_CONFIG,
    OFFICIAL_MSTCN2_HEAD,
    OFFICIAL_SOURCE_FILES,
    QUEUE_ASSIGNMENTS,
    canonical_digest,
    file_sha256,
    verify_manifest,
)


def test_phase_b_matrix_is_exactly_thirteen_unique_official_folds() -> None:
    cells = [cell for queue in QUEUE_ASSIGNMENTS.values() for cell in queue]
    expected = {
        (dataset, fold)
        for dataset, config in DATASETS.items()
        for fold in range(1, int(config["folds"]) + 1)
    }
    assert len(cells) == 13
    assert set(cells) == expected


def test_every_gpu_lane_has_three_or_four_tasks() -> None:
    assert set(QUEUE_ASSIGNMENTS) == {0, 1, 2, 3}
    assert sorted(len(queue) for queue in QUEUE_ASSIGNMENTS.values()) == [3, 3, 3, 4]


def test_official_training_config_is_frozen() -> None:
    assert OFFICIAL_CONFIG == {
        "num_epochs": 100,
        "features_dim": 2048,
        "batch_size": 1,
        "learning_rate": 0.0005,
        "num_f_maps": 64,
        "num_layers_PG": 11,
        "num_layers_R": 10,
        "num_R": 3,
        "seed": 1538574472,
    }


def test_official_source_contract_is_minimal_and_pinned() -> None:
    assert OFFICIAL_MSTCN2_HEAD == "f423a9e65f4ccb1cd7322eb9f94946a19e787993"
    assert set(OFFICIAL_SOURCE_FILES) == {"main.py", "model.py", "batch_gen.py", "eval.py", "train.sh"}


def test_manifest_verification_is_hash_strict(tmp_path: Path) -> None:
    payload = tmp_path / "input.bin"
    payload.write_bytes(b"locked")
    row = {
        "role": "input",
        "path": str(payload),
        "size_bytes": payload.stat().st_size,
        "sha256": file_sha256(payload),
    }
    manifest = {
        "files": [row],
        "file_count": 1,
        "manifest_digest": canonical_digest([{"role": "input", "sha256": row["sha256"]}]),
    }
    assert verify_manifest(manifest, full_hash=True)["input"] == payload
    payload.write_bytes(b"drift!")
    with pytest.raises(RuntimeError, match="hash drift"):
        verify_manifest(manifest, full_hash=True)


def test_reconciliation_thresholds_match_phase_a() -> None:
    assert reconciliation_status({key: 0.9 for key in ("acc", "edit", "f1@10", "f1@25", "f1@50")}) == "PASS"
    assert reconciliation_status({key: 1.4 for key in ("acc", "edit", "f1@10", "f1@25", "f1@50")}) == "PASS_WITH_NOTES"
    assert reconciliation_status({key: 3.1 for key in ("acc", "edit", "f1@10", "f1@25", "f1@50")}) == "FAIL"


def test_readme_keeps_phase_c_closed() -> None:
    readme = (SCRIPT / "README.md").read_text()
    assert "stays closed until" in readme
    assert "two consecutive free-GPU checks" in readme
