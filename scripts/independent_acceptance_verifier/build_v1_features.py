#!/usr/bin/env python3
"""Build the hash-verified 80-bin temporal feature cache for V1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from v1_common import (
    FEATURE_DIM,
    TOTAL_BINS,
    atomic_write_json,
    canonical_digest,
    file_sha256,
    load_json,
    temporal_span_view,
    verify_flat_manifest,
    verify_nested_manifest,
    verify_source_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    return parser.parse_args()


def execute(study_dir: Path) -> None:
    metadata = load_json(study_dir / "study_metadata.json")
    config = load_json(study_dir / "study_config.json")
    manifest = load_json(study_dir / "input_manifest.json")
    verify_source_provenance(metadata["source_provenance"])
    paths = verify_flat_manifest(manifest)
    if not config["feature_build_allowed"] or config["outer_test_open_allowed"]:
        raise RuntimeError("V1 feature build is not authorized in this study")
    nested = load_json(paths["features/nested_manifest"])
    verify_nested_manifest(nested)
    feature_entries = {str(row["case_id"]): row for row in nested["features"]}
    spans = pd.read_csv(paths["v0/results/flagged_oof_spans.csv"]).sort_values("v0_span_id")
    if len(spans) != 1289 or spans.v0_span_id.duplicated().any():
        raise RuntimeError("Frozen V0 span coverage drift")

    cache_path = study_dir / "cache" / "temporal_span_features.npy"
    cache = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float16, shape=(len(spans), TOTAL_BINS, FEATURE_DIM)
    )
    index_rows: list[dict[str, object]] = []
    verified: set[str] = set()
    loaded_case: str | None = None
    loaded_feature: np.ndarray | None = None
    ordered = spans.sort_values(["case_id", "v0_span_id"], kind="mergesort")
    for count, row in enumerate(ordered.itertuples(index=False), start=1):
        case_id = str(row.case_id)
        entry = feature_entries[case_id]
        path = Path(entry["path"])
        if case_id not in verified:
            if not path.is_file() or int(path.stat().st_size) != int(entry["bytes"]):
                raise RuntimeError(f"Feature missing/size drift: {case_id}")
            if file_sha256(path) != entry["sha256"]:
                raise RuntimeError(f"Feature hash drift: {case_id}")
            verified.add(case_id)
        if loaded_case != case_id:
            loaded_feature = np.load(path, allow_pickle=False)
            loaded_case = case_id
        assert loaded_feature is not None
        cache_index = int(row.v0_span_id)
        if cache_index < 0 or cache_index >= len(spans):
            raise RuntimeError("V0 span IDs must be contiguous cache indices")
        cache[cache_index] = temporal_span_view(
            loaded_feature, int(row.selected_start), int(row.selected_end)
        ).astype(np.float16)
        index_rows.append(
            {
                "v0_span_id": cache_index,
                "cache_index": cache_index,
                "outer_fold": int(row.outer_fold),
                "inner_fold": int(row.inner_fold),
                "case_id": case_id,
            }
        )
        if count % 100 == 0:
            cache.flush()
    cache.flush()
    del cache
    index = pd.DataFrame(index_rows).sort_values("v0_span_id")
    if list(index.v0_span_id.astype(int)) != list(range(len(spans))):
        raise RuntimeError("V1 cache index is not a complete contiguous span map")
    index_path = study_dir / "cache" / "temporal_span_index.csv"
    index.to_csv(index_path, index=False)
    complete = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "shape": [len(spans), TOTAL_BINS, FEATURE_DIM],
        "dtype": "float16",
        "spans": len(spans),
        "unique_feature_files_verified": len(verified),
        "feature_manifest_digest": nested["manifest_digest"],
        "cache_sha256": file_sha256(cache_path),
        "index_sha256": file_sha256(index_path),
    }
    complete["completion_digest"] = canonical_digest(complete)
    atomic_write_json(study_dir / "cache" / "feature_cache_complete.json", complete)


def main() -> int:
    args = parse_args()
    execute(args.study_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
