#!/usr/bin/env python3
"""
Fix a results CSV's `ground_truth` column by re-attaching labels from a dataset log CSV.

This is useful when a results CSV was generated with an incorrect ground-truth alignment
but still has correct `case:concept:name` ordering.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _load_ground_truth_map(dataset_log_csv: Path) -> Dict[str, List[str]]:
    gt: Dict[str, List[str]] = defaultdict(list)
    with dataset_log_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{dataset_log_csv} has no header row")
        required = {"case:concept:name", "concept:name"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{dataset_log_csv} missing required columns: {sorted(missing)}")
        for row in reader:
            gt[str(row["case:concept:name"])].append(str(row["concept:name"]))
    return dict(gt)


def fix_ground_truth_csv_in_place(results_csv: Path, gt_map: Dict[str, List[str]], *, backup: bool) -> None:
    tmp_path = results_csv.with_suffix(results_csv.suffix + ".tmp")
    bak_path = results_csv.with_suffix(results_csv.suffix + ".bak")

    counters: Dict[str, int] = defaultdict(int)
    seen_cases = set()

    with results_csv.open("r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError(f"{results_csv} has no header row")
        if "ground_truth" not in reader.fieldnames:
            raise ValueError(f"{results_csv} has no 'ground_truth' column")
        if "case:concept:name" not in reader.fieldnames:
            raise ValueError(f"{results_csv} has no 'case:concept:name' column")

        with tmp_path.open("w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                case_id = str(row["case:concept:name"])
                seen_cases.add(case_id)
                idx = counters[case_id]
                if case_id not in gt_map:
                    raise KeyError(f"Case '{case_id}' not found in dataset log (needed for {results_csv})")
                if idx >= len(gt_map[case_id]):
                    raise IndexError(
                        f"Case '{case_id}' in {results_csv} has more rows than dataset log "
                        f"({idx+1} > {len(gt_map[case_id])})"
                    )
                row["ground_truth"] = gt_map[case_id][idx]
                counters[case_id] = idx + 1
                writer.writerow(row)

    # Basic sanity: ensure each seen case consumed the entire trace in the log.
    # If your results were sampled (not full-length), this will fail by design.
    mismatches = []
    for case_id in sorted(seen_cases, key=lambda x: int(x) if x.isdigit() else x):
        if counters[case_id] != len(gt_map.get(case_id, [])):
            mismatches.append((case_id, counters[case_id], len(gt_map.get(case_id, []))))
    if mismatches:
        msg = "; ".join(f"{cid}: {have}/{need}" for cid, have, need in mismatches[:10])
        raise ValueError(
            f"{results_csv}: ground truth length mismatch for {len(mismatches)} case(s) "
            f"(this script assumes full-length traces). Examples: {msg}"
        )

    if backup:
        results_csv.replace(bak_path)
        tmp_path.replace(results_csv)
    else:
        tmp_path.replace(results_csv)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-log-csv", type=Path, required=True)
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("results_csv", type=Path, nargs="+")
    args = ap.parse_args()

    gt_map = _load_ground_truth_map(args.dataset_log_csv)
    for p in args.results_csv:
        fix_ground_truth_csv_in_place(p, gt_map, backup=(not args.no_backup))
        print(f"✓ fixed ground_truth: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

