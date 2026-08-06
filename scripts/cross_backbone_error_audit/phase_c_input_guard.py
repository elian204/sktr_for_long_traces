#!/usr/bin/env python3
"""Fail-closed digest and path gate for future Phase-C loaders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

from phase_b_selection_common import file_sha256, load_json, validate_phase_c_inputs


def verify_phase_c_records(records: Iterable[Mapping[str, str]], allowed_roots: Iterable[Path]) -> list[Path]:
    values = list(records)
    paths = validate_phase_c_inputs((Path(row["path"]) for row in values), allowed_roots)
    for path, row in zip(paths, values):
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != row["sha256"]:
            raise RuntimeError(f"Phase-C digest drift: {path}")
    return paths


def verify_phase_c_study_records(study: Path, records: Iterable[Mapping[str, str]]) -> list[Path]:
    """Use only roots frozen in the reviewed Step-0 config, never caller-supplied roots."""
    config = load_json(study.resolve() / "study_config.json")
    policy = config["phase_c_input_policy"]
    if policy.get("digest_verification_required") is not True:
        raise RuntimeError("Phase-C digest policy is not fail-closed")
    roots = [Path(value) for value in policy["allowed_roots_after_digest_review"]]
    return verify_phase_c_records(records, roots)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True, help="JSON list of {path, sha256}")
    args = parser.parse_args()
    records = json.loads(args.records.read_text())
    verified = verify_phase_c_study_records(args.study_dir, records)
    print(f"verified_phase_c_inputs={len(verified)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
