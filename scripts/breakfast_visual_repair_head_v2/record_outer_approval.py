#!/usr/bin/env python3
"""Record Fable's explicit exact-digest approval for the single outer run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from common import PROTOCOL_VERSION, atomic_write_json, load_json, verify_self_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--expected-frozen-digest", required=True)
    parser.add_argument("--reviewer", default="Fable")
    parser.add_argument("--approval-statement", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    path = study_dir / "outer_review_approval.json"
    if path.exists():
        raise FileExistsError(path)
    metadata = load_json(study_dir / "study_metadata.json")
    frozen = load_json(study_dir / "frozen_outer_config.json")
    verify_self_digest(frozen, "frozen_config_digest")
    digest = frozen["frozen_config_digest"]
    if digest != args.expected_frozen_digest or digest != metadata["frozen_config_digest"]:
        raise RuntimeError("Approval digest does not match the staged frozen configuration")
    atomic_write_json(
        path,
        {
            "protocol_version": PROTOCOL_VERSION,
            "reviewer": args.reviewer,
            "approved_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_config_digest": digest,
            "source_digest": metadata["source_provenance"]["source_digest"],
            "outer_input_manifest_digest": metadata["outer_input_manifest_digest"],
            "approval_statement": args.approval_statement,
            "authorizes_exactly_one_outer_evaluation": True,
        },
    )
    print(f"Recorded exact-digest outer approval: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
