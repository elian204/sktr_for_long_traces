#!/usr/bin/env python3
"""Fail closed unless the Task-1 multi-fold decision explicitly opens the gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = load_json(args.study_dir.resolve() / "study_metadata.json")
    contract = metadata["multifold_launch_dependency"]
    decision_path = Path(contract["decision_path"])
    if not decision_path.is_file():
        raise FileNotFoundError(
            f"Task-2 launch remains gated; Task-1 decision is absent: {decision_path}"
        )
    decision = load_json(decision_path)
    field = contract["required_field"]
    required = contract["required_value"]
    if decision.get(field) != required:
        raise RuntimeError(
            f"Task-2 launch remains gated: {field}={decision.get(field)!r}, "
            f"required {required!r}"
        )
    print(f"Task-1 launch gate passed: {decision_path}")


if __name__ == "__main__":
    main()
