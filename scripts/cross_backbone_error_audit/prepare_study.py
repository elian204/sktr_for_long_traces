#!/usr/bin/env python3
"""Create the immutable Phase-A inventory/reconciliation review study."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    BACKBONES,
    DATASETS,
    DEFAULT_ASFORMER_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_MSTCN2_ROOT,
    DEFAULT_STUDY_DIR,
    PROTOCOL_VERSION,
    atomic_write_json,
    canonical_digest,
    cell_paths,
    file_sha256,
    git_provenance,
    source_provenance,
    study_config,
    write_lines,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--mstcn2-root", type=Path, default=DEFAULT_MSTCN2_ROOT)
    parser.add_argument("--asformer-root", type=Path, default=DEFAULT_ASFORMER_ROOT)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def add_input(rows: list[dict[str, Any]], role: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {role}: {path}")
    rows.append(
        {
            "role": role,
            "path": str(path.resolve()),
            "size_bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
    )


def build_manifest(config: dict[str, Any]) -> dict[str, Any]:
    data_root = Path(config["data_root"])
    roots = {key: Path(value) for key, value in config["backbone_roots"].items()}
    rows: list[dict[str, Any]] = []
    for dataset, dataset_config in DATASETS.items():
        add_input(rows, f"data/{dataset}/mapping", data_root / dataset / "mapping.txt")
        for fold in range(1, int(dataset_config["folds"]) + 1):
            add_input(rows, f"data/{dataset}/fold{fold}/train", data_root / dataset / "splits" / f"train.split{fold}.bundle")
            add_input(rows, f"data/{dataset}/fold{fold}/test", data_root / dataset / "splits" / f"test.split{fold}.bundle")
        for gt_path in sorted((data_root / dataset / "groundTruth").glob("*")):
            if gt_path.is_file():
                add_input(rows, f"data/{dataset}/ground_truth/{gt_path.name}", gt_path)
    for backbone in BACKBONES:
        root = roots[backbone]
        add_input(rows, f"{backbone}/source/main", root / "main.py")
        add_input(rows, f"{backbone}/source/model", root / "model.py")
        add_input(rows, f"{backbone}/source/eval", root / "eval.py")
        for dataset, dataset_config in DATASETS.items():
            common_paths = cell_paths(backbone, dataset, 1, roots)
            add_input(rows, f"{backbone}/{dataset}/video_index_map", common_paths["video_index_map"])
            add_input(rows, f"{backbone}/{dataset}/stored_mapping", common_paths["stored_mapping"])
            for probability in sorted(common_paths["softmax_dir"].glob("*.npy")):
                if probability.stem.isdigit():
                    add_input(rows, f"{backbone}/{dataset}/probability/{probability.name}", probability)
            for fold in range(1, int(dataset_config["folds"]) + 1):
                paths = cell_paths(backbone, dataset, fold, roots)
                add_input(rows, f"{backbone}/{dataset}/fold{fold}/checkpoint", paths["checkpoint"])
    rows.sort(key=lambda row: row["role"])
    return {
        "protocol_version": PROTOCOL_VERSION,
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(row["size_bytes"] for row in rows),
        "manifest_digest": canonical_digest(
            [{"role": row["role"], "sha256": row["sha256"]} for row in rows]
        ),
    }


def main() -> int:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    if study_dir.exists():
        if not args.replace:
            raise FileExistsError(study_dir)
        shutil.rmtree(study_dir)
    study_dir.mkdir(parents=True)
    (study_dir / "results").mkdir()
    (study_dir / "logs").mkdir()

    config = study_config(
        data_root=args.data_root,
        mstcn2_root=args.mstcn2_root,
        asformer_root=args.asformer_root,
    )
    config["config_digest"] = canonical_digest(config)
    manifest = build_manifest(config)
    provenance = source_provenance()
    external_git = {
        backbone: git_provenance(Path(root))
        for backbone, root in config["backbone_roots"].items()
    }
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study_dir),
        "config": config,
        "input_manifest_digest": manifest["manifest_digest"],
        "source_provenance": provenance,
        "external_git_provenance": external_git,
        "review_state": "PHASE_A_READY_NOT_PHASE_C",
        "gpu_launched": False,
        "sealed_studies_touched": False,
    }
    metadata["spec_sha256"] = canonical_digest(metadata)
    atomic_write_json(study_dir / "study_config.json", config)
    atomic_write_json(study_dir / "input_manifest.json", manifest)
    atomic_write_json(study_dir / "study_metadata.json", metadata)
    write_lines(
        study_dir / "DO_NOT_EDIT.txt",
        [
            "Immutable review study: cross-backbone error audit Phase A.",
            "Do not edit inputs, metadata, or results in place.",
            "Regenerate a new version after any source or protocol change.",
            "Phase C and all GPU training remain disabled pending Fable review.",
        ],
    )
    run_script = study_dir / "run_phase_a.sh"
    run_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"/usr/bin/python {Path(__file__).resolve().parent / 'run_phase_a.py'} "
        f"--study-dir {study_dir} 2>&1 | tee {study_dir / 'logs' / 'phase_a.log'}\n",
        encoding="utf-8",
    )
    run_script.chmod(0o755)
    print(study_dir)
    print(f"inputs={manifest['file_count']} bytes={manifest['total_bytes']}")
    print(f"spec_sha256={metadata['spec_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
