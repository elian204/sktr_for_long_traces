#!/usr/bin/env python3
"""Prepare author-checkpoint inference before any Phase-B retraining."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase_b_option0_common import (
    ASFORMER_ARCHIVE_SHA256,
    ASFORMER_ARCHIVE_URL,
    ASFORMER_OFFICIAL_GIT_HEAD,
    DEFAULT_ASFORMER_SOURCE,
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_FEATURE_MANIFEST,
    DEFAULT_PHASE_A_DIR,
    DEFAULT_STUDY_DIR,
    OUTER_FOLDS,
    PROTOCOL_VERSION,
    atomic_write_json,
    canonical_digest,
    file_sha256,
    git_head,
    manifest_digest,
    source_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--phase-a-dir", type=Path, default=DEFAULT_PHASE_A_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--asformer-source", type=Path, default=DEFAULT_ASFORMER_SOURCE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def add_input(rows: list[dict[str, Any]], role: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Option-0 input {role}: {path}")
    rows.append(
        {
            "role": role,
            "path": str(path.resolve()),
            "size_bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
    )


def main() -> int:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    if study_dir.exists():
        if not args.replace:
            raise FileExistsError(study_dir)
        shutil.rmtree(study_dir)
    for name in ("results", "logs", "exports", "status"):
        (study_dir / name).mkdir(parents=True, exist_ok=True)

    source_root = args.asformer_source.resolve()
    if git_head(source_root) != ASFORMER_OFFICIAL_GIT_HEAD:
        raise RuntimeError("Clean ASFormer source is not the official archived Git HEAD")
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is required for the fail-closed launcher")

    cache = args.cache_dir.resolve()
    archive = cache / "ASFormer" / "models.zip"
    if file_sha256(archive) != ASFORMER_ARCHIVE_SHA256:
        raise RuntimeError("Downloaded ASFormer author archive hash drift")

    rows: list[dict[str, Any]] = []
    phase_a = args.phase_a_dir.resolve()
    for name in ("phase_a_decision.json", "paper_reconciliation.csv", "phase_a_complete.json"):
        add_input(rows, f"phase_a/{name}", phase_a / "results" / name)
    add_input(rows, "official/asformer/archive", archive)
    add_input(rows, "official/asformer/readme_snapshot", cache / "asformer_official_README.md")
    add_input(rows, "official/asformer/source/model", source_root / "model.py")
    add_input(rows, "official/asformer/source/eval", source_root / "eval.py")
    add_input(rows, "official/asformer/source/readme", source_root / "README.md")
    for fold in OUTER_FOLDS:
        add_input(
            rows,
            f"official/asformer/breakfast/fold{fold}/checkpoint",
            cache
            / "ASFormer"
            / "extracted"
            / "models"
            / "breakfast"
            / f"split_{fold}"
            / "epoch-120.model",
        )
    add_input(rows, "official/mstcn2/readme_snapshot", cache / "mstcn2_official_README.md")
    add_input(rows, "official/mstcn2/github_releases", cache / "mstcn2_github_releases.json")
    add_input(rows, "official/mstcn2/github_tags", cache / "mstcn2_github_tags.json")
    add_input(rows, "breakfast/feature_manifest", args.feature_manifest.resolve())
    add_input(rows, "breakfast/mapping", args.data_root.resolve() / "breakfast" / "mapping.txt")
    for fold in OUTER_FOLDS:
        add_input(
            rows,
            f"breakfast/fold{fold}/test_bundle",
            args.data_root.resolve() / "breakfast" / "splits" / f"test.split{fold}.bundle",
        )
    rows.sort(key=lambda row: row["role"])
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "manifest_digest": manifest_digest(rows),
    }
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "phase": "B_option0_author_checkpoint_inference",
        "phase_a_dir": str(phase_a),
        "data_root": str(args.data_root.resolve()),
        "asformer_source": str(source_root),
        "asformer_source_head": ASFORMER_OFFICIAL_GIT_HEAD,
        "asformer_archive_url": ASFORMER_ARCHIVE_URL,
        "asformer_archive_sha256": ASFORMER_ARCHIVE_SHA256,
        "target": {"backbone": "asformer", "dataset": "breakfast", "folds": list(OUTER_FOLDS)},
        "official_checkpoint_inventory": {
            "asformer": {
                "status": "AVAILABLE",
                "archive_contains_folds": {"gtea": 4, "50salads": 5, "breakfast": 4},
            },
            "mstcn2": {
                "status": "UNAVAILABLE_IN_OFFICIAL_REPOSITORY",
                "github_releases": 0,
                "github_tags": 0,
                "readme_checkpoint_link": False,
            },
        },
        "gpu_inference_allowed": True,
        "gpu_training_allowed": False,
        "phase_c_allowed": False,
        "residual_training_launch_allowed": False,
        "sealed_studies_opened": False,
    }
    config["config_digest"] = canonical_digest(config)
    provenance = source_provenance()
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_dir": str(study_dir),
        "review_state": "OPTION0_INFERENCE_READY_TRAINING_DISABLED",
        "input_manifest_digest": manifest["manifest_digest"],
        "source_provenance": provenance,
        "gpu_training_launched": False,
        "sealed_studies_opened": False,
    }
    metadata["spec_sha256"] = canonical_digest(metadata)
    atomic_write_json(study_dir / "study_config.json", config)
    atomic_write_json(study_dir / "input_manifest.json", manifest)
    atomic_write_json(study_dir / "study_metadata.json", metadata)
    (study_dir / "DO_NOT_EDIT.txt").write_text(
        "Immutable Phase-B Option-0 study. Author-checkpoint inference only.\n"
        "Training and Phase C are disabled; regenerate after any source change.\n"
    )

    devices = {fold: fold - 1 for fold in OUTER_FOLDS}
    for fold, device in devices.items():
        script = study_dir / f"run_fold{fold}.sh"
        script.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"/usr/bin/python {Path(__file__).resolve().parent / 'export_asformer_official.py'} "
            f"--study-dir {study_dir} --fold {fold} --device {device} "
            f"2>&1 | tee {study_dir / 'logs' / f'fold{fold}.log'}\n"
        )
        script.chmod(0o755)
    finalize = study_dir / "finalize.sh"
    finalize.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"/usr/bin/python {Path(__file__).resolve().parent / 'finalize_phase_b_option0.py'} "
        f"--study-dir {study_dir} 2>&1 | tee {study_dir / 'logs' / 'finalize.log'}\n"
    )
    finalize.chmod(0o755)
    launcher = study_dir / "launch_tmux.sh"
    checks = []
    launches = []
    for fold, device in devices.items():
        checks.append(
            f"if [[ \"$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader -i {device} | sed '/^$/d' | wc -l)\" -ne 0 ]]; then echo 'GPU {device} occupied; refusing launch'; exit 1; fi"
        )
        launches.append(
            f"tmux new-session -d -s cb_option0_asf_bf_f{fold} 'cd {study_dir} && ./run_fold{fold}.sh'"
        )
    launcher.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p logs\n"
        + "\n".join(checks)
        + "\n"
        + "\n".join(launches)
        + "\n"
    )
    launcher.chmod(0o755)
    print(study_dir)
    print(f"spec_sha256={metadata['spec_sha256']}")
    print(f"input_manifest_digest={manifest['manifest_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
