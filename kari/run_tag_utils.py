"""
Helper utilities to generate and prepare per-run output folders.

The run tag now defaults to just the date (YYYYMMDD), optionally prefixed.
`KARI_RUN_TAG` can still override it explicitly.

If a config dict is provided to `prepare_run_dirs`, it is automatically
written to each run folder (results and grammar) as JSON.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def build_run_tag(
    dataset: str,
    use_topK: bool,
    topK_K: Optional[int] = None,
    prefix: Optional[str] = None,
    explicit: Optional[str] = None,
) -> str:
    """
    Build a run tag (or return explicit/env override).

    NOTE: `dataset`, `use_topK`, and `topK_K` are retained for backward
    compatibility but no longer affect the generated tag; the default tag is
    simply the current date (YYYYMMDD), optionally prefixed.
    """
    if explicit:
        return explicit
    env_tag = os.environ.get("KARI_RUN_TAG")
    if env_tag:
        return env_tag

    date_str = datetime.now().strftime("%Y%m%d")
    return f"{prefix}_{date_str}" if prefix else date_str


def prepare_run_dirs(
    base_dir: Path | str,
    dataset: str,
    use_topK: bool,
    topK_K: Optional[int] = None,
    prefix: Optional[str] = None,
    run_tag: Optional[str] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
    config_filename: str = "config.json",
) -> Tuple[str, Path, Path]:
    """
    Compute (and create) per-run results/grammar directories.

    Returns
    -------
    run_tag : str
        The tag used (auto-generated or provided).
    results_dir : Path
        Where to save prediction pickles for this run.
    grammar_dir : Path
        Where to save induced grammars for this run.

    Notes
    -----
    A config JSON is written to both `results_dir` and `grammar_dir` using
    `config_filename`. It always contains run metadata (dataset, run_tag,
    use_topK, topK_K, prefix when set). If `config` is provided, it is stored
    under the "hyperparameters" key.
    """
    base = Path(base_dir)
    tag = build_run_tag(dataset, use_topK=use_topK, topK_K=topK_K, prefix=prefix, explicit=run_tag)

    results_dir = base / "results" / "kari_experiments" / dataset / tag
    grammar_dir = base / "results" / "induced_grammars_experiment" / dataset / tag

    results_dir.mkdir(parents=True, exist_ok=True)
    grammar_dir.mkdir(parents=True, exist_ok=True)

    cfg_payload: Dict[str, Any] = {
        "run_tag": tag,
        "dataset": dataset,
        "use_topK": use_topK,
        "topK_K": topK_K,
    }
    if prefix is not None:
        cfg_payload["prefix"] = prefix
    if config is not None:
        cfg_payload["hyperparameters"] = config

    def _write_cfg(target_dir: Path) -> None:
        cfg_path = target_dir / config_filename
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_payload, f, indent=2, sort_keys=True, default=str)

    _write_cfg(results_dir)
    _write_cfg(grammar_dir)

    return tag, results_dir, grammar_dir
