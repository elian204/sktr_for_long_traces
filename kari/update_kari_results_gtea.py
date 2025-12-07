"""
Recompute KARI TAS metrics for GTEA runs and refresh CSV summaries.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RESULTS_DIR = REPO_ROOT / "results" / "gtea"
KARI_EXP_DIR = HERE / "results" / "kari_experiments" / "gtea"

# Local import after adjusting path
from compute_kari_tas import compute_kari_tas


def _latest_pred_file(n_traces: int) -> Path:
    pattern = f"results_gtea_n{n_traces}_seed101_*.pkl"
    # Search recursively so runs can be organized in subfolders
    candidates = list(KARI_EXP_DIR.rglob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No prediction files found for n_traces={n_traces} in {KARI_EXP_DIR}")

    def _date_key(p: Path) -> int:
        return int(p.stem.split("_")[-1])

    return max(candidates, key=_date_key)


def compute_all_metrics() -> List[Dict]:
    rows: List[Dict] = []
    for n in range(7, 0, -1):  # run from max traces down to min
        pred_path = _latest_pred_file(n)
        metrics = compute_kari_tas(
            pred_path,
            dataset="gtea",
            allow_remap=False,
            verbose=True,
        )
        rows.append(
            {
                "dataset_name": "gtea",
                "n_train_traces": n,
                "kari_acc_micro": metrics["acc_micro"],
                "kari_edit": metrics["edit"],
                "kari_f1@10": metrics["f1@10"],
                "kari_f1@25": metrics["f1@25"],
                "kari_f1@50": metrics["f1@50"],
            }
        )
    return rows


def write_kari_only(rows: List[Dict]) -> None:
    out_path = RESULTS_DIR / "gtea_kari_ntraces_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote KARI metrics to {out_path}")


def update_combined(rows: List[Dict]) -> None:
    combined_path = RESULTS_DIR / "gtea_combined_results_with_kari.csv"
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined results CSV not found: {combined_path}")

    combined = pd.read_csv(combined_path)
    for row in rows:
        mask = (combined["dataset_name"] == "gtea") & (combined["n_train_traces"] == row["n_train_traces"])
        if not mask.any():
            print(f"Warning: no combined row found for n_train_traces={row['n_train_traces']}")
            continue
        for key in ("kari_acc_micro", "kari_edit", "kari_f1@10", "kari_f1@25", "kari_f1@50"):
            combined.loc[mask, key] = row[key]
    combined.to_csv(combined_path, index=False)
    print(f"Updated combined CSV at {combined_path}")


def main():
    rows = compute_all_metrics()
    write_kari_only(rows)
    update_combined(rows)


if __name__ == "__main__":
    main()
