#!/usr/bin/env python3
"""
Plot SKTR vs KARI (plus Argmax) kfold learning curves from aggregated_results.csv.
Generates a 2x3 grid: 5 metrics + legend (no time plot).
"""
from src.evaluation import compute_sktr_vs_argmax_metrics
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib
matplotlib.use('Agg')


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_sktr_fold_results(results_dir: Path, dataset: str) -> Optional[pd.DataFrame]:
    all_results_path = results_dir / 'all_results.csv'
    if all_results_path.exists():
        df = pd.read_csv(all_results_path)
        if {'fold', 'k', 'sktr_acc'}.issubset(df.columns):
            return df

    result_files = sorted(results_dir.glob('fold*_k*_results.csv'))
    if not result_files:
        return None

    rows = []
    pattern = re.compile(r'^fold(\d+)_k(\d+)_results\.csv$')
    for csv_path in result_files:
        match = pattern.match(csv_path.name)
        if not match:
            continue
        fold_val = int(match.group(1))
        k_val = int(match.group(2))
        metrics = compute_sktr_vs_argmax_metrics(
            str(csv_path),
            case_col='case:concept:name',
            sktr_pred_col='sktr_activity',
            argmax_pred_col='argmax_activity',
            gt_col='ground_truth',
            background=None,
            dataset_name=dataset,
        )
        rows.append({
            'fold': fold_val,
            'k': k_val,
            'sktr_acc': metrics['sktr']['acc'],
            'sktr_edit': metrics['sktr']['edit'],
            'sktr_f1@10': metrics['sktr']['f1@10'],
            'sktr_f1@25': metrics['sktr']['f1@25'],
            'sktr_f1@50': metrics['sktr']['f1@50'],
            'argmax_acc': metrics['argmax']['acc'],
            'argmax_edit': metrics['argmax']['edit'],
            'argmax_f1@10': metrics['argmax']['f1@10'],
            'argmax_f1@25': metrics['argmax']['f1@25'],
            'argmax_f1@50': metrics['argmax']['f1@50'],
        })

    if not rows:
        return None

    return pd.DataFrame(rows)


def _aggregate_fold_results(df: pd.DataFrame, metric_prefixes: list) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    metric_cols = [c for c in df.columns if any(
        c.startswith(pfx) for pfx in metric_prefixes)]
    grouped = df.groupby('k')[metric_cols].mean(
        numeric_only=True).reset_index()
    rename = {col: f'{col}_mean' for col in grouped.columns if col != 'k'}
    return grouped.rename(columns=rename)


def _compute_constant_baseline(
    kari_all_path: Path,
    sktr_fold_df: Optional[pd.DataFrame],
) -> Optional[dict]:
    baseline_cols = ['argmax_acc', 'argmax_edit',
                     'argmax_f1@10', 'argmax_f1@25', 'argmax_f1@50']
    if kari_all_path.exists():
        df = pd.read_csv(kari_all_path)
        available = [c for c in baseline_cols if c in df.columns]
        if not available:
            return None
        per_fold = df.groupby('fold')[available].mean(numeric_only=True)
        mean_vals = per_fold.mean()
        return mean_vals.to_dict()

    if sktr_fold_df is None or sktr_fold_df.empty:
        return None
    available = [c for c in baseline_cols if c in sktr_fold_df.columns]
    if not available:
        return None
    per_fold = sktr_fold_df.groupby('fold')[available].mean(numeric_only=True)
    mean_vals = per_fold.mean()
    return mean_vals.to_dict()


def plot_kfold_comparison(
    dataset: str,
    model: str,
    sktr_results_root: Path,
    kari_results_root: Path,
    ours_label: str,
    title: Optional[str] = None,
    exclude_k: Optional[set] = None,
    replace_k: Optional[dict] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    baseline_constant: bool = False,
    save_pdf: bool = False,
    output_path: Optional[Path] = None,
) -> Path:
    sktr_results_dir = sktr_results_root / dataset / 'kfold_learning_curve' / model
    kari_results_dir = kari_results_root / 'kfold_learning_curve' / dataset / model

    sktr_path = sktr_results_dir / 'aggregated_results.csv'
    kari_path = kari_results_dir / 'aggregated_results.csv'

    sktr_fold_df = _load_sktr_fold_results(sktr_results_dir, dataset)
    kari_all_path = kari_results_dir / 'all_results.csv'
    kari_fold_df = _load_csv(kari_all_path)

    if sktr_fold_df is None:
        sktr_df = _load_csv(sktr_path)
    else:
        sktr_df = _aggregate_fold_results(sktr_fold_df, ['sktr_', 'argmax_'])

    if kari_fold_df is not None:
        if sktr_fold_df is not None:
            pairs = sktr_fold_df[['fold', 'k']].drop_duplicates()
            kari_fold_df = kari_fold_df.merge(pairs, on=['fold', 'k'])
        kari_df = _aggregate_fold_results(kari_fold_df, ['kari_', 'argmax_'])
    else:
        kari_df = _load_csv(kari_path)

    if sktr_df is None and kari_df is None:
        raise FileNotFoundError(
            f"No aggregated_results.csv found for {dataset}/{model}")

    if exclude_k:
        if sktr_fold_df is not None:
            sktr_fold_df = sktr_fold_df[~sktr_fold_df['k'].isin(exclude_k)]
            sktr_df = _aggregate_fold_results(
                sktr_fold_df, ['sktr_', 'argmax_'])
        elif sktr_df is not None and 'k' in sktr_df.columns:
            sktr_df = sktr_df[~sktr_df['k'].isin(
                exclude_k)].reset_index(drop=True)

        if kari_fold_df is not None:
            kari_fold_df = kari_fold_df[~kari_fold_df['k'].isin(exclude_k)]
            kari_df = _aggregate_fold_results(
                kari_fold_df, ['kari_', 'argmax_'])
        elif kari_df is not None and 'k' in kari_df.columns:
            kari_df = kari_df[~kari_df['k'].isin(
                exclude_k)].reset_index(drop=True)

    if replace_k:
        if sktr_df is not None and 'k' in sktr_df.columns:
            sktr_df = sktr_df.copy()
            sktr_df['k'] = sktr_df['k'].replace(replace_k)
        if kari_df is not None and 'k' in kari_df.columns:
            kari_df = kari_df.copy()
            kari_df['k'] = kari_df['k'].replace(replace_k)

    if baseline_constant:
        baseline = _compute_constant_baseline(kari_all_path, sktr_fold_df)
        if baseline and sktr_df is not None:
            for key, value in baseline.items():
                col = f'{key}_mean' if not key.endswith('_mean') else key
                if col in sktr_df.columns:
                    sktr_df[col] = value
        if baseline and kari_df is not None:
            for key, value in baseline.items():
                col = f'{key}_mean' if not key.endswith('_mean') else key
                if col in kari_df.columns:
                    kari_df[col] = value

    sns.set_theme(style='whitegrid', context='notebook', palette='deep')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    metrics_config = [
        ('acc', 'Accuracy'),
        ('edit', 'Edit Score'),
        ('f1@10', 'F1@10'),
        ('f1@25', 'F1@25'),
        ('f1@50', 'F1@50'),
    ]

    colors = {
        'sktr': '#1f77b4',
        'kari': '#ff7f0e',
        'argmax': '#4d4d4d',
    }

    all_series = []
    for metric_suffix, _ in metrics_config:
        if sktr_df is not None and f'sktr_{metric_suffix}_mean' in sktr_df.columns:
            all_series.append(sktr_df[f'sktr_{metric_suffix}_mean'])
        if kari_df is not None and f'kari_{metric_suffix}_mean' in kari_df.columns:
            all_series.append(kari_df[f'kari_{metric_suffix}_mean'])
        if sktr_df is not None and f'argmax_{metric_suffix}_mean' in sktr_df.columns:
            all_series.append(sktr_df[f'argmax_{metric_suffix}_mean'])
        elif kari_df is not None and f'argmax_{metric_suffix}_mean' in kari_df.columns:
            all_series.append(kari_df[f'argmax_{metric_suffix}_mean'])

    if y_min is not None and y_max is not None:
        y_min_global, y_max_global = y_min, y_max
    elif all_series:
        y_min_global = min(series.min() for series in all_series)
        y_max_global = max(series.max() for series in all_series)
        pad = (y_max_global - y_min_global) * \
            0.05 if y_max_global > y_min_global else 0.5
        y_min_global -= pad
        y_max_global += pad
    else:
        y_min_global, y_max_global = None, None

    for idx, (metric_suffix, metric_label) in enumerate(metrics_config):
        ax = axes[idx]

        if sktr_df is not None and f'sktr_{metric_suffix}_mean' in sktr_df.columns:
            sns.lineplot(
                x=sktr_df['k'],
                y=sktr_df[f'sktr_{metric_suffix}_mean'],
                ax=ax,
                label=ours_label,
                color=colors['sktr'],
                marker='o',
                linewidth=2.5,
                markersize=7,
            )

        if kari_df is not None and f'kari_{metric_suffix}_mean' in kari_df.columns:
            sns.lineplot(
                x=kari_df['k'],
                y=kari_df[f'kari_{metric_suffix}_mean'],
                ax=ax,
                label='KARI',
                color=colors['kari'],
                marker='^',
                linewidth=2.5,
                markersize=7,
            )

        if sktr_df is not None and f'argmax_{metric_suffix}_mean' in sktr_df.columns:
            sns.lineplot(
                x=sktr_df['k'],
                y=sktr_df[f'argmax_{metric_suffix}_mean'],
                ax=ax,
                label='Argmax',
                color=colors['argmax'],
                marker='s',
                linewidth=2,
                markersize=6,
                linestyle='--',
            )
        elif kari_df is not None and f'argmax_{metric_suffix}_mean' in kari_df.columns:
            sns.lineplot(
                x=kari_df['k'],
                y=kari_df[f'argmax_{metric_suffix}_mean'],
                ax=ax,
                label='Argmax',
                color=colors['argmax'],
                marker='s',
                linewidth=2,
                markersize=6,
                linestyle='--',
            )

        if y_min_global is not None and y_max_global is not None:
            ax.set_ylim(y_min_global, y_max_global)

        ax.set_title(metric_label, fontsize=13, fontweight='bold')
        ax.set_xlabel('N_train')
        ax.set_ylabel('Score')
        ax.legend().remove()

        x_vals = []
        if sktr_df is not None and 'k' in sktr_df.columns:
            x_vals.extend(sktr_df['k'].tolist())
        if kari_df is not None and 'k' in kari_df.columns:
            x_vals.extend(kari_df['k'].tolist())
        if x_vals:
            ticks = sorted(set(int(x) for x in x_vals))
            ax.set_xticks(ticks)

    ax_legend = axes[5]
    ax_legend.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    ax_legend.legend(
        handles, labels,
        loc='center', title='Method',
        fontsize=12, title_fontsize=13,
        frameon=True, fancybox=True, shadow=True,
    )

    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path is None:
        output_path = sktr_results_root / dataset / \
            'kfold_learning_curve' / model / 'learning_curves_kari_sktr.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if save_pdf:
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot SKTR vs KARI kfold learning curves (no time plot).'
    )
    parser.add_argument('-d', '--dataset', required=True,
                        choices=['50salads', 'gtea', 'breakfast'])
    parser.add_argument('-m', '--model', required=True,
                        choices=['asformer', 'mstcn2', 'diffact'])
    parser.add_argument(
        '--sktr-root',
        default=str(Path('~/sktr_for_long_traces/results').expanduser()),
        help='Root directory for SKTR results',
    )
    parser.add_argument(
        '--kari-root',
        default=str(Path('~/kari/results').expanduser()),
        help='Root directory for KARI results',
    )
    parser.add_argument('--output', default=None,
                        help='Output path for the plot')
    parser.add_argument('--ours-label', default='SKTR',
                        help='Label for SKTR series (e.g., Ours)')
    parser.add_argument('--no-title', action='store_true',
                        help='Disable plot title')
    parser.add_argument('--exclude-k', default=None,
                        help='Comma-separated k values to exclude')
    parser.add_argument('--replace-k', default=None,
                        help='Comma-separated k remaps like 199:200')
    parser.add_argument('--ymin', type=float, default=None,
                        help='Fixed y-axis minimum')
    parser.add_argument('--ymax', type=float, default=None,
                        help='Fixed y-axis maximum')
    parser.add_argument(
        '--baseline-constant',
        action='store_true',
        help='Use a constant argmax baseline across all k values',
    )
    parser.add_argument('--save-pdf', action='store_true',
                        help='Also save a PDF copy of the plot')
    args = parser.parse_args()

    output_path = Path(args.output).expanduser() if args.output else None
    title = None if args.no_title else f'{args.dataset.upper()} / {args.model.upper()} - {args.ours_label} vs KARI'
    exclude_k = None
    if args.exclude_k:
        exclude_k = {int(k) for k in args.exclude_k.split(',') if k.strip()}
    replace_k = None
    if args.replace_k:
        replace_k = {}
        for pair in args.replace_k.split(','):
            pair = pair.strip()
            if not pair:
                continue
            old, new = pair.split(':', 1)
            replace_k[int(old.strip())] = int(new.strip())
    plot_path = plot_kfold_comparison(
        dataset=args.dataset,
        model=args.model,
        sktr_results_root=Path(args.sktr_root).expanduser(),
        kari_results_root=Path(args.kari_root).expanduser(),
        ours_label=args.ours_label,
        title=title,
        exclude_k=exclude_k,
        replace_k=replace_k,
        y_min=args.ymin,
        y_max=args.ymax,
        baseline_constant=args.baseline_constant,
        save_pdf=args.save_pdf,
        output_path=output_path,
    )
    print(f"Saved plot to {plot_path}")


if __name__ == '__main__':
    main()
