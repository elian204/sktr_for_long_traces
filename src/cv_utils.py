"""
Cross-validation utilities for k-fold learning curve experiments.

This module provides functions for:
- Loading train/test splits from bundle files
- Mapping video names to case IDs
- Stratified sampling of training traces
- Dataset-specific CV configurations
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict

import pandas as pd


# Default data root (can be overridden via environment variable)
DEFAULT_DATA_ROOT = os.environ.get(
    'DATA_ROOT', '/home/dsi/eli-bogdanov/data/data')


def load_split_bundle(bundle_path: str) -> List[str]:
    """
    Load video names from a split bundle file.

    Parameters
    ----------
    bundle_path : str
        Path to .bundle file containing one video name per line.

    Returns
    -------
    List[str]
        List of video names (e.g., ["rgb-01-1.txt", "rgb-01-2.txt"])
    """
    with open(bundle_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def build_video_to_case_mapping(
    dataset_name: str,
    model_source: str = 'asformer',
    model_base_path: Optional[str] = None,
    *,
    video_index_map_path: Optional[Path] = None,
) -> Dict[str, str]:
    """
    Build mapping from video names to case IDs by loading the saved video_index_map.txt
    from model outputs. This ensures exact match with how prepare_df_from_model assigns case IDs.

    Parameters
    ----------
    dataset_name : str
        One of: '50salads', 'gtea', 'breakfast'
    model_source : str
        Model source: 'asformer', 'mstcn2', or 'diffact'. Default 'asformer'.
    model_base_path : str, optional
        Base path to model results. If None, uses ~/ASFormer or ~/MS-TCN2.
    video_index_map_path : pathlib.Path, optional
        If set, load this file instead of the default
        ``.../results/<dataset>/softmax/video_index_map.txt`` (e.g. per-fold DiffAct bundles).

    Returns
    -------
    Dict[str, str]
        Mapping from video name (e.g., "rgb-01-1.txt") to case ID (e.g., "0").
        Keys include the .txt extension to match bundle file format.
    """
    # Resolve model base path
    if model_base_path is None:
        home = Path.home()
        src = model_source.lower()
        if src == 'asformer':
            model_base_path = home / 'ASFormer'
        elif src == 'diffact':
            model_base_path = Path(__file__).resolve(
            ).parent.parent / 'baselines' / 'DiffAct'
        else:
            model_base_path = home / 'MS-TCN2'
    else:
        model_base_path = Path(model_base_path)

    # Path to video_index_map.txt
    if video_index_map_path is not None:
        map_file = Path(video_index_map_path)
    else:
        map_file = model_base_path / 'results' / \
            dataset_name / 'softmax' / 'video_index_map.txt'

    if not map_file.exists():
        raise FileNotFoundError(
            f"video_index_map.txt not found at {map_file}. "
            f"Ensure softmax exports exist for {dataset_name} (ASFormer, MS-TCN2, or DiffAct)."
        )

    # Load mapping: format is "idx\tname" per line
    # Bundle files use names with .txt extension, map file uses names without
    mapping = {}
    with open(map_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                case_id, video_name = parts
                # Bundle files have .txt extension - add it only if not already present
                if not video_name.endswith('.txt'):
                    video_name = video_name + '.txt'
                mapping[video_name] = case_id

    if not mapping:
        raise ValueError(f"No entries found in {map_file}")

    return mapping


def video_names_to_case_ids(video_names: List[str], mapping: Dict[str, str]) -> List[str]:
    """
    Convert video names to case IDs using the mapping.

    Parameters
    ----------
    video_names : List[str]
        List of video names from split bundle file.
    mapping : Dict[str, str]
        Mapping from video names to case IDs.

    Returns
    -------
    List[str]
        List of case IDs corresponding to the video names.

    Raises
    ------
    KeyError
        If a video name is not found in the mapping.
    """
    missing = [name for name in video_names if name not in mapping]
    if missing:
        raise KeyError(
            f"Video names not found in mapping: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    return [mapping[name] for name in video_names]


def stratified_sample_k_traces(
    train_case_ids: List[str],
    k: int,
    variant_df: pd.DataFrame,
    seed: int = 42
) -> List[str]:
    """
    Stratified sampling of k traces from training pool preserving variant diversity.

    Strategy:
    1. Group available train cases by variant
    2. Round-robin: select one case from each variant until k or variants exhausted
    3. Fill remaining slots randomly from remaining pool

    Parameters
    ----------
    train_case_ids : List[str]
        Pool of available training case IDs (from split file).
    k : int
        Number of traces to sample.
    variant_df : pd.DataFrame
        DataFrame from get_variant_info() with columns:
        - variant_id: Integer ID
        - case_ids: List of case IDs belonging to this variant
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    List[str]
        k case IDs sampled with stratification by variant.
    """
    # If k >= pool size, return all
    if k >= len(train_case_ids):
        return list(train_case_ids)

    rng = random.Random(seed)
    train_set = set(train_case_ids)

    # Build mapping from case_id to variant_id
    case_to_variant: Dict[str, int] = {}
    for _, row in variant_df.iterrows():
        for case_id in row['case_ids']:
            case_str = str(case_id)
            if case_str in train_set:
                case_to_variant[case_str] = row['variant_id']

    # Group train cases by variant
    variant_to_cases: Dict[int, List[str]] = defaultdict(list)
    for case_id in train_case_ids:
        if case_id in case_to_variant:
            variant_to_cases[case_to_variant[case_id]].append(case_id)
        else:
            # Case not in variant_df (shouldn't happen, but handle gracefully)
            variant_to_cases[-1].append(case_id)

    # Shuffle cases within each variant for randomness
    for vid in variant_to_cases:
        rng.shuffle(variant_to_cases[vid])

    # Shuffle variant order to avoid bias toward frequent variants (variant_id 0 = most frequent)
    # This ensures fair representation when k < #variants
    sorted_variants = list(variant_to_cases.keys())
    rng.shuffle(sorted_variants)

    selected: List[str] = []
    used: Set[str] = set()

    # Round-robin selection: one case per variant
    variant_idx = 0
    while len(selected) < k and variant_to_cases:
        vid = sorted_variants[variant_idx % len(sorted_variants)]

        # Find an unused case from this variant
        cases = variant_to_cases[vid]
        while cases and cases[-1] in used:
            cases.pop()

        if cases:
            case = cases.pop()
            selected.append(case)
            used.add(case)

        variant_idx += 1

        # If we've cycled through all variants, check if any have remaining cases
        if variant_idx % len(sorted_variants) == 0:
            # Remove empty variants
            sorted_variants = [
                v for v in sorted_variants if variant_to_cases[v]]
            if not sorted_variants:
                break

    # If still need more cases (unlikely), sample from remaining pool
    if len(selected) < k:
        remaining = [c for c in train_case_ids if c not in used]
        n_more = k - len(selected)
        if remaining:
            selected.extend(rng.sample(remaining, min(n_more, len(remaining))))

    return selected[:k]


def get_unique_representatives(
    case_ids: List[str],
    variant_df: pd.DataFrame,
    seed: int = 42,
) -> List[str]:
    """
    Filter case_ids to keep only one representative per unique variant.

    Parameters
    ----------
    case_ids : List[str]
        List of case IDs to filter.
    variant_df : pd.DataFrame
        DataFrame with 'variant_id' and 'case_ids' columns from get_variant_info().
    seed : int
        Random seed for reproducible selection.

    Returns
    -------
    List[str]
        Filtered list with one representative per variant.
    """
    rng = random.Random(seed)

    # Build case_id -> variant_id mapping
    case_to_variant: Dict[str, Any] = {}
    for _, row in variant_df.iterrows():
        for cid in row['case_ids']:
            case_to_variant[str(cid)] = row['variant_id']

    # Group input case_ids by variant
    variant_to_cases: Dict[Any, List[str]] = {}
    for cid in case_ids:
        cid_str = str(cid)
        # Use case_id as variant if not found
        vid = case_to_variant.get(cid_str, cid_str)
        if vid not in variant_to_cases:
            variant_to_cases[vid] = []
        variant_to_cases[vid].append(cid_str)

    # Pick one representative per variant (first one after shuffle for reproducibility)
    representatives: List[str] = []
    for vid in sorted(variant_to_cases.keys()):
        cases = variant_to_cases[vid]
        rng.shuffle(cases)
        representatives.append(cases[0])

    return representatives


def get_dataset_cv_config(dataset_name: str, data_root: Optional[str] = None) -> Dict:
    """
    Return cross-validation configuration for each dataset.

    Parameters
    ----------
    dataset_name : str
        One of: '50salads', 'gtea', 'breakfast'
    data_root : str, optional
        Root directory containing dataset folders.

    Returns
    -------
    Dict
        Configuration with keys:
        - n_folds: Number of CV folds
        - k_values: List of training set sizes to evaluate
        - splits_dir: Path to split bundle files
    """
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT

    configs = {
        '50salads': {
            'n_folds': 5,
            'k_values': [1, 5, 10, 20, 30, 40],
            'splits_dir': str(Path(data_root) / '50salads' / 'splits'),
        },
        'breakfast': {
            'n_folds': 4,
            'k_values': [1, 10, 50, 100, 150, 199],
            'splits_dir': str(Path(data_root) / 'breakfast' / 'splits'),
        },
        'gtea': {
            'n_folds': 4,
            'k_values': [1, 5, 10, 15, 20],
            'splits_dir': str(Path(data_root) / 'gtea' / 'splits'),
        },
    }

    if dataset_name not in configs:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Expected one of: {list(configs.keys())}")

    return configs[dataset_name]


def get_fold_split_paths(dataset_name: str, fold: int, data_root: Optional[str] = None) -> Dict[str, str]:
    """
    Get paths to train and test bundle files for a specific fold.

    Parameters
    ----------
    dataset_name : str
        One of: '50salads', 'gtea', 'breakfast'
    fold : int
        Fold number (1-indexed).
    data_root : str, optional
        Root directory containing dataset folders.

    Returns
    -------
    Dict[str, str]
        Dictionary with 'train' and 'test' paths.
    """
    config = get_dataset_cv_config(dataset_name, data_root)
    splits_dir = config['splits_dir']

    return {
        'train': str(Path(splits_dir) / f'train.split{fold}.bundle'),
        'test': str(Path(splits_dir) / f'test.split{fold}.bundle'),
    }


def load_fold_case_ids(
    dataset_name: str,
    fold: int,
    video_to_case: Dict[str, str],
    data_root: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Load train and test case IDs for a specific fold.

    Parameters
    ----------
    dataset_name : str
        One of: '50salads', 'gtea', 'breakfast'
    fold : int
        Fold number (1-indexed).
    video_to_case : Dict[str, str]
        Mapping from video names to case IDs.
    data_root : str, optional
        Root directory containing dataset folders.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary with 'train' and 'test' case ID lists.
    """
    paths = get_fold_split_paths(dataset_name, fold, data_root)

    train_videos = load_split_bundle(paths['train'])
    test_videos = load_split_bundle(paths['test'])

    return {
        'train': video_names_to_case_ids(train_videos, video_to_case),
        'test': video_names_to_case_ids(test_videos, video_to_case),
    }
