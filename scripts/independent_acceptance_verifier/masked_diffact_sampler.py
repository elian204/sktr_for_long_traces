#!/usr/bin/env python3
"""Corrected masked DDIM sampler used only as V2 candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as torch_f
from scipy.ndimage import median_filter


@dataclass(frozen=True)
class RegionMasks:
    core: np.ndarray
    halo: np.ndarray
    exterior: np.ndarray

    @property
    def sampling_free(self) -> np.ndarray:
        return self.core | self.halo


@dataclass(frozen=True)
class SamplerOutput:
    probabilities: np.ndarray
    pre_restore_labels: np.ndarray
    labels: np.ndarray
    masks: RegionMasks
    time_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CandidateBatch:
    samples: tuple[SamplerOutput, ...]
    clusters: tuple[tuple[int, ...], ...]
    medoid_indices: tuple[int, ...]
    stopped_early: bool


def merge_intervals(intervals: Iterable[tuple[int, int]], length: int) -> list[tuple[int, int]]:
    normalized = sorted(
        (max(0, int(start)), min(int(end), int(length)))
        for start, end in intervals
        if int(end) > int(start) and int(end) > 0 and int(start) < int(length)
    )
    merged: list[list[int]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def build_region_masks(
    length: int, core_intervals: Sequence[tuple[int, int]], halo_width: int
) -> RegionMasks:
    if length <= 0 or halo_width < 0:
        raise ValueError("Mask length must be positive and halo non-negative")
    core = np.zeros(length, dtype=bool)
    merged_core = merge_intervals(core_intervals, length)
    for start, end in merged_core:
        core[start:end] = True
    expanded = merge_intervals(
        [(start - halo_width, end + halo_width) for start, end in merged_core], length
    )
    free = np.zeros(length, dtype=bool)
    for start, end in expanded:
        free[start:end] = True
    halo = free & ~core
    exterior = ~free
    if np.any(core & halo) or np.any(core & exterior) or np.any(halo & exterior):
        raise AssertionError("Three-region masks overlap")
    if not np.all(core | halo | exterior):
        raise AssertionError("Three-region masks do not partition time")
    return RegionMasks(core=core, halo=halo, exterior=exterior)


def ddim_time_pairs(
    total_timesteps: int, sampling_timesteps: int, t_start: int
) -> tuple[tuple[int, int], ...]:
    if not 0 <= t_start < total_timesteps:
        raise ValueError(f"t_start must be in [0,{total_timesteps - 1}]")
    base = list(
        reversed(
            torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
            .int()
            .tolist()
        )
    )
    times = [int(t_start), *[int(value) for value in base if int(value) < int(t_start)]]
    deduplicated: list[int] = []
    for value in times:
        if not deduplicated or value != deduplicated[-1]:
            deduplicated.append(value)
    if deduplicated[-1] != -1:
        deduplicated.append(-1)
    pairs = tuple(zip(deduplicated[:-1], deduplicated[1:]))
    if not pairs or pairs[0][0] != t_start or pairs[-1][1] != -1:
        raise AssertionError("Invalid restart DDIM schedule")
    if any(current <= following for current, following in pairs):
        raise AssertionError("DDIM time pairs must be strictly descending")
    return pairs


def hard_incumbent_latent(
    labels: torch.Tensor, num_classes: int, scale: float
) -> torch.Tensor:
    if labels.ndim != 1:
        raise ValueError("Incumbent labels must be one-dimensional")
    if int(labels.min()) < 0 or int(labels.max()) >= num_classes:
        raise ValueError("Incumbent labels fall outside the class vocabulary")
    one_hot = torch_f.one_hot(labels.long(), num_classes=num_classes).T.unsqueeze(0).float()
    return (one_hot * 2.0 - 1.0) * float(scale)


def q_from_alpha_bar(
    x_start: torch.Tensor,
    alpha_bar: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    return alpha_bar.sqrt() * x_start + (1.0 - alpha_bar).sqrt() * noise


def official_postprocess_probabilities(
    probabilities: np.ndarray, postprocess: dict[str, Any]
) -> np.ndarray:
    if probabilities.ndim != 2:
        raise ValueError("Expected classes-by-time probabilities")
    post_type = postprocess.get("type")
    if post_type == "median":
        value = int(postprocess["value"])
        smoothed = np.zeros_like(probabilities)
        for class_id in range(probabilities.shape[0]):
            smoothed[class_id] = median_filter(probabilities[class_id], size=value)
        probabilities = smoothed / np.maximum(smoothed.sum(axis=0, keepdims=True), 1e-12)
    elif post_type not in (None, "none"):
        raise ValueError(
            "V2 B0 currently supports the Breakfast median-probability postprocess only"
        )
    return probabilities.argmax(axis=0).astype(np.int16)


def _randn_like(value: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        value.shape, device=value.device, dtype=value.dtype, generator=generator
    )


@torch.no_grad()
def masked_ddim_sample(
    model: Any,
    backbone_feats: torch.Tensor,
    incumbent_labels: np.ndarray,
    core_intervals: Sequence[tuple[int, int]],
    *,
    halo_width: int,
    t_start: int,
    pure_noise: bool,
    seed: int,
    postprocess: dict[str, Any],
) -> SamplerOutput:
    incumbent = np.asarray(incumbent_labels, dtype=np.int64)
    masks = build_region_masks(len(incumbent), core_intervals, halo_width)
    if not masks.core.any():
        probabilities = np.eye(int(model.num_classes), dtype=np.float32)[incumbent].T
        return SamplerOutput(
            probabilities=probabilities,
            pre_restore_labels=incumbent.astype(np.int16),
            labels=incumbent.astype(np.int16),
            masks=masks,
            time_pairs=tuple(),
        )
    if backbone_feats.ndim != 3 or backbone_feats.shape[-1] != len(incumbent):
        raise ValueError("Backbone features and incumbent sequence are not frame-aligned")
    device = backbone_feats.device
    labels_tensor = torch.as_tensor(incumbent, device=device, dtype=torch.long)
    x_incumbent = hard_incumbent_latent(
        labels_tensor, int(model.num_classes), float(model.scale)
    ).to(device=device, dtype=backbone_feats.dtype)
    pairs = ddim_time_pairs(int(model.num_timesteps), int(model.sampling_timesteps), t_start)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    context_noise = _randn_like(x_incumbent, generator)
    if pure_noise:
        x_time = _randn_like(x_incumbent, generator)
    else:
        x_time = q_from_alpha_bar(
            x_incumbent, model.alphas_cumprod[int(t_start)], context_noise
        )
    exterior = torch.as_tensor(masks.exterior, device=device).view(1, 1, -1)
    x_start = x_incumbent
    for time, time_next in pairs:
        context_now = q_from_alpha_bar(
            x_incumbent, model.alphas_cumprod[int(time)], context_noise
        )
        x_time = torch.where(exterior, context_now, x_time)
        time_cond = torch.full((1,), int(time), device=device, dtype=torch.long)
        pred_noise, x_start = model.model_predictions(backbone_feats, x_time, time_cond)
        if time_next < 0:
            x_time = torch.where(exterior, x_incumbent, x_start)
            continue
        alpha = model.alphas_cumprod[int(time)]
        alpha_next = model.alphas_cumprod[int(time_next)]
        eta = float(model.ddim_sampling_eta)
        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        coefficient = (1 - alpha_next - sigma**2).clamp_min(0).sqrt()
        step_noise = _randn_like(x_time, generator)
        x_next = x_start * alpha_next.sqrt() + coefficient * pred_noise + sigma * step_noise
        context_next = q_from_alpha_bar(x_incumbent, alpha_next, context_noise)
        x_time = torch.where(exterior, context_next, x_next)

    probabilities_tensor = ((x_time / float(model.scale)) + 1.0) / 2.0
    probabilities_tensor = probabilities_tensor.clamp_min(0)
    probabilities_tensor = probabilities_tensor / probabilities_tensor.sum(dim=1, keepdim=True).clamp_min(1e-12)
    probabilities = probabilities_tensor[0].float().cpu().numpy()
    pre_restore = official_postprocess_probabilities(probabilities, postprocess)
    deployed = pre_restore.copy()
    # The halo is a generative/postprocessing buffer, not a deployable edit.
    # Restoring all non-core frames is the post-postprocessing invariance contract.
    deployed[~masks.core] = incumbent[~masks.core]
    return SamplerOutput(
        probabilities=probabilities,
        pre_restore_labels=pre_restore,
        labels=deployed.astype(np.int16),
        masks=masks,
        time_pairs=pairs,
    )


def collapse_labels(labels: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    for value in labels:
        value = int(value)
        if not result or value != result[-1]:
            result.append(value)
    return tuple(result)


def segmental_edit_distance(left: Sequence[int], right: Sequence[int]) -> float:
    a = collapse_labels(left)
    b = collapse_labels(right)
    if not a and not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, value_a in enumerate(a, start=1):
        current = [i]
        for j, value_b in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (value_a != value_b),
                )
            )
        previous = current
    return float(previous[-1] / max(len(a), len(b), 1))


def cluster_medoid_indices(
    sequences: Sequence[Sequence[int]], threshold: float = 0.25
) -> list[tuple[list[int], int]]:
    if not sequences:
        return []
    size = len(sequences)
    distances = np.zeros((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(left + 1, size):
            value = segmental_edit_distance(sequences[left], sequences[right])
            distances[left, right] = distances[right, left] = value
    remaining = set(range(size))
    clusters: list[list[int]] = []
    while remaining:
        root = min(remaining)
        component = {root}
        frontier = [root]
        remaining.remove(root)
        while frontier:
            current = frontier.pop()
            neighbors = [
                other for other in sorted(remaining) if distances[current, other] <= threshold
            ]
            for other in neighbors:
                remaining.remove(other)
                component.add(other)
                frontier.append(other)
        clusters.append(sorted(component))
    result: list[tuple[list[int], int]] = []
    for cluster in clusters:
        medoid = min(cluster, key=lambda index: (float(distances[index, cluster].sum()), index))
        result.append((cluster, medoid))
    return result


def sample_candidate_medoids(
    model: Any,
    backbone_feats: torch.Tensor,
    incumbent_labels: np.ndarray,
    core_intervals: Sequence[tuple[int, int]],
    *,
    halo_width: int,
    t_start: int,
    pure_noise: bool,
    base_seed: int,
    postprocess: dict[str, Any],
    max_samples: int = 15,
    min_samples: int = 7,
    no_new_patience: int = 5,
    cluster_threshold: float = 0.25,
) -> CandidateBatch:
    """Sequentially sample, stop on trace saturation, and return cluster medoids."""
    if not 1 <= min_samples <= max_samples or no_new_patience < 1:
        raise ValueError("Invalid sequential-sampling early-stop contract")
    outputs: list[SamplerOutput] = []
    seen: set[tuple[int, ...]] = set()
    consecutive_no_new = 0
    stopped_early = False
    for sample_index in range(max_samples):
        output = masked_ddim_sample(
            model,
            backbone_feats,
            incumbent_labels,
            core_intervals,
            halo_width=halo_width,
            t_start=t_start,
            pure_noise=pure_noise,
            seed=int(base_seed) + sample_index,
            postprocess=postprocess,
        )
        outputs.append(output)
        trace = collapse_labels(output.labels)
        if trace in seen:
            consecutive_no_new += 1
        else:
            seen.add(trace)
            consecutive_no_new = 0
        if len(outputs) >= min_samples and consecutive_no_new >= no_new_patience:
            stopped_early = True
            break
    clustered = cluster_medoid_indices(
        [output.labels for output in outputs], threshold=cluster_threshold
    )
    return CandidateBatch(
        samples=tuple(outputs),
        clusters=tuple(tuple(cluster) for cluster, _ in clustered),
        medoid_indices=tuple(int(medoid) for _, medoid in clustered),
        stopped_early=stopped_early,
    )
