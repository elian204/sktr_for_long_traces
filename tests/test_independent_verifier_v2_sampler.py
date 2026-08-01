from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "independent_acceptance_verifier"
sys.path.insert(0, str(SCRIPT_DIR))

from masked_diffact_sampler import (  # noqa: E402
    build_region_masks,
    cluster_medoid_indices,
    ddim_time_pairs,
    hard_incumbent_latent,
    masked_ddim_sample,
    merge_intervals,
    q_from_alpha_bar,
    sample_candidate_medoids,
)


class FakeModel:
    num_classes = 3
    scale = 0.5
    num_timesteps = 1000
    sampling_timesteps = 25
    ddim_sampling_eta = 1.0

    def __init__(self) -> None:
        self.alphas_cumprod = torch.linspace(0.999, 0.001, 1000)
        self.calls = 0

    def model_predictions(self, backbone, value, time):
        self.calls += 1
        start = value.clamp(-self.scale, self.scale) * 0.5
        noise = torch.zeros_like(value)
        return noise, start


def test_intervals_merge_and_three_regions_partition() -> None:
    assert merge_intervals([(5, 10), (9, 12), (12, 14)], 20) == [(5, 14)]
    masks = build_region_masks(20, [(5, 10), (9, 12)], 2)
    assert int(masks.core.sum()) == 7
    assert int(masks.halo.sum()) == 4
    assert np.all(masks.core | masks.halo | masks.exterior)


def test_restart_schedule_starts_exactly_at_requested_timestep() -> None:
    for start in (250, 500, 750, 999):
        pairs = ddim_time_pairs(1000, 25, start)
        assert pairs[0][0] == start
        assert pairs[-1][1] == -1
        assert all(left > right for left, right in pairs)


def test_alpha_bar_forward_noise_formula() -> None:
    start = torch.ones(1, 2, 3)
    noise = torch.zeros_like(start)
    result = q_from_alpha_bar(start, torch.tensor(0.25), noise)
    assert torch.allclose(result, torch.full_like(start, 0.5))


def test_hard_incumbent_is_exact_scaled_one_hot() -> None:
    result = hard_incumbent_latent(torch.tensor([0, 2]), 3, 0.5)
    assert result.shape == (1, 3, 2)
    assert torch.equal(result[0, :, 0], torch.tensor([0.5, -0.5, -0.5]))


def test_empty_mask_is_identity_without_model_call() -> None:
    model = FakeModel()
    incumbent = np.asarray([0, 0, 1, 1, 2, 2])
    output = masked_ddim_sample(
        model,
        torch.zeros(1, 4, len(incumbent)),
        incumbent,
        [],
        halo_width=8,
        t_start=500,
        pure_noise=False,
        seed=1,
        postprocess={"type": "median", "value": 3},
    )
    assert model.calls == 0
    assert np.array_equal(output.labels, incumbent)


def test_seeded_replay_and_postprocess_exterior_invariance() -> None:
    incumbent = np.asarray([0] * 8 + [1] * 8 + [2] * 8)
    kwargs = dict(
        model=FakeModel(),
        backbone_feats=torch.zeros(1, 4, len(incumbent)),
        incumbent_labels=incumbent,
        core_intervals=[(9, 14)],
        halo_width=8,
        t_start=500,
        pure_noise=False,
        seed=42,
        postprocess={"type": "median", "value": 3},
    )
    first = masked_ddim_sample(**kwargs)
    kwargs["model"] = FakeModel()
    second = masked_ddim_sample(**kwargs)
    assert np.array_equal(first.probabilities, second.probabilities)
    assert np.array_equal(first.labels, second.labels)
    assert np.array_equal(first.labels[~first.masks.core], incumbent[~first.masks.core])


def test_segmental_cluster_uses_medoid_not_frame_vote() -> None:
    sequences = [[1, 1, 2, 2], [1, 2], [1, 1, 3, 3]]
    clusters = cluster_medoid_indices(sequences, threshold=0.25)
    assert clusters[0][0] == [0, 1]
    assert clusters[0][1] == 0
    assert clusters[1][0] == [2]


def test_sequential_sampling_stops_after_trace_saturation() -> None:
    incumbent = np.asarray([0] * 8 + [1] * 8)
    batch = sample_candidate_medoids(
        FakeModel(),
        torch.zeros(1, 4, len(incumbent)),
        incumbent,
        [(4, 8)],
        halo_width=2,
        t_start=250,
        pure_noise=False,
        base_seed=10,
        postprocess={"type": "median", "value": 3},
        max_samples=15,
        min_samples=7,
        no_new_patience=5,
    )
    assert 7 <= len(batch.samples) <= 15
    assert len(batch.medoid_indices) == len(batch.clusters)
    assert all(index < len(batch.samples) for index in batch.medoid_indices)
