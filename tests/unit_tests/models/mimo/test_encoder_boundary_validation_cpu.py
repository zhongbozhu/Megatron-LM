# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Gloo checks the explicit encoder boundary without CUDA or model construction."""

import math
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from examples.mimo.full_gradient_validation import validate_encoder_boundary


def _check_boundary(rank, directory):
    root = Path(directory)
    dist.init_process_group(
        'gloo', init_method=f'file://{root / "rendezvous"}', rank=rank, world_size=2
    )
    try:
        # The empty rank must still join every comparison and failure collective.
        features = torch.arange(1, 7, dtype=torch.bfloat16).reshape(2, 3)[: 2 if rank == 0 else 0]
        features.requires_grad_()
        gradient = torch.full(features.shape, 8.0, dtype=torch.float32)
        media = [
            dict(image_id=i, source_id=str(i), offset=i, length=1) for i in range(len(features))
        ]
        original_features, original_gradient = features.detach().clone(), gradient.clone()

        def check(current=features, incoming=gradient, images=media, **kwargs):
            return validate_encoder_boundary(
                current, incoming, images, root / 'reference', step=3, num_tokens=8, **kwargs
            )

        written = check(write_reference=True)
        assert written['components']['features']['elements'] == 6
        assert written['components']['features']['l2'] == pytest.approx(math.sqrt(91))
        assert written['components']['gradient']['l2'] == pytest.approx(math.sqrt(6))
        saved = torch.load(
            root / 'reference' / f'step00000003-rank{rank:05d}.pt', weights_only=True
        )
        torch.testing.assert_close(saved['gradient'], torch.ones_like(gradient), rtol=0, atol=0)
        exact = check()
        assert exact['all_finite']
        assert all(value['relative_l2'] == 0 for value in exact['components'].values())
        assert features.grad is None
        torch.testing.assert_close(features.detach(), original_features, rtol=0, atol=0)
        torch.testing.assert_close(gradient, original_gradient, rtol=0, atol=0)

        changed = features.detach().clone()
        if rank == 0:
            changed[0, 0] += 0.5
        difference = check(current=changed, incoming=gradient * 2)
        assert difference['components']['features']['relative_l2'] == pytest.approx(
            0.5 / math.sqrt(91)
        )
        assert difference['components']['features']['max_abs_error'] == 0.5
        assert difference['components']['gradient']['relative_l2'] == 1
        if rank == 0:
            changed[0, 0] = math.nan
        assert not check(current=changed)['all_finite']

        # A single producer's wrong image identity must fail on both ranks.
        reordered = list(reversed(media))
        with pytest.raises(RuntimeError, match='Encoder boundary validation failed on 1 ranks'):
            check(images=reordered)
        with pytest.raises(RuntimeError, match='Encoder boundary validation failed on 2 ranks'):
            check(write_reference=True)
        if rank == 0:
            (root / 'reference' / 'step00000003-rank00000.pt').unlink()
        dist.barrier()
        with pytest.raises(RuntimeError, match='Encoder boundary validation failed on 1 ranks'):
            check()
    finally:
        dist.destroy_process_group()


def test_encoder_boundary_gloo_empty_producer_normalization_and_failures(tmp_path):
    mp.spawn(_check_boundary, args=(str(tmp_path),), nprocs=2, join=True)
