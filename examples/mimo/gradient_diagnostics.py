# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Small, deterministic diagnostics before native DDP gradient finalization.

These sampled linear projections are useful for parallel-layout comparisons;
they are not a replacement for the optimizer's actual global gradient norm.
"""

import hashlib
import math
import random
from functools import lru_cache

import torch
import torch.distributed as dist

_COMPONENTS = ('vision', 'merger', 'decoder', 'routers', 'experts', 'shared_experts', 'mtp')


def _component_names(name):
    while name.startswith('module.'):
        name = name[len('module.') :]
    if name.startswith('modality_submodules.'):
        return name, ('merger',) if '.merger.' in name else ('vision',)
    components = ['decoder']
    if '.router.' in name:
        components.append('routers')
    if '.shared_experts.' in name:
        components.append('shared_experts')
    elif '.experts.' in name:
        components.append('experts')
    if '.mtp.' in name or name.startswith('mtp.'):
        components.append('mtp')
    return name, tuple(components)


@lru_cache(maxsize=8192)
def _sampling_plan(name, numel, expert_owner, samples, projections):
    """CPU-only plans; no retained gradient, model or GPU tensor references."""
    identity = f'{name}|expert_owner={expert_owner}'.encode()
    seed = int.from_bytes(hashlib.blake2b(identity, digest_size=8).digest(), 'little')
    rng = random.Random(seed)
    count = min(samples, numel)
    indices = tuple(range(numel)) if count == numel else tuple(rng.sample(range(numel), count))
    # The first projection is an unsigned sampled checksum. Independent signed
    # projections make cancellation or expert-shard swaps easier to detect.
    signs = [(1.0,) * count]
    signs.extend(
        tuple(1.0 if rng.getrandbits(1) else -1.0 for _ in indices) for _ in range(projections - 1)
    )
    return indices, tuple(signs)


def _local_squared_l2(gradient, chunk_elements=1 << 20):
    """Bound temporary memory even if a backend casts FP32 input to FP64."""
    if gradient.numel() <= chunk_elements:
        return torch.linalg.vector_norm(gradient, dtype=torch.float64).square()
    if gradient.is_contiguous():
        chunks = gradient.view(-1).split(chunk_elements)
    else:
        dimension = max(range(gradient.ndim), key=lambda dim: gradient.shape[dim])
        elements_per_slice = gradient.numel() // gradient.shape[dimension]
        width = max(1, chunk_elements // elements_per_slice)
        chunks = gradient.split(width, dim=dimension)
    total = torch.zeros((), device=gradient.device, dtype=torch.float64)
    for chunk in chunks:
        total.add_(_local_squared_l2(chunk, chunk_elements))
    return total


@torch.no_grad()
def collect_gradient_diagnostics(
    model,
    dp_cp_group,
    global_supervised_tokens,
    *,
    ep_group,
    samples_per_parameter=32,
    projections=8,
    required_components=(),
):
    """Fingerprint raw, unscaled main gradients without changing native buffers.

    Call after decoder and encoder backwards, before ``finalize_model_grads``,
    with ``calculate_per_token_loss=True`` and grad-reduce overlap disabled.
    Native DDP then still holds local raw token-sum contributions in main_grad.
    This helper SUM-reduces only small projection vectors and divides them by
    the same step-global supervised count as native gradient finalization.

    TP=PP=1 is assumed. Expert parameters use their EP owner in the sampling
    identity: distinct shards remain distinct, while expert-DP replicas project
    the same coordinates. Comparisons must keep the model names and EP layout
    fixed. Dense parameter contributions use identical sampling on every rank.

    Components overlap intentionally: decoder includes its MTP/MoE submodules.
    ``local_raw_l2`` is this rank's norm *before* normalization/reduction and is
    explicitly not a global gradient norm. Finite checking covers all entries,
    including entries outside the sparse sampling plan. Required components must
    exist and have a nonzero contribution on at least one rank.
    """
    if samples_per_parameter <= 0 or projections <= 0:
        raise ValueError('Gradient diagnostics need positive sample and projection counts')
    unknown = set(required_components) - set(_COMPONENTS)
    if unknown:
        raise ValueError(f'Unknown gradient diagnostic components: {sorted(unknown)}')
    named_parameters = list(model.named_parameters())
    if not named_parameters:
        raise ValueError('Gradient diagnostics require model parameters')
    device = named_parameters[0][1].device
    denominator = torch.as_tensor(global_supervised_tokens, device=device, dtype=torch.float64)
    if denominator.numel() != 1 or not torch.isfinite(denominator).all() or denominator.item() <= 0:
        raise ValueError('global_supervised_tokens must be a positive finite scalar')
    values = torch.zeros((len(_COMPONENTS), projections), device=device, dtype=torch.float64)
    local_squares = torch.zeros(len(_COMPONENTS), device=device, dtype=torch.float64)
    # Parameter count, gradient count, sampled values, nonfinite parameters.
    statistics = torch.zeros((len(_COMPONENTS), 4), device=device, dtype=torch.float64)
    component_index = {name: index for index, name in enumerate(_COMPONENTS)}

    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        name, components = _component_names(name)
        indices = [component_index[component] for component in components]
        statistics[indices, 0] += 1
        gradient = getattr(parameter, 'main_grad', None)
        if gradient is None:
            gradient = parameter.grad
        if gradient is None or gradient.numel() == 0:
            continue
        # Full-entry finite checking and local norms use bounded-size FP64
        # reductions; diagnostics must not allocate a giant FP64 embedding copy.
        squared_norm = _local_squared_l2(gradient)
        local_squares[indices] += squared_norm
        statistics[indices, 1] += 1
        statistics[indices, 3] += (~torch.isfinite(squared_norm)).to(torch.float64)
        is_expert = not getattr(parameter, 'allreduce', True)
        if is_expert and ep_group is None:
            raise ValueError('An explicit EP group is required for expert gradient diagnostics')
        owner = ep_group.rank() if is_expert else None
        selected, signs = _sampling_plan(
            name, gradient.numel(), owner, samples_per_parameter, projections
        )
        selected = torch.tensor(selected, device=device, dtype=torch.long)
        if gradient.is_contiguous():
            sampled = gradient.view(-1).index_select(0, selected)
        else:
            # Avoid materializing a potentially huge contiguous gradient copy.
            sampled = gradient[torch.unravel_index(selected, gradient.shape)]
        projected = torch.tensor(signs, device=device, dtype=torch.float64) @ sampled.double()
        values[indices] += projected
        statistics[indices, 2] += sampled.numel()

    local_norms = local_squares.sqrt().tolist()
    active = (local_squares > 0).to(torch.float64)
    # One compact collective; no parameter-gradient all-gather or mutation.
    reduced = torch.cat((values, statistics, active[:, None]), dim=1)
    dist.all_reduce(reduced, group=dp_cp_group)
    values = reduced[:, :projections] / denominator
    statistics = reduced[:, projections : projections + 4]
    active = reduced[:, -1]
    result = {
        'kind': 'sampled_linear_gradient_projection',
        'samples_per_parameter': samples_per_parameter,
        'num_projections': projections,
        'supervised_tokens': int(denominator.item()),
        'local_rank': dist.get_rank(),
        'components': {},
    }
    failed = []
    for index, component in enumerate(_COMPONENTS):
        counts = statistics[index].tolist()
        finite = counts[3] == 0 and all(math.isfinite(value) for value in values[index].tolist())
        result['components'][component] = {
            'projections': values[index].tolist(),
            'all_finite': finite,
            'local_raw_l2': local_norms[index],
            'parameter_instances': int(counts[0]),
            'gradient_instances': int(counts[1]),
            'sampled_values': int(counts[2]),
            'active_ranks': int(active[index].item()),
        }
        if not finite:
            failed.append(f'{component}: nonfinite gradient')
        elif component in required_components and (counts[1] == 0 or active[index].item() == 0):
            failed.append(f'{component}: missing or zero gradient contributions')
    if failed:
        raise RuntimeError('Gradient diagnostics failed: ' + '; '.join(failed))
    return result
