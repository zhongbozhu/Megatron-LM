# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Opt-in, complete native checkpoint/resume state observation.

Set ``MIMO_RESUME_DIAGNOSTICS_DIR`` for the native MIMO entrypoint. Every tensor
is hashed in full, in bounded CPU chunks; no sampled values stand in for model
parameters or Adam moments. Ownership metadata makes comparisons rank/shard
aware. This observer does not serialize a replacement checkpoint or change RNG,
optimizer, scheduling, or training math.
"""

import argparse
import hashlib
import importlib
import json
import os
import random
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from examples.mimo.optimizer_diagnostics import _children, _owned_optimizer_shards
from megatron.core import tensor_parallel


def _describe(value):
    """Canonical, complete contents; device addresses and tensor IDs are excluded."""
    if isinstance(value, torch.Tensor):
        digest = hashlib.sha256()
        flat = value.detach().reshape(-1)
        for start in range(0, flat.numel(), 1 << 20):
            chunk = flat[start : start + (1 << 20)].to(device='cpu', copy=True).contiguous()
            digest.update(memoryview(chunk.view(torch.uint8).numpy()))
        return dict(dtype=str(value.dtype), shape=list(value.shape), sha256=digest.hexdigest())
    if isinstance(value, np.ndarray):
        return dict(
            dtype=str(value.dtype),
            shape=list(value.shape),
            sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest(),
        )
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _describe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_describe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported resume state type: {type(value).__name__}')


def _rng():
    return _describe(
        dict(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state(),
            cuda_tracker=tensor_parallel.get_cuda_rng_tracker().get_states(),
        )
    )


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite resume evidence: {path}')
    temporary = path.with_suffix('.incomplete')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@torch.no_grad()
def _snapshot(models, optimizer, scheduler, args, iteration):
    parameters, buffers = {}, {}
    parameter_elements = 0
    for index, model in enumerate(models):
        for name, parameter in model.named_parameters():
            parameters[f'{index}/{name}'] = _describe(parameter)
            parameter_elements += parameter.numel()
        # Persistent buffers belong to checkpoints; transient attention caches do not.
        for module_name, module in model.named_modules():
            for name, buffer in module.named_buffers(recurse=False):
                if name not in module._non_persistent_buffers_set:
                    buffers[f'{index}/{module_name}/{name}'] = _describe(buffer)
    shards, groups = _owned_optimizer_shards(models, optimizer)
    owned = []
    for shard in shards:
        owned.append(
            dict(
                ownership={k: v for k, v in shard.metadata.items() if k != 'grad_norm_group'},
                master=_describe(shard.master),
                state=_describe(shard.inner.state.get(shard.master, {})),
            )
        )
    return dict(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        iteration=iteration,
        consumed_train_samples=args.consumed_train_samples,
        consumed_valid_samples=args.consumed_valid_samples,
        skipped_train_samples=args.skipped_train_samples,
        coverage=dict(
            model_parameter_tensors=len(parameters),
            model_parameter_elements=parameter_elements,
            persistent_buffer_tensors=len(buffers),
            optimizer_owned_shards=len(shards),
            optimizer_master_elements=sum(shard.master.numel() for shard in shards),
            optimizer_moment_elements={
                key: sum(
                    shard.inner.state.get(shard.master, {}).get(key, torch.empty(0)).numel()
                    for shard in shards
                )
                for key in ('exp_avg', 'exp_avg_sq')
            },
        ),
        parameters=parameters,
        persistent_buffers=buffers,
        optimizer_shards=owned,
        optimizer_groups=groups,
        optimizer_scalers=[
            _describe(child.grad_scaler.state_dict()) if child.grad_scaler is not None else None
            for child in _children(optimizer)
        ],
        scheduler=_describe(scheduler.state_dict()),
        rng=_rng(),
    )


@contextmanager
def observe_native_resume(args):
    """Observe exact save/load boundaries and actual source batches/DCP decisions."""
    directory = os.environ.get('MIMO_RESUME_DIAGNOSTICS_DIR')
    if not directory:
        with observe_source_batches(args):
            yield
        return
    if args.model_provider != 'qwen35_native':
        raise ValueError('Resume diagnostics require native Qwen35 MIMO')
    target = int(os.environ.get('MIMO_RESUME_CHECKPOINT_STEP', '52'))
    steps = {
        int(item)
        for item in os.environ.get('MIMO_RESUME_DIAGNOSTIC_STEPS', '52,53,54,55').split(',')
    }
    from examples.mimo.native_step import NativeSourceIterator
    from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler
    from megatron.training import get_args

    training = importlib.import_module('megatron.training.training')
    originals = dict(
        setup=training.setup_model_and_optimizer,
        save=training.save_checkpoint,
        train_step=training.train_step,
        source=NativeSourceIterator.__next__,
        schedule=DefaultDynamicCPScheduler.get_groups_and_subsamples,
    )
    state = {}

    def write(label, iteration, content):
        path = Path(directory) / f'rank{dist.get_rank():05d}' / f'{label}-{iteration:08d}.json'
        _write(path, content)

    @wraps(originals['setup'])
    def setup(*values, **kwargs):
        models, optimizer, scheduler = originals['setup'](*values, **kwargs)
        state.update(models=models, optimizer=optimizer, scheduler=scheduler)
        runtime_args = get_args()
        write(
            'loaded',
            runtime_args.iteration,
            _snapshot(models, optimizer, scheduler, runtime_args, runtime_args.iteration),
        )
        return models, optimizer, scheduler

    @wraps(originals['save'])
    def save(iteration, models, optimizer, scheduler, *values, **kwargs):
        if iteration == target:
            write(
                'saving', iteration, _snapshot(models, optimizer, scheduler, get_args(), iteration)
            )
        return originals['save'](iteration, models, optimizer, scheduler, *values, **kwargs)

    @wraps(originals['train_step'])
    def train_step(*values, **kwargs):
        iteration = kwargs.get('iteration')
        state['iteration'] = iteration
        try:
            if iteration in steps:
                write(
                    'before',
                    iteration,
                    _snapshot(
                        **{k: state[k] for k in ('models', 'optimizer', 'scheduler')},
                        args=get_args(),
                        iteration=iteration,
                    ),
                )
            result = originals['train_step'](*values, **kwargs)
            if iteration in steps:
                write(
                    'outcome',
                    iteration,
                    _describe(
                        dict(
                            loss=result[0],
                            skipped=result[1],
                            grad_norm=result[5],
                            zero_gradients=result[6],
                            scheduled_microbatches=result[8],
                        )
                    ),
                )
            return result
        finally:
            state.pop('iteration', None)

    @wraps(originals['source'])
    def source(iterator):
        result = originals['source'](iterator)
        if state.get('iteration') in steps:
            write('source', state['iteration'], _describe(result))
        return result

    @wraps(originals['schedule'])
    def schedule(scheduler, lengths):
        result = originals['schedule'](scheduler, lengths)
        if state.get('iteration') in steps:
            write(
                'schedule', state['iteration'], _describe(dict(lengths=lengths, assignments=result))
            )
        return result

    training.setup_model_and_optimizer = setup
    training.save_checkpoint = save
    training.train_step = train_step
    NativeSourceIterator.__next__ = source
    DefaultDynamicCPScheduler.get_groups_and_subsamples = schedule
    try:
        yield
    finally:
        training.setup_model_and_optimizer = originals['setup']
        training.save_checkpoint = originals['save']
        training.train_step = originals['train_step']
        NativeSourceIterator.__next__ = originals['source']
        DefaultDynamicCPScheduler.get_groups_and_subsamples = originals['schedule']


@contextmanager
def observe_source_batches(args):
    """Lightweight source/plan evidence for natural routing and performance runs."""
    directory = os.environ.get('MIMO_SOURCE_AUDIT_DIR')
    if not directory:
        yield
        return
    if args.model_provider != 'qwen35_native':
        raise ValueError('Source auditing requires native Qwen35 MIMO')
    from examples.mimo.native_step import NativeSourceIterator
    from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler, DpBalancedScheduler

    original_source = NativeSourceIterator.__next__
    originals = {
        cls: cls.get_groups_and_subsamples
        for cls in (DpBalancedScheduler, DefaultDynamicCPScheduler)
    }
    pending, visits = {}, {}

    @wraps(original_source)
    def source(iterator):
        result = original_source(iterator)
        if dist.get_rank() == 0:
            if pending:
                raise RuntimeError('A source batch was not observed by its packing scheduler')
            description = _describe(result)
            pending.update(
                step=result['step'],
                split=getattr(iterator.dataset, 'split', 'train'),
                source=description,
                source_sha256=hashlib.sha256(
                    json.dumps(description, sort_keys=True, separators=(',', ':')).encode()
                ).hexdigest(),
            )
        return result

    def observe_schedule(original):
        @wraps(original)
        def schedule(scheduler, lengths):
            result = original(scheduler, lengths)
            if dist.get_rank() == 0 and pending:
                record = dict(pending, assignments=_describe(result), lengths=_describe(lengths))
                address = (record['split'], record['step'])
                occurrence = visits.get(address, 0)
                visits[address] = occurrence + 1
                record['occurrence'] = occurrence
                # Validation and final test may deliberately revisit the same heldout batch.
                suffix = f'-repeat{occurrence:03d}' if occurrence else ''
                path = Path(directory) / f'{address[0]}-{address[1]:08d}{suffix}.json'
                _write(path, record)
                pending.clear()
            return result

        return schedule

    NativeSourceIterator.__next__ = source
    for cls, original in originals.items():
        cls.get_groups_and_subsamples = observe_schedule(original)
    try:
        yield
    finally:
        NativeSourceIterator.__next__ = original_source
        for cls, original in originals.items():
            cls.get_groups_and_subsamples = original


def _differences(left, right, path=''):
    if type(left) is not type(right):
        return [path + ': type']
    if isinstance(left, dict):
        result = [path + ': keys'] if left.keys() != right.keys() else []
        for key in left.keys() & right.keys():
            result.extend(_differences(left[key], right[key], f'{path}/{key}'))
        return result
    if isinstance(left, list):
        result = [path + ': length'] if len(left) != len(right) else []
        for index, (a, b) in enumerate(zip(left, right)):
            result.extend(_differences(a, b, f'{path}/{index}'))
        return result
    return [] if left == right else [path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--continuous', type=Path, required=True)
    parser.add_argument('--resumed', type=Path, required=True)
    parser.add_argument('--checkpoint-step', type=int, default=52)
    parser.add_argument('--steps', type=int, nargs='+', default=[52, 53, 54, 55])
    parser.add_argument('--world-size', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(exact_restore=True, exact_source_schedule=True, comparisons=[])
    for rank in range(args.world_size):

        def compare(label, step, left_label=None, right_label=None):
            name = f'rank{rank:05d}'
            a = args.continuous / name / f'{left_label or label}-{step:08d}.json'
            b = args.resumed / name / f'{right_label or label}-{step:08d}.json'
            differences = _differences(json.loads(a.read_text()), json.loads(b.read_text()))
            report['comparisons'].append(
                dict(
                    rank=rank,
                    label=label,
                    step=step,
                    exact=not differences,
                    difference_count=len(differences),
                    first_differences=differences[:30],
                )
            )
            return not differences

        report['exact_restore'] &= compare('restore', args.checkpoint_step, 'saving', 'loaded')
        for step in args.steps:
            for label in ('source', 'schedule'):
                report['exact_source_schedule'] &= compare(label, step)
            for label in ('before', 'outcome'):
                compare(label, step)
    report['passed_state_and_progress_contract'] = (
        report['exact_restore'] and report['exact_source_schedule']
    )
    report['interpretation'] = (
        'Restore and source/schedule require exact equality. Subsequent state/outcome equality is diagnostic; natural-routing numerical drift is not automatically a checkpoint bug.'
    )
    _write(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'comparisons'}, indent=2))
    raise SystemExit(0 if report['passed_state_and_progress_contract'] else 1)


if __name__ == '__main__':
    main()
