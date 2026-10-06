# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Opt-in per-rank timing of native MIMO steps, without changing their math.

Stage intervals are nested (prepare includes encoder and auxiliary forward;
finalize includes reverse transport and reductions), so they must not be summed.
The synchronized whole-step wall time includes source preprocessing and optimizer.
"""

import gzip
import importlib
import json
import os
import shutil
import time
from contextlib import contextmanager, nullcontext
from functools import wraps
from pathlib import Path


@contextmanager
def observe_native_performance(args):
    """Opt-in timings and one CPU/CUDA trace of an absolute source batch step.

    MIMO_TORCH_PROFILE_DIR and MIMO_TORCH_PROFILE_STEP enable the trace. The
    profiled update and its successor are excluded from throughput comparisons.
    """
    directory = os.environ.get('MIMO_PERFORMANCE_DIR')
    profile_directory = os.environ.get('MIMO_TORCH_PROFILE_DIR')
    if not directory and not profile_directory:
        yield
        return
    profile_step = None
    if profile_directory:
        profile_step = int(os.environ['MIMO_TORCH_PROFILE_STEP'])
        if profile_step < 0:
            raise ValueError('MIMO_TORCH_PROFILE_STEP must be a nonnegative source step')

    import torch
    import torch.distributed as dist

    from megatron.core.utils import get_attr_wrapped_model

    training = importlib.import_module('megatron.training.training')
    original_setup = training.setup_model_and_optimizer
    original_step = training.train_step
    installed = []
    coordinator = None
    events = None
    profiling = False
    profile_complete = False
    profile_context = {}
    module_scopes = []
    round_metadata = []

    def scope(label):
        if not profile_directory:
            return nullcontext()
        context = ','.join(f'{key}={value}' for key, value in profile_context.items())
        return torch.profiler.record_function(f'mimo::{label}[{context}]')

    def install_profile_scopes():
        """Install once before warmup so capture does not change callable identities."""

        def install(obj, name, wrapper):
            original = getattr(obj, name)
            setattr(obj, name, wrapper(original))
            installed.append((obj, name, original))

        def annotate_module(label, *, disable_compile=False):
            def decorate(original):
                @wraps(original)
                def annotated(*positional, **keywords):
                    with scope(label):
                        return original(*positional, **keywords)

                return torch.compiler.disable(annotated) if disable_compile else annotated

            return decorate

        def annotate_transfer(original):
            @wraps(original)
            def annotated(item):
                profile_context.update(
                    round=item['round_id'],
                    cp=item['cp_group'].size(),
                    phase='feature_transfer',
                    local_tokens=item['kwargs']['input_ids'].numel() // item['cp_group'].size(),
                    samples=':'.join(str(sid) for sid in item['diagnostic_sample_ids']),
                )
                return original(item)

            return annotated

        def annotate_round(original):
            @wraps(original)
            def annotated(item, phase, transfer):
                local_tokens = item['kwargs']['input_ids'].numel() // item['cp_group'].size()
                sample_ids = list(item['diagnostic_sample_ids'])
                profile_context.update(
                    round=item['round_id'],
                    cp=item['cp_group'].size(),
                    phase=phase,
                    local_tokens=local_tokens,
                    samples=':'.join(str(sid) for sid in sample_ids),
                )
                if profiling:
                    round_metadata.append(
                        dict(
                            round=item['round_id'],
                            cp=item['cp_group'].size(),
                            phase=phase,
                            local_tokens=local_tokens,
                            sample_ids=sample_ids,
                        )
                    )
                return original(item, phase, transfer)

            return annotated

        install(coordinator, '_feature_transfer', annotate_transfer)
        install(coordinator, '_begin_diagnostics', annotate_round)
        for name, module in coordinator.model.language_model.named_modules():
            kind = type(module).__name__
            if kind in ('GatedDeltaNet', 'GatedDeltaNet2', 'SelfAttention', 'MoELayer') or (
                name.rsplit('.', 1)[-1] in ('router', 'experts', 'shared_experts')
            ):
                label = f'{kind}:{name}'
                module_scopes.append(label)
                install(module, 'forward', annotate_module(label))
            if kind == 'MoELayer':
                dispatcher = module.token_dispatcher
                if type(dispatcher).__name__ == 'MoEFlexTokenDispatcher':
                    for method in (
                        'dispatch_preprocess',
                        'token_dispatch',
                        'dispatch_postprocess',
                        'combine_preprocess',
                        'token_combine',
                        'combine_postprocess',
                    ):
                        label = f'{name}.dispatcher.{method}'
                        module_scopes.append(label)
                        install(dispatcher, method, annotate_module(label))
                    manager = dispatcher._comm_manager
                    if type(manager).__name__ == '_HybridEPManager':
                        # Metadata includes the EP-wide MAX and host scalar read.
                        for method in ('setup_metadata', 'dispatch', 'combine'):
                            label = f'{name}.hybridep.{method}'
                            module_scopes.append(label)
                            install(
                                manager,
                                method,
                                # Preserve setup_metadata's existing eager boundary;
                                # other module/dispatcher paths remain compilable.
                                annotate_module(label, disable_compile=method == 'setup_metadata'),
                            )

    @contextmanager
    def capture_profile(source_step):
        nonlocal profiling, profile_complete
        if not profile_directory or profile_complete or source_step != profile_step:
            with scope('native_train_step'):
                yield None
            return
        try:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=False,
                profile_memory=False,
            ) as profiler:
                profiling = True
                with scope('native_train_step'):
                    yield profiler
            # Keep metadata attached to this profile, independent of the next round.
            profiler.mimo_module_scopes = module_scopes
            profiler.mimo_round_metadata = round_metadata
            profile_complete = True
        finally:
            profiling = False

    def export_profile(profiler, record):
        """Export outside the measured update, then align ranks before continuing."""
        source_step = record['source_step']
        if source_step != profile_step:
            raise RuntimeError(
                f'Profile captured source step {source_step}, expected {profile_step}'
            )
        path = Path(profile_directory)
        path.mkdir(parents=True, exist_ok=True)
        prefix = path / f'rank{dist.get_rank():05d}-step{source_step:08d}'
        trace = Path(f'{prefix}.trace.json')
        profiler.export_chrome_trace(str(trace))
        compressed = Path(f'{trace}.gz')
        with (
            trace.open('rb') as source,
            gzip.open(compressed, 'wb', compresslevel=1) as destination,
        ):
            shutil.copyfileobj(source, destination)
        trace.unlink()
        averages = profiler.key_averages(group_by_input_shape=True)
        operators = [
            dict(
                name=item.key,
                input_shapes=item.input_shapes,
                calls=item.count,
                cpu_time_total_us=item.cpu_time_total,
                self_cpu_time_total_us=item.self_cpu_time_total,
                device_time_total_us=item.device_time_total,
                self_device_time_total_us=item.self_device_time_total,
            )
            for item in averages
        ]
        Path(f'{prefix}.operators.json').write_text(json.dumps(operators) + '\n')
        Path(f'{prefix}.operators.txt').write_text(
            averages.table(sort_by='self_device_time_total', row_limit=100) + '\n'
        )
        metadata = dict(
            rank=dist.get_rank(),
            source_step=source_step,
            torch_version=torch.__version__,
            activities=['CPU', 'CUDA'],
            record_shapes=True,
            with_stack=False,
            profile_memory=False,
            trace=compressed.name,
            module_scopes=profiler.mimo_module_scopes,
            decoder_calls=profiler.mimo_round_metadata,
            timing=record,
            throughput_comparable=False,
            notes=[
                'One full native update, including auxiliary forward and backward recomputation.',
                'Module forward scopes nested under autograd checkpoint backward identify recompute.',
                'Operator totals overlap; scopes and communication waits are not additive kernel cost.',
                'Associate kernels with CPU scopes through correlation/external IDs, not time overlap.',
                'Identical profiling wrappers/scopes are installed before all warmup updates.',
                'Export and the subsequent explicit DPxCP barrier occur outside measured update time.',
            ],
        )
        Path(f'{prefix}.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        dist.barrier(group=coordinator.pg.dp_cp)
        torch.cuda.synchronize()

    def wrap_stage(obj, name, label):
        original = getattr(obj, name)

        @wraps(original)
        def measured(*positional, **keywords):
            if events is None:
                return original(*positional, **keywords)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            wall = time.perf_counter()
            with scope(label):
                result = original(*positional, **keywords)
            end.record()
            events.append((label, begin, end, time.perf_counter() - wall))
            return result

        setattr(obj, name, measured)
        installed.append((obj, name, original))

    @wraps(original_setup)
    def setup(*positional, **keywords):
        nonlocal coordinator
        models, optimizer, scheduler = original_setup(*positional, **keywords)
        coordinator = get_attr_wrapped_model(models[0], 'native_mimo_step')
        wrap_stage(coordinator, 'prepare', 'prepare_including_encoder_and_auxiliary')
        wrap_stage(coordinator, 'finalize', 'finalize_including_encoder_backward_and_reduction')
        # config stores bound methods before wrapping; update those entrypoints too.
        coordinator.config.sequence_packing_data_adapter = coordinator.prepare
        coordinator.config.finalize_model_grads_func = coordinator.finalize
        wrap_stage(coordinator.model, 'encode_modalities', 'encoder_forward')
        wrap_stage(coordinator.model.language_model, 'forward', 'decoder_forward_all_passes')
        wrap_stage(coordinator.bridge, 'forward', 'feature_forward_transport')
        wrap_stage(coordinator.bridge, 'backward', 'feature_backward_transport')
        wrap_stage(optimizer, 'step', 'optimizer')
        if profile_directory:
            install_profile_scopes()
        return models, optimizer, scheduler

    @wraps(original_step)
    def step(*positional, **keywords):
        nonlocal events
        source_step = args.consumed_train_samples // args.global_batch_size
        if profile_directory:
            # Keep changing absolute step IDs out of compiler-visible scope strings.
            profile_context.clear()
            profile_context.update(rank=dist.get_rank(), round=-1, cp=0, phase='prepare')
            round_metadata.clear()
        with capture_profile(source_step) as profiler:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            events = []
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            started = time.perf_counter()
            result = original_step(*positional, **keywords)
            end.record()
            torch.cuda.synchronize()
            wall = time.perf_counter() - started
        stages = {}
        for label, first, last, cpu_seconds in events:
            record = stages.setdefault(label, dict(calls=0, cuda_ms=0.0, host_ms=0.0))
            record['calls'] += 1
            record['cuda_ms'] += first.elapsed_time(last)
            record['host_ms'] += cpu_seconds * 1000
        events = None
        record = dict(
            rank=dist.get_rank(),
            source_step=coordinator.step,
            wall_ms=wall * 1000,
            cuda_ms=begin.elapsed_time(end),
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            stages=stages,
            stages_are_nested=True,
            metrics={
                key: coordinator.metrics[key]
                for key in (
                    'source_samples',
                    'source_images',
                    'source_tokens',
                    'supervised_tokens',
                    'cp_sizes',
                    'loss',
                )
                if key in coordinator.metrics
            },
        )
        if profile_directory:
            record['torch_profiled'] = profiler is not None
            record['throughput_comparable'] = source_step not in (profile_step, profile_step + 1)
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            with (path / f'rank{dist.get_rank():05d}.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
        if profiler is not None:
            export_profile(profiler, record)
        return result

    training.setup_model_and_optimizer = setup
    training.train_step = step
    try:
        yield
    finally:
        events = None
        training.setup_model_and_optimizer = original_setup
        training.train_step = original_step
        for obj, name, original in reversed(installed):
            setattr(obj, name, original)
        if coordinator is not None:
            coordinator.config.sequence_packing_data_adapter = coordinator.prepare
            coordinator.config.finalize_model_grads_func = coordinator.finalize


def summarize(suite, output):
    """Compare matched source steps in the four controlled ABBA runs."""
    import math
    import statistics

    runs, identity, by_mode = {}, None, {'static': [], 'dynamic': []}
    for case in ('static_r1', 'dynamic_r1', 'dynamic_r2', 'static_r2'):
        name = f'performance_{case}'
        paths = sorted((suite / 'performance' / name).glob('rank*.jsonl'))
        assert len(paths) == 16, f'{name}: incomplete rank coverage'
        ranks = [[json.loads(line) for line in path.read_text().splitlines()] for path in paths]
        assert all(len(rows) == 10 for rows in ranks), f'{name}: incomplete step coverage'
        assert all([row['source_step'] for row in rows] == list(range(52, 62)) for rows in ranks)
        hashes = [
            json.loads((suite / 'source-audit' / name / f'train-{step:08d}.json').read_text())[
                'source_sha256'
            ]
            for step in range(52, 62)
        ]
        if identity is None:
            identity = hashes
        assert hashes == identity, f'{name}: source batch mismatch'
        steps = []
        for index in range(10):
            rows = [rank[index] for rank in ranks]
            for rank, row in enumerate(rows):
                assert row['rank'] == rank
                assert math.isfinite(row['metrics']['loss'])
                assert all(
                    math.isfinite(row[key]) and row[key] > 0
                    for key in ('wall_ms', 'peak_allocated_gib', 'peak_reserved_gib')
                )
                assert row['metrics']['source_tokens'] == rows[0]['metrics']['source_tokens']
            steps.append(
                dict(
                    source_step=rows[0]['source_step'],
                    wall_ms=max(row['wall_ms'] for row in rows),
                    source_tokens=rows[0]['metrics']['source_tokens'],
                    cp_sizes=rows[0]['metrics']['cp_sizes'],
                    peak_allocated_gib=max(row['peak_allocated_gib'] for row in rows),
                    peak_reserved_gib=max(row['peak_reserved_gib'] for row in rows),
                    rank_peak_allocated_gib=[row['peak_allocated_gib'] for row in rows],
                    stage_max_cuda_ms={
                        stage: max(row['stages'].get(stage, {}).get('cuda_ms', 0) for row in rows)
                        for stage in sorted({key for row in rows for key in row['stages']})
                    },
                )
            )
        measured = steps[2:]
        seconds = sum(step['wall_ms'] for step in measured) / 1000
        runs[case] = dict(
            warmup_steps=steps[:2],
            measured_steps=measured,
            mean_step_seconds=seconds / len(measured),
            source_tokens_per_second=sum(step['source_tokens'] for step in measured) / seconds,
            peak_allocated_gib=max(step['peak_allocated_gib'] for step in measured),
            peak_reserved_gib=max(step['peak_reserved_gib'] for step in measured),
        )
        by_mode[case.rsplit('_', 1)[0]].extend(measured)

    totals = {
        mode: sum(step['wall_ms'] for step in steps) / 1000 for mode, steps in by_mode.items()
    }
    per_source = []
    for index in range(8):
        times = {
            mode: statistics.mean(
                runs[f'{mode}_r{repeat}']['measured_steps'][index]['wall_ms'] for repeat in (1, 2)
            )
            for mode in by_mode
        }
        per_source.append(dict(source_step=index + 54, speedup=times['static'] / times['dynamic']))
    result = dict(
        runs=runs,
        source_identity_exact=True,
        measured_sources=list(range(54, 62)),
        dynamic_speedup_over_static_cp8=totals['static'] / totals['dynamic'],
        per_source_speedup=per_source,
        scope=(
            'Two ABBA repeats on the same 16 GPUs, native optimizer and objective, natural routing. '
            'First two steps excluded; later unseen shapes may still compile. Whole-step timing '
            'includes CPU source audit and preprocessing. Stage intervals overlap and cannot be '
            'summed. Memory is PyTorch allocated/reserved, not total device memory. This compares '
            'static CP8 with DCP at fixed encoder DP16; it does not measure encoder-layout gains.'
        ),
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    summarize(options.suite, options.output)
