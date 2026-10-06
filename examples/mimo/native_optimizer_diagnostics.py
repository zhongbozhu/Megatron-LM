# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Connect optional AdamW observation to the existing native MIMO entrypoint."""

import importlib
from contextlib import ExitStack, contextmanager
from functools import wraps


@contextmanager
def observe_native_optimizer(args):
    """Install opt-in observations around the otherwise unchanged native entrypoint."""
    from examples.mimo.performance_diagnostics import observe_native_performance
    from examples.mimo.resume_diagnostics import observe_native_resume

    with (
        observe_native_resume(args),
        observe_native_performance(args),
        _observe_native_optimizer(args),
    ):
        yield


@contextmanager
def _observe_native_optimizer(args):
    """Observe native setup/step without replacing training or optimizer math.

    The setup function is temporarily wrapped only in explicitly enabled runs.
    Both setup and the optimizer's step method are restored on normal/error exit.
    """
    reference = getattr(args, 'mimo_optimizer_reference', None)
    if not reference:
        yield
        return
    if args.model_provider != 'qwen35_native':
        raise ValueError('Optimizer diagnostics require the native Qwen35 MIMO provider')
    expected = getattr(args, 'mimo_optimizer_expected_step', None)
    if expected is None or expected < 0:
        raise ValueError('Optimizer diagnostics require an explicit nonnegative expected step')
    if not args.mimo_optimizer_report or not args.mimo_optimizer_scratch_dir:
        raise ValueError('Optimizer diagnostics require report and scratch paths')
    from examples.mimo.optimizer_diagnostics import optimizer_step_diagnostics
    from megatron.core.utils import get_attr_wrapped_model

    training = importlib.import_module('megatron.training.training')
    original = training.setup_model_and_optimizer
    installed = False
    with ExitStack() as stack:

        @wraps(original)
        def setup(*setup_args, **setup_kwargs):
            nonlocal installed
            if installed:
                raise RuntimeError('Optimizer diagnostics expect a single native setup')
            models, optimizer, scheduler = original(*setup_args, **setup_kwargs)
            coordinator = get_attr_wrapped_model(models[0], 'native_mimo_step')

            def context():
                # Exclude intentionally changed CP/packing geometry. Starting
                # weights/moments and optimizer settings are checked separately.
                metrics = coordinator.metrics
                keys = (
                    'source_samples',
                    'source_images',
                    'source_tokens',
                    'supervised_tokens',
                    'mtp_tokens',
                )
                result = dict(
                    source_step=coordinator.step,
                    source={key: metrics[key] for key in keys if key in metrics},
                    scheduler=scheduler.state_dict(),
                )
                if 'fixed_routing' in metrics:
                    result['route_sha256'] = metrics['fixed_routing']['route_sha256']
                return result

            stack.enter_context(
                optimizer_step_diagnostics(
                    models,
                    optimizer,
                    reference_dir=reference,
                    write_reference=args.mimo_write_optimizer_reference,
                    scratch_dir=args.mimo_optimizer_scratch_dir,
                    report_path=args.mimo_optimizer_report,
                    expected_step=expected,
                    cold_moment_bands=getattr(args, 'mimo_optimizer_cold_moment_bands', False),
                    group=coordinator.pg.dp_cp,
                    context=context,
                )
            )
            installed = True
            return models, optimizer, scheduler

        training.setup_model_and_optimizer = setup
        try:
            yield
        finally:
            training.setup_model_and_optimizer = original
