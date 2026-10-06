# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""The opt-in adapter must leave native setup and math under native ownership."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from examples.mimo.native_optimizer_diagnostics import observe_native_optimizer


def test_disabled_diagnostics_do_not_import_or_wrap_training(monkeypatch):
    from examples.mimo import native_optimizer_diagnostics as module

    def unexpected(*args):
        raise AssertionError('Disabled diagnostics imported native training')

    monkeypatch.setattr(module.importlib, 'import_module', unexpected)
    with observe_native_optimizer(SimpleNamespace()):
        pass


@pytest.mark.parametrize('raise_inside', [False, True])
def test_native_setup_result_context_and_restoration(monkeypatch, tmp_path, raise_inside):
    observed, events = {}, []
    coordinator = SimpleNamespace(
        step=52,
        metrics={'source_tokens': 123, 'supervised_tokens': 100, 'cp_sizes': [8]},
        pg=SimpleNamespace(dp_cp='group'),
    )
    models = [SimpleNamespace(native_mimo_step=coordinator)]
    optimizer = object()
    scheduler = SimpleNamespace(state_dict=lambda: {'num_steps': 3328})

    def original(*args, **kwargs):
        events.append('native_setup')
        return models, optimizer, scheduler

    @contextmanager
    def observer(got_models, got_optimizer, **kwargs):
        assert got_models is models and got_optimizer is optimizer
        observed.update(kwargs)
        events.append('observe')
        try:
            yield
        finally:
            events.append('restore_optimizer')

    fake_training = SimpleNamespace(setup_model_and_optimizer=original)
    monkeypatch.setitem(sys.modules, 'megatron.training.training', fake_training)
    monkeypatch.setitem(
        sys.modules,
        'examples.mimo.optimizer_diagnostics',
        SimpleNamespace(optimizer_step_diagnostics=observer),
    )
    args = SimpleNamespace(
        model_provider='qwen35_native',
        mimo_optimizer_reference=str(tmp_path),
        mimo_optimizer_expected_step=52,
        mimo_optimizer_report=str(tmp_path / 'report.json'),
        mimo_optimizer_scratch_dir=str(tmp_path / 'scratch'),
        mimo_write_optimizer_reference=False,
    )
    try:
        with observe_native_optimizer(args):
            assert fake_training.setup_model_and_optimizer is not original
            assert fake_training.setup_model_and_optimizer() == (models, optimizer, scheduler)
            assert observed['group'] == 'group'
            assert observed['cold_moment_bands'] is False
            assert observed['context']() == dict(
                source_step=52,
                source={'source_tokens': 123, 'supervised_tokens': 100},
                scheduler={'num_steps': 3328},
            )
            if raise_inside:
                raise RuntimeError('test failure')
    except RuntimeError:
        assert raise_inside
    assert fake_training.setup_model_and_optimizer is original
    assert events == ['native_setup', 'observe', 'restore_optimizer']
