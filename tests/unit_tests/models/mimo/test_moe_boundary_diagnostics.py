# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU checks of full MoE observations, graph preservation and hook lifetime."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from examples.mimo.moe_boundary_diagnostics import MoEBoundaryDiagnostics


class _Router(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4) / 20)

    def gating(self, input):
        return input @ self.weight

    def forward(self, input):
        logits = self.gating(input)
        probs = logits.reshape(-1, 4).softmax(-1)
        return probs, probs > 0.25


class _Dispatcher:
    def combine_postprocess(self, value):
        return value.reshape(-1, 1, 4)


class _MoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = _Router()
        self.shared_experts = torch.nn.Linear(4, 4, bias=False)
        torch.nn.init.constant_(self.shared_experts.weight, 0.125)
        self.token_dispatcher = _Dispatcher()
        self.shared_expert_overlap = False

    def forward(self, hidden_states):
        probs, _ = self.router(input=hidden_states)
        shared = self.shared_experts(hidden_states)
        routed = hidden_states * probs[:, 0].reshape(-1, 1, 1)
        routed = self.token_dispatcher.combine_postprocess(routed)
        return routed + shared, None


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_mlp_layernorm = torch.nn.LayerNorm(4)
        self.mlp = _MoE()

    def forward(self, hidden_states):
        normalized = self.pre_mlp_layernorm(hidden_states)
        return self.mlp(hidden_states=normalized)[0] + hidden_states


def _run(capture, *, recompute=False):
    layer = _Layer()
    hidden = (torch.arange(24, dtype=torch.float32).reshape(6, 1, 4) / 7).requires_grad_()
    state = dict(phase='training', local_tokens=6)
    observer = MoEBoundaryDiagnostics([layer], (0,) if capture else (), lambda: state)
    if recompute:
        output = checkpoint(layer, hidden, use_reentrant=True)
    else:
        output = layer(hidden)
    output.square().sum().backward()
    gradient = hidden.grad.clone()
    parameters = {name: value.grad.clone() for name, value in layer.named_parameters()}
    observer.close()
    return state, output.detach(), gradient, parameters


@pytest.mark.parametrize('recompute', [False, True])
def test_full_boundaries_preserve_outputs_and_gradients(recompute):
    actual = _run(True, recompute=recompute)
    reference = _run(False, recompute=recompute)
    assert torch.equal(actual[1], reference[1])
    assert torch.equal(actual[2], reference[2])
    assert actual[3].keys() == reference[3].keys()
    assert all(torch.equal(value, reference[3][name]) for name, value in actual[3].items())
    entry = actual[0]['moe_layers']['0']
    expected = {
        'pre_mlp_norm_input',
        'pre_mlp_norm_output',
        'moe_input',
        'moe_output',
        'router_input',
        'router_logits',
        'router_probs',
        'routing_map',
        'shared_output',
        'routed_output',
    }
    assert set(entry['installed_boundaries']) == set(entry['boundaries']) == expected
    for name, records in entry['boundaries'].items():
        assert len(records) == (2 if recompute else 1)
        for invocation, record in enumerate(records):
            assert record['invocation'] == invocation
            assert record['kind'] == ('original' if invocation == 0 else 'recompute_or_repeat')
            assert record['grad_enabled'] == (not recompute or invocation == 1)
            assert record['value'].device.type == 'cpu'
            assert not record['value'].requires_grad
            assert record['value'].reshape(6, -1).shape == (6, 4)
            if name != 'routing_map' and (not recompute or invocation == 1):
                assert record['gradient'].shape == record['value'].shape
                assert not record['gradient'].requires_grad
            elif name == 'routing_map':
                assert 'gradient' not in record
            # A checkpoint's original no-grad call can still observe gradients
            # on its input leaf; grad mode alone does not decide hook execution.
            if 'gradient' in record:
                assert record['requires_grad']
                assert record['gradient'].shape == record['value'].shape
        if recompute:
            assert torch.equal(records[0]['value'], records[1]['value'])
    records = entry['boundaries']
    assert torch.equal(records['pre_mlp_norm_output'][0]['value'], records['moe_input'][0]['value'])
    assert torch.equal(
        records['moe_output'][0]['value'],
        records['shared_output'][0]['value'] + records['routed_output'][0]['value'],
    )


def test_statistics_and_training_keep_separate_records_and_preserve_saved_values():
    layer = _Layer()
    state = dict(phase='statistics', local_tokens=2)
    observer = MoEBoundaryDiagnostics([layer], (0,), lambda: state)
    value = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)
    with torch.no_grad():
        layer(value)
    statistics = state
    state = dict(phase='training', local_tokens=2)
    layer(value.requires_grad_()).sum().backward()
    before = statistics['moe_layers']['0']['boundaries']['pre_mlp_norm_input'][0]['value']
    with torch.no_grad():
        value.fill_(100)
    assert torch.equal(before, torch.arange(8, dtype=torch.float32).reshape(2, 1, 4))
    for name, records in statistics['moe_layers']['0']['boundaries'].items():
        assert len(records) == 1
        assert not records[0]['grad_enabled'] and 'gradient' not in records[0]
        assert len(state['moe_layers']['0']['boundaries'][name]) == 1
    observer.close()


def test_selective_norm_recompute_does_not_shift_router_or_moe_invocations():
    layer = _Layer()
    state = dict(phase='training', local_tokens=2)
    observer = MoEBoundaryDiagnostics([layer], (0,), lambda: state)
    value = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4).requires_grad_()
    normalized = checkpoint(layer.pre_mlp_layernorm, value, use_reentrant=True)
    layer.mlp(normalized)[0].square().sum().backward()
    boundaries = state['moe_layers']['0']['boundaries']
    assert len(boundaries['pre_mlp_norm_input']) == 2
    assert len(boundaries['pre_mlp_norm_output']) == 2
    for name in ('moe_input', 'moe_output', 'router_input', 'router_logits', 'router_probs'):
        assert len(boundaries[name]) == 1
        assert boundaries[name][0]['kind'] == 'original'
        assert 'gradient' in boundaries[name][0]
    assert boundaries['pre_mlp_norm_output'][1]['kind'] == 'recompute_or_repeat'
    observer.close()


def test_close_restores_class_and_existing_instance_methods_and_is_idempotent():
    layer = _Layer()
    router = layer.mlp.router
    dispatcher = layer.mlp.token_dispatcher
    original_gating = router.gating
    original_combine = dispatcher.combine_postprocess
    # Cover restoring an explicitly overridden instance method as well as a
    # normal class method, which must not retain an instance shadow afterwards.
    dispatcher.combine_postprocess = original_combine
    observer = MoEBoundaryDiagnostics([layer], (0,), lambda: None)
    assert 'gating' in vars(router)
    assert dispatcher.combine_postprocess is not original_combine
    tensor = torch.ones(2, 1, 4)
    assert torch.equal(router.gating(tensor), original_gating(tensor))
    assert dispatcher.combine_postprocess(tensor) is not None
    observer.close()
    observer.close()
    assert 'gating' not in vars(router)
    assert router.gating == original_gating
    assert dispatcher.combine_postprocess is original_combine
    for module in layer.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks


def test_default_empty_selection_has_no_hooks_or_methods():
    layer = _Layer()
    state = dict(phase='training', local_tokens=2)
    observer = MoEBoundaryDiagnostics([layer], (), lambda: state)
    layer(torch.ones(2, 1, 4)).sum().backward()
    assert 'moe_layers' not in state
    assert not observer.handles and not observer.methods
    assert 'gating' not in vars(layer.mlp.router)
    observer.close()


@pytest.mark.parametrize('indices', [(-1,), (1,)])
def test_invalid_layer_does_not_leave_installed_hooks(indices):
    layer = _Layer()
    with pytest.raises(ValueError, match='Invalid MoE diagnostic layer index'):
        MoEBoundaryDiagnostics([layer], indices, lambda: None)
    assert 'gating' not in vars(layer.mlp.router)
    assert not layer.mlp._forward_hooks


def test_token_layout_mismatch_is_an_error_not_silent_partial_coverage():
    layer = _Layer()
    state = dict(phase='training', local_tokens=3)
    observer = MoEBoundaryDiagnostics([layer], (0,), lambda: state)
    try:
        with pytest.raises(ValueError, match='unexpected token layout'):
            layer(torch.ones(2, 1, 4))
    finally:
        observer.close()


def test_overlap_does_not_mislabel_routed_plus_shared_as_routed_only():
    layer = _Layer()
    layer.mlp.shared_expert_overlap = True
    observer = MoEBoundaryDiagnostics([layer], (0,), lambda: None)
    assert 'routed_output' not in observer.installed[0]
    assert 'combine_postprocess' not in vars(layer.mlp.token_dispatcher)
    observer.close()
