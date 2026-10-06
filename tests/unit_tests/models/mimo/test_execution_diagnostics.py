# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU checks that sampled observations preserve autograd and logical identity."""

from types import SimpleNamespace

import pytest
import torch

from examples.mimo.execution_diagnostics import (
    ExecutionDiagnostics,
    compare_execution_directories,
    probe_positions,
)


class _Router(torch.nn.Module):
    def forward(self, value):
        probs = value.reshape(-1, value.shape[-1]).softmax(-1)
        return probs, probs > 0.5


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = torch.nn.Identity()

    def forward(self, hidden_states):
        return self.in_proj(hidden_states)


class _MoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = _Router()

    def forward(self, hidden_states):
        probs, _ = self.router(hidden_states)
        return hidden_states * 2 + probs.unsqueeze(1)


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = _Attention()
        self.pre_mlp_layernorm = torch.nn.Identity()
        self.mlp = _MoE()

    def forward(self, hidden_states):
        hidden_states = self.self_attention(hidden_states)
        return self.mlp(self.pre_mlp_layernorm(hidden_states))


class _Partition:
    def shard(self, embeddings, labels, mask, packed):
        return embeddings, labels, mask, packed


def _run(
    directory,
    order,
    *,
    partition=True,
    num_layers=1,
    gdn_layers=(0,),
    moe_layers=(),
    statistics=False,
    repeat=False,
    return_output=False,
):
    model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    model.language_model.decoder = torch.nn.Module()
    model.language_model.decoder.layers = torch.nn.ModuleList([_Layer() for _ in range(num_layers)])
    model.partition_adapter = _Partition() if partition else None
    model.special_token_ids = {'images': 99}
    samples = {sid: dict(original_seq_len=3, tokens=torch.tensor([sid, 99, sid])) for sid in (0, 1)}
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])[list(order)].requires_grad_()
    hidden = torch.stack(
        [torch.tensor([float(sid), -float(sid)]) for sid in order for _ in range(3)]
    )
    hidden[[1, 4]] = features
    hidden = hidden.unsqueeze(1)
    tokens = torch.cat([samples[sid]['tokens'] for sid in order]).unsqueeze(0)
    transfer = SimpleNamespace(
        features=features, plans=[SimpleNamespace(cp_ranks=(0,))], local_plan=0
    )
    item = dict(
        round_id=0,
        diagnostic_sample_ids=order,
        kwargs=dict(
            input_ids=tokens,
            packing_kwargs=dict(
                cu_seqlens_q=torch.tensor([0, 3, 6], dtype=torch.int32),
                cu_seqlens_q_padded=torch.tensor([0, 3, 6], dtype=torch.int32),
            ),
        ),
    )
    tracer = (
        ExecutionDiagnostics(
            model, directory, 0, 7, samples, gdn_layers=gdn_layers, moe_layers=moe_layers
        )
        if directory
        else None
    )

    def forward():
        value = hidden
        for layer in model.language_model.decoder.layers:
            value = layer(value)
        return value

    if tracer and statistics:
        tracer.begin(item, 'statistics', transfer)
        with torch.no_grad():
            forward()
        tracer.flush()
    if tracer:
        tracer.begin(item, 'training', transfer)
    if repeat:
        with torch.no_grad():
            forward()
    output = forward()
    output.square().sum().backward()
    gradient = features.grad.clone()
    if tracer:
        tracer.close()
    return (gradient, output.detach()) if return_output else gradient


def test_probes_do_not_include_padding():
    assert probe_positions(0) == []
    assert probe_positions(3, (1,)) == [0, 1, 2]
    assert max(probe_positions(137, (60, 61))) == 136


@pytest.mark.parametrize('partition', [True, False])
def test_probes_preserve_gradient_and_align_reordered_samples(tmp_path, partition):
    first, second = tmp_path / 'first', tmp_path / 'second'
    actual = _run(first, (0, 1), partition=partition)
    assert torch.equal(actual, _run(None, (0, 1), partition=partition))
    _run(second, (1, 0), partition=partition)
    result = compare_execution_directories(first, second, 7)
    assert result['missing_records'] == result['extra_records'] == []
    assert result['first_changed_activation'] is None
    assert result['cp_vision_inputs_exact']
    assert result['local_vision_gradients_exact']
    assert result['nonowner_vision_gradients_zero']
    assert all(value['exact'] for value in result['statistics'].values())
    captured = torch.load(first / 'step00000007/rank00000/training-round0000.pt', weights_only=True)
    assert captured['gdn_input'].shape == captured['gdn_output_gradient'].shape == (6, 1, 2)
    assert not captured['gdn_input'].requires_grad


def test_selected_gdn_full_snapshots_preserve_output_and_gradient(tmp_path):
    options = dict(num_layers=2, gdn_layers=(0, 1), repeat=True, return_output=True)
    actual = _run(tmp_path, (0, 1), statistics=True, **options)
    reference = _run(None, (0, 1), **options)
    assert all(torch.equal(a, b) for a, b in zip(actual, reference))
    directory = tmp_path / 'step00000007/rank00000'
    for phase in ('statistics', 'training'):
        data = torch.load(directory / f'{phase}-round0000.pt', weights_only=True)
        first, second = data['gdn_layers']['0'], data['gdn_layers']['1']
        assert torch.equal(first['input'], data['gdn_input'])
        for index, entry in data['gdn_layers'].items():
            assert entry['input'].shape == entry['output'].shape == (6, 1, 2)
            assert not entry['input'].requires_grad and not entry['output'].requires_grad
            assert torch.equal(
                entry['input'],
                data['records'][f'layer{int(index):02d}.gdn_input'][0]['value'].unsqueeze(1),
            )
            if phase == 'training':
                assert entry['output_gradient'].shape == entry['output'].shape
                assert entry['recompute_checks'] == [
                    dict(
                        input_exact=True,
                        input_max_abs=0.0,
                        output_observed=True,
                        output_exact=True,
                        output_max_abs=0.0,
                    )
                ]
            else:
                assert 'output_gradient' not in entry
                assert entry['recompute_checks'] == []
        if phase == 'training':
            assert torch.equal(first['output_gradient'], data['gdn_output_gradient'])
        assert not torch.equal(first['input'], second['input'])


def test_moe_snapshots_share_round_metadata_and_keep_repeated_calls(tmp_path):
    actual = _run(tmp_path, (1, 0), moe_layers=(0,), statistics=True, repeat=True)
    reference = _run(None, (1, 0), repeat=True)
    assert torch.equal(actual, reference)
    for phase in ('statistics', 'training'):
        path = tmp_path / 'step00000007/rank00000' / f'{phase}-round0000.pt'
        data = torch.load(path, weights_only=True)
        assert data['sample_ids'] == [1, 0]
        assert data['sample_lengths'] == [3, 3]
        assert torch.equal(data['physical_indices'], torch.arange(6))
        assert data['padded_boundaries'] == [0, 3, 6]
        boundary = data['moe_layers']['0']['boundaries']['moe_output']
        assert len(boundary) == (2 if phase == 'training' else 1)
        assert boundary[0]['value'].shape == (6, 1, 2)
        if phase == 'training':
            assert boundary[1]['gradient'].shape == (6, 1, 2)


@pytest.mark.parametrize('indices', [(-1,), (2,)])
def test_invalid_gdn_layer_indices_fail_clearly(tmp_path, indices):
    with pytest.raises(ValueError, match='Invalid GDN diagnostic layer index'):
        _run(tmp_path, (0, 1), num_layers=2, gdn_layers=indices)


def test_non_gdn_selection_fails_clearly(tmp_path):
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            decoder=SimpleNamespace(layers=[SimpleNamespace(self_attention=torch.nn.Identity())])
        )
    )
    with pytest.raises(ValueError, match='is not a GDN layer'):
        ExecutionDiagnostics(model, tmp_path, 0, 0, {}, gdn_layers=(0,))


def test_changed_full_recompute_is_saved_without_changing_graph(tmp_path):
    tracer = object.__new__(ExecutionDiagnostics)
    tracer.current = dict(phase='training', records={}, gdn_layers={}, local_tokens=2)
    tracer.rows = torch.tensor([0, 1])
    tracer.gdn_names = {'layer04.attention': '4'}
    before, after = tracer._gdn_input_hook(4), tracer._output_hook('layer04.attention')
    first = torch.ones(2, 1, 3, requires_grad=True)
    second = torch.full((2, 1, 3), 2.0, requires_grad=True)
    before(None, (first,), {})
    after(None, (), first * 2)
    before(None, (second,), {})
    output = second * 3
    after(None, (), output)
    output.sum().backward()
    entry = tracer.current['gdn_layers']['4']
    check = entry['recompute_checks'][0]
    assert not check['input_exact'] and check['input_max_abs'] == 1.0
    assert not check['output_exact'] and check['output_max_abs'] == 4.0
    assert torch.equal(check['input'], second)
    assert torch.equal(check['output'], output)
    assert not check['input'].requires_grad and not check['output'].requires_grad
    assert torch.equal(second.grad, torch.full_like(second, 3.0))
    assert torch.equal(entry['output_gradient'], torch.ones_like(output))


def test_nonowner_gradient_is_checked_independently(tmp_path):
    tracer = object.__new__(ExecutionDiagnostics)
    tracer.directory = tmp_path
    tracer.vision_feature_rows = torch.tensor([0])
    features = torch.zeros(2, 3, requires_grad=True)
    features.grad = torch.tensor([[2.0, 3.0, 4.0], [0.0, 1.0, 0.0]])
    tracer.transfer = SimpleNamespace(features=features)
    tracer.current = dict(phase='training', round=0, cp_input=[], records={}, keys=[])
    tracer.flush()
    saved = torch.load(tmp_path / 'training-round0000.pt', weights_only=True)
    assert saved['nonowner_gradient'] == dict(rows=1, nonzero_elements=1, max_abs=1.0)


def test_missing_input_observation_is_not_a_pass(tmp_path):
    _run(tmp_path, (0, 1))
    path = tmp_path / 'step00000007/rank00000/training-round0000.pt'
    saved = torch.load(path, weights_only=True)
    saved['cp_input'] = []
    torch.save(saved, path)
    report = compare_execution_directories(tmp_path, tmp_path, 7)
    assert report['cp_vision_inputs_exact'] is None
    assert report['local_vision_gradients_exact'] is None
    assert report['cp_input_coverage']['missing_input_rounds'] == 1


def test_gdn_input_is_reported_before_input_projection(tmp_path):
    reference, candidate = tmp_path / 'reference', tmp_path / 'candidate'
    _run(reference, (0, 1))
    _run(candidate, (0, 1))
    path = candidate / 'step00000007/rank00000/training-round0000.pt'
    saved = torch.load(path, weights_only=True)
    for name in ('layer00.gdn_input', 'layer00.input_projection'):
        saved['records'][name][0]['value'] = saved['records'][name][0]['value'] + 1
    torch.save(saved, path)
    report = compare_execution_directories(reference, candidate, 7)
    assert report['first_changed_activation'] == 'layer00.gdn_input/value'
    assert not report['statistics']['layer00.input_projection/value']['exact']
