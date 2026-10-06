# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU-only checks for the diagnostic formulas, checkpoint slices and adjoints."""

import pytest
import torch
import torch.nn.functional as F

from examples.mimo.moe_numerical_reference import (
    ExpertWeights,
    QwenMoECheckpoint,
    TorchDistSliceReader,
    canonical_combine,
    canonical_dispatch,
    capture_rows,
    expert_reference,
    frozen_bf16_values,
    load_capture_selection,
    moe_reference,
    router_reference,
    shared_reference,
)


def _fixture():
    generator = torch.Generator().manual_seed(43)

    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64) / 3

    inputs, router, gate = rand(3, 4), rand(5, 4), rand(1, 4)
    shared = ExpertWeights(rand(6, 4), rand(4, 3))
    experts = {i: ExpertWeights(rand(6, 4), rand(4, 3)) for i in range(5)}
    return inputs, router, gate, shared, experts


def test_bf16_rounding_does_not_recover_checkpoint_precision():
    value = torch.tensor([1.001, 1.007, -0.999], dtype=torch.float32)
    actual = frozen_bf16_values(value)
    assert actual.dtype == torch.float64
    assert torch.equal(actual, value.bfloat16().double())
    assert not torch.equal(actual, value.double())


def test_router_fixed_ids_remain_differentiable_and_ties_are_explicit():
    value = torch.tensor([[1.0, 2.0]], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    logits, ids, probs, margin = router_reference(value, weight, 2)
    assert ids.tolist() == [[0, 1]]
    assert margin.tolist() == [1.0]
    fixed = torch.tensor([[2, 0]])
    _, selected, probabilities, _ = router_reference(value, weight, 2, fixed)
    assert torch.equal(selected, fixed)
    torch.testing.assert_close(probabilities, logits[:, [2, 0]].softmax(-1))
    probabilities[:, 0].sum().backward()
    assert value.grad[0, 0] != 0


@pytest.mark.parametrize(
    'ids',
    [
        torch.tensor([[0, 0]]),
        torch.tensor([[-1, 0]]),
        torch.tensor([[0, 3]]),
        torch.tensor([[0.0, 1.0]]),
    ],
)
def test_router_rejects_invalid_fixed_experts(ids):
    with pytest.raises(ValueError):
        router_reference(torch.ones(1, 2), torch.ones(3, 2), 2, ids)


def test_weighted_expert_and_shared_formulas_and_parameter_vjp():
    inputs, _, gate, shared, experts = _fixture()
    expert = experts[0]
    values = (
        inputs.requires_grad_(),
        expert.fc1.requires_grad_(),
        expert.fc2.requires_grad_(),
        torch.tensor([0.1, 0.4, 0.9], dtype=torch.float64, requires_grad=True),
    )

    def func(x, fc1, fc2, probs):
        return expert_reference(x, ExpertWeights(fc1, fc2), probs)

    assert torch.autograd.gradcheck(func, values)
    probability = values[-1]
    gate_part, up_part = F.linear(inputs, expert.fc1).chunk(2, -1)
    manual = F.linear((F.silu(gate_part) * up_part) * probability[:, None], expert.fc2)
    torch.testing.assert_close(func(*values), manual, atol=0, rtol=0)
    torch.testing.assert_close(
        shared_reference(inputs, shared, gate),
        expert_reference(inputs, shared) * F.linear(inputs, gate).sigmoid(),
    )


def test_moe_vjp_fixed_routing_matches_explicit_per_token_formula():
    inputs, router, gate, shared, experts = _fixture()
    inputs.requires_grad_()
    ids = torch.tensor([[3, 1], [0, 4], [2, 1]])
    probabilities = torch.tensor([[0.4, 0.6], [0.2, 0.8], [0.7, 0.3]], dtype=torch.float64)
    actual = moe_reference(inputs, router, shared, gate, experts.__getitem__, 2, ids, probabilities)
    manual_rows = []
    for row, selected in enumerate(ids.tolist()):
        contributions = [
            expert_reference(
                inputs[row : row + 1], experts[index], probabilities[row, slot : slot + 1]
            )
            for slot, index in enumerate(selected)
        ]
        manual_rows.append(
            sum(contributions) + shared_reference(inputs[row : row + 1], shared, gate)
        )
    expected = torch.cat(manual_rows)
    torch.testing.assert_close(actual['output'], expected, atol=1e-16, rtol=1e-14)
    cotangent = torch.arange(12, dtype=torch.float64).reshape(3, 4) / 10
    (a,) = torch.autograd.grad(actual['output'], inputs, cotangent)
    (b,) = torch.autograd.grad(expected, inputs, cotangent)
    torch.testing.assert_close(a, b, atol=1e-16, rtol=1e-13)


def test_frozen_probabilities_exclude_router_vjp_but_fixed_ids_do_not():
    inputs, router, gate, shared, experts = _fixture()
    router.requires_grad_()
    ids = torch.tensor([[0, 1], [1, 2], [3, 4]])
    fixed_ids = moe_reference(inputs, router, shared, gate, experts.__getitem__, 2, ids)
    (gradient,) = torch.autograd.grad(fixed_ids['output'].sum(), router)
    assert gradient.abs().sum() > 0
    inputs.requires_grad_()
    frozen = moe_reference(
        inputs,
        router,
        shared,
        gate,
        experts.__getitem__,
        2,
        ids,
        torch.full((3, 2), 0.5, dtype=torch.float64),
    )
    (gradient,) = torch.autograd.grad(frozen['output'].sum(), router, allow_unused=True)
    assert gradient is None


def test_integer_dispatch_and_combine_preserve_routes_exactly():
    tokens = torch.arange(15, dtype=torch.int64).reshape(5, 3)
    routing = torch.tensor(
        [[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 0], [0, 0, 1]], dtype=torch.bool
    )
    dispatched, indices, experts = canonical_dispatch(tokens, routing)
    assert list(zip(experts.tolist(), indices.tolist())) == sorted(
        (expert, token) for token, expert in routing.nonzero().tolist()
    )
    assert torch.equal(dispatched, tokens[indices])
    actual = canonical_combine(dispatched, indices, len(tokens))
    assert torch.equal(actual, tokens * routing.sum(-1, keepdim=True))
    wrong = indices.roll(1)
    assert not torch.equal(canonical_combine(dispatched, wrong, len(tokens)), actual)


def test_dispatch_combine_adjoint_and_local_zero_image_equivalent_rows():
    inputs = torch.arange(12, dtype=torch.float64).reshape(4, 3).requires_grad_()
    routing = torch.tensor([[1, 1], [0, 0], [1, 0], [0, 1]], dtype=torch.bool)
    dispatched, indices, _ = canonical_dispatch(inputs, routing)
    probe = torch.arange(dispatched.numel(), dtype=torch.float64).reshape_as(dispatched) - 4
    (dispatch_gradient,) = torch.autograd.grad((dispatched * probe).sum(), inputs)
    assert torch.equal(dispatch_gradient, canonical_combine(probe, indices, len(inputs)))
    assert not dispatch_gradient[1].any()
    values = probe.clone().requires_grad_()
    combined = canonical_combine(values, indices, len(inputs))
    (combine_gradient,) = torch.autograd.grad((combined * inputs.detach()).sum(), values)
    assert torch.equal(combine_gradient, inputs.detach()[indices])
    assert torch.equal(
        (dispatched.detach() * probe).sum(), (inputs.detach() * dispatch_gradient).sum()
    )


def test_empty_dispatch_has_correct_zero_adjoint():
    inputs = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    dispatched, indices, _ = canonical_dispatch(inputs, torch.zeros(3, 2, dtype=torch.bool))
    output = canonical_combine(dispatched, indices, 3)
    assert output.shape == inputs.shape and not output.any()
    output.sum().backward()
    assert torch.equal(inputs.grad, torch.zeros_like(inputs))


def _capture():
    value = torch.arange(12).reshape(3, 1, 4).to(torch.bfloat16)
    routing = torch.tensor([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=torch.bool)
    tensors = dict(
        moe_input=value,
        router_logits=torch.zeros(3, 3),
        router_probs=routing.float() / 2,
        routing_map=routing,
        shared_output=value,
        routed_output=value,
        moe_output=value,
    )
    boundaries = {
        name: [dict(value=tensor, invocation=0, kind='original', grad_enabled=False)]
        for name, tensor in tensors.items()
    }
    boundaries['moe_output'].append(
        dict(
            value=value.clone(),
            gradient=torch.ones_like(value),
            invocation=1,
            kind='recompute_or_repeat',
            grad_enabled=True,
        )
    )
    boundaries['moe_input'].append(
        dict(value=value.clone(), invocation=1, kind='recompute_or_repeat', grad_enabled=True)
    )
    return dict(
        step=0,
        phase='training',
        physical_indices=torch.tensor([0, 2, 3]),
        sample_ids=[9],
        sample_lengths=[3],
        padded_boundaries=[0, 4],
        moe_layers={'0': dict(schema_version=1, boundaries=boundaries)},
    )


def test_capture_selection_handles_cp_indices_and_excludes_padding(tmp_path):
    snapshot = _capture()
    keys, values = capture_rows(snapshot, [(9, 2), (9, 3)])
    assert keys == [(9, 2)]
    assert torch.equal(
        values['moe_input'], snapshot['moe_layers']['0']['boundaries']['moe_input'][0]['value'][1]
    )
    directory = tmp_path / 'rank00000'
    directory.mkdir()
    torch.save(snapshot, directory / 'training-round0000.pt')
    selected = load_capture_selection(tmp_path, [(9, 2), (9, 0)])
    assert selected['moe_input'][:, 0].tolist() == [4, 0]
    with pytest.raises(ValueError, match='Missing 1'):
        load_capture_selection(tmp_path, [(9, 3)])


def test_capture_refuses_mixed_recompute_cotangent_and_missing_fields():
    snapshot = _capture()
    snapshot['moe_layers']['0']['boundaries']['moe_output'][1]['value'][0, 0, 0] += 1
    with pytest.raises(ValueError, match='cotangent-bearing'):
        capture_rows(snapshot, [(9, 0)])
    snapshot = _capture()
    del snapshot['moe_layers']['0']['boundaries']['router_logits']
    with pytest.raises(ValueError, match='Missing original'):
        capture_rows(snapshot, [(9, 0)])


def test_checkpoint_slice_and_native_bf16_values(tmp_path):
    import torch.distributed.checkpoint as dcp

    prefix = 'language_model.decoder.layers.0.mlp.'
    generator = torch.Generator().manual_seed(1)
    shapes = {
        'router.weight': (5, 4),
        'shared_experts.linear_fc1.weight': (6, 4),
        'shared_experts.linear_fc2.weight': (4, 3),
        'shared_experts.gate_weight': (1, 4),
        'experts.experts.linear_fc1.weight': (5, 6, 4),
        'experts.experts.linear_fc2.weight': (5, 4, 3),
    }
    tensors = {
        prefix + name: torch.randn(shape, generator=generator) for name, shape in shapes.items()
    }
    dcp.save(tensors, checkpoint_id=tmp_path)
    reader = TorchDistSliceReader(tmp_path)
    key = prefix + 'experts.experts.linear_fc1.weight'
    assert torch.equal(
        reader.tensor(key, starts=(2, 1, 0), sizes=(1, 3, 4)), tensors[key][2:3, 1:4]
    )
    weights = QwenMoECheckpoint(tmp_path)
    assert torch.equal(weights.router(), tensors[prefix + 'router.weight'].bfloat16().double())
    assert torch.equal(weights.expert(3).fc1, tensors[key][3].bfloat16().double())
    assert weights.shared().fc1.shape == (6, 4)
    assert weights.shared_gate().shape == (1, 4)
    with pytest.raises(KeyError):
        reader.tensor('absent')
    with pytest.raises(ValueError):
        weights.expert(5)
