# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import pytest
import torch

from examples.mimo.shared_gate_numerical_replay import (
    bf16_reduction_mode,
    embedded_geometry,
    gate_forward,
    metric,
    selected_local_rows,
    tensor_sha256,
)


@pytest.mark.parametrize('original', [False, True])
@pytest.mark.parametrize('mode', ['default', 'on', 'off'])
def test_bf16_reduction_mode_restores_backend_without_cuda(original, mode):
    backend = torch.backends.cuda.matmul
    previous = backend.allow_bf16_reduced_precision_reduction
    initialized = torch.cuda.is_initialized()
    try:
        backend.allow_bf16_reduced_precision_reduction = original
        with bf16_reduction_mode(mode) as state:
            expected = original if mode == 'default' else mode == 'on'
            assert backend.allow_bf16_reduced_precision_reduction == expected
            assert state == dict(requested=mode, original=original, active=expected)
        assert state['restored'] == original
        assert backend.allow_bf16_reduced_precision_reduction == original
        assert torch.cuda.is_initialized() == initialized
    finally:
        backend.allow_bf16_reduced_precision_reduction = previous


def test_bf16_reduction_mode_restores_after_failure():
    backend = torch.backends.cuda.matmul
    original = backend.allow_bf16_reduced_precision_reduction
    initialized = torch.cuda.is_initialized()
    with pytest.raises(RuntimeError, match='replay failed'):
        with bf16_reduction_mode('off' if original else 'on') as state:
            assert backend.allow_bf16_reduced_precision_reduction != original
            raise RuntimeError('replay failed')
    assert state['restored'] == original
    assert backend.allow_bf16_reduced_precision_reduction == original
    assert torch.cuda.is_initialized() == initialized


def test_tensor_fingerprint_preserves_values_dtype_and_shape():
    value = torch.arange(6).reshape(2, 3).bfloat16()
    assert tensor_sha256(value) == tensor_sha256(value.clone())
    assert tensor_sha256(value) != tensor_sha256(value.float())
    assert tensor_sha256(value) != tensor_sha256(value.reshape(3, 2))
    assert tensor_sha256(value) != tensor_sha256(value + 1)


def test_gate_reference_and_adjoint():
    x = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([[0.25, -0.5]], dtype=torch.float64, requires_grad=True)
    result = gate_forward(x, weight)
    assert torch.equal(result['logits'], x @ weight.T)
    assert torch.equal(result['sigmoid'], (x @ weight.T).sigmoid())
    assert torch.autograd.gradcheck(lambda a, b: gate_forward(a, b)['sigmoid'], (x, weight))


@pytest.mark.parametrize('placement', ['front', 'spread'])
def test_geometry_preserves_inputs_and_zero_filler(placement):
    value = torch.arange(15).reshape(3, 5).bfloat16()
    result, rows = embedded_geometry(value, 19, placement)
    assert result.shape == (19, 1, 5)
    assert torch.equal(result[rows, 0], value)
    mask = torch.ones(19, dtype=torch.bool)
    mask[rows] = False
    assert not result[mask].any()
    assert len(set(rows.tolist())) == 3


def test_geometry_rejects_inadequate_capacity():
    with pytest.raises(ValueError, match='enough rows'):
        embedded_geometry(torch.zeros(3, 5), 2)


def test_canonical_rows_exclude_padding_and_follow_cp_ownership():
    snapshot = dict(
        physical_indices=torch.tensor([0, 1, 2, 4, 6, 7]),
        padded_boundaries=[0, 4, 8],
        sample_ids=[12, 25],
        sample_lengths=[2, 3],
    )
    keys, rows = selected_local_rows(snapshot, [(12, 1), (12, 2), (25, 2), (25, 3)])
    assert keys == [(12, 1), (25, 2)]
    assert rows.tolist() == [1, 4]


def test_difference_localizes_changed_keys():
    value = torch.tensor([[0.0], [1.0], [2.0]])
    other = value.clone()
    other[1] = 1.5
    result = metric(other, value, [(3, 2), (5, 0), (7, 9)])
    assert result['changed_rows'] == 1
    assert result['changed_keys'] == [[5, 0]]
