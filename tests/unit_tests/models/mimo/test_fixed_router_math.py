# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU checks for fixed expert IDs with live dispatch and auxiliary gradients."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    compute_routing_scores_for_aux_loss,
    topk_routing_with_score_function,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig


class _FixedIDs:
    def __init__(self, indices=None):
        self.indices = indices
        self.calls = []

    def select(self, router, logits, default_selector, padding_mask, packed_seq_params):
        self.calls.append((padding_mask, packed_seq_params))
        if self.indices is None:
            self.indices = default_selector().clone()
        return self.indices


def _router(*, score_function="softmax", pre_softmax=False, topk=2, aux=True):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=3,
        num_attention_heads=1,
        num_moe_experts=4,
        moe_router_topk=topk,
        moe_router_pre_softmax=pre_softmax,
        moe_router_score_function=score_function,
        moe_router_load_balancing_type="global_aux_loss",
        moe_aux_loss_coeff=0.07 if aux else 0.0,
    )
    singleton = SimpleNamespace(size=lambda: 1)
    groups = SimpleNamespace(tp=singleton, cp=singleton, tp_cp=singleton, tp_dp_cp=singleton)
    # Buffer allocation is the only constructor use of CUDA; routing runs on CPU logits.
    with patch.object(torch.cuda, "current_device", return_value="cpu"):
        router = TopKRouter(config, pg_collection=groups)
    return router


def _normalized_scores(logits, score_function):
    if score_function == "softmax":
        return logits.softmax(-1)
    scores = logits.sigmoid() if score_function == "sigmoid" else F.softplus(logits).sqrt()
    return scores / scores.sum(-1, keepdim=True)


@pytest.mark.parametrize("score_function", ["softmax", "sigmoid", "sqrtsoftplus"])
def test_aux_scores_keep_all_logits_but_counts_use_fixed_ids(score_function):
    logits = torch.tensor(
        [[8.0, 2.0, 0.5, -3.0], [0.2, 1.3, -0.4, 0.7], [1.0, 3.0, 2.0, -1.0]], requires_grad=True
    )
    # These deliberately differ from the natural top-k.
    indices = torch.tensor([[3, 2], [2, 0], [0, 3]])
    padding = torch.tensor([False, True, False])
    routing, scores = compute_routing_scores_for_aux_loss(
        logits, 2, score_function, padding_mask=padding, precomputed_indices=indices
    )
    expected_map = torch.zeros_like(logits, dtype=torch.bool).scatter(1, indices, True)
    expected_map[padding] = False
    assert torch.equal(routing, expected_map)
    expected_scores = _normalized_scores(logits, score_function) * (~padding)[:, None]
    torch.testing.assert_close(scores, expected_scores)
    assert scores[0, 0] > 0  # An unselected expert must still participate in the aux objective.

    weights = torch.tensor([0.2, -1.1, 0.8, 2.0])
    (actual_grad,) = torch.autograd.grad((scores * weights).sum(), logits, retain_graph=True)
    (expected_grad,) = torch.autograd.grad((expected_scores * weights).sum(), logits)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert actual_grad[0, 0] != 0
    assert torch.count_nonzero(actual_grad[padding]) == 0


class _AuxProbe:
    """Observe the production router's aux handoff without distributed reductions."""

    collecting_statistics = False
    auditing_training = False
    audit_token_assignments = True

    def observe_dispatch(self, router, routing_map, padding_mask):
        self.dispatch = routing_map.detach().clone()

    def apply(self, router, probs, scores, routing_map):
        self.scores, self.routing = scores, routing_map
        count = routing_map.sum().to(scores.dtype) / router.topk
        frequency = routing_map.sum(0) / (count * router.topk)
        self.loss = (
            router.config.moe_aux_loss_coeff
            * router.num_experts
            * (scores.sum(0) / count * frequency).sum()
        )
        if not torch.is_grad_enabled():
            return probs
        return MoEAuxLossAutoScaler.apply(probs, self.loss)


@pytest.mark.parametrize(
    "score_function,pre_softmax,topk",
    [
        ("softmax", False, 2),
        ("softmax", True, 2),
        ("sigmoid", False, 2),
        ("sqrtsoftplus", False, 2),
        ("sigmoid", False, 1),
    ],
)
def test_real_router_keeps_dispatch_aux_and_router_weight_gradients_consistent(
    score_function, pre_softmax, topk, monkeypatch
):
    router = _router(score_function=score_function, pre_softmax=pre_softmax, topk=topk)
    # Fixed mode must bypass both fused routing kernels even when the config enables them.
    router.config.moe_router_fusion = True
    router.config.moe_router_aux_loss_fusion = True
    indices = torch.tensor([[3, 0], [1, 0], [2, 0]])[:, :topk].contiguous()
    controller = _FixedIDs(indices)
    router._fixed_routing = controller
    probe = _AuxProbe()
    router._global_aux_loss_step = probe
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))

    hidden = torch.tensor([[0.5, 1.0, -0.2], [1.0, -0.5, 0.3], [-0.4, 0.8, 1.2]])
    weight = torch.tensor([[1.0, 0.3, 0.1], [-0.4, 0.6, 0.2], [0.1, -0.5, 0.7], [-0.7, 0.2, -0.8]])
    with torch.no_grad():
        router.weight.copy_(weight)
    logits = F.linear(hidden, router.weight)
    padding = torch.tensor([[False], [True], [False]])
    packed = object()
    probs, routing = router.routing(logits[:, None, :], padding, packed)
    expected_map = torch.zeros_like(logits, dtype=torch.bool).scatter(1, indices, True)
    assert torch.equal(routing, expected_map)
    assert torch.equal(probe.dispatch, expected_map)
    assert torch.equal(probe.routing, expected_map & (~padding))
    assert len(controller.calls) == 1  # Aux must reuse IDs rather than select again.
    assert controller.calls[0][1] is packed
    assert torch.equal(controller.calls[0][0], padding.flatten())

    reference_weight = weight.clone().requires_grad_()
    reference_logits = F.linear(hidden, reference_weight)
    if score_function == "softmax":
        selected = (
            reference_logits.softmax(-1).gather(1, indices)
            if pre_softmax
            else reference_logits.gather(1, indices).softmax(-1)
        )
    else:
        scores = (
            reference_logits.sigmoid()
            if score_function == "sigmoid"
            else F.softplus(reference_logits).sqrt()
        )
        selected = scores.gather(1, indices)
        if topk > 1:
            selected = selected / selected.sum(-1, keepdim=True)
    expected_probs = torch.zeros_like(reference_logits).scatter(1, indices, selected)
    torch.testing.assert_close(probs, expected_probs)
    reference_scores = _normalized_scores(reference_logits, score_function) * (~padding)
    frequency = (expected_map & (~padding)).sum(0) / (2 * topk)
    reference_aux = 0.07 * 4 * (reference_scores.sum(0) / 2 * frequency).sum()
    torch.testing.assert_close(probe.loss, reference_aux)

    cotangent = torch.tensor([[0.2, -0.7, 1.1, -0.3], [0.0, 0.0, 0.0, 0.0], [1.2, -0.5, 0.6, 0.3]])
    (actual_grad,) = torch.autograd.grad((probs * cotangent).sum(), router.weight)
    (expected_grad,) = torch.autograd.grad(
        (expected_probs * cotangent).sum() + reference_aux, reference_weight
    )
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-7, rtol=2e-6)
    assert actual_grad.abs().sum() > 0


def test_recorded_choices_survive_statistics_checkpoint_and_recompute(monkeypatch):
    router = _router()
    controller = _FixedIDs()
    router._fixed_routing = controller
    probe = _AuxProbe()
    probe.collecting_statistics = True
    router._global_aux_loss_step = probe
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))
    logits = torch.tensor([[[5.0, 2.0, -3.0, -1.0]], [[0.0, 1.0, 4.0, 2.0]]])
    with torch.no_grad():
        _, original_map = router.routing(logits)
    assert torch.equal(probe.routing, original_map)
    probe.collecting_statistics = False
    changed = logits.flip(-1).requires_grad_()
    probs = checkpoint(lambda value: router.routing(value)[0], changed, use_reentrant=True)
    changed_map = probe.dispatch
    natural_probs, natural_map = topk_routing_with_score_function(changed.flatten(0, 1), 2)
    assert torch.equal(original_map, changed_map)
    assert not torch.equal(changed_map, natural_map)
    assert not torch.equal(probs, natural_probs)
    probs[:, 0].sum().backward()
    assert changed.grad.abs().sum() > 0
    assert torch.equal(probe.routing, original_map)
    assert torch.equal(probe.dispatch, original_map)
    assert len(controller.calls) == 3  # Statistics, checkpoint original and recompute.


def test_without_controller_preserves_configured_fused_dispatch(monkeypatch):
    router = _router(aux=False)
    router.config.moe_router_fusion = True
    expected = (torch.tensor([[0.3, 0.0, 0.7, 0.0]]), torch.tensor([[True, False, True, False]]))
    calls = []

    def fused_probe(*args, **kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(
        "megatron.core.transformer.moe.router.topk_routing_with_score_function", fused_probe
    )
    actual = router.routing(torch.zeros(1, 1, 4))
    assert actual[0] is expected[0] and actual[1] is expected[1]
    assert calls[0]["fused"] is True
    assert calls[0]["precomputed_indices"] is None


@pytest.mark.parametrize(
    "option,value",
    [
        ("moe_expert_capacity_factor", 1.0),
        ("moe_expert_rank_capacity_factor", 1.0),
        ("moe_router_num_groups", 2),
        ("moe_router_group_topk", 1),
    ],
)
def test_fixed_routing_rejects_unsupported_dispatch_options(option, value):
    router = _router(aux=False)
    router._fixed_routing = _FixedIDs(torch.tensor([[0, 1]]))
    setattr(router.config, option, value)
    with pytest.raises(ValueError, match="ordinary, dropless"):
        router.routing(torch.zeros(1, 1, 4))


@pytest.mark.parametrize("indices", [torch.tensor([[0.0, 1.0]]), torch.tensor([[0]])])
def test_fixed_routing_rejects_controller_shape_or_dtype_mismatch(indices):
    router = _router(aux=False)
    router._fixed_routing = _FixedIDs(indices)
    with pytest.raises(ValueError, match="long"):
        router.routing(torch.zeros(1, 1, 4))


def test_aux_fixed_ids_reject_fused_kernel():
    with pytest.raises(ValueError, match="fused=False"):
        compute_routing_scores_for_aux_loss(
            torch.zeros(1, 4), 2, "softmax", fused=True, precomputed_indices=torch.tensor([[0, 1]])
        )
