# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Optional optimizer-step scope for exact global MoE load-balancing loss."""

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    switch_load_balancing_loss_func,
)
from megatron.core.transformer.moe.router import TopKRouter


class GlobalAuxLossStep:
    """Accumulate router statistics independently of packing and runtime CP groups.

    This opt-in TP1 training scope retains forward graphs until ``finalize``. Each
    unique token contributes once on its CP owner, including prompt/vision tokens
    and excluding padding. Only ``finalize`` communicates, over the explicitly
    supplied group, in stable module order. Ordinary router behavior is unchanged
    outside the context. The default retained-graph mode does not support
    activation recomputation. ``two_pass=True`` instead collects detached
    statistics in a first pass, then attaches the objective during a
    normal sequential forward/backward pass. This mode supports recomputation
    and expert parallelism: router statistics belong to the input token owners,
    before expert dispatch, and are reduced over the full DP x CP domain.
    A shared MTP router repeated across multiple depths is also unsupported:
    each depth needs separate statistics before averaging its auxiliary loss.

    Usage::

        with GlobalAuxLossStep(model, reduction_group) as auxiliary:
            losses = []
            for round_id, batch in enumerate(batches):
                auxiliary.set_round(round_id)
                losses.append(forward_raw_token_loss(batch))
            auxiliary.finalize()
            for round_id, loss in enumerate(losses):
                (loss + global_supervised_tokens * auxiliary.loss_for_round(round_id)).backward()

    The example assumes parameter gradients are subsequently divided by
    ``global_supervised_tokens``. The resulting objective is the mean LM loss
    plus the sum of global, coefficient-weighted router losses. Backward remains
    explicitly ordered by round, rather than relying on cross-round autograd
    traversal to schedule context-parallel collectives.

    For native per-token training, without retaining a global batch of graphs::

        with GlobalAuxLossStep(model, reduction_group, two_pass=True) as auxiliary:
            with torch.enable_grad():
                for round_id, batch in enumerate(batches):
                    auxiliary.set_round(round_id)
                    forward(batch)
            auxiliary.finalize()
            auxiliary.begin_training(global_supervised_tokens)
            native_forward_backward(batches)

    The second pass must replay the same batches, weights and stochastic state.
    Match training's grad mode when kernels specialize on it. Discard each
    statistics-pass graph, including any module-owned checkpoint references;
    only detached statistics are retained by this scope. No backward is needed
    for that pass.
    Its router losses are multiplied by ``global_supervised_tokens`` because
    native per-token gradient finalization divides by that count. No router
    statistics are changed, and no auxiliary-loss communication occurs, during
    the training pass or activation recomputation. Keep this context active
    until all backwards are complete. Log the step-wide ``metrics`` directly;
    they are already globally reduced and must not be averaged by microbatch.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        reduction_group: dist.ProcessGroup,
        *,
        two_pass: bool = False,
        audit_replay: bool = False,
        audit_token_assignments: bool = False,
    ) -> None:
        self.reduction_group = reduction_group
        self.two_pass = two_pass
        if audit_replay and not two_pass:
            raise ValueError("Replay diagnostics require two_pass=True")
        if audit_token_assignments and not audit_replay:
            raise ValueError("Token assignment diagnostics require audit_replay=True")
        self.audit_replay = audit_replay
        self.audit_token_assignments = audit_token_assignments
        self._replay_records = {}
        self._recompute_records = []
        self._assignment_records = {"auxiliary": {}, "dispatch": {}}
        self._training_assignments = {"auxiliary": {}, "dispatch": {}}
        self._assignment_comparisons = []
        self.routers = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, TopKRouter) and module.get_aux_loss_coeff("global_aux_loss") > 0
        ]
        self.device = next(model.parameters()).device
        self._records = {router: {} for _, router in self.routers}
        self._global_counts = {}
        self._rounds = set()
        self._round = None
        self._finalized = False
        self._entered = False
        self._active = False
        self._training_pass = False
        self._global_supervised_tokens = None
        self.metrics = {"loss": 0.0, "layers": {}}

    def __enter__(self):
        if self._entered:
            raise RuntimeError("GlobalAuxLossStep is a single-use context")
        for _, router in self.routers:
            if getattr(router, "_global_aux_loss_step", None) is not None:
                raise RuntimeError("A router already belongs to a global auxiliary-loss step")
            if router.tp_group.size() != 1:
                raise ValueError("Step-scoped global auxiliary loss currently requires TP1")
            if (
                router.is_mtp_layer
                and router.config.mtp_use_repeated_layer
                and (router.config.mtp_num_layers or 0) > 1
            ):
                raise ValueError(
                    "Repeated MTP layers require separate per-depth auxiliary-loss statistics; "
                    "GlobalAuxLossStep does not support a shared router across multiple depths"
                )
            if self.two_pass and not router.calculate_per_token_loss:
                raise ValueError("Two-pass auxiliary loss requires calculate_per_token_loss=True")
            if not self.two_pass and router.config.recompute_granularity is not None:
                raise ValueError(
                    "Step-scoped global auxiliary loss requires retained forward graphs"
                )
        for _, router in self.routers:
            router._global_aux_loss_step = self
        self._entered = True
        self._active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for _, router in self.routers:
            del router._global_aux_loss_step
        self._active = False

    def set_round(self, round_id: int) -> None:
        """Select the decoder round before its forward; no communication occurs."""
        if self._active and self._training_pass:
            if round_id not in self._rounds:
                raise ValueError("The training pass must replay a collected decoder round")
            self._round = round_id
            return
        if not self._active or self._finalized:
            raise RuntimeError("Select rounds inside an unfinished auxiliary-loss context")
        if round_id in self._rounds:
            raise ValueError("Each decoder round must be selected exactly once")
        self._rounds.add(round_id)
        self._round = round_id

    def record(self, router: TopKRouter, scores: torch.Tensor, routing_map: torch.Tensor) -> None:
        """Called by the router with already padding-masked scores and assignments."""
        if self._round is None or self._finalized:
            raise RuntimeError("Router statistics require a selected, unfinished decoder round")
        score_sum = scores.sum(dim=0)
        if self.two_pass:
            score_sum = score_sum.detach()
        counts = routing_map.detach().sum(dim=0)
        records = self._records[router]
        if self._round in records:
            if self.audit_replay:
                raise RuntimeError(
                    "Replay diagnostics require one statistics call per router/round; "
                    "a shared router needs separate call identifiers"
                )
            old_scores, old_counts = records[self._round]
            score_sum = old_scores + score_sum
            counts = old_counts + counts
        records[self._round] = (score_sum, counts)
        if self.audit_token_assignments:
            self._observe_assignments("auxiliary", router, routing_map)

    @property
    def collecting_statistics(self) -> bool:
        """Whether routers must collect global statistics even under no-grad."""
        return self.two_pass and self._active and not self._finalized

    @property
    def auditing_training(self) -> bool:
        """Include a checkpoint's original no-grad forward in diagnostic observations."""
        return self.audit_replay and self._active and self._training_pass

    @staticmethod
    def _pack_assignments(routing_map: torch.Tensor) -> torch.Tensor:
        """Keep exact assignment sets on CPU at one bit per token/expert, not a hash."""
        rows, experts = routing_map.shape
        padded_experts = (experts + 7) // 8 * 8
        values = torch.nn.functional.pad(routing_map.detach(), (0, padded_experts - experts))
        values = values.reshape(rows, padded_experts // 8, 8).to(torch.uint8)
        weights = 2 ** torch.arange(8, device=values.device, dtype=torch.int16)
        return (values * weights).sum(-1).to(device="cpu", dtype=torch.uint8)

    @torch.no_grad()
    def observe_dispatch(self, router, routing_map, padding_mask=None) -> None:
        """Audit the actual dispatch map separately from the auxiliary top-k map.

        Bias, grouped routing, and token dropping can make these maps different.
        This observation never modifies the map used by expert dispatch.
        """
        if not self.audit_token_assignments:
            return
        if padding_mask is not None:
            routing_map = routing_map & ~padding_mask.reshape(-1, 1)
        self._observe_assignments("dispatch", router, routing_map)

    def _observe_assignments(self, kind, router, routing_map):
        key = (router, self._round)
        observed = self._pack_assignments(routing_map)
        if not self._training_pass:
            self._assignment_records[kind][key] = observed
            return
        training = self._training_assignments[kind]
        recompute = key in training
        expected = (training if recompute else self._assignment_records[kind]).get(key)
        if not recompute:
            training[key] = observed
        shape_mismatch = expected is None or expected.shape != observed.shape
        if shape_mismatch:
            different_rows = torch.arange(observed.shape[0])
        else:
            different_rows = (observed != expected).any(dim=1).nonzero().flatten()
        examples = []
        for row in different_rows[:4].tolist():

            def experts_in(bits):
                if bits is None or row >= bits.shape[0]:
                    return None
                return [
                    byte * 8 + bit
                    for byte, value in enumerate(bits[row].tolist())
                    for bit in range(8)
                    if value & (1 << bit) and byte * 8 + bit < router.num_experts
                ]

            examples.append(
                {"row": row, "expected": experts_in(expected), "actual": experts_in(observed)}
            )
        self._assignment_comparisons.append(
            {
                "router": router,
                "round": self._round,
                "kind": kind,
                "phase": "recompute" if recompute else "training",
                "rows": observed.shape[0],
                "different_rows": different_rows.numel(),
                "shape_mismatch": shape_mismatch,
                "examples": examples,
            }
        )

    def begin_training(self, global_supervised_tokens: torch.Tensor) -> None:
        """Freeze statistics and enable native per-token autograd loss attachment.

        ``global_supervised_tokens`` is the unreplicated optimizer-step count
        used by native ``finalize_model_grads``; it is not a pack's token count.
        This method performs no collective. Replaying checkpointed layers only
        recomputes a differentiable local score sum against the fixed counts.
        """
        if not self.two_pass or not self._active or not self._finalized or self._training_pass:
            raise RuntimeError("Begin training once after finalizing a two-pass statistics scope")
        count = torch.as_tensor(global_supervised_tokens, device=self.device).detach()
        if count.numel() != 1 or not torch.isfinite(count).all() or count.item() < 0:
            raise ValueError("global_supervised_tokens must be a finite nonnegative scalar")
        self._global_supervised_tokens = count.reshape(()).clone()
        # Only O(num_routers * num_experts) fixed counts are needed in training.
        if not self.audit_replay:
            self._records.clear()
        self._training_pass = True

    def apply(
        self,
        router: TopKRouter,
        probs: torch.Tensor,
        scores: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> torch.Tensor:
        """Collect once in pass one, or attach an immutable objective in pass two."""
        if not self._training_pass:
            self.record(router, scores, routing_map)
            return probs
        if self.audit_replay:
            records = self._replay_records.setdefault(router, {})
            observed = (scores.detach().sum(dim=0), routing_map.detach().sum(dim=0))
            if self._round in records:
                # Recompute is a second observation of the same tokens, never a
                # second contribution to step-global training counts.
                self._recompute_records.append(
                    (router, self._round, records[self._round], observed)
                )
            else:
                records[self._round] = observed
            if self.audit_token_assignments:
                self._observe_assignments("auxiliary", router, routing_map)
        if not torch.is_grad_enabled():
            # Reentrant checkpoint's original forward has no graph; its grad-enabled
            # recomputation below attaches the auxiliary objective exactly once.
            return probs
        score_sum = scores.sum(dim=0)
        loss = self._loss(router, score_sum) * self._global_supervised_tokens
        return MoEAuxLossAutoScaler.apply(probs, loss)

    @torch.no_grad()
    def replay_metrics(self) -> dict:
        """Compare statistics, original training forwards, and recomputations.

        Call collectively after all backwards. The original training forward
        contributes to ``C_training`` exactly once, including under no-grad
        checkpointing. Recomputations are audited separately, never summed into
        that count. All diagnostic communication occurs here, not in routers.
        Exact token-set checks are opt-in and retain bit-packed CPU maps; no
        activations, logits, autograd graphs, or dense routing maps are retained.
        """
        if not self.audit_replay or not self._training_pass or not self._active:
            raise RuntimeError("Replay metrics require an active diagnostic training pass")
        fields = (
            "pairs",
            "missing",
            "unexpected",
            "histogram_mismatches",
            "histogram_l1",
            "score_difference_squared",
            "score_reference_squared",
        )
        totals = torch.zeros(
            (len(self.routers), len(fields)), device=self.device, dtype=torch.float64
        )
        maxima = torch.zeros(len(self.routers), device=self.device, dtype=torch.float64)
        recompute_totals = torch.zeros_like(totals)
        recompute_maxima = torch.zeros_like(maxima)
        global_training_counts = torch.zeros(
            (len(self.routers), max((router.num_experts for _, router in self.routers), default=0)),
            device=self.device,
            dtype=torch.float64,
        )
        local_mismatches = []

        def mismatch_order(item):
            return (
                item["round"],
                item["router_index"],
                item.get("rank", 0),
                item["phase"] != "training",
                item["kind"],
            )

        def keep_mismatch(item):
            # Capping in router iteration order could hide an earlier round in
            # a later layer. Keep the earliest candidates before gathering.
            local_mismatches.append(item)
            local_mismatches.sort(key=mismatch_order)
            del local_mismatches[32:]

        def compare(index, name, round_id, reference, actual, aggregate, maximum, phase):
            expected_scores, expected_counts = reference
            scores, counts = actual
            count_delta = (counts.double() - expected_counts.double()).abs().sum()
            score_delta = scores.double() - expected_scores.double()
            aggregate[index, 3] += count_delta.ne(0)
            aggregate[index, 4] += count_delta
            aggregate[index, 5] += score_delta.square().sum()
            aggregate[index, 6] += expected_scores.double().square().sum()
            score_maximum = score_delta.abs().max()
            maximum[index] = torch.maximum(maximum[index], score_maximum)
            # Scalar transfers are confined to this post-backward diagnostic call.
            if count_delta.item() or score_maximum.item():
                keep_mismatch(
                    {
                        "router": name,
                        "router_index": index,
                        "round": round_id,
                        "phase": phase,
                        "kind": "auxiliary_statistics",
                        "histogram_l1": count_delta.item(),
                        "score_sum_max_abs": score_maximum.item(),
                    }
                )

        for index, (_, router) in enumerate(self.routers):
            name = self.routers[index][0]
            reference = self._records[router]
            actual = self._replay_records.get(router, {})
            totals[index, :3] = torch.tensor(
                [
                    len(reference.keys() | actual.keys()),
                    len(reference.keys() - actual.keys()),
                    len(actual.keys() - reference.keys()),
                ],
                device=self.device,
            )
            zero = torch.zeros(router.num_experts, device=self.device, dtype=torch.float64)
            for round_id in sorted(reference.keys() | actual.keys()):
                compare(
                    index,
                    name,
                    round_id,
                    reference.get(round_id, (zero, zero)),
                    actual.get(round_id, (zero, zero)),
                    totals,
                    maxima,
                    "training",
                )
                if round_id in actual:
                    global_training_counts[index, : router.num_experts] += actual[round_id][1]
        router_indices = {router: index for index, (_, router) in enumerate(self.routers)}
        for router, round_id, reference, actual in self._recompute_records:
            index = router_indices[router]
            recompute_totals[index, 0] += 1
            compare(
                index,
                self.routers[index][0],
                round_id,
                reference,
                actual,
                recompute_totals,
                recompute_maxima,
                "recompute",
            )
        dist.all_reduce(totals, group=self.reduction_group)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX, group=self.reduction_group)
        dist.all_reduce(global_training_counts, group=self.reduction_group)
        dist.all_reduce(recompute_totals, group=self.reduction_group)
        dist.all_reduce(recompute_maxima, op=dist.ReduceOp.MAX, group=self.reduction_group)
        totals, maxima = totals.cpu(), maxima.cpu()

        def summarize(aggregate, maximum):
            aggregate, maximum = aggregate.cpu(), maximum.cpu()
            layers = {}
            for (name, _), values, peak in zip(self.routers, aggregate.tolist(), maximum.tolist()):
                result = dict(zip(fields, values))
                result["score_sum_relative_l2"] = (
                    (values[5] / values[6]) ** 0.5
                    if values[6]
                    else (0.0 if not values[5] else None)
                )
                result["score_sum_max_abs"] = peak
                layers[name] = result
            return {
                "layers": layers,
                **{key: int(aggregate[:, i].sum()) for i, key in enumerate(fields[:5])},
                "score_sum_max_abs": float(maximum.max()) if maximum.numel() else 0.0,
            }

        result = summarize(totals, maxima)
        result["scope"] = "statistics pass vs original training forward; recomputations separate"
        result["recompute"] = summarize(recompute_totals, recompute_maxima)
        result["recompute"][
            "scope"
        ] = "observed router calls only; an early-stopped checkpoint may not replay every router"
        result["global_histogram_mismatches"] = 0
        result["global_histogram_l1"] = 0
        for index, (name, router) in enumerate(self.routers):
            expected = self._global_counts[router][0].cpu()
            actual = global_training_counts[index, : router.num_experts].cpu()
            count_delta = int((actual - expected).abs().sum())
            result["global_histogram_mismatches"] += int(count_delta != 0)
            result["global_histogram_l1"] += count_delta
            result["layers"][name]["global_histogram_l1"] = count_delta
            if count_delta:
                result["layers"][name]["C_stats"] = expected.to(torch.int64).tolist()
                result["layers"][name]["C_training"] = actual.to(torch.int64).tolist()

        assignment_keys = [
            (kind, phase)
            for kind in ("auxiliary", "dispatch")
            for phase in ("training", "recompute")
        ]
        assignment_fields = (
            "pairs",
            "rows",
            "mismatched_pairs",
            "different_rows",
            "shape_mismatches",
        )
        assignments = torch.zeros((4, 5), device=self.device, dtype=torch.int64)
        for comparison in self._assignment_comparisons:
            index = assignment_keys.index((comparison["kind"], comparison["phase"]))
            different = bool(comparison["different_rows"] or comparison["shape_mismatch"])
            assignments[index] += torch.tensor(
                [
                    1,
                    comparison["rows"],
                    different,
                    comparison["different_rows"],
                    comparison["shape_mismatch"],
                ],
                device=self.device,
            )
            if different:
                keep_mismatch(
                    {
                        **comparison,
                        "router": self.routers[router_indices[comparison["router"]]][0],
                        "router_index": router_indices[comparison["router"]],
                    }
                )
        dist.all_reduce(assignments, group=self.reduction_group)
        result["token_assignments"] = {
            "enabled": self.audit_token_assignments,
            "representation": "exact bit-packed expert sets, including padding rows",
            **{
                f"{kind}_{phase}": dict(zip(assignment_fields, values))
                for (kind, phase), values in zip(assignment_keys, assignments.cpu().tolist())
            },
        }
        for mismatch in local_mismatches:
            mismatch["rank"] = dist.get_rank()
        gathered = [None] * dist.get_world_size(self.reduction_group)
        dist.all_gather_object(gathered, local_mismatches, group=self.reduction_group)
        result["first_mismatches"] = sorted(
            (item for rank_records in gathered for item in rank_records), key=mismatch_order
        )[:32]
        result["first_mismatches_limit"] = 32
        return result

    @staticmethod
    def _coefficient(router):
        coefficient = router.get_aux_loss_coeff("global_aux_loss")
        if router.is_mtp_layer and router.config.mtp_use_repeated_layer:
            coefficient /= router.config.mtp_num_layers
        return coefficient

    def _loss(self, router, score_sum):
        counts, num_tokens = self._global_counts[router]
        return switch_load_balancing_loss_func(
            probs=score_sum.unsqueeze(0),
            tokens_per_expert=counts,
            total_num_tokens=num_tokens.clamp_min(1),
            topk=router.topk,
            num_experts=router.num_experts,
            moe_aux_loss_coeff=self._coefficient(router),
        )

    @torch.no_grad()
    def finalize(self) -> dict:
        """Reduce sufficient statistics once per router, after all decoder forwards.

        All ranks, including those with no valid routed tokens, must call this
        once in the same step. No decoder pack or CP-group collective is used.
        Returns global coefficient-weighted loss and routing counts for logging.
        """
        if not self._active or self._finalized:
            raise RuntimeError("Finalize an active auxiliary-loss step exactly once")
        for name, router in self.routers:
            # One small reduction carries both exact counts and detached probability
            # sums. Float64 preserves integer counts and improves metric accumulation.
            statistics = torch.zeros(
                (2, router.num_experts), device=self.device, dtype=torch.float64
            )
            for scores, counts in self._records[router].values():
                statistics[0].add_(counts)
                statistics[1].add_(scores.detach())
            dist.all_reduce(statistics, group=self.reduction_group)
            num_tokens = statistics[0].sum() / router.topk
            self._global_counts[router] = (statistics[0], num_tokens)
            loss = self._loss(router, statistics[1]).item()
            self.metrics["layers"][name] = {
                "loss": loss,
                "routed_tokens": int(num_tokens.item()),
                "tokens_per_expert": statistics[0].tolist(),
            }
            self.metrics["loss"] += loss
        self._finalized = True
        return self.metrics

    def loss_for_round(self, round_id: int) -> torch.Tensor:
        """Return this rank/round's differentiable contribution to the global loss."""
        if self.two_pass:
            raise RuntimeError("Two-pass losses are attached by the router during training")
        if not self._finalized or round_id not in self._rounds:
            raise RuntimeError("Round losses are available after statistics have been finalized")
        loss = torch.zeros((), device=self.device, dtype=torch.float64)
        for _, router in self.routers:
            record = self._records[router].get(round_id)
            if record is not None:
                loss = loss + self._loss(router, record[0])
        return loss
