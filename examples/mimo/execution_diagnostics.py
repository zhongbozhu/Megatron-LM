# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Opt-in sample-aligned decoder probes; never changes tensors or gradients.

Files contain sampled real-token activations and gradients, not whole-model
equivalence evidence. CP input vision rows are checked exhaustively against the
received boundary leaf. Recomputed invocations are kept separate from forward.
"""

import math
from pathlib import Path

import torch

from examples.mimo.moe_boundary_diagnostics import MoEBoundaryDiagnostics
from megatron.core.packed_seq_params import PackedSeqParams


def probe_positions(length, vision_positions=(), count=32):
    """Stable sample-relative probes, independent of pack offsets and CP size."""
    if length <= 0:
        return []
    positions = set(torch.linspace(0, length - 1, min(length, count)).long().tolist())
    positions.update(range(min(length, 8)))
    positions.update(range(max(0, length - 8), length))
    vision_positions = list(vision_positions)
    if vision_positions:
        positions.update((vision_positions[0], vision_positions[-1]))
    return sorted(positions)


class ExecutionDiagnostics:
    """Install read-only hooks for one native source step, then remove them."""

    def __init__(
        self,
        model,
        directory,
        rank,
        step,
        samples,
        gdn_layers=(0,),
        moe_layers=(),
        capture_full_input=False,
    ):
        self.model = model
        self.capture_full_input = capture_full_input
        layers = model.language_model.decoder.layers
        self.gdn_layers = tuple(dict.fromkeys(gdn_layers))
        for index in self.gdn_layers:
            if not isinstance(index, int) or not 0 <= index < len(layers):
                raise ValueError(f'Invalid GDN diagnostic layer index: {index}')
            if not hasattr(getattr(layers[index], 'self_attention', None), 'in_proj'):
                raise ValueError(f'Diagnostic layer {index} is not a GDN layer')
        self.gdn_names = {f'layer{index:02d}.attention': str(index) for index in self.gdn_layers}
        self.directory = Path(directory) / f'step{step:08d}' / f'rank{rank:05d}'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.samples = samples
        self.step = step
        self.current = None
        self.handles = []
        self.moe_diagnostics = MoEBoundaryDiagnostics(layers, moe_layers, lambda: self.current)
        for index, layer in enumerate(model.language_model.decoder.layers):
            name = f'layer{index:02d}'
            self.handles.append(layer.register_forward_hook(self._output_hook(name)))
            if hasattr(layer, 'self_attention'):
                self.handles.append(
                    layer.self_attention.register_forward_hook(
                        self._output_hook(name + '.attention')
                    )
                )
                # Sample the first GDN input projections to separate GEMM
                # differences from recurrence differences without dumping all QKV.
                if (index < 2 or index in self.gdn_layers) and hasattr(
                    layer.self_attention, 'in_proj'
                ):
                    self.handles.append(
                        layer.self_attention.in_proj.register_forward_hook(
                            self._output_hook(name + '.input_projection')
                        )
                    )
            self.handles.append(
                layer.mlp.router.register_forward_hook(self._output_hook(name + '.router'))
            )
            self.handles.append(
                layer.mlp.router.register_forward_pre_hook(
                    self._router_input_hook(name + '.router_input'), with_kwargs=True
                )
            )
        self.handles.append(
            model.language_model.decoder.layers[0].register_forward_pre_hook(
                self._input_hook, with_kwargs=True
            )
        )
        for index in self.gdn_layers:
            self.handles.append(
                layers[index].self_attention.register_forward_pre_hook(
                    self._gdn_input_hook(index), with_kwargs=True
                )
            )

    def begin(self, item, phase, transfer):
        if self.current is not None:
            raise RuntimeError('Flush previous execution diagnostic round first')
        kwargs = item['kwargs']
        tokens = kwargs['input_ids'].flatten()
        packed = PackedSeqParams(qkv_format='thd', **kwargs['packing_kwargs'])
        # Reuse the actual partition adapter, including its runtime CP group.
        index = torch.arange(tokens.numel(), device=tokens.device).unsqueeze(0)
        local = index
        if self.model.partition_adapter is not None:
            _, local, _, _ = self.model.partition_adapter.shard(None, index, None, packed)
        local = local.flatten()
        inverse = torch.full_like(tokens, -1)
        inverse[local] = torch.arange(local.numel(), device=tokens.device)
        sample_ids = item['diagnostic_sample_ids']
        boundaries = packed.cu_seqlens_q_padded.tolist()
        rows, keys = [], []
        image_token = self.model.special_token_ids['images']
        for slot, sid in enumerate(sample_ids):
            sample = self.samples[sid]
            length = int(sample['original_seq_len'])
            image_positions = (
                (sample['tokens'][:length] == image_token).nonzero().flatten().tolist()
            )
            for position in probe_positions(length, image_positions):
                local_row = int(inverse[boundaries[slot] + position])
                if local_row >= 0:
                    rows.append(local_row)
                    keys.append((sid, position))
        image_positions = (tokens == image_token).nonzero().flatten()
        vision_local = inverse[image_positions]
        owned = vision_local >= 0
        self.rows = torch.tensor(rows, device=tokens.device, dtype=torch.long)
        self.vision_rows = vision_local[owned]
        self.vision_feature_rows = torch.arange(image_positions.numel(), device=tokens.device)[
            owned
        ]
        self.transfer = transfer
        self.current = dict(
            step=self.step,
            round=item['round_id'],
            phase=phase,
            cp_ranks=list(transfer.plans[transfer.local_plan].cp_ranks),
            keys=keys,
            local_tokens=local.numel(),
            sample_ids=list(sample_ids),
            sample_lengths=[int(self.samples[sid]['original_seq_len']) for sid in sample_ids],
            physical_indices=local.cpu(),
            padded_boundaries=boundaries,
            logical_boundaries=packed.cu_seqlens_q.tolist(),
            records={},
            cp_input=[],
            gdn_layers={},
        )

    @staticmethod
    def _full_difference(actual, expected):
        exact = torch.equal(actual, expected)
        return exact, 0.0 if exact else float((actual.float() - expected.float()).abs().max())

    def _gdn_input_hook(self, index):
        def capture(module, args, kwargs):
            if self.current is None or self.current['phase'] not in ('statistics', 'training'):
                return
            hidden = kwargs.get('hidden_states', args[0] if args else None)
            self._record(f'layer{index:02d}.gdn_input', hidden)
            value = hidden.detach().cpu()
            entry = self.current['gdn_layers'].setdefault(str(index), {})
            if 'input' not in entry:
                # Recurrence replay needs the whole original physical pack.
                entry['input'] = value
                entry['recompute_checks'] = []
                if index == 0:
                    self.current['gdn_input'] = value
            else:
                exact, maximum = self._full_difference(value, entry['input'])
                check = dict(input_exact=exact, input_max_abs=maximum, output_observed=False)
                if not exact:
                    check['input'] = value
                entry['recompute_checks'].append(check)

        return capture

    def _record(self, name, tensor):
        if self.current is None or not isinstance(tensor, torch.Tensor):
            return
        if tensor.ndim < 2:
            return
        value = tensor.reshape(-1, tensor.shape[-1])
        if value.shape[0] != self.current['local_tokens']:
            raise ValueError(f'Probe token layout changed at {name}: {tensor.shape}')
        record = dict(
            grad_enabled=torch.is_grad_enabled(),
            value=value.detach().index_select(0, self.rows).cpu(),
        )
        self.current['records'].setdefault(name, []).append(record)
        if self.current['phase'] == 'training' and tensor.requires_grad:
            rows = self.rows
            current = self.current

            def save_gradient(gradient):
                record['gradient'] = (
                    gradient.detach().reshape(-1, gradient.shape[-1]).index_select(0, rows).cpu()
                )
                if name in self.gdn_names:
                    index = self.gdn_names[name]
                    entry = current['gdn_layers'].get(index)
                    if entry is not None:
                        entry['output_gradient'] = gradient.detach().cpu()
                        if index == '0':
                            current['gdn_output_gradient'] = entry['output_gradient']

            tensor.register_hook(save_gradient)

    def _output_hook(self, name):
        def capture(module, args, output):
            values = output if isinstance(output, tuple) else (output,)
            if self.current is not None and name in self.gdn_names:
                entry = self.current['gdn_layers'].get(self.gdn_names[name])
                if entry is not None:
                    value = values[0].detach().cpu()
                    if 'output' not in entry:
                        entry['output'] = value
                    else:
                        exact, maximum = self._full_difference(value, entry['output'])
                        check = entry['recompute_checks'][-1]
                        check.update(
                            output_observed=True, output_exact=exact, output_max_abs=maximum
                        )
                        if not exact:
                            check['output'] = value
            self._record(name, values[0])
            if name.endswith('.router'):
                self._record(name + '.choices', values[1])

        return capture

    def _router_input_hook(self, name):
        def capture(module, args, kwargs):
            self._record(name, kwargs.get('input', args[0] if args else None))

        return capture

    def _input_hook(self, module, args, kwargs):
        if self.current is None:
            return
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        self._record('decoder_input', hidden)
        if self.capture_full_input:
            full_record = dict(value=hidden.detach().cpu(), grad_enabled=torch.is_grad_enabled())
            self.current.setdefault('full_decoder_input', []).append(full_record)
            if self.current['phase'] == 'training' and hidden.requires_grad:

                def capture_gradient(gradient):
                    full_record['gradient'] = gradient.detach().cpu()

                hidden.register_hook(capture_gradient)
        value = hidden.reshape(-1, hidden.shape[-1])
        actual = value.detach().index_select(0, self.vision_rows)
        expected = self.transfer.features.detach().index_select(0, self.vision_feature_rows)
        if actual.shape != expected.shape:
            raise ValueError('CP vision rows do not match receiver feature rows')
        record = dict(
            feature_rows=self.vision_feature_rows.cpu(),
            elements=actual.numel(),
            exact=torch.equal(actual, expected),
            max_abs_error=(float((actual - expected).abs().max()) if actual.numel() else 0.0),
        )
        self.current['cp_input'].append(record)
        if self.current['phase'] == 'training' and hidden.requires_grad:
            rows = self.vision_rows

            def save_gradient(gradient):
                record['gradient'] = (
                    gradient.detach().reshape(-1, gradient.shape[-1]).index_select(0, rows).cpu()
                )

            hidden.register_hook(save_gradient)

    def flush(self):
        if self.current is None:
            return
        path = self.directory / f"{self.current['phase']}-round{self.current['round']:04d}.pt"
        if path.exists():
            raise FileExistsError(f'Execution diagnostic already exists: {path}')
        # Compare the actual first decoder input gradient to the local leaf,
        # before any reverse P2P/CP reduction; padding/nonowners must be zero.
        if self.current['phase'] == 'training':
            leaf_gradient = self.transfer.features.grad
            nonowners = torch.ones(
                self.transfer.features.shape[0],
                device=self.transfer.features.device,
                dtype=torch.bool,
            )
            nonowners[self.vision_feature_rows] = False
            unwanted = None if leaf_gradient is None else leaf_gradient.detach()[nonowners]
            self.current['nonowner_gradient'] = dict(
                rows=int(nonowners.sum()),
                nonzero_elements=int(torch.count_nonzero(unwanted)) if unwanted is not None else 0,
                max_abs=(
                    float(unwanted.abs().max())
                    if unwanted is not None and unwanted.numel()
                    else 0.0
                ),
            )
            for record in self.current['cp_input']:
                if 'gradient' not in record:
                    continue
                actual = record['gradient']
                expected = (
                    torch.zeros_like(actual)
                    if leaf_gradient is None
                    else leaf_gradient.detach().cpu().index_select(0, record['feature_rows'])
                )
                record['leaf_gradient_exact'] = torch.equal(actual, expected)
                record['leaf_gradient_max_abs_error'] = (
                    float((actual - expected).abs().max()) if actual.numel() else 0.0
                )
        torch.save(self.current, path)
        self.current = None
        self.transfer = None

    def close(self):
        try:
            self.flush()
        finally:
            self.moe_diagnostics.close()
            for handle in self.handles:
                handle.remove()
            self.handles = []


def _load_probes(directory, step, phase):
    result = {}
    inputs = []
    nonowners = []
    coverage = dict(rounds=0, missing_input_rounds=0, missing_visual_gradient_rounds=0)
    paths = sorted((Path(directory) / f'step{step:08d}').glob(f'rank*/{phase}-round*.pt'))
    if not paths:
        raise FileNotFoundError(f'No {phase} probes for step {step} in {directory}')
    for path in paths:
        data = torch.load(path, map_location='cpu', weights_only=True)
        coverage['rounds'] += 1
        coverage['missing_input_rounds'] += not bool(data['cp_input'])
        coverage['missing_visual_gradient_rounds'] += any(
            record['elements'] for record in data['cp_input']
        ) and not any('leaf_gradient_exact' in record for record in data['cp_input'])
        inputs.extend(data['cp_input'])
        if 'nonowner_gradient' in data:
            nonowners.append(data['nonowner_gradient'])
        for name, invocations in data['records'].items():
            values = dict(value=invocations[0]['value'])
            backwards = [record['gradient'] for record in invocations if 'gradient' in record]
            if backwards:
                values['gradient'] = backwards[-1]
            for kind, value in values.items():
                target = result.setdefault(name + '/' + kind, {})
                for key, row in zip(data['keys'], value):
                    key = tuple(key)
                    if key in target:
                        raise ValueError(f'Duplicate sample-token owner in {path}: {key}')
                    target[key] = row.clone()
    return result, inputs, nonowners, coverage


def compare_execution_directories(
    reference, candidate, step, *, phase='training', reference_phase=None
):
    """Compare probes by logical sample/token identity, independent of ranks/packs.

    Gradients are raw pre-finalization derivatives; callers must use identical
    global loss scaling/counts. This function does not assign a pass tolerance.
    """
    expected, _, _, _ = _load_probes(reference, step, reference_phase or phase)
    observed, inputs, nonowners, coverage = _load_probes(candidate, step, phase)
    statistics = {}
    for name in sorted(expected.keys() & observed.keys()):
        ref, cur = expected[name], observed[name]
        if ref.keys() != cur.keys():
            raise ValueError(f'Sample-token coverage differs at {name}')
        ref_sq = diff_sq = cur_sq = maximum = 0.0
        elements = changed_rows = 0
        for key, ref_row in ref.items():
            cur_row = cur[key]
            if ref_row.shape != cur_row.shape or ref_row.dtype != cur_row.dtype:
                raise ValueError(f'Tensor schema differs at {name}/{key}')
            a, b = ref_row.double(), cur_row.double()
            delta = b - a
            ref_sq += float(a.square().sum())
            cur_sq += float(b.square().sum())
            diff_sq += float(delta.square().sum())
            maximum = max(maximum, float(delta.abs().max()))
            changed_rows += not torch.equal(ref_row, cur_row)
            elements += a.numel()
        statistics[name] = dict(
            elements=elements,
            tokens=len(ref),
            changed_tokens=changed_rows,
            exact=changed_rows == 0,
            max_abs_error=maximum,
            relative_l2=math.sqrt(diff_sq / ref_sq) if ref_sq else (math.inf if diff_sq else 0.0),
            reference_l2=math.sqrt(ref_sq),
            candidate_l2=math.sqrt(cur_sq),
        )
    changed = [
        name for name, value in statistics.items() if name.endswith('/value') and not value['exact']
    ]

    def execution_order(name):
        if name.startswith('decoder_input'):
            return (-1, 0)
        index = int(name[5:7])
        stage = next(
            (
                i
                for i, part in enumerate(
                    (
                        '.gdn_input/',
                        '.input_projection/',
                        '.attention/',
                        '.router_input/',
                        '.router',
                        '/',
                    )
                )
                if part in name
            ),
            5,
        )
        return (index, stage)

    changed.sort(key=execution_order)
    gradients = [record for record in inputs if 'leaf_gradient_exact' in record]
    return dict(
        kind='sampled_decoder_execution',
        step=step,
        phase=phase,
        reference_phase=reference_phase or phase,
        first_changed_activation=changed[0] if changed else None,
        missing_records=sorted(expected.keys() - observed.keys()),
        extra_records=sorted(observed.keys() - expected.keys()),
        cp_input_coverage=coverage,
        cp_vision_inputs_exact=(
            all(record['exact'] for record in inputs) and not coverage['missing_input_rounds']
            if inputs
            else None
        ),
        cp_vision_input_elements=sum(record['elements'] for record in inputs),
        local_vision_gradient_records=len(gradients),
        local_vision_gradient_elements=sum(record['elements'] for record in gradients),
        local_vision_gradients_exact=(
            all(record['leaf_gradient_exact'] for record in gradients)
            and not coverage['missing_visual_gradient_rounds']
            if gradients
            else None
        ),
        nonowner_vision_gradients_zero=(
            all(record['nonzero_elements'] == 0 for record in nonowners)
            and len(nonowners) == coverage['rounds']
            if nonowners
            else None
        ),
        nonowner_vision_rows=sum(record['rows'] for record in nonowners),
        statistics=statistics,
    )
