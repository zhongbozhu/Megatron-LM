# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Opt-in observation of one actual native AdamW update, with bounded host memory.

Install after native optimizer/checkpoint setup and before training. The next
``optimizer.step()`` is observed; all later calls are unchanged. Gradients must
already have passed native finalization. This does not implement an optimizer.

References store only post-step FP32 master weights and Adam moments (12 bytes
per uniquely owned element). Pre-step weights use temporary disk space (4 bytes
per element); initial moments are hashed, never copied in full. Comparisons
require identical optimizer ownership and starting state. Run the existing
``validate_full_gradients`` separately for elementwise *unclipped* gradient
comparison; here gradients are only hashed and their complete norms observed.
"""

import hashlib
import json
import math
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from examples.mimo.full_gradient_validation import _accumulate_statistics, _owned_gradients
from examples.mimo.gradient_diagnostics import _COMPONENTS, _component_names

_FIELDS = ('master_update', 'decay_removed_update', 'exp_avg', 'exp_avg_sq')
_MOMENTS = ('exp_avg', 'exp_avg_sq')


class _ColdMomentBands:
    """Attribute actual update error using observed moments, not a simulated Adam."""

    bands = ('reference_zero', '(0,0.1]', '(0.1,1]', '(1,10]', '(10,100]', '(100,inf)')
    signs = ('both_zero', 'reference_zero_only', 'current_zero_only', 'opposite', 'same')
    fields = (
        'elements',
        'reference_m_square',
        'delta_m_square',
        'reference_update_square',
        'delta_update_square',
    )

    def __init__(self):
        self.values = torch.zeros((1 + len(_COMPONENTS), 30, 5), dtype=torch.float64)
        self.edges = torch.tensor([0.1, 1.0, 10.0, 100.0], dtype=torch.float64)

    def add(self, metadata, current, reference, update, reference_update, beta1, eps):
        scale = (1 - beta1) * eps
        if not 0 <= beta1 < 1 or not math.isfinite(scale) or scale <= 0:
            raise ValueError('Cold moment bands require beta1 in [0,1) and positive epsilon')
        current, reference = current.double(), reference.double()
        current_zero, reference_zero = current == 0, reference == 0
        band = torch.bucketize(reference.abs() / scale, self.edges) + 1
        band[reference_zero] = 0
        sign = torch.full_like(band, 4)
        sign[(current > 0) != (reference > 0)] = 3
        sign[current_zero] = 2
        sign[reference_zero] = 1
        sign[current_zero & reference_zero] = 0
        cells = band * 5 + sign
        statistics = torch.stack(
            [torch.bincount(cells, minlength=30).double()]
            + [
                torch.bincount(cells, weights=weight, minlength=30)
                for weight in (
                    reference.square(),
                    (current - reference).square(),
                    reference_update.double().square(),
                    (update.double() - reference_update.double()).square(),
                )
            ],
            dim=-1,
        )
        indices = [0] + [1 + _COMPONENTS.index(name) for name in metadata['components']]
        self.values[indices] += statistics

    def finish(self, group, device):
        if dist.is_initialized():
            values = self.values.to(device)
            dist.all_reduce(values, group=group)
            self.values = values.cpu()

        def report(values):
            cells = values.reshape(6, 5, 5)
            count = cells[:, :, 0].sum().item()
            active = count - cells[:, 0, 0].sum().item()
            error = cells[:, :, 4].sum().item()
            return dict(
                elements=int(count),
                elements_excluding_joint_zero=int(active),
                opposite_sign_fraction=cells[:, 3, 0].sum().item() / active if active else None,
                delta_update_square=error,
                opposite_sign_update_error_fraction=(
                    cells[:, 3, 4].sum().item() / error if error else None
                ),
                cells=[
                    dict(
                        band=self.bands[b],
                        sign=self.signs[s],
                        **dict(zip(self.fields, cells[b, s].tolist())),
                    )
                    for b in range(6)
                    for s in range(5)
                    if cells[b, s, 0] > 0
                ],
            )

        return dict(
            interpretation='rho=abs(native FP32 post-step m_ref)/((1-beta1)*eps): nominal cold '
            'clipped-gradient scale proxy, not an exact inversion of FusedAdam FP32 arithmetic '
            'or the pre-clipping gradient. Signs describe stored moments. Actual master-update '
            'error includes native rounding; no optimizer formula is substituted. Bands are '
            'asymmetric and reference-defined; bucket association alone does not establish cause.',
            totals=report(self.values[0]),
            components={
                name: report(self.values[index + 1]) for index, name in enumerate(_COMPONENTS)
            },
        )


def _json_value(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError('Optimizer group metadata must contain only scalar tensors')
        value = value.item()
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite optimizer/context scalar metadata')
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'Unsupported optimizer metadata: {type(value).__name__}')


def _chunks(tensor, count):
    if tensor.dtype != torch.float32 or not tensor.is_contiguous():
        raise ValueError('Diagnostics require contiguous FP32 master parameters, moments and grads')
    flat = tensor.detach().view(-1)
    for start in range(0, flat.numel(), count):
        yield flat[start : start + count].to(device='cpu', copy=True)


def _fingerprint(tensor, count, stream=None):
    digest = hashlib.sha256()
    square = 0.0
    for chunk in _chunks(tensor, count):
        if not torch.isfinite(chunk).all():
            raise ValueError('Nonfinite optimizer input/state')
        raw = memoryview(chunk.numpy())
        digest.update(raw)
        if stream is not None:
            stream.write(raw)
        square += chunk.double().square().sum().item()
    return dict(sha256=digest.hexdigest(), elements=tensor.numel(), square=square)


def _read_chunk(stream, elements):
    raw = bytearray(stream.read(elements * 4))
    if len(raw) != elements * 4:
        raise ValueError('Truncated optimizer diagnostic vector')
    return torch.frombuffer(raw, dtype=torch.float32)


def _children(optimizer):
    children = getattr(optimizer, 'chained_optimizers', None)
    if children is None:
        return [optimizer]
    return [child for parent in children for child in _children(parent)]


@dataclass
class _Shard:
    metadata: dict
    master: torch.Tensor
    gradient: torch.Tensor
    inner: object
    param_group: dict


def _owned_optimizer_shards(models, optimizer):
    """Cross-check native DDP ownership against native DistributedOptimizer views."""
    parameters = {
        (index, _component_names(name)[0]): parameter
        for index, model in enumerate(models)
        for name, parameter in model.named_parameters()
    }
    by_parameter, groups, native_parameters = {}, [], set()
    for child_index, child in enumerate(_children(optimizer)):
        config = child.config
        if (
            getattr(config, 'optimizer', None) != 'adam'
            or not getattr(config, 'decoupled_weight_decay', False)
            or getattr(config, 'use_precision_aware_optimizer', False)
            or getattr(config, 'optimizer_cpu_offload', False)
            or getattr(config, 'optimizer_cuda_graph', False)
            or getattr(child, 'grad_scaler', None) is not None
        ):
            raise ValueError('Diagnostics support ordinary FP32-master, unscaled native AdamW only')
        if getattr(child, 'is_stub_optimizer', False):
            continue
        inner = child.optimizer
        is_fused_adam = type(inner).__name__ == 'FusedAdam' and type(inner).__module__.startswith(
            ('transformer_engine.', 'apex.')
        )
        if not isinstance(inner, torch.optim.AdamW) and not is_fused_adam:
            raise ValueError(f'Unsupported native optimizer: {type(inner)}')
        if is_fused_adam and not getattr(inner, 'adam_w_mode', False):
            raise ValueError('FusedAdam must use decoupled AdamW weight decay')
        if getattr(inner, 'capturable', False) or getattr(inner, 'use_decoupled_grad', False):
            raise ValueError(
                'Capturable/decoupled-gradient optimizers are outside diagnostic scope'
            )
        locations = {}
        for group_index, group in enumerate(inner.param_groups):
            if group.get('amsgrad', False) or group.get('maximize', False):
                raise ValueError('AMSGrad/maximize variants are outside diagnostic scope')
            groups.append(
                dict(
                    optimizer=child_index,
                    group=group_index,
                    implementation=f'{type(inner).__module__}.{type(inner).__name__}',
                    clip_grad=float(config.clip_grad),
                    settings=_json_value({k: v for k, v in group.items() if k != 'params'}),
                    elements=sum(p.numel() for p in group['params']),
                )
            )
            for master in group['params']:
                if id(master) in native_parameters:
                    raise ValueError('A master parameter belongs to multiple optimizer groups')
                native_parameters.add(id(master))
                locations[id(master)] = (group_index, group)
        for model_groups, master_groups in (
            (child.model_float16_groups, child.shard_fp32_from_float16_groups),
            (child.model_fp32_groups, child.shard_fp32_groups),
        ):
            if len(model_groups) != len(master_groups):
                raise ValueError('Native model/master group count mismatch')
            for model_group, master_group in zip(model_groups, master_groups):
                if len(model_group) != len(master_group):
                    raise ValueError('Native model/master parameter count mismatch')
                for parameter, master in zip(model_group, master_group):
                    if parameter in by_parameter or id(master) not in locations:
                        raise ValueError('Duplicate or unregistered native optimizer owner')
                    group_index, group = locations[id(master)]
                    owned = child._get_model_param_range_map(parameter)['param']
                    by_parameter[parameter] = (
                        master,
                        inner,
                        group,
                        child_index,
                        group_index,
                        owned.start,
                        owned.end,
                    )
    shards = []
    for metadata, gradient in _owned_gradients(models):
        parameter = parameters[(metadata['model'], metadata['name'])]
        if parameter not in by_parameter:
            raise ValueError(f'Missing native optimizer owner: {metadata["name"]}')
        master, inner, group, child, group_index, start, end = by_parameter.pop(parameter)
        if (
            start != metadata['parameter_offset']
            or end - start != metadata['elements']
            or master.numel() != metadata['elements']
        ):
            raise ValueError(f'DDP/optimizer ownership mismatch: {metadata["name"]}')
        metadata = dict(
            metadata,
            optimizer=child,
            optimizer_group=group_index,
            grad_norm_group=getattr(master, 'grad_norm_group', None),
        )
        shards.append(_Shard(metadata, master, gradient, inner, group))
    if by_parameter or {id(shard.master) for shard in shards} != native_parameters:
        raise ValueError(
            'DDP ownership does not cover every native optimizer parameter exactly once'
        )
    return shards, groups


def _state_description(shard, expected_step, chunk_elements, *, after=False):
    """Absent cold state is explicit evidence, never a substitute for warm state."""
    state = shard.inner.state.get(shard.master, {})
    allowed = {*_MOMENTS, 'step'}
    if set(state) - allowed:
        raise ValueError(f'Unsupported Adam state keys: {set(state) - allowed}')
    parameter_step = _json_value(state.get('step'))
    group_step = _json_value(shard.param_group.get('step'))
    is_torch_adam = isinstance(shard.inner, torch.optim.AdamW)
    counter = parameter_step if is_torch_adam else group_step
    if counter is not None and counter != expected_step:
        raise ValueError(f'Adam step mismatch: expected {expected_step}, observed {counter}')
    cold = expected_step == 0 and not after
    if counter is None and not cold:
        raise ValueError('Missing native Adam step state')
    present = [name in state for name in _MOMENTS]
    if any(present) != all(present) or (not all(present) and not cold):
        raise ValueError('Missing/incomplete native Adam moment state')
    if not all(present) and state:
        raise ValueError('Missing Adam moments in nonempty state; this is not native lazy state')
    if isinstance(shard.inner, torch.optim.AdamW) and all(present) and parameter_step is None:
        raise ValueError('Missing native torch AdamW per-parameter step state')
    if not isinstance(shard.inner, torch.optim.AdamW) and not cold and group_step is None:
        raise ValueError('Missing native FusedAdam group step state')
    moments = {}
    for name in _MOMENTS:
        if name not in state:
            moments[name] = None
            continue
        if state[name].shape != shard.master.shape:
            raise ValueError(f'Adam {name} shape mismatch')
        # Post-state bytes are hashed in the comparison pass; avoid copying the
        # same complete moments from the GPU twice.
        moments[name] = (
            {'elements': state[name].numel()}
            if after
            else _fingerprint(state[name], chunk_elements)
        )
        if cold and moments[name].get('square') != 0:
            raise ValueError('Explicit cold initialization requires zero Adam moments')
        moments[name].pop('square', None)
    return dict(
        parameter_step=parameter_step,
        group_step=group_step,
        counter_source='parameter' if is_torch_adam else 'group',
        initialization='native_lazy' if not all(present) else 'materialized',
        moments=moments,
    )


def _snapshot_groups(optimizer, groups, expected_step):
    """FusedAdam advances group counters even when a local group owns no parameters."""
    children = _children(optimizer)
    snapshot = []
    for entry in groups:
        group = children[entry['optimizer']].optimizer.param_groups[entry['group']]
        settings = _json_value({key: value for key, value in group.items() if key != 'params'})
        if entry['implementation'].endswith('.FusedAdam'):
            step = settings.get('step')
            if step != expected_step and not (step is None and expected_step == 0):
                raise ValueError(
                    f'FusedAdam group step mismatch at optimizer {entry["optimizer"]} '
                    f'group {entry["group"]}: expected {expected_step}, observed {step}'
                )
        snapshot.append(dict(entry, settings=settings))
    return snapshot


class _Comparisons:
    def __init__(self):
        self.values = torch.zeros((len(_FIELDS), 1 + len(_COMPONENTS), 10), dtype=torch.float64)

    def add(self, field, metadata, current, reference):
        indices = [0] + [1 + _COMPONENTS.index(name) for name in metadata['components']]
        values = self.values[_FIELDS.index(field)]
        _accumulate_statistics(values[:, :9], indices, current, reference)
        values[indices, 9] += (current.double() * reference.double()).sum().item()

    def finish(self, group, device):
        if dist.is_initialized():
            values = self.values.to(device)
            sums = values[:, :, [0, 1, 2, 3, 4, 5, 9]].contiguous()
            maxima = values[:, :, 6:9].contiguous()
            dist.all_reduce(sums, group=group)
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX, group=group)
            self.values[:, :, [0, 1, 2, 3, 4, 5, 9]] = sums.cpu()
            self.values[:, :, 6:9] = maxima.cpu()

        def report(row):
            count, square, ref_square, delta, bad, ref_bad, error, maximum, ref_max, dot = row
            norm, ref_norm = math.sqrt(square), math.sqrt(ref_square)
            return dict(
                elements=int(count),
                l2=norm,
                reference_l2=ref_norm,
                zero_current_norm=norm == 0,
                zero_reference_norm=ref_norm == 0,
                delta_l2=math.sqrt(delta),
                relative_l2=math.sqrt(delta / ref_square) if ref_square else None,
                cosine=max(-1.0, min(1.0, dot / (norm * ref_norm))) if norm * ref_norm else None,
                scale_projection=dot / ref_square if ref_square else None,
                dot_product=dot,
                norm_ratio=norm / ref_norm if ref_norm else None,
                max_abs_error=error,
                max_abs=maximum,
                reference_max_abs=ref_max,
                all_finite=bad == 0 and ref_bad == 0,
            )

        return {
            field: {
                'totals': report(rows[0].tolist()),
                'components': {
                    name: report(rows[index + 1].tolist()) for index, name in enumerate(_COMPONENTS)
                },
            }
            for field, rows in zip(_FIELDS, self.values)
        }


def _collective_failure(failure, group, device):
    failed = torch.tensor([failure is not None], device=device, dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(failed, group=group)
    if failed.item():
        raise RuntimeError(
            f'Native optimizer diagnostics failed on {failed.item()} ranks: '
            f'{failure or "see failing peer log"}'
        )


@contextmanager
def _replace_step(optimizer, replacement):
    had_instance_step = 'step' in vars(optimizer)
    previous = vars(optimizer).get('step')
    optimizer.step = replacement
    try:
        yield
    finally:
        if had_instance_step:
            optimizer.step = previous
        else:
            del optimizer.step


class NativeOptimizerStepDiagnostics:
    """One-shot step wrapper; ``report`` stays None until the complete audit succeeds.

    ``expected_step`` is the number of *successful optimizer updates* in the
    starting checkpoint, not a scheduler or dataset step. ``context`` is optional
    JSON metadata (or a callback returning it), e.g. scheduler state and source
    batch identity; it must agree exactly with the reference. LR, weight decay,
    betas and all other parameter-group settings are always checked separately.
    """

    def __init__(
        self,
        models,
        optimizer,
        *,
        reference_dir,
        write_reference,
        scratch_dir,
        report_path,
        expected_step,
        group=None,
        chunk_elements=1 << 20,
        context=None,
        cold_moment_bands=False,
    ):
        if expected_step < 0 or chunk_elements <= 0:
            raise ValueError('expected_step must be nonnegative and chunk_elements positive')
        self.models = list(models) if isinstance(models, (list, tuple)) else [models]
        self.optimizer = optimizer
        self.reference_dir = Path(reference_dir)
        self.write_reference = write_reference
        self.scratch_dir = Path(scratch_dir)
        self.report_path = Path(report_path)
        self.expected_step = expected_step
        self.group, self.chunk_elements, self.context = group, chunk_elements, context
        self.report = None
        self._used = False
        self._cold_bands = _ColdMomentBands() if cold_moment_bands and expected_step == 0 else None

    @contextmanager
    def install(self):
        original = self.optimizer.step

        def step(*args, **kwargs):
            if self._used:
                return original(*args, **kwargs)
            self._used = True
            return self._observe(original, *args, **kwargs)

        with _replace_step(self.optimizer, step):
            yield self

    @torch.no_grad()
    def _observe(self, original, *args, **kwargs):
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size(self.group) if dist.is_initialized() else 1
        device = next(self.models[0].parameters()).device
        manifest_path = self.reference_dir / f'rank{rank:05d}.json'
        candidate_manifest = self.report_path.with_name(
            f'{self.report_path.stem}.rank{rank:05d}.json'
        )
        vector_path = self.reference_dir / f'rank{rank:05d}.f32'
        temporary_vector = vector_path.with_suffix('.f32.incomplete')
        scratch = None
        failure = None
        try:
            if self.report_path.exists():
                raise FileExistsError(f'Refusing existing optimizer report: {self.report_path}')
            if not self.write_reference and candidate_manifest.exists():
                raise FileExistsError(f'Refusing existing optimizer manifest: {candidate_manifest}')
            self.scratch_dir.mkdir(parents=True, exist_ok=True)
            scratch = tempfile.TemporaryFile(dir=self.scratch_dir, prefix=f'adam-rank{rank:05d}-')
            shards, groups = _owned_optimizer_shards(self.models, self.optimizer)
            groups = _snapshot_groups(self.optimizer, groups, self.expected_step)
            context = self.context() if callable(self.context) else self.context
            manifest = dict(
                format='mimo-native-adamw-step-v1',
                rank=rank,
                world_size=world_size,
                expected_step=self.expected_step,
                context=_json_value(context),
                groups=groups,
                entries=[],
            )
            before_gradients = []
            for shard in shards:
                initial = _state_description(shard, self.expected_step, self.chunk_elements)
                initial['master'] = _fingerprint(shard.master, self.chunk_elements, scratch)
                initial['master'].pop('square')
                manifest['entries'].append(dict(metadata=shard.metadata, initial=initial))
                before_gradients.append(_fingerprint(shard.gradient, self.chunk_elements))
            scratch.flush()
            if self.write_reference:
                self.reference_dir.mkdir(parents=True, exist_ok=True)
                if manifest_path.exists() or vector_path.exists() or temporary_vector.exists():
                    raise FileExistsError(f'Refusing existing optimizer reference: {manifest_path}')
                reference = None
            else:
                reference = json.loads(manifest_path.read_text())
                initial_reference = {key: reference[key] for key in manifest}
                if initial_reference != manifest:
                    raise ValueError('Starting optimizer state/ownership/hyperparameters differ')
                expected_bytes = 12 * sum(shard.master.numel() for shard in shards)
                if vector_path.stat().st_size != expected_bytes:
                    raise ValueError('Optimizer reference vector byte count mismatch')
        except Exception as error:
            failure = f'{type(error).__name__}: {error}'
        try:
            # Reject incompatible/missing starting state on every rank BEFORE mutation.
            _collective_failure(failure, self.group, device)
            observed_gradients = {}
            hook_failure = []
            with ExitStack() as stack:
                for child_index, child in enumerate(_children(self.optimizer)):
                    if getattr(child, 'is_stub_optimizer', False):
                        continue
                    inner = child.optimizer
                    owned = [s for s in shards if s.metadata['optimizer'] == child_index]
                    original_inner = inner.step

                    def observe_inner(
                        *inner_args, _owned=owned, _step=original_inner, **inner_kwargs
                    ):
                        # No collectives here: ranks may have empty/stub optimizers.
                        # A local I/O/observation failure is reported after native step;
                        # it must not strand peers partway through the optimizer chain.
                        try:
                            for shard in _owned:
                                if id(shard.master) in observed_gradients:
                                    raise ValueError('Native inner Adam stepped a parameter twice')
                                if shard.master.grad is None:
                                    raise ValueError('Missing actual gradient at native Adam step')
                                observed_gradients[id(shard.master)] = _fingerprint(
                                    shard.master.grad, self.chunk_elements
                                )
                        except Exception as error:
                            hook_failure.append(f'{type(error).__name__}: {error}')
                        return _step(*inner_args, **inner_kwargs)

                    stack.enter_context(_replace_step(inner, observe_inner))
                try:
                    result = original(*args, **kwargs)
                    native_failure = None
                except Exception as error:
                    native_failure = f'Native step raised {type(error).__name__}: {error}'
            # This rendezvous covers exceptions after native work returns. It
            # cannot recover an error that strands a peer inside a native CUDA
            # collective; the process-group timeout remains authoritative there.
            _collective_failure(native_failure, self.group, device)
            failure = None
            comparisons = _Comparisons()
            try:
                if hook_failure:
                    raise ValueError('; '.join(hook_failure))
                if not isinstance(result, tuple) or len(result) != 3 or not result[0]:
                    raise ValueError(
                        f'Native optimizer did not report a successful update: {result}'
                    )
                if len(observed_gradients) != len(shards):
                    raise ValueError('Not every owned parameter reached native Adam.step')
                manifest['post_state'] = [
                    _state_description(
                        shard, self.expected_step + 1, self.chunk_elements, after=True
                    )
                    for shard in shards
                ]
                manifest['post_groups'] = _snapshot_groups(
                    self.optimizer, groups, self.expected_step + 1
                )
                if reference is not None and manifest['post_groups'] != reference['post_groups']:
                    raise ValueError('Post-step optimizer group settings/step differ')
                manifest['native_result'] = _json_value(result)
                manifest['grad_norms_by_group'] = _json_value(
                    getattr(self.optimizer, 'grad_norms_by_group', {})
                )
                manifest['gradients'] = dict(
                    before_clipping=before_gradients,
                    at_native_adam=[observed_gradients[id(s.master)] for s in shards],
                )
                self._compare_post_step(
                    shards, manifest, reference, scratch, vector_path, temporary_vector, comparisons
                )
            except Exception as error:
                failure = f'{type(error).__name__}: {error}'
            _collective_failure(failure, self.group, device)
            metrics = comparisons.finish(self.group, device)
            local = dict(
                rank=rank,
                manifest_path=str(manifest_path if self.write_reference else candidate_manifest),
                native_result=manifest['native_result'],
                grad_norms_by_group=manifest['grad_norms_by_group'],
                gradients=self._gradient_norms(shards, manifest['gradients']),
                learning_rates=[
                    dict(
                        optimizer=entry['optimizer'],
                        group=entry['group'],
                        lr=entry['settings']['lr'],
                        elements=entry['elements'],
                    )
                    for entry in groups
                ],
                cold_lazy_entries=sum(
                    entry['initial']['initialization'] == 'native_lazy'
                    for entry in manifest['entries']
                ),
            )
            rank_reports = [None] * world_size
            if dist.is_initialized():
                dist.all_gather_object(rank_reports, local, group=self.group)
            else:
                rank_reports[0] = local
            gradient_squares = {}
            for entry in rank_reports:
                for phase, norms in entry['gradients']['squared_l2_by_grad_norm_group'].items():
                    totals = gradient_squares.setdefault(phase, {})
                    for name, square in norms.items():
                        totals[name] = totals.get(name, 0.0) + square
            cold_bands = (
                self._cold_bands.finish(self.group, device)
                if self._cold_bands is not None
                else None
            )
            if cold_bands is not None:
                full = metrics['master_update']['totals']
                if cold_bands['totals']['elements'] != full['elements']:
                    raise ValueError('Cold moment bands do not cover every owned optimizer element')
                full_square = full['delta_l2'] ** 2
                difference = abs(cold_bands['totals']['delta_update_square'] - full_square)
                cold_bands['accounting'] = dict(
                    element_count_matches=True,
                    full_update_error_square=full_square,
                    band_update_error_square=cold_bands['totals']['delta_update_square'],
                    absolute_disagreement=difference,
                    relative_disagreement=(
                        difference / full_square
                        if full_square
                        else (0.0 if difference == 0 else None)
                    ),
                )
            report = dict(
                kind='full_native_adamw_update',
                mode='write' if self.write_reference else 'compare',
                expected_step=self.expected_step,
                reference_dir=str(self.reference_dir),
                initial_state_exact=True,
                initial_state_check='SHA256 of every owned FP32 master/moment plus metadata',
                cold_lazy_entries=sum(entry['cold_lazy_entries'] for entry in rank_reports),
                full_update_elements=metrics['master_update']['totals']['elements'],
                positive_lr_owned_elements=sum(
                    group['elements']
                    for entry in rank_reports
                    for group in entry['learning_rates']
                    if group['lr'] > 0
                ),
                zero_lr_owned_elements=sum(
                    group['elements']
                    for entry in rank_reports
                    for group in entry['learning_rates']
                    if group['lr'] == 0
                ),
                nonzero_master_update=metrics['master_update']['totals']['l2'] > 0,
                observed_gradient_l2={
                    phase: {name: math.sqrt(square) for name, square in norms.items()}
                    for phase, norms in gradient_squares.items()
                },
                metrics=metrics,
                cold_moment_bands=cold_bands,
                ranks=rank_reports,
                tolerance_pass=None,
                scope='Numerical evidence only; no tolerance inferred from the candidate',
                decay_removal='FP64 (actual_master_after - before) + lr * weight_decay * before; '
                'includes native floating-point update/decay rounding',
                gradient_scope='Complete observed norms/hashes; use full_gradient_validation '
                'for elementwise pre-clipping gradient comparison',
            )
            failure = None
            try:
                if self.write_reference:
                    temporary_vector.replace(vector_path)
                    temporary_manifest = manifest_path.with_suffix('.json.incomplete')
                    temporary_manifest.write_text(
                        json.dumps(manifest, indent=2, allow_nan=False) + '\n'
                    )
                    temporary_manifest.replace(manifest_path)
                else:
                    candidate_manifest.parent.mkdir(parents=True, exist_ok=True)
                    temporary_manifest = candidate_manifest.with_suffix('.json.incomplete')
                    temporary_manifest.write_text(
                        json.dumps(
                            dict(
                                manifest,
                                reference_dir=str(self.reference_dir),
                                vectors_stored=False,
                            ),
                            indent=2,
                            allow_nan=False,
                        )
                        + '\n'
                    )
                    temporary_manifest.replace(candidate_manifest)
            except Exception as error:
                failure = f'{type(error).__name__}: {error}'
            # Never publish a success-shaped rank-zero report until every
            # reference shard and manifest has been published successfully.
            _collective_failure(failure, self.group, device)
            failure = None
            try:
                if rank == 0:
                    self.report_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary_report = self.report_path.with_suffix('.json.incomplete')
                    temporary_report.write_text(
                        json.dumps(report, indent=2, allow_nan=False) + '\n'
                    )
                    temporary_report.replace(self.report_path)
            except Exception as error:
                failure = f'{type(error).__name__}: {error}'
            _collective_failure(failure, self.group, device)
            self.report = report
            return result
        finally:
            if scratch is not None:
                scratch.close()

    def _compare_post_step(
        self, shards, manifest, reference, scratch, vector_path, temporary_vector, comparisons
    ):
        scratch.seek(0)
        manifest['post_vector_sha256'] = []
        path = temporary_vector if self.write_reference else vector_path
        with ExitStack() as stack:
            vectors = stack.enter_context(path.open('xb' if self.write_reference else 'rb'))
            moment_reference = (
                stack.enter_context(vector_path.open('rb'))
                if self._cold_bands is not None and not self.write_reference
                else None
            )
            for index, shard in enumerate(shards):
                state = shard.inner.state[shard.master]
                moment_chunks = _chunks(state['exp_avg'], self.chunk_elements)
                if moment_reference is not None:
                    moment_reference.seek(vectors.tell() + shard.master.numel() * 4)
                current_hashes, reference_hashes = {}, {}
                for field in ('master', *_MOMENTS):
                    current_digest, reference_digest = hashlib.sha256(), hashlib.sha256()
                    tensor = shard.master if field == 'master' else state[field]
                    for current in _chunks(tensor, self.chunk_elements):
                        if not torch.isfinite(current).all():
                            raise ValueError('Nonfinite actual native optimizer output')
                        current_digest.update(memoryview(current.numpy()))
                        if self.write_reference:
                            vectors.write(memoryview(current.numpy()))
                            original = current
                        else:
                            original = _read_chunk(vectors, current.numel())
                        reference_digest.update(memoryview(original.numpy()))
                        if field == 'master':
                            before = _read_chunk(scratch, current.numel()).double()
                            update, reference_update = (
                                current.double() - before,
                                original.double() - before,
                            )
                            comparisons.add(
                                'master_update', shard.metadata, update, reference_update
                            )
                            if self._cold_bands is not None:
                                moment = next(moment_chunks)
                                ref_moment = (
                                    _read_chunk(moment_reference, moment.numel())
                                    if moment_reference is not None
                                    else moment
                                )
                                self._cold_bands.add(
                                    shard.metadata,
                                    moment,
                                    ref_moment,
                                    update,
                                    reference_update,
                                    float(shard.param_group['betas'][0]),
                                    float(shard.param_group['eps']),
                                )
                            decay = (
                                before
                                * float(shard.param_group['lr'])
                                * float(shard.param_group['weight_decay'])
                            )
                            comparisons.add(
                                'decay_removed_update',
                                shard.metadata,
                                update + decay,
                                reference_update + decay,
                            )
                        else:
                            comparisons.add(field, shard.metadata, current, original)
                    current_hashes[field] = current_digest.hexdigest()
                    reference_hashes[field] = reference_digest.hexdigest()
                manifest['post_vector_sha256'].append(current_hashes)
                if reference is not None:
                    if reference_hashes != reference['post_vector_sha256'][index]:
                        raise ValueError('Optimizer reference vector checksum mismatch')
                    for key in ('parameter_step', 'group_step', 'counter_source', 'initialization'):
                        if (
                            manifest['post_state'][index][key]
                            != reference['post_state'][index][key]
                        ):
                            raise ValueError(f'Post-step Adam state metadata mismatch: {key}')
            if not self.write_reference and vectors.read(1):
                raise ValueError('Trailing optimizer reference data')

    @staticmethod
    def _gradient_norms(shards, gradients):
        norms = {}
        for name, entries in gradients.items():
            grouped = {}
            for shard, entry in zip(shards, entries):
                group_name = shard.metadata['grad_norm_group'] or 'main'
                grouped[group_name] = grouped.get(group_name, 0.0) + entry['square']
            norms[name] = grouped
        return dict(
            squared_l2_by_grad_norm_group=norms,
            note='Local uniquely owned shards; sum squared norms across ranks. '
            'Native returned norm is recorded separately, not reconstructed or substituted.',
        )


@contextmanager
def optimizer_step_diagnostics(models, optimizer, **kwargs):
    """Restore the optimizer instance's original step on normal or exceptional exit."""
    observer = NativeOptimizerStepDiagnostics(models, optimizer, **kwargs)
    with observer.install():
        yield observer
