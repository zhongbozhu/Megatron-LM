# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Diagnostic expert-ID replay keyed by source sample and logical token position.

The statistics pass records one immutable table per source step. Subsequent
statistics, training and checkpoint recomputation reuse expert IDs while the
router computes probabilities and their gradients from its current logits. This
is an opt-in conditioned objective, not an assertion of natural-route equality.
Only IDs and compact metadata are persisted; no activation gather is introduced.
"""

import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.moe.router import TopKRouter

_SCHEMA = 1
_SOURCE_FIELDS = ('tokens', 'labels', 'loss_mask', 'position_ids')
_ROUTER_FIELDS = (
    'moe_router_score_function',
    'moe_router_pre_softmax',
    'moe_router_topk_scaling_factor',
    'moe_router_dtype',
    'moe_router_load_balancing_type',
    'moe_aux_loss_coeff',
    'moe_z_loss_coeff',
    'calculate_per_token_loss',
    'mtp_num_layers',
    'mtp_use_repeated_layer',
)


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _tensor_digest(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(_json_bytes([str(value.dtype), list(value.shape)]))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_fingerprint(samples, media=None):
    """Hash real source fields, excluding layout-dependent padding and packing."""
    result = {}
    for sid, sample in sorted(samples.items()):
        length = int(sample['original_seq_len'])
        if length <= 0:
            raise ValueError('Fixed routing requires positive source sequence lengths')
        fields = {}
        for name in _SOURCE_FIELDS:
            value = sample[name]
            if not isinstance(value, torch.Tensor) or value.shape[-1] < length:
                raise ValueError(f'Invalid source field {name} for sample {sid}')
            fields[name] = _tensor_digest(value[..., :length])
        fields['length'] = length
        for name in ('media_ids', 'source_ids', 'source_id'):
            if name in sample:
                fields[name] = sample[name]
        result[str(sid)] = fields
    images = []
    for image in sorted(media or (), key=lambda item: item['image_id']):
        images.append(
            {
                name: (
                    _tensor_digest(image[name])
                    if isinstance(image[name], torch.Tensor)
                    else image[name]
                )
                for name in ('image_id', 'source_id', 'size', 'length', 'grid', 'pixel_values')
            }
        )
    return hashlib.sha256(_json_bytes({'samples': result, 'media': images})).hexdigest()


def _validate_ids(ids, rows, router):
    if ids.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError('Expert IDs must be integers')
    if tuple(ids.shape) != (rows, router.topk):
        raise ValueError('Expert ID table has the wrong shape')
    if ids.numel() and ((ids < 0).any() or (ids >= router.num_experts).any()):
        raise ValueError('Expert IDs are outside the configured expert range')
    if router.topk > 1 and (ids.sort(-1).values.diff(dim=-1) == 0).any():
        raise ValueError('A token cannot select the same expert twice')


class FixedRouting:
    """Single-source-step routing scope; retain round bindings through backward.

    ``begin_round`` must precede each decoder forward. The caller must finish
    that round's backward before switching bindings. Recording and replay use the
    model's actual partition adapter for each runtime CP group. Partial recording
    tables are assembled on CPU before publishing the existing full-table format.
    TP/PP sharing and repeated MTP depths are deliberately unsupported.
    """

    def __init__(self, model, directory, mode, step, samples, group=None, context=None, media=None):
        self.model = model
        self.directory = Path(directory) / f'step{step:08d}'
        self.mode = mode
        self.step = int(step)
        self.samples = samples
        self.group = group
        self.rank = dist.get_rank() if group is not None else 0
        self.group_rank = dist.get_rank(group) if group is not None else 0
        self.world_size = dist.get_world_size(group) if group is not None else 1
        self.closed = False
        self.current = None
        self.cache = {}
        self.records = {}
        self.loaded_shards = {}
        self.verified = set()
        self.recorded_rounds = set()
        self.record_cp_sizes = set()
        self.record_positions = {}
        self.sample_owners = {}
        self.manifest = None
        self.metrics = {}
        self.routers = []

        def initialize():
            if mode not in ('record', 'replay'):
                raise ValueError('Fixed routing mode must be record or replay')
            # remove_duplicate=False detects reused routers that need call/depth IDs.
            all_routers = [
                (name, module)
                for name, module in model.named_modules(remove_duplicate=False)
                if isinstance(module, TopKRouter)
            ]
            if not all_routers or len({id(router) for _, router in all_routers}) != len(
                all_routers
            ):
                raise ValueError('Fixed routing requires distinct, non-shared TopKRouters')
            config = {}
            for name, router in all_routers:
                if getattr(router, '_fixed_routing', None) is not None:
                    raise RuntimeError('Router already belongs to a fixed-routing scope')
                if router.tp_group.size() != 1:
                    raise ValueError('Fixed routing initially requires TP1')
                if any(
                    getattr(router.config, field, 'zigzag') != 'zigzag'
                    for field in ('linear_cp_layout', 'attention_cp_layout')
                ):
                    raise ValueError(
                        'Fixed routing currently requires zigzag layouts at every router'
                    )
                unsupported = (
                    getattr(router, 'routing_type', None) in ('sinkhorn', 'quantile_balancing')
                    or getattr(router, 'expert_bias', None) is not None
                    or getattr(router, 'router_replay', None) is not None
                    or getattr(router.config, 'moe_router_force_load_balancing', False)
                    or any(
                        getattr(router.config, field, None) is not None
                        for field in (
                            'moe_router_num_groups',
                            'moe_router_group_topk',
                            'moe_expert_capacity_factor',
                            'moe_expert_rank_capacity_factor',
                            'moe_router_force_biased',
                            'moe_input_jitter_eps',
                        )
                    )
                )
                if unsupported:
                    raise ValueError(
                        'Fixed routing requires ordinary dropless routing without router noise or bias'
                    )
                if (
                    router.is_mtp_layer
                    and getattr(router.config, 'mtp_use_repeated_layer', False)
                    and (getattr(router.config, 'mtp_num_layers', 0) or 0) > 1
                ):
                    raise ValueError('Repeated MTP depths require separate canonical router IDs')
                config[name] = {
                    'experts': int(router.num_experts),
                    'topk': int(router.topk),
                    'mtp': bool(router.is_mtp_layer),
                    **{field: getattr(router.config, field, None) for field in _ROUTER_FIELDS},
                }
            self.identity = dict(
                schema=_SCHEMA,
                step=self.step,
                source_sha256=source_fingerprint(samples, media),
                lengths={
                    str(sid): int(sample['original_seq_len']) for sid, sample in samples.items()
                },
                routers=config,
                context=context or {},
            )
            # Normalize tuples to JSON lists so record/replay compare identically.
            self.identity = json.loads(_json_bytes(self.identity))
            self.routers = all_routers
            self.names = {router: name for name, router in all_routers}
            if mode == 'replay':
                self.manifest = json.loads((self.directory / 'manifest.json').read_text())
                if self.manifest['identity'] != self.identity:
                    raise ValueError(
                        'Fixed routing source, checkpoint context or router configuration changed'
                    )
                self.sample_owners = self.manifest['sample_owners']
                self.metrics = dict(
                    self.manifest['metrics'], source_sha256=self.identity['source_sha256']
                )
            elif self.directory.exists():
                raise FileExistsError(f'Refusing to overwrite fixed routing: {self.directory}')

        self._collective(initialize)
        if mode == 'record':
            self._collective(
                lambda: self.directory.mkdir(parents=True) if self.group_rank == 0 else None
            )
        for _, router in self.routers:
            router._fixed_routing = self

    def _collective(self, operation):
        """Publish rank-local failures before any rank leaves a phase boundary."""
        try:
            result = {'value': operation(), 'error': None}
        except Exception as error:
            result = {'value': None, 'error': f'{type(error).__name__}: {error}'}
        gathered = [result]
        if self.group is not None:
            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, result, group=self.group)
        errors = [f'rank {i}: {item["error"]}' for i, item in enumerate(gathered) if item['error']]
        if errors:
            raise RuntimeError('Fixed routing phase failed: ' + '; '.join(errors))
        return [item['value'] for item in gathered]

    def begin_round(self, item):
        """Bind real rows using the same CP partition code as decoder inputs."""
        if self.closed:
            raise RuntimeError('Fixed routing scope is closed')
        kwargs = item['kwargs']
        tokens = kwargs['input_ids'].flatten()
        packed = PackedSeqParams(qkv_format='thd', **kwargs['packing_kwargs'])
        cp_size = packed.local_cp_size
        if cp_size is None:
            cp_size = packed.cp_group.size() if packed.cp_group is not None else 1
        if cp_size <= 0:
            raise ValueError('Runtime CP size must be positive')
        index = torch.arange(tokens.numel(), device=tokens.device).unsqueeze(0)
        if self.model.partition_adapter is not None:
            _, index, _, _ = self.model.partition_adapter.shard(None, index, None, packed)
        index = index.flatten()
        physical = index.detach().cpu()
        if physical.numel() and (physical.min() < 0 or physical.max() >= tokens.numel()):
            raise ValueError('Partition adapter emitted out-of-range physical rows')
        if physical.unique().numel() != physical.numel():
            raise ValueError('Partition adapter emitted duplicate physical rows')
        boundaries = packed.cu_seqlens_q_padded.detach().cpu().tolist()
        sample_ids = tuple(item['diagnostic_sample_ids'])
        if len(set(sample_ids)) != len(sample_ids) or len(boundaries) < len(sample_ids) + 1:
            raise ValueError('Invalid sample IDs or padded boundaries')
        if boundaries[0] != 0 or boundaries[-1] != tokens.numel():
            raise ValueError('Padded boundaries must cover the full decoder input')
        if any(right < left for left, right in zip(boundaries, boundaries[1:])):
            raise ValueError('Padded boundaries must be monotonic')
        logical = packed.cu_seqlens_q.detach().cpu().tolist()
        if len(logical) != len(boundaries) or len(boundaries) not in (
            len(sample_ids) + 1,
            len(sample_ids) + 2,
        ):
            raise ValueError(
                'Packed boundaries may contain only real samples and one trailing dummy'
            )
        canonical_real = torch.zeros(tokens.numel(), dtype=torch.bool)
        packed_tokens = tokens.detach().cpu()
        parts = []
        for slot, sid in enumerate(sample_ids):
            length = int(self.samples[sid]['original_seq_len'])
            start, end = boundaries[slot : slot + 2]
            if logical[slot + 1] - logical[slot] != length:
                raise ValueError('Logical packed boundary disagrees with source sample length')
            if not torch.equal(
                packed_tokens[start : start + length], self.samples[sid]['tokens'][:length].cpu()
            ):
                raise ValueError('Packed token values disagree with canonical source identity')
            if end - start < length:
                raise ValueError('Real sample exceeds its padded physical boundary')
            canonical_real[start : start + length] = True
            rows = ((physical >= start) & (physical < start + length)).nonzero().flatten()
            positions = physical[rows] - start
            if (
                self.mode == 'record'
                and cp_size == 1
                and not torch.equal(positions, torch.arange(length))
            ):
                raise ValueError(
                    'CP1 record must own every real token exactly once in source order'
                )
            parts.append((sid, rows, positions))
        padding = kwargs.get('padding_mask')
        if padding is not None:
            padding = padding.detach().cpu().flatten().bool()
            if not torch.equal(padding, ~canonical_real):
                raise ValueError('Decoder padding mask disagrees with canonical real token rows')
        elif len(boundaries) != len(sample_ids) + 1:
            raise ValueError('Trailing dummy sequence requires an explicit padding mask')
        if self.mode == 'record':
            self.record_cp_sizes.add(cp_size)
            if cp_size > 1:
                for sid, _, positions in parts:
                    if sid in self.record_positions:
                        raise ValueError(f'Sample {sid} appeared in multiple recording rounds')
                    self.record_positions[sid] = positions
        self.current = dict(round=int(item['round_id']), rows=index.numel(), parts=parts)
        self.cache = {}
        # mmap handles from previous rounds need not stay alive after IDs are copied.
        self.loaded_shards = {}

    def _read_sample(self, name, sid, router):
        owner = self.sample_owners[str(sid)]
        if owner not in self.loaded_shards:
            path = self.directory / f'rank{owner:05d}.pt'
            self.loaded_shards[owner] = torch.load(
                path, map_location='cpu', weights_only=True, mmap=True
            )
        entry = self.loaded_shards[owner][name][sid]
        key = (name, sid)
        if key not in self.verified:
            _validate_ids(entry, int(self.samples[sid]['original_seq_len']), router)
            expected = self.manifest['digests'][name][str(sid)]
            if _tensor_digest(entry) != expected:
                raise ValueError(f'Fixed routing data checksum mismatch: {name}, sample {sid}')
            self.verified.add(key)
        return entry

    def select(self, router, logits, default_selector, padding_mask=None, packed_seq_params=None):
        """Return detached ordered expert IDs; the router retains probability math."""
        if self.closed or self.current is None or router not in self.names:
            raise RuntimeError('Fixed routing requires an active round and attached router')
        rows = self.current['rows']
        if logits.ndim != 2 or logits.shape != (rows, router.num_experts):
            raise ValueError('Router rows do not match the canonical CP-local token mapping')
        if router in self.cache:
            if self.mode == 'record':
                raise RuntimeError('Statistics recording requires one invocation per router/round')
            return self.cache[router]
        name = self.names[router]
        # Physical padding has no canonical sample identity. Its choices are stable
        # and auxiliary objectives still exclude it through their existing masks.
        result = torch.arange(router.topk, device=logits.device).expand(rows, -1).clone()
        if self.mode == 'record':
            selected = default_selector().detach()
            _validate_ids(selected, rows, router)
            selected = selected.to(
                device='cpu', dtype=torch.int16 if router.num_experts <= 32768 else torch.int32
            )
            records = self.records.setdefault(name, {})
            for sid, local_rows, positions in self.current['parts']:
                if sid in records:
                    raise ValueError(f'Sample {sid} recorded more than once for {name}')
                values = selected[local_rows].contiguous()
                records[sid] = values
                result[local_rows.to(logits.device)] = values.to(
                    device=logits.device, dtype=torch.long
                )
            self.recorded_rounds.add(self.current['round'])
        else:
            for sid, local_rows, positions in self.current['parts']:
                if local_rows.numel():
                    values = self._read_sample(name, sid, router).index_select(0, positions)
                    result[local_rows.to(logits.device)] = values.to(
                        device=logits.device, dtype=torch.long
                    )
        self.cache[router] = result
        return result

    def seal_recording(self):
        """Validate full coverage, publish an immutable manifest, then replay."""
        if self.closed:
            raise RuntimeError('Fixed routing scope is closed')
        if self.mode != 'record':
            return
        # Every rank takes the same branch even when runtime CP varies by pack.
        cp_sizes = sorted(
            {
                size
                for values in self._collective(lambda: sorted(self.record_cp_sizes))
                for size in values
            }
        )
        if any(size > 1 for size in cp_sizes):
            from examples.mimo.fixed_routing_shards import assemble_canonical_records

            positions = dict(self.record_positions)
            # CP1 rounds already own complete samples; avoid retaining duplicate
            # position tensors on the ordinary CP1 recording path.
            for sid in next(iter(self.records.values()), {}):
                if sid not in positions:
                    positions[sid] = torch.arange(int(self.samples[sid]['original_seq_len']))
            self.records = assemble_canonical_records(
                self.records,
                positions,
                {sid: int(sample['original_seq_len']) for sid, sample in self.samples.items()},
                self.group,
                self.directory / 'cp_parts',
            )

        def write_rank():
            names = {name for name, _ in self.routers}
            if set(self.records) != names:
                raise ValueError('Not every decoder/MTP router was recorded')
            sample_sets = [set(self.records[name]) for name in names]
            if any(ids != sample_sets[0] for ids in sample_sets):
                raise ValueError('Routers did not observe the same source sample set')
            digests = {
                name: {str(sid): _tensor_digest(ids) for sid, ids in table.items()}
                for name, table in self.records.items()
            }
            temporary = self.directory / f'rank{self.rank:05d}.tmp'
            torch.save(self.records, temporary)
            temporary.replace(self.directory / f'rank{self.rank:05d}.pt')
            return dict(
                rank=self.rank,
                samples=sorted(sample_sets[0]),
                digests=digests,
                identity=self.identity,
            )

        ranks = self._collective(write_rank)

        def publish():
            owners = {}
            digests = {name: {} for name, _ in self.routers}
            for record in ranks:
                if record['identity'] != self.identity:
                    raise ValueError('Ranks disagree on fixed-routing source or configuration')
                for sid in record['samples']:
                    if str(sid) in owners:
                        raise ValueError(f'Source sample {sid} recorded by multiple ranks')
                    owners[str(sid)] = record['rank']
                for name, values in record['digests'].items():
                    digests[name].update(values)
            if set(owners) != set(self.identity['lengths']):
                raise ValueError('Fixed-routing recording omitted source samples')
            metrics = dict(
                source_step=self.step,
                source_samples=len(owners),
                source_tokens=sum(self.identity['lengths'].values()),
                routers=len(digests),
                mtp_routers=sum(config['mtp'] for config in self.identity['routers'].values()),
                route_sha256=hashlib.sha256(_json_bytes(digests)).hexdigest(),
                recording_cp=cp_sizes[0] if len(cp_sizes) == 1 else None,
                recording_cp_sizes=cp_sizes,
                source_sha256=self.identity['source_sha256'],
            )
            manifest = dict(
                identity=self.identity, sample_owners=owners, digests=digests, metrics=metrics
            )
            if self.group_rank == 0:
                temporary = self.directory / 'manifest.tmp'
                temporary.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n'
                )
                temporary.replace(self.directory / 'manifest.json')
            return manifest

        manifests = self._collective(publish)
        self.manifest = manifests[0]
        self.sample_owners = self.manifest['sample_owners']
        self.metrics = self.manifest['metrics']
        self.records = {}
        self.record_positions = {}
        self.cache = {}
        self.mode = 'replay'

    def close(self):
        """Restore ordinary routing after all forwards/backwards finish."""
        if self.closed:
            return
        for _, router in self.routers:
            if getattr(router, '_fixed_routing', None) is self:
                del router._fixed_routing
        self.cache.clear()
        self.loaded_shards.clear()
        self.records.clear()
        self.record_positions.clear()
        self.current = None
        self.closed = True
