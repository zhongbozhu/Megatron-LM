# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Opt-in snapshots of the explicit vision boundary, without diagnostic collectives.

Files contain image features only, not model parameters or full decoder activations.
Analyze a completed step offline: never read another rank's files during training.
Receiver gradients must be saved *before* ``PackFeatureBridge.backward``; source
gradients are saved after every round has returned and before encoder backward.
"""

import math
from pathlib import Path

import torch


class BoundaryDiagnostics:
    """Write detached CPU snapshots, preserving rows and refusing stale-file reuse."""

    def __init__(self, output_dir, rank, *, selected_steps=None):
        self.output_dir = Path(output_dir)
        self.rank = int(rank)
        self.selected_steps = None if selected_steps is None else frozenset(selected_steps)

    def enabled(self, step):
        """Whether this source step was selected for inspection."""
        return self.selected_steps is None or step in self.selected_steps

    def _write(self, step, kind, tensor, *, round_id=None, phase=None, normalizer=1.0, **metadata):
        if not self.enabled(step):
            return
        scale = float(normalizer)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('Boundary gradient normalizer must be finite and positive')
        if tensor.ndim != 2 or not tensor.is_floating_point():
            raise ValueError('Boundary snapshots require floating-point [rows, hidden] tensors')
        suffix = ''
        if round_id is not None:
            if not phase or not phase.replace('_', '').isalnum():
                raise ValueError('Boundary phase must contain only letters, digits and underscores')
            suffix = f'-round{round_id:05d}-{phase}'
        directory = self.output_dir / f'step{step:08d}'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'{kind}-rank{self.rank:05d}{suffix}.pt'
        # CPU copies also detach the snapshot from the live encoder/decoder graph.
        value = tensor.detach().to(device='cpu', copy=True)
        if kind.endswith('gradient'):
            value = value.float().div_(scale)
        record = dict(
            version=1,
            kind=kind,
            step=int(step),
            rank=self.rank,
            round_id=round_id,
            phase=phase,
            normalizer=scale,
            values=value,
            **metadata,
        )
        with path.open('xb') as stream:
            torch.save(record, stream)

    def save_producer(self, step, features, media, *, num_rounds=None):
        """Save local encoder rows and image identities, including empty producers.

        ``media`` contains ``image_id``, ``source_id``, ``offset`` and ``length``
        in local producer order. These records make physical offsets interpretable
        by image/source identity when packs or CP sizes change between experiments.
        """
        metadata = [
            {key: item[key] for key in ('image_id', 'source_id', 'offset', 'length')}
            for item in media
        ]
        self._write(step, 'producer', features, media=metadata, num_rounds=num_rounds)

    @staticmethod
    def _route(transfer):
        plan = transfer.plans[transfer.local_plan]
        return dict(
            cp_ranks=list(plan.cp_ranks),
            slices=[
                dict(producer_rank=piece.producer_rank, offset=piece.offset, length=piece.length)
                for piece in plan.features
            ],
        )

    def save_receiver(self, step, round_id, phase, transfer):
        """Save the actual received/broadcast leaf, before modality insertion."""
        self._write(
            step,
            'receiver',
            transfer.features,
            round_id=round_id,
            phase=phase,
            **self._route(transfer),
        )

    def save_receiver_gradient(self, step, round_id, transfer, *, normalizer=1.0, phase='training'):
        """Save the local leaf gradient before CP SUM and reverse P2P.

        Use the same step-global token normalizer as ``save_producer_gradient``.
        A missing leaf gradient contributes zeros, exactly as the bridge does.
        """
        if not self.enabled(step):
            return
        gradient = transfer.features.grad
        if gradient is None:
            gradient = torch.zeros_like(transfer.features, dtype=torch.float32)
        self._write(
            step,
            'receiver_gradient',
            gradient,
            round_id=round_id,
            phase=phase,
            normalizer=normalizer,
            missing_leaf_gradient=transfer.features.grad is None,
            **self._route(transfer),
        )

    def save_producer_gradient(self, step, gradient, *, normalizer=1.0):
        """Save the total returned gradient after all decoder rounds."""
        self._write(step, 'producer_gradient', gradient, normalizer=normalizer)


def _difference(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(
            f'Boundary shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}'
        )
    actual, expected = actual.double(), expected.double()
    delta = actual - expected
    squared_error = delta.square().sum().item()
    squared_reference = expected.square().sum().item()
    return dict(
        elements=actual.numel(),
        exact=torch.equal(actual, expected),
        all_finite=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
        squared_error=squared_error,
        squared_reference=squared_reference,
        relative_l2=(
            math.sqrt(squared_error / squared_reference)
            if squared_reference
            else (0.0 if not squared_error else math.inf)
        ),
        max_abs_error=delta.abs().max().item() if delta.numel() else 0.0,
    )


def compare_boundary_directories(reference, candidate, step, *, expected_images=None):
    """Compare every encoder row and normalized returned gradient by image identity.

    Producer rank and local offset may change. Source/image identity, image length,
    dtype and the step-global gradient normalizer must agree. This compares the
    *returned* gradients across runs; ``analyze_boundary_directory`` independently
    checks each run's communication adjoint against its pre-communication leaves.
    """

    def load(directory):
        directory = Path(directory) / f'step{step:08d}'
        records = {}
        ranks = {}
        normalizers = set()
        for kind in ('producer', 'producer_gradient'):
            ranks[kind] = {}
            for path in sorted(directory.glob(f'{kind}-rank*.pt')):
                record = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
                rank = record['rank']
                if (
                    record['kind'] != kind
                    or record['step'] != step
                    or path.name != f'{kind}-rank{rank:05d}.pt'
                    or rank in ranks[kind]
                    or record['version'] != 1
                ):
                    raise ValueError('Boundary producer snapshot identity mismatch')
                values = record['values']
                if values.ndim != 2 or not values.is_floating_point():
                    raise ValueError('Boundary producer values must be floating-point rows')
                if not torch.isfinite(values).all():
                    raise ValueError(f'Nonfinite boundary values in {path}')
                ranks[kind][rank] = record
        if not ranks['producer'] or ranks['producer'].keys() != ranks['producer_gradient'].keys():
            raise ValueError('Missing producer or returned-gradient snapshot')
        for rank, producer in ranks['producer'].items():
            gradient = ranks['producer_gradient'][rank]
            scale = float(gradient['normalizer'])
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError('Invalid returned-gradient normalizer')
            normalizers.add(scale)
            if gradient['values'].shape != producer['values'].shape:
                raise ValueError('Returned gradient shape differs from producer features')
            offset = 0
            for item in producer['media']:
                key = (item['source_id'], item['image_id'])
                length = int(item['length'])
                if key in records or item['offset'] != offset or length <= 0:
                    raise ValueError('Ambiguous image identity or noncontiguous producer offsets')
                if offset + length > producer['values'].shape[0]:
                    raise ValueError('Image length exceeds producer feature rows')
                records[key] = dict(
                    rank=rank,
                    offset=offset,
                    length=length,
                    features=producer['values'][offset : offset + length],
                    gradient=gradient['values'][offset : offset + length],
                )
                offset += length
            if offset != producer['values'].shape[0]:
                raise ValueError('Image metadata does not cover every producer row')
        if len(normalizers) != 1:
            raise ValueError('Producers disagree on the global gradient normalizer')
        return records, normalizers.pop(), len(ranks['producer'])

    expected, expected_scale, expected_ranks = load(reference)
    observed, observed_scale, observed_ranks = load(candidate)
    if expected.keys() != observed.keys():
        raise ValueError('Source/image coverage differs between boundary directories')
    if expected_images is not None and len(expected) != expected_images:
        raise ValueError('Boundary image count differs from expected source coverage')
    if expected_scale != observed_scale:
        raise ValueError('Boundary gradient normalizers differ between runs')
    images, totals = [], {}
    for key, ref in expected.items():
        cur = observed[key]
        if ref['length'] != cur['length']:
            raise ValueError(f'Vision token count changed for image {key}')
        image = dict(
            source_id=key[0],
            image_id=key[1],
            length=ref['length'],
            reference_rank=ref['rank'],
            candidate_rank=cur['rank'],
        )
        for field in ('features', 'gradient'):
            if ref[field].dtype != cur[field].dtype:
                raise ValueError(f'Boundary dtype differs for {key}/{field}')
            metrics = _difference(cur[field], ref[field])
            metrics['squared_candidate'] = cur[field].double().square().sum().item()
            if not math.isfinite(metrics['relative_l2']):
                metrics['relative_l2'] = None
            image[field] = metrics
            total = totals.setdefault(
                field,
                dict(
                    elements=0,
                    squared_error=0.0,
                    squared_reference=0.0,
                    squared_candidate=0.0,
                    max_abs_error=0.0,
                    exact=True,
                ),
            )
            for name in ('elements', 'squared_error', 'squared_reference', 'squared_candidate'):
                total[name] += metrics[name]
            total['max_abs_error'] = max(total['max_abs_error'], metrics['max_abs_error'])
            total['exact'] &= metrics['exact']
        images.append(image)
    if not images:
        raise ValueError('Cross-layout boundary comparison requires at least one source image')
    for metrics in totals.values():
        error, ref, cur = (
            metrics[name] for name in ('squared_error', 'squared_reference', 'squared_candidate')
        )
        metrics.update(
            relative_l2=math.sqrt(error / ref) if ref else (0.0 if error == 0 else None),
            reference_l2=math.sqrt(ref),
            candidate_l2=math.sqrt(cur),
            cosine=(
                (ref + cur - error) / (2 * math.sqrt(ref) * math.sqrt(cur)) if ref and cur else None
            ),
        )
    return dict(
        kind='full_encoder_boundary_across_layouts',
        step=step,
        reference=str(reference),
        candidate=str(candidate),
        images=len(images),
        reference_producers=expected_ranks,
        candidate_producers=observed_ranks,
        normalizer=expected_scale,
        all_finite=True,
        all_exact=all(value['exact'] for value in totals.values()),
        totals=totals,
        image_comparisons=images,
    )


def analyze_cp_image_coverage(boundary_dir, execution_dir, step, *, expected_images=None):
    """Count actual image-row owners after CP slicing, including split images.

    The saved execution indices select rows in the received pack feature table.
    Resolve those rows through the saved bridge slices to producer image identity;
    require every original image row to be consumed once across the source step.
    """
    boundary_dir = Path(boundary_dir) / f'step{step:08d}'
    execution_dir = Path(execution_dir) / f'step{step:08d}'
    source_rows, images = {}, {}
    for path in sorted(boundary_dir.glob('producer-rank*.pt')):
        producer = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        rank, offset = producer['rank'], 0
        for media in producer['media']:
            key = (media['source_id'], media['image_id'])
            length = media['length']
            if key in images or media['offset'] != offset or length <= 0:
                raise ValueError('Ambiguous producer image mapping for CP coverage')
            images[key] = dict(length=length, owners=set(), counts=[0] * length)
            for row in range(length):
                source_rows[(rank, offset + row)] = (key, row)
            offset += length
        if offset != producer['values'].shape[0]:
            raise ValueError('Image metadata does not cover every producer row')
    if not images or (expected_images is not None and len(images) != expected_images):
        raise ValueError('CP image coverage differs from expected source coverage')
    rank_rounds = empty_rank_rounds = empty_in_image_packs = 0
    receivers = set(boundary_dir.glob('receiver-rank*-training.pt'))
    for path in sorted(execution_dir.glob('rank*/training-round*.pt')):
        execution = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        rank, round_id = int(path.parent.name[4:]), execution['round']
        receiver_path = boundary_dir / f'receiver-rank{rank:05d}-round{round_id:05d}-training.pt'
        if receiver_path not in receivers:
            raise ValueError('Missing or duplicate CP receiver record')
        receivers.remove(receiver_path)
        receiver = torch.load(receiver_path, map_location='cpu', weights_only=True, mmap=True)
        if (
            execution['step'] != step
            or execution['phase'] != 'training'
            or receiver['step'] != step
            or receiver['rank'] != rank
            or receiver['round_id'] != round_id
            or execution['cp_ranks'] != receiver['cp_ranks']
            or rank not in execution['cp_ranks']
            or not execution['cp_input']
        ):
            raise ValueError('Missing or mismatched CP input ownership record')
        pack_rows = []
        for piece in receiver['slices']:
            for row in range(piece['offset'], piece['offset'] + piece['length']):
                identity = source_rows.get((piece['producer_rank'], row))
                if identity is None:
                    raise ValueError('Receiver slice has no producer image row')
                pack_rows.append(identity)
        if len(pack_rows) != receiver['values'].shape[0]:
            raise ValueError('Receiver slices do not cover its feature rows')
        # Recompute repeats the same partition; count the original forward once.
        rows = execution['cp_input'][0]['feature_rows'].tolist()
        rank_rounds += 1
        empty_rank_rounds += not rows
        empty_in_image_packs += bool(pack_rows) and not rows
        for row in rows:
            if not isinstance(row, int) or not 0 <= row < len(pack_rows):
                raise ValueError('CP feature index is outside the received pack')
            key, image_row = pack_rows[row]
            images[key]['owners'].add(rank)
            images[key]['counts'][image_row] += 1
    if receivers:
        raise ValueError('Missing CP execution record, including empty local vision ranks')
    if any(any(count != 1 for count in image['counts']) for image in images.values()):
        raise ValueError('CP consumers duplicate or omit producer image rows')
    return dict(
        step=step,
        images=len(images),
        rank_rounds=rank_rounds,
        visual_rows=sum(image['length'] for image in images.values()),
        split_images=sum(len(image['owners']) > 1 for image in images.values()),
        max_image_owners=max(len(image['owners']) for image in images.values()),
        empty_local_vision_rank_rounds=empty_rank_rounds,
        empty_local_vision_in_image_packs=empty_in_image_packs,
        all_image_rows_owned_once=True,
        image_owners=[
            dict(
                source_id=key[0],
                image_id=key[1],
                length=value['length'],
                owners=sorted(value['owners']),
            )
            for key, value in images.items()
        ],
    )


def analyze_boundary_directory(output_dir, step, *, require_gradients=True):
    """Check receiver copies and the bridge adjoint from a completed step's files.

    The CPU reference sums each rank's *pre-communication* leaf gradient in FP64
    by producer row, including repeated slices and rounds. This is independent
    of the bridge's CP reduction/P2P implementation. An ordinary FP32 accumulation
    can have rounding differences for multiply consumed rows; inspect reported
    errors rather than assuming bitwise equality is mandatory in that case.

    Raises on missing/ambiguous records so a partially written run cannot pass.
    Numerical differences are returned, with per-image producer-gradient metrics.
    """
    directory = Path(output_dir) / f'step{step:08d}'
    # Map payloads on demand: CP replicas must not all occupy anonymous CPU RAM.
    records = [
        torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        for path in sorted(directory.glob('*.pt'))
    ]
    if not records:
        raise ValueError(f'No boundary records for source step {step}')
    indexed = {}
    for record in records:
        if record['version'] != 1 or record['step'] != step:
            raise ValueError('Unexpected boundary record version or source step')
        key = (record['kind'], record['rank'], record['round_id'], record['phase'])
        if key in indexed:
            raise ValueError(f'Duplicate boundary record: {key}')
        indexed[key] = record
    producers = {r['rank']: r for r in records if r['kind'] == 'producer'}
    if not producers:
        raise ValueError('Missing producer snapshots')
    image_ids = set()
    for producer in producers.values():
        offset = 0
        for image in producer['media']:
            if image['image_id'] in image_ids or image['offset'] != offset or image['length'] <= 0:
                raise ValueError('Image IDs and producer offsets must uniquely cover producer rows')
            image_ids.add(image['image_id'])
            offset += image['length']
        if offset != len(producer['values']):
            raise ValueError('Image metadata does not cover every producer row')
    receivers = [r for r in records if r['kind'] == 'receiver']
    if not receivers:
        raise ValueError('Missing receiver snapshots')
    totals = {
        rank: torch.zeros_like(r['values'], dtype=torch.float64) for rank, r in producers.items()
    }
    comparisons, seen_gradients = [], set()
    rounds = {}
    for receiver in receivers:
        rank, round_id, phase = (receiver[k] for k in ('rank', 'round_id', 'phase'))
        if rank not in receiver['cp_ranks'] or len(set(receiver['cp_ranks'])) != len(
            receiver['cp_ranks']
        ):
            raise ValueError('Invalid receiver CP membership')
        rounds.setdefault((round_id, phase), set()).add(rank)
        # Every member must have saved the same route, even if it has no vision tokens.
        for peer in receiver['cp_ranks']:
            other = indexed.get(('receiver', peer, round_id, phase))
            if other is None or any(other[k] != receiver[k] for k in ('cp_ranks', 'slices')):
                raise ValueError('Missing CP receiver or inconsistent saved route')
        expected_pieces = []
        for piece in receiver['slices']:
            producer = producers.get(piece['producer_rank'])
            start, length = piece['offset'], piece['length']
            if (
                producer is None
                or start < 0
                or length < 0
                or start + length > len(producer['values'])
            ):
                raise ValueError('Receiver refers to an absent producer or invalid source rows')
            expected_pieces.append(producer['values'][start : start + length])
        expected = (
            torch.cat(expected_pieces)
            if expected_pieces
            else receiver['values'].new_empty((0, receiver['values'].shape[1]))
        )
        comparisons.append(
            dict(
                rank=rank,
                round_id=round_id,
                phase=phase,
                **_difference(receiver['values'], expected),
            )
        )
        if not require_gradients or phase != 'training':
            continue
        key = ('receiver_gradient', rank, round_id, phase)
        gradient = indexed.get(key)
        if gradient is None:
            raise ValueError(f'Missing pre-communication receiver gradient: {key}')
        seen_gradients.add(key)
        if any(gradient[k] != receiver[k] for k in ('cp_ranks', 'slices')):
            raise ValueError('Forward and backward routes do not match')
        if gradient['values'].shape != receiver['values'].shape:
            raise ValueError('Receiver gradient shape does not match forward features')
        offset = 0
        for piece in gradient['slices']:
            rank, start, length = (piece[k] for k in ('producer_rank', 'offset', 'length'))
            returned = indexed.get(('producer_gradient', rank, None, None))
            if returned is None or returned['normalizer'] != gradient['normalizer']:
                raise ValueError('Missing returned producer gradient or inconsistent normalization')
            totals[rank][start : start + length].add_(
                gradient['values'][offset : offset + length].double()
            )
            offset += length
    if any(ranks != set(producers) for ranks in rounds.values()):
        raise ValueError('Each receiver round must cover all saved producer ranks')
    expected_rounds = {record['num_rounds'] for record in producers.values()}
    if len(expected_rounds) != 1:
        raise ValueError('Producers disagree on the number of decoder rounds')
    num_rounds = expected_rounds.pop()
    if num_rounds is not None:
        for phase in {phase for _, phase in rounds}:
            if {round_id for round_id, name in rounds if name == phase} != set(range(num_rounds)):
                raise ValueError('Missing complete decoder rounds')
    gradients, images = [], []
    if require_gradients:
        saved_gradients = {key for key in indexed if key[0] == 'receiver_gradient'}
        if not seen_gradients or seen_gradients != saved_gradients:
            raise ValueError('Missing training receivers or unmatched receiver gradient records')
        for rank, producer in producers.items():
            returned = indexed.get(('producer_gradient', rank, None, None))
            if returned is None:
                raise ValueError(f'Missing returned gradient for producer rank {rank}')
            actual, expected = returned['values'], totals[rank]
            gradients.append(dict(rank=rank, **_difference(actual, expected)))
            for image in producer['media']:
                start, length = image['offset'], image['length']
                images.append(
                    dict(
                        **image,
                        rank=rank,
                        **_difference(
                            actual[start : start + length], expected[start : start + length]
                        ),
                    )
                )
    results = comparisons + gradients
    return dict(
        step=step,
        producers=len(producers),
        receiver_comparisons=comparisons,
        producer_gradient_comparisons=gradients,
        image_gradient_comparisons=images,
        all_exact=all(result['exact'] for result in results),
        all_finite=all(result['all_finite'] for result in results),
    )
