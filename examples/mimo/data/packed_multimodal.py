# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Join existing decoder packing schedules with independent encoder image tasks.

The scheduler owns sample packing and runtime CP sizes. This adapter preserves
media identity and constructs the feature order consumed by the MIMO bridge;
token concatenation remains in ``data_schedule_utils.build_packed_microbatches``.
"""

from megatron.core.models.mimo.comm.pack_bridge import FeatureSlice, PackFeaturePlan


def assign_encoder_media(media_specs, producer_ranks):
    """Greedily balance whole images, then establish each producer's row offsets.

    Squared pixel area is proportional to squared patch count for a common patch
    size. Images are encoded in image-ID order on each producer; callers must
    preserve that order when concatenating the projected outputs.
    """
    producer_ranks = tuple(producer_ranks)
    if len(set(producer_ranks)) != len(producer_ranks):
        raise ValueError("Encoder producer ranks must be unique")
    if media_specs and not producer_ranks:
        raise ValueError("Images require at least one encoder producer")
    if len({media["image_id"] for media in media_specs}) != len(media_specs):
        raise ValueError("Each encoder image task must have a unique image ID")

    def cost(media):
        size = media["size"]
        height, width = (size, size) if isinstance(size, int) else size
        if min(height, width, media["length"]) <= 0:
            raise ValueError("Image dimensions and projected feature length must be positive")
        return (height * width) ** 2

    assignments = {rank: [] for rank in producer_ranks}
    loads = [0] * len(producer_ranks)
    for media in sorted(media_specs, key=lambda item: (-cost(item), item["image_id"])):
        producer_index = min(range(len(producer_ranks)), key=lambda index: (loads[index], index))
        assignments[producer_ranks[producer_index]].append(media)
        loads[producer_index] += cost(media)

    feature_slices = {}
    for rank, tasks in assignments.items():
        tasks.sort(key=lambda media: media["image_id"])
        offset = 0
        for media in tasks:
            length = media["length"]
            feature_slices[media["image_id"]] = FeatureSlice(rank, offset, length)
            offset += length
    return assignments, feature_slices


def build_round_plans(sample_ids_by_rank, samples, feature_slices, domain_ranks):
    """Convert one decoder round to bridge plans in deterministic domain order.

    Identical ordered sample tuples identify ranks sharing one decoder pack.
    ``media_ids`` must already follow placeholder order within each sample.
    Runtime process-group lookup remains the caller's responsibility.
    """
    domain_ranks = tuple(domain_ranks)
    if len(sample_ids_by_rank) != len(domain_ranks) or len(set(domain_ranks)) != len(domain_ranks):
        raise ValueError("Round assignments must match a domain of unique global ranks")
    packs = {}
    for rank, sample_ids in zip(domain_ranks, sample_ids_by_rank):
        sample_ids = tuple(sample_ids)
        if not sample_ids or any(sample_id not in samples for sample_id in sample_ids):
            raise ValueError("Each decoder rank needs a nonempty pack of known sample IDs")
        packs.setdefault(sample_ids, []).append(rank)
    plans = []
    for sample_ids, ranks in packs.items():
        features = tuple(
            feature_slices[image_id]
            for sample_id in sample_ids
            for image_id in samples[sample_id]["media_ids"]
        )
        plans.append(PackFeaturePlan(tuple(ranks), features))
    return tuple(plans)
