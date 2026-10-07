# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Read tokenized SFT packs without re-tokenizing or crossing sample boundaries.

The three data columns match Megatron Bridge's offline packing format. Each
constituent stores its complete, unpadded token stream, including the extra
token needed for next-token prediction. ``loss_mask`` is already label-aligned;
``seq_start_id`` contains *storage* offsets, not shifted runtime offsets.
"""

import bisect
import json
from pathlib import Path

import torch

IGNORE_INDEX = -100
PACKED_SFT_METADATA_KEY = b"megatron_sft_contract"
PACKED_SFT_CONTRACT = {
    "version": 1,
    "input_ids": "unshifted_unpadded",
    "loss_mask": "label_aligned_last_position_zero",
    "seq_start_id": "storage_offsets",
}
PACKED_SFT_COLUMNS = ("input_ids", "loss_mask", "seq_start_id")


def get_sft_padding_divisor(
    *, context_parallel_size, data_parallel_size, sequence_parallel_size, dynamic_context_parallel
):
    """Return the same per-constituent alignment for online and offline SFT."""
    cp_size = context_parallel_size
    if dynamic_context_parallel:
        cp_size *= data_parallel_size
    cp_alignment = 2 * cp_size if cp_size > 1 or dynamic_context_parallel else 1
    return cp_alignment * max(sequence_parallel_size, 1)


def packed_row_to_tensors(row, *, pad_token_id, padding_divisor, sequence_length):
    """Shift every constituent once, then add explicitly masked physical padding.

    Real EOS tokens remain valid, even when EOS is also the padding token ID.
    Lengths come exclusively from stored boundaries, never from token values or
    the supervision mask (which also masks real prompt tokens).
    """
    ids, mask, starts = (row[key] for key in PACKED_SFT_COLUMNS)
    if len(ids) != len(mask) or not starts or starts[0] != 0:
        raise ValueError("Packed SFT requires equally sized ids/mask and offsets starting at zero")
    if padding_divisor < 1:
        raise ValueError("Packed SFT padding divisor must be positive")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in ids):
        raise ValueError("Packed SFT token IDs must be nonnegative integers")
    if any(value not in (0, 1) for value in mask):
        raise ValueError("Packed SFT loss_mask must contain only zero or one")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in starts):
        raise ValueError("Packed SFT offsets must be integers")
    boundaries = [*starts, len(ids)]
    if any(end - start < 2 for start, end in zip(boundaries, boundaries[1:])):
        raise ValueError("Every packed SFT constituent must contain at least two stored tokens")

    tokens, labels, losses, positions = [], [], [], []
    logical, physical = [0], [0]
    for start, end in zip(boundaries, boundaries[1:]):
        if mask[end - 1] != 0:
            raise ValueError("Packed SFT constituent's final stored loss_mask must be zero")
        real_length = end - start - 1
        if real_length > sequence_length:
            raise ValueError("Packed SFT constituent exceeds the configured sequence length")
        padded_length = real_length + (-real_length % padding_divisor)
        padding = padded_length - real_length
        tokens.extend(ids[start : end - 1])
        labels.extend(
            token if supervise else IGNORE_INDEX
            for token, supervise in zip(ids[start + 1 : end], mask[start : end - 1])
        )
        losses.extend(mask[start : end - 1])
        tokens.extend([pad_token_id] * padding)
        labels.extend([pad_token_id] * padding)
        losses.extend([0] * padding)
        positions.extend(range(padded_length))
        logical.append(logical[-1] + real_length)
        physical.append(physical[-1] + padded_length)

    if physical[-1] > sequence_length:
        raise ValueError(
            f"Packed row needs {physical[-1]} tokens after alignment, exceeding "
            f"sequence_length={sequence_length}; rebuild packs for this topology"
        )
    return {
        "tokens": torch.tensor(tokens, dtype=torch.int64),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "loss_mask": torch.tensor(losses, dtype=torch.float32),
        "position_ids": torch.tensor(positions, dtype=torch.int64),
        "cu_seqlens": torch.tensor(logical, dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor(physical, dtype=torch.int32),
        "max_seqlen": torch.tensor(
            max(b - a for a, b in zip(physical, physical[1:])), dtype=torch.int32
        ),
    }


class PackedSFTLowLevelDataset:
    """Lazy Parquet row-group reader; each DataLoader worker owns its reader.

    A schema contract is required because legacy Bridge files may have padding
    inside constituents without recording their real lengths. Such padding
    cannot safely be recovered by scanning for EOS/pad IDs or zero loss masks.
    """

    def __init__(self, path):
        import pyarrow.parquet as pq

        self.path = str(Path(path))
        reader = pq.ParquetFile(self.path)
        if not set(PACKED_SFT_COLUMNS).issubset(reader.schema_arrow.names):
            raise ValueError(f"Packed SFT requires columns {PACKED_SFT_COLUMNS}: {self.path}")
        metadata = reader.schema_arrow.metadata or {}
        contract = json.loads(metadata.get(PACKED_SFT_METADATA_KEY, b"null"))
        if contract != PACKED_SFT_CONTRACT:
            raise ValueError(
                "Packed SFT requires explicit unpadded storage/label-mask metadata; "
                "rebuild using the agentic SFT packing script. Legacy padded Bridge "
                "artifacts do not preserve the real constituent lengths."
            )
        self.row_offsets = [0]
        for index in range(reader.num_row_groups):
            self.row_offsets.append(
                self.row_offsets[-1] + reader.metadata.row_group(index).num_rows
            )
        if not self.row_offsets[-1]:
            raise ValueError(f"Empty packed SFT dataset: {self.path}")
        self._reader = None
        self._row_group_index = None
        self._row_group = None

    def __len__(self):
        return self.row_offsets[-1]

    def __getitem__(self, index):
        import pyarrow.parquet as pq

        if not 0 <= index < len(self):
            raise IndexError(index)
        group = bisect.bisect_right(self.row_offsets, index) - 1
        if self._reader is None:
            self._reader = pq.ParquetFile(self.path)
        if self._row_group_index != group:
            self._row_group = self._reader.read_row_group(group, columns=PACKED_SFT_COLUMNS)
            self._row_group_index = group
        row = index - self.row_offsets[group]
        return {name: self._row_group[name][row].as_py() for name in PACKED_SFT_COLUMNS}

    def __getstate__(self):
        return {**self.__dict__, "_reader": None, "_row_group_index": None, "_row_group": None}
