# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Packing algorithm and producer/consumer contracts for the agentic SFT example."""

import random

import pytest

from examples.agentic_sft.packing import first_fit_decreasing, make_packed_row, shuffle_packed_rows
from megatron.core.datasets.data_schedule_utils import _unpack_batch
from megatron.training.datasets.packed_sft_dataset import packed_row_to_tensors


def sample_pair():
    # EOS==pad==9 is a real supervised token in both constituents. Their masks
    # also contain real prompt tokens, so neither masks nor IDs imply lengths.
    return [
        {"input_ids": [1, 2, 3, 9], "targets": [-100, -100, 3, 9]},
        {"input_ids": [4, 5, 6, 7, 8, 9], "targets": [-100, -100, 6, 7, 8, 9]},
    ]


@pytest.mark.parametrize("divisor", [1, 2, 4, 8])
def test_offline_roundtrip_preserves_online_labels_masks_and_lengths(divisor):
    originals = sample_pair()
    row = make_packed_row(originals)
    batch = packed_row_to_tensors(row, pad_token_id=9, padding_divisor=divisor, sequence_length=32)
    unpacked = _unpack_batch([batch])
    assert len(unpacked) == 2
    for original, restored in zip(originals, unpacked):
        length = len(original["input_ids"]) - 1
        physical = length + (-length % divisor)
        assert restored["original_seq_len"].item() == length
        assert restored["padded_seq_len"].item() == physical
        assert restored["tokens"][:length].tolist() == original["input_ids"][:-1]
        assert restored["labels"][:length].tolist() == original["targets"][1:]
        assert restored["loss_mask"][:length].tolist() == [
            int(target != -100) for target in original["targets"][1:]
        ]
        assert not restored["loss_mask"][length:].any()
        assert restored["position_ids"].tolist() == list(range(physical))
        assert restored["labels"][length - 1] == 9
        assert restored["loss_mask"][length - 1] == 1


def test_first_fit_decreasing_preserves_samples_and_accounts_for_padding():
    bins = first_fit_decreasing([8, 16, 8, 24, 32], 32)
    assert bins == [[4], [3, 0], [1, 2]]
    assert sorted(index for row in bins for index in row) == list(range(5))
    with pytest.raises(ValueError, match="Every aligned"):
        first_fit_decreasing([33], 32)


def test_row_shuffle_is_reproducible_and_preserves_bin_membership():
    assignments = [[0], [1, 2], [3], [4, 5], [6], [7, 8], [9], [10, 11]]
    original = tuple(tuple(row) for row in assignments)
    rng_state = random.getstate()
    shuffled = shuffle_packed_rows(assignments, seed=1234)
    assert shuffled == shuffle_packed_rows(assignments, seed=1234)
    assert shuffled != shuffle_packed_rows(assignments, seed=4321)
    assert tuple(tuple(row) for row in assignments) == original
    assert sorted(tuple(row) for row in shuffled) == sorted(original)
    assert random.getstate() == rng_state
