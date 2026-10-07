# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU contracts for Bridge-style offline packs entering the DCP scheduler."""

import json
import pickle
from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets.data_schedule_utils import (
    _pack_sequences,
    _unpack_batch,
    pad_packed_batch_before_cp_slice,
)
from megatron.training.datasets.packed_sft_dataset import (
    PACKED_SFT_CONTRACT,
    PACKED_SFT_METADATA_KEY,
    PackedSFTLowLevelDataset,
    get_sft_padding_divisor,
    packed_row_to_tensors,
)
from megatron.training.datasets.sft_dataset import SFTDataset


def sample_pair():
    # EOS==pad==9 is a real supervised token in both constituents. Their masks
    # also contain real prompt tokens, so neither masks nor IDs imply lengths.
    return [
        {"input_ids": [1, 2, 3, 9], "targets": [-100, -100, 3, 9]},
        {"input_ids": [4, 5, 6, 7, 8, 9], "targets": [-100, -100, 6, 7, 8, 9]},
    ]


def packed_pair():
    # An independently specified artifact, so reader tests do not depend on the
    # example producer or reproduce its mask/offset construction algorithm.
    return {
        "input_ids": [1, 2, 3, 9, 4, 5, 6, 7, 8, 9],
        "loss_mask": [0, 1, 1, 0, 0, 1, 1, 1, 1, 0],
        "seq_start_id": [0, 4],
    }


@pytest.mark.parametrize("divisor", [1, 2, 4, 8])
def test_packed_reader_preserves_labels_masks_and_lengths(divisor):
    originals = sample_pair()
    row = packed_pair()
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


def test_bridge_storage_mask_is_not_shifted_twice():
    row = packed_pair()
    assert row["seq_start_id"] == [0, 4]
    assert row["loss_mask"] == [0, 1, 1, 0, 0, 1, 1, 1, 1, 0]
    batch = packed_row_to_tensors(row, pad_token_id=9, padding_divisor=1, sequence_length=8)
    assert batch["tokens"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert batch["labels"].tolist() == [-100, 3, 9, -100, 6, 7, 8, 9]
    assert batch["cu_seqlens"].tolist() == [0, 3, 8]


def test_full_loss_roundtrip_and_no_cross_sample_target():
    row = {
        "input_ids": [1, 2, 3, 10, 11, 12],
        "loss_mask": [1, 1, 0, 1, 1, 0],
        "seq_start_id": [0, 3],
    }
    batch = packed_row_to_tensors(row, pad_token_id=0, padding_divisor=1, sequence_length=4)
    assert batch["tokens"].tolist() == [1, 2, 10, 11]
    assert batch["labels"].tolist() == [2, 3, 11, 12]
    assert batch["loss_mask"].tolist() == [1, 1, 1, 1]


def test_64k_pack_budget_counts_shifted_tokens_not_storage_tokens():
    row = {
        "input_ids": list(range(32769)) + list(range(40000, 72769)),
        "loss_mask": ([1] * 32768 + [0]) * 2,
        "seq_start_id": [0, 32769],
    }
    assert len(row["input_ids"]) == 65538
    batch = packed_row_to_tensors(row, pad_token_id=0, padding_divisor=8, sequence_length=65536)
    assert batch["tokens"].numel() == 65536
    assert batch["cu_seqlens"].tolist() == [0, 32768, 65536]
    assert batch["cu_seqlens_padded"].tolist() == [0, 32768, 65536]
    assert batch["labels"][32767].item() == 32768
    assert batch["tokens"][32768].item() == 40000
    assert batch["labels"][-1].item() == 72768
    assert batch["loss_mask"].sum().item() == 65536


def test_padding_alignment_applies_to_packed_cp_local_width():
    batch = packed_row_to_tensors(
        packed_pair(), pad_token_id=9, padding_divisor=8, sequence_length=32
    )
    samples = _unpack_batch([batch])
    repacked = _pack_sequences(
        samples,
        torch.cat([sample["padded_seq_len"] for sample in samples]),
        torch.cat([sample["original_seq_len"] for sample in samples]),
        None,
        torch.device("cpu"),
    )
    config = SimpleNamespace(
        pad_packed_seq_alignment=32,
        fp8="e4m3",
        fp8_recipe="mxfp8",
        sequence_parallel=False,
        max_seqlen_per_dp_cp_rank=32,
    )
    pad_packed_batch_before_cp_slice(repacked, config, cp_size=4, tp_size=1)
    assert repacked["tokens"].numel() == 128
    # The individual physical sequences remain 8 tokens each, not 32 each.
    assert repacked["cu_seqlens_padded"][:3].tolist() == [0, 8, 16]
    assert repacked["cu_seqlens"][:3].tolist() == [0, 3, 8]
    assert not repacked["loss_mask"][16:].any()
    assert repacked["loss_mask"].sum().item() == 6


def test_mtp_roll_does_not_cross_constituents_or_padding():
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.multi_token_prediction import _roll_tensor_packed_seq

    batch = packed_row_to_tensors(
        packed_pair(), pad_token_id=9, padding_divisor=8, sequence_length=32
    )
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=batch["cu_seqlens"],
        cu_seqlens_q_padded=batch["cu_seqlens_padded"],
    )
    rolled, count = _roll_tensor_packed_seq(batch["loss_mask"].unsqueeze(0), -1, -1, params)
    # Each sample is rolled independently and its physical padding stays zero.
    assert rolled.tolist() == [[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]]
    assert count.item() == 6


@pytest.mark.parametrize(
    "cp,dp,sp,dynamic,expected",
    [(1, 1, 1, False, 1), (4, 1, 1, False, 8), (2, 2, 1, True, 8), (1, 4, 2, True, 16)],
)
def test_topology_alignment_matches_online(cp, dp, sp, dynamic, expected):
    config = SimpleNamespace(
        context_parallel_size=cp,
        data_parallel_size=dp,
        sequence_parallel_size=sp,
        dynamic_context_parallel=dynamic,
    )
    dataset = object.__new__(SFTDataset)
    dataset.config = config
    assert dataset._calculate_padding_divisor() == expected
    assert get_sft_padding_divisor(**vars(config)) == expected


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


@pytest.mark.parametrize(
    "mutate,error",
    [
        (lambda row: row.update(seq_start_id=[1]), "starting at zero"),
        (lambda row: row.update(seq_start_id=[0, 1]), "at least two"),
        (lambda row: row.update(loss_mask=[1] * len(row["input_ids"])), "final stored"),
        (lambda row: row.update(loss_mask=[2] * len(row["input_ids"])), "zero or one"),
    ],
)
def test_malformed_artifact_is_rejected(mutate, error):
    row = packed_pair()
    mutate(row)
    with pytest.raises(ValueError, match=error):
        packed_row_to_tensors(row, pad_token_id=9, padding_divisor=4, sequence_length=32)


def test_repacking_for_larger_topology_fails_clearly():
    with pytest.raises(ValueError, match="rebuild packs"):
        packed_row_to_tensors(packed_pair(), pad_token_id=9, padding_divisor=8, sequence_length=8)


def test_unpack_preserves_legacy_physical_boundaries():
    batch = packed_row_to_tensors(
        packed_pair(), pad_token_id=9, padding_divisor=1, sequence_length=32
    )
    batch.pop("cu_seqlens_padded")
    restored = _unpack_batch([batch])
    assert [sample["original_seq_len"].item() for sample in restored] == [3, 5]


def test_unpack_rejects_inconsistent_logical_physical_lengths():
    batch = packed_row_to_tensors(
        packed_pair(), pad_token_id=9, padding_divisor=1, sequence_length=32
    )
    batch["cu_seqlens"] = torch.tensor([0, 4, 8])
    with pytest.raises(ValueError, match="boundaries"):
        _unpack_batch([batch])


def test_parquet_loader_row_groups_and_worker_pickling(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    row = packed_pair()
    table = pa.Table.from_pylist([row, row, row]).replace_schema_metadata(
        {PACKED_SFT_METADATA_KEY: json.dumps(PACKED_SFT_CONTRACT).encode()}
    )
    path = tmp_path / "packed.parquet"
    pq.write_table(table, path, row_group_size=1)
    dataset = SFTDataset.build_low_level_dataset(str(path), None)
    assert isinstance(dataset, PackedSFTLowLevelDataset)
    assert len(dataset) == 3
    assert dataset[2] == row
    assert dataset[0] == row
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored[1] == row
    with pytest.raises(IndexError):
        restored[3]


def test_parquet_loader_rejects_unknown_padding_contract(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "legacy.parquet"
    pq.write_table(pa.Table.from_pylist([packed_pair()]), path)
    with pytest.raises(ValueError, match="real constituent lengths"):
        PackedSFTLowLevelDataset(path)
