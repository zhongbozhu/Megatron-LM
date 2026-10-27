# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The dataset must pad for the largest runtime CP group, including GTP peers."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import pretrain_gpt
from megatron.training.arguments import parse_args
from megatron.training.datasets.packed_sft_dataset import packed_row_to_tensors
from megatron.training.datasets.varlen_dataset import MockVarlenDataset


@pytest.mark.parametrize(
    "world,tp,cp,gtp,sp,dynamic,expected_divisor",
    [
        (4, 1, 2, 1, False, True, 8),
        (4, 1, 2, 2, False, True, 8),
        (8, 1, 2, 2, False, True, 16),
        (8, 2, 2, 2, True, True, 16),
        (4, 1, 2, 2, False, False, 4),
    ],
)
def test_dataset_entry_aligns_online_and_offline_for_full_runtime_cp(
    monkeypatch, world, tp, cp, gtp, sp, dynamic, expected_divisor
):
    monkeypatch.setattr(sys, "argv", ["test_gtp_dataset_alignment.py"])
    args = parse_args()
    args.seq_length = 128
    args.context_parallel_size = cp
    args.tensor_model_parallel_size = tp
    args.sequence_parallel = sp
    args.dynamic_context_parallel = dynamic
    args.data_parallel_size = world // (tp * cp * gtp)
    args.batch_parallel_size = world // (tp * cp)
    tokenizer = SimpleNamespace(eod=0, pad=0, vocab_size=128)
    monkeypatch.setattr(pretrain_gpt, "build_tokenizer", lambda args: tokenizer)
    monkeypatch.setattr(pretrain_gpt, "get_blend_and_blend_per_split", lambda args: (None, None))
    monkeypatch.setattr(
        pretrain_gpt,
        "get_run_config",
        lambda: SimpleNamespace(
            rng=SimpleNamespace(seed=1234),
            validation=SimpleNamespace(multiple_validation_sets=False, full_validation=False),
        ),
    )
    config = pretrain_gpt.core_gpt_dataset_config_from_args(args)

    # This physical length fits the former GTP-excluded divisor, but cannot
    # supply both zigzag chunks on every rank of the full dynamic CP group.
    logical_length = expected_divisor * 3 // 2
    dataset = MockVarlenDataset.__new__(MockVarlenDataset)
    dataset.config = config
    dataset.indices = np.asarray([0])
    dataset.dataset = [np.arange(1, logical_length + 1)]
    assert dataset._calculate_padding_divisor() == expected_divisor
    online = dataset[0]
    physical_length = expected_divisor * 2
    assert online['original_seq_len'].item() == logical_length
    assert online['padded_seq_len'].item() == physical_length
    assert online['loss_mask'].sum().item() == logical_length
    assert not online['loss_mask'][logical_length:].any()

    offline = packed_row_to_tensors(
        {
            'input_ids': [*range(1, logical_length + 1), tokenizer.eod],
            'loss_mask': [1] * logical_length + [0],
            'seq_start_id': [0],
        },
        pad_token_id=tokenizer.pad,
        padding_divisor=dataset._calculate_padding_divisor(),
        sequence_length=config.sequence_length,
    )
    for key in ('tokens', 'labels', 'loss_mask', 'position_ids'):
        torch.testing.assert_close(online[key], offline[key], rtol=0, atol=0)
    assert offline['cu_seqlens'].tolist() == [0, logical_length]
    assert offline['cu_seqlens_padded'].tolist() == [0, physical_length]

    runtime_cp = world // tp if dynamic else cp
    assert physical_length % (2 * runtime_cp * (tp if sp else 1)) == 0
