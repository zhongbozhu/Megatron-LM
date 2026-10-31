# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Qwen data adapter boundaries: image row order and native source resume."""

from argparse import ArgumentParser
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")

from examples.mimo.data import qwen35_native
from examples.mimo.training.args import add_mimo_dcp_args
from megatron import training
from megatron.training import global_vars


@pytest.mark.parametrize(
    "schedule", (None, "whole-step", "per-round"), ids=("default", "whole-step", "per-round")
)
def test_qwen_image_assignment_preserves_thd_row_order(monkeypatch, schedule):
    cli = [] if schedule is None else ["--mimo-dcp-encoder-schedule", schedule]
    args = add_mimo_dcp_args(ArgumentParser()).parse_args(cli)
    assert args.mimo_dcp_encoder_schedule == (schedule or "whole-step")
    per_round = args.mimo_dcp_encoder_schedule == "per-round"
    source_kwargs = {"per_round": per_round} if schedule is not None else {}
    media = []
    for image_id, grid_size in (
        (91, (2, 2)),
        (4, (2, 4)),
        (27, (4, 4)),
        (101, (2, 2)),
        (9, (2, 2)),
    ):
        height, width = grid_size
        media.append(
            dict(
                image_id=image_id,
                size=(16 * height, 16 * width),
                length=height * width // 4,
                grid=torch.tensor([[1, height, width]]),
                pixel_values=torch.full((height * width, 12), image_id),
            )
        )
    samples = {"A": {"media_ids": [91, 4, 27]}, "B": {"media_ids": [101, 9, 91]}}
    source = dict(samples=samples, media=media)
    expected = {0: [27], 1: [4], 2: [91, 9], 3: [101]}
    by_id = {item["image_id"]: item for item in media}
    materialized = []
    build_vision_inputs = qwen35_native.build_vision_inputs

    def observe_materialization(items, device):
        materialized.append([item["image_id"] for item in items])
        assert all(item["pixel_values"].device.type == "cpu" for item in items)
        return build_vision_inputs(items, device)

    monkeypatch.setattr(qwen35_native, "build_vision_inputs", observe_materialization)
    for rank, ids in expected.items():
        calls_before = len(materialized)
        actual_samples, slices, encoder_inputs = qwen35_native.prepare_source(
            source, producer_ranks=range(4), rank=rank, device="cpu", **source_kwargs
        )
        assert actual_samples is samples
        assert callable(encoder_inputs) is per_round
        assert len(materialized) == calls_before + (not per_round)
        for requested in (ids, ids[::-1], ids[1:], []) if per_round else (ids,):
            inputs = encoder_inputs(requested) if per_round else encoder_inputs
            assert materialized[-1] == requested
            if not requested:
                assert inputs == {}
                continue
            vision = inputs["images"]["qwen35"]
            torch.testing.assert_close(
                vision["hidden_states"],
                torch.cat([by_id[key]["pixel_values"] for key in requested]).bfloat16(),
            )
            torch.testing.assert_close(
                vision["grid_thw"], torch.cat([by_id[key]["grid"] for key in requested])
            )
        if per_round:
            foreign_id = next(key for key in by_id if key not in ids)
            with pytest.raises(ValueError, match="belong to this encoder producer"):
                encoder_inputs([foreign_id])
        offset = 0
        for key in ids:
            assert (slices[key].producer_rank, slices[key].offset, slices[key].length) == (
                rank,
                offset,
                by_id[key]["length"],
            )
            offset += by_id[key]["length"]
    _, _, encoder_inputs = qwen35_native.prepare_source(
        source, producer_ranks=range(8), rank=7, device="cpu", **source_kwargs
    )
    assert (encoder_inputs([]) if per_round else encoder_inputs) == {}
    assert all(item["pixel_values"].device.type == "cpu" for item in media)
    assert samples["B"]["media_ids"] == [101, 9, 91]


def test_qwen_provider_uses_native_train_and_eval_resume(monkeypatch):
    class Dataset:
        def __init__(self, *args, split, **kwargs):
            self.split = split

        def build_global_batch(self, step, size):
            return (self.split, tuple(range(step * size, (step + 1) * size))), []

    args = SimpleNamespace(
        dataloader_type="external",
        mimo_manifest="manifest",
        mimo_hf_model="hf",
        seq_length=131072,
        seed=1234,
        mimo_heldout_images=64,
        mimo_data_split_seed=1234,
    )
    config = SimpleNamespace(
        rng=SimpleNamespace(seed=1234),
        train=SimpleNamespace(global_batch_size=64),
        validation=SimpleNamespace(
            full_validation=False, eval_iters=1, eval_global_batch_size=16, skip_train=False
        ),
    )
    state = SimpleNamespace(consumed_train_samples=192, consumed_valid_samples=32)
    monkeypatch.setattr(qwen35_native, "Qwen35Dataset", Dataset)
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(global_vars, "get_run_config", lambda: config)
    monkeypatch.setattr(global_vars, "get_train_state", lambda: state)
    train, valid, test = qwen35_native.train_valid_test_datasets_provider(None)
    assert next(train)["samples"] == ("train", tuple(range(192, 256)))
    assert next(valid)["samples"] == ("valid", tuple(range(32, 48)))
    assert test is None
    config.validation.skip_train = True
    _, valid, _ = qwen35_native.train_valid_test_datasets_provider(None)
    assert next(valid)["samples"] == ("valid", tuple(range(16)))
    config.validation.skip_train = False
    state.consumed_valid_samples = 33
    with pytest.raises(ValueError, match="complete source global batch"):
        qwen35_native.train_valid_test_datasets_provider(None)
