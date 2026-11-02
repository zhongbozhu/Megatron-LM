# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Qwen data adapter boundaries: image row order and native source resume."""

import json
import os
from argparse import ArgumentParser
from functools import lru_cache
from threading import Event
from types import SimpleNamespace

import pytest
import torch
from PIL import ImageFile

pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")

from examples.mimo.data import qwen35_native
from examples.mimo.data.source_batch import SourceBatchIterator
from examples.mimo.training.args import add_mimo_dcp_args
from megatron import training
from megatron.training import global_vars


def test_qwen_native_multiimage_document(tmp_path, monkeypatch):
    snapshot = os.environ.get("MIMO_HF_MODEL")
    if not snapshot:
        pytest.skip("Set MIMO_HF_MODEL to a local Qwen3.5 processor snapshot")
    paths = []
    for index, size in enumerate(((65, 97), (129, 65), (257, 193))):
        path = tmp_path / f"page-{index}.png"
        width, height = size
        pixels = (torch.arange(height * width * 3) + index * 37).remainder(256)
        qwen35_native.Image.fromarray(pixels.reshape(height, width, 3).byte().numpy()).save(path)
        paths.append(path)
    records, metadata = [], []
    for index, pages in enumerate((paths[:2], paths[2:])):
        records.append(
            dict(
                id=f"document-{index}",
                messages=[
                    dict(
                        role="user",
                        content=[
                            *[dict(type="image", image=path.name) for path in pages],
                            "Describe these pages.",
                        ],
                    ),
                    dict(role="assistant", content=["These are colored pages."]),
                ],
            )
        )
        metadata.append(
            dict(
                record_index=index,
                split_group=f"document-{index}",
                image_paths=[path.name for path in pages],
                images=[dict(path=path.name, sha256=path.stem) for path in pages],
            )
        )
    (tmp_path / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(records_file="records.jsonl", samples=metadata)))
    dataset = qwen35_native.Qwen35Dataset(manifest, snapshot, heldout_images=1)

    def forbid_pixels(patch):
        def fail(*args, **kwargs):
            raise AssertionError("Metadata planning or a decoder rank processed image pixels")

        patch.setattr(qwen35_native.Image.Image, "convert", fail)
        patch.setattr(ImageFile.ImageFile, "load", fail)
        patch.setattr(dataset.processor.image_processor, "preprocess", fail)

    reference_pixels = {}
    for index, pages in enumerate((paths[:2], paths[2:])):
        with monkeypatch.context() as patch:
            forbid_pixels(patch)
            sample, media = dataset._sample(index)
        assert len(media) == len(pages)
        assert all("pixel_values" not in item and callable(item["load_pixels"]) for item in media)
        messages = dataset.examples[index]["messages"]
        rendered = dataset.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        images = []
        for path in pages:
            with qwen35_native.Image.open(path) as image:
                images.append(image.convert("RGB"))
        try:
            reference = dataset.processor(text=[rendered], images=images, return_tensors="pt")
        finally:
            for image in images:
                image.close()
        logical = sample["original_seq_len"]
        ids = reference["input_ids"]
        assert logical == ids.shape[1] - 1
        torch.testing.assert_close(sample["tokens"][:logical], ids[0, :-1], rtol=0, atol=0)
        torch.testing.assert_close(sample["labels"][:logical], ids[0, 1:], rtol=0, atol=0)
        for item, path, grid, pixels in zip(
            media,
            pages,
            reference["image_grid_thw"],
            reference["pixel_values"].split(reference["image_grid_thw"].prod(-1).tolist()),
        ):
            assert item["path"] == str(path)
            torch.testing.assert_close(item["grid"][0], grid, rtol=0, atol=0)
            reference_pixels[str(path)] = pixels
        torch.testing.assert_close(
            media[0]["load_pixels"](media), reference["pixel_values"], rtol=0, atol=0
        )
        assert int((sample["tokens"] == dataset.image_token_id).sum()) == sum(
            m["length"] for m in media
        )
        positions, _ = qwen35_native.Qwen3_5MoeModel.get_rope_index(
            dataset.position_helper,
            input_ids=ids,
            mm_token_type_ids=(ids == dataset.image_token_id).int(),
            image_grid_thw=reference["image_grid_thw"],
        )
        torch.testing.assert_close(
            sample["position_ids"][:, :logical], positions[:, 0, :-1], rtol=0, atol=0
        )

    # Each accepted sample is prepared once even when the source batch exceeds
    # the cache. Keep this assertion independent of the processor's own caches.
    uncached = dataset._sample.__wrapped__
    prepared = []

    def observe(index):
        prepared.append(index)
        return uncached(index)

    monkeypatch.setattr(dataset, "_sample", observe)
    dataset.indices["train"] = [0, 1]
    with monkeypatch.context() as patch:
        forbid_pixels(patch)
        samples, media = dataset.build_global_batch(0, 9)
    assert prepared == [0, 1, 0, 1, 0, 1, 0, 1, 0]
    assert [samples[i]["media_ids"] for i in range(3)] == [[0, 1], [2], [3, 4]]
    assert [item["image_id"] for item in media] == list(range(len(media)))
    assert [item["path"] for item in media[:5]] == [str(paths[i]) for i in (0, 1, 2, 0, 1)]
    # The same metadata-only source serves both encoder schedules. Opening and
    # normalizing pixels is confined to the producer's requested image chunk.
    source = dict(samples=samples, media=media)
    by_id = {item["image_id"]: item for item in media}
    original_open = qwen35_native.Image.open
    for per_round in (False, True):
        with monkeypatch.context() as patch:
            forbid_pixels(patch)
            _, _, inputs = qwen35_native.prepare_source(
                source, producer_ranks=range(2), rank=2, device="cpu", per_round=per_round
            )
            assert (inputs([]) if per_round else inputs) == {}
        for rank in range(2):
            opened = []

            def observe_open(path, *args, **kwargs):
                opened.append(str(path))
                return original_open(path, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(qwen35_native.Image, "open", observe_open)
                _, slices, inputs = qwen35_native.prepare_source(
                    source, producer_ranks=range(2), rank=rank, device="cpu", per_round=per_round
                )
                owned = sorted(
                    (key for key, value in slices.items() if value.producer_rank == rank),
                    key=lambda key: slices[key].offset,
                )
                if per_round:
                    assert opened == []
                    # Deliberately change chunk order; no foreign page may be decoded.
                    requested = owned[::-1][:2]
                    vision = inputs(requested)["images"]["qwen35"]
                else:
                    requested = owned
                    vision = inputs["images"]["qwen35"]
            assert opened == [by_id[key]["path"] for key in requested]
            torch.testing.assert_close(
                vision["hidden_states"],
                torch.cat([reference_pixels[by_id[key]["path"]] for key in requested]).bfloat16(),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                vision["grid_thw"],
                torch.cat([by_id[key]["grid"] for key in requested]),
                rtol=0,
                atol=0,
            )
    dataset.max_length = 1
    with monkeypatch.context() as patch:
        forbid_pixels(patch)
        assert uncached(0) is None  # Expanded vision tokens participate in admission.
    records[0]["messages"][0]["content"][:2] = reversed(records[0]["messages"][0]["content"][:2])
    (tmp_path / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    with pytest.raises(ValueError, match="markers and manifest paths disagree"):
        qwen35_native.Qwen35Dataset(manifest, snapshot, heldout_images=1)


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
    decoder_samples, decoder_slices, decoder_inputs = qwen35_native.prepare_source(
        source, producer_ranks=range(4), rank=7, device="cpu", **source_kwargs
    )
    assert decoder_samples is samples
    assert decoder_slices == slices
    assert (decoder_inputs([]) if per_round else decoder_inputs) == {}
    assert samples["B"]["media_ids"] == [101, 9, 91]


@pytest.mark.parametrize("cache_size", [0, 8])
@pytest.mark.parametrize("prefetch", [False, True])
def test_qwen_source_cache_prefetch_and_resume(cache_size, prefetch):
    class Dataset(qwen35_native.Qwen35Dataset):
        def __init__(self, cache_size, *, block=False, fail=False):
            self.split = "train"
            self.indices = {"train": [0, 1, 2]}
            self._accepted_indices, self._source_cursors = {"train": []}, {"train": 0}
            self.calls, self.block, self.fail = [], block, fail
            self.started, self.release = Event(), Event()
            self._sample = lru_cache(maxsize=cache_size)(self._sample)

        def _sample(self, index):
            if index == 1:  # A complete conversation exceeding the context limit.
                return None
            tokens = torch.arange(8) + index * 10
            sample = dict(
                tokens=tokens,
                labels=tokens + 1,
                loss_mask=torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
                position_ids=torch.arange(24).reshape(3, 8) + index,
                original_seq_len=6,
                padded_seq_len=8,
            )
            media = [
                dict(
                    path=f"{index}/{page}.png",
                    size=(32, 32),
                    length=4,
                    grid=torch.tensor([[1, 4, 4]]),
                    source_id=str(index),
                )
                for page in range(index + 2)
            ]
            return sample, media

        def build_global_batch(self, step, size):
            self.calls.append(step)
            if step == 1 and self.block:
                self.started.set()
                if not self.release.wait(10):
                    raise TimeoutError("Test did not release source producer")
            if step == 1 and self.fail:
                raise ValueError("source failure")
            return super().build_global_batch(step, size)

    def values(item):
        if isinstance(item, torch.Tensor):
            assert item.device.type == "cpu"
            return item.tolist()
        if isinstance(item, dict):
            return {key: values(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [values(value) for value in item]
        return item

    dataset = Dataset(cache_size, block=prefetch)
    source = SourceBatchIterator(dataset, 3, prefetch=prefetch)
    baseline = SourceBatchIterator(Dataset(0), 3)
    resumed = None
    try:
        assert dataset.calls == []  # No work until the first consumer request.
        assert values(next(source)) == values(next(baseline))
        assert source.step == 1
        if prefetch:
            assert dataset.started.wait(10)
            assert dataset.calls == [0, 1]  # Only the single next batch is in flight.
            dataset.release.set()
            source._future.result(timeout=10)
            assert dataset.calls == [0, 1] and source.step == 1
        source.close()  # Discard an already prepared but unconsumed batch.
        source.close()
        with pytest.raises(RuntimeError, match="closed"):
            next(source)
        resumed = SourceBatchIterator(Dataset(cache_size), 3, consumed_samples=3, prefetch=prefetch)
        for _ in range(4):  # Multiple wraps of the filtered, two-record stream.
            assert values(next(resumed)) == values(next(baseline))
        assert resumed.step == baseline.step == 5
        assert bool(resumed.dataset._sample.cache_info().hits) == bool(cache_size)
    finally:
        dataset.release.set()
        source.close()
        baseline.close()
        if resumed is not None:
            resumed.close()

    broken = SourceBatchIterator(Dataset(cache_size, fail=True), 3, prefetch=prefetch)
    next(broken)
    with pytest.raises(ValueError, match="source failure"):
        next(broken)
    assert broken.step == 1
    broken.close()
    with pytest.raises(RuntimeError, match="closed"):
        next(broken)


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
