# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU synthetic length controls with unchanged frozen MIMO image inputs.

The repeated ordinary text is artificial: these fixtures measure scheduling and
performance, not language quality or convergence. Supply the resulting file to
``--mimo-diagnostic-source-batch``. No model or scheduler is instantiated here.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel

from examples.mimo.data.joint_cp_fixture import FrozenSourceDataset
from examples.mimo.data.qwen35_native import Qwen35Dataset
from examples.mimo.fixed_routing import source_fingerprint
from examples.mimo.resume_diagnostics import _describe

_TEXT = "Describe the colored shapes and compare their positions in the image. "
_PREFIX = 32
_ALIGNMENT = 32


def _file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _assert_cpu(value):
    if isinstance(value, torch.Tensor):
        assert value.device.type == "cpu"
    elif isinstance(value, dict):
        for item in value.values():
            _assert_cpu(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_cpu(item)


def _image_starts(tokens, image_token_id):
    is_image = tokens == image_token_id
    previous = torch.cat((is_image.new_zeros(1), is_image[:-1]))
    return (is_image & ~previous).nonzero().flatten().tolist()


def _sample_summary(sample, image_token_id):
    length = int(sample["original_seq_len"])
    tokens = sample["tokens"][:length]
    return dict(
        real_tokens=length,
        padded_tokens=int(sample["padded_seq_len"]),
        supervised_tokens=int(sample["loss_mask"].sum()),
        image_tokens=int((tokens == image_token_id).sum()),
        image_token_starts=_image_starts(tokens, image_token_id),
        media_ids=sample["media_ids"],
    )


def _read_lengths(path, sample_ids):
    lengths = json.loads(Path(path).read_text())
    if isinstance(lengths, dict):
        if set(lengths) != {str(sid) for sid in sample_ids}:
            raise ValueError("Length mapping must contain exactly the source sample IDs")
        lengths = [lengths[str(sid)] for sid in sample_ids]
    if not isinstance(lengths, list) or len(lengths) != len(sample_ids):
        raise ValueError("Lengths must be a list or sample-ID mapping matching the source batch")
    if any(type(length) is not int or length <= 0 for length in lengths):
        raise ValueError("Target lengths must be positive integers, excluding padding")
    return dict(zip(sample_ids, lengths))


def create_fixture(source, hf_model, lengths_json, output):
    """Render a length-controlled batch, preserving every source image byte."""
    output = Path(output)
    metadata_path = output.with_suffix(".json")
    if output == metadata_path or output.exists() or metadata_path.exists():
        raise FileExistsError("Use a new non-JSON output path and an unused JSON sidecar")
    if torch.cuda.is_initialized():
        raise RuntimeError("Fixture preparation must run without initializing CUDA")
    torch.set_num_threads(1)
    source_dataset = FrozenSourceDataset(source)
    original, media = source_dataset.build_global_batch(
        step=0, global_batch_size=len(source_dataset.payload["samples"])
    )
    _assert_cpu((original, media))
    lengths = _read_lengths(lengths_json, sorted(original))
    config = AutoConfig.from_pretrained(hf_model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_model, local_files_only=True)
    image_token_id = config.image_token_id
    vision_start = config.vision_start_token_id
    vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None or tokenizer.convert_ids_to_tokens(vision_end) != "<|vision_end|>":
        raise ValueError("Tokenizer is missing the required Qwen image or padding tokens")
    ordinary = tokenizer.encode(_TEXT, add_special_tokens=False)
    if not ordinary or set(ordinary) & set(tokenizer.all_special_ids):
        raise ValueError("Repeated text must consist entirely of ordinary token IDs")
    helper = SimpleNamespace(config=config)
    helper.get_vision_position_ids = MethodType(Qwen3_5MoeModel.get_vision_position_ids, helper)
    padding = SimpleNamespace(alignment=_ALIGNMENT, pad_token_id=pad_token_id)
    by_id = {image["image_id"]: image for image in media}
    if len(by_id) != len(media):
        raise ValueError("Frozen source contains duplicate image IDs")
    image_description = _describe(media)
    source_description = _describe(original)
    samples, records = {}, {}
    for sid, real in lengths.items():
        prior = original[sid]
        images = [by_id[mid] for mid in prior["media_ids"]]
        expected_rows = sum(int(image["length"]) for image in images)
        assert _sample_summary(prior, image_token_id)["image_tokens"] == expected_rows
        ids = torch.tensor((ordinary * ((real + 1) // len(ordinary) + 1))[: real + 1])
        supervised = torch.ones(real + 1, dtype=torch.float32)
        supervised[:_PREFIX] = 0
        cursor, spans = _PREFIX, []
        for image in images:
            grid, rows = image["grid"], int(image["length"])
            assert grid.shape == (1, 3) and int(grid[0, 0]) == 1
            assert int(grid.prod()) // config.vision_config.spatial_merge_size**2 == rows
            end = cursor + rows + 1
            if end >= real - 1:
                raise ValueError(f"Sample {sid} length {real} cannot retain its complete images")
            ids[cursor] = vision_start
            ids[cursor + 1 : end] = image_token_id
            ids[end] = vision_end
            supervised[cursor : end + 1] = 0
            spans.append(dict(image_id=image["image_id"], start=cursor + 1, rows=rows))
            cursor = end + 17
        grids = torch.cat([image["grid"] for image in images]) if images else None
        positions, _ = Qwen3_5MoeModel.get_rope_index(
            helper,
            input_ids=ids[None],
            mm_token_type_ids=(ids == image_token_id).int()[None],
            image_grid_thw=grids,
        )
        sample = Qwen35Dataset._pad_sample(padding, ids, supervised, positions[:, 0])
        for field in ("media_ids", "source_ids", "source_id"):
            if field in prior:
                sample[field] = prior[field]
        physical = sample["padded_seq_len"]
        assert sample["original_seq_len"] == real
        assert physical == (real + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
        assert all(
            sample[field].shape == (physical,) for field in ("tokens", "labels", "loss_mask")
        )
        assert sample["position_ids"].shape == (3, physical)
        assert torch.equal(sample["tokens"][:real], ids[:-1])
        assert torch.equal(sample["labels"][:real], ids[1:])
        assert torch.equal(sample["loss_mask"][:real], supervised[1:])
        assert torch.equal(sample["position_ids"][:, :real], positions[:, 0, :-1])
        assert not sample["loss_mask"][real:].any()
        assert not sample["position_ids"][:, real:].any()
        assert (sample["tokens"][real:] == pad_token_id).all()
        assert (sample["labels"][real:] == pad_token_id).all()
        assert sample["loss_mask"].sum() > 0
        assert int((sample["tokens"] == image_token_id).sum()) == expected_rows
        assert int(ids.min()) >= 0 and int(ids.max()) < config.text_config.vocab_size
        for span in spans:
            start, rows = span["start"], span["rows"]
            assert sample["tokens"][start - 1] == vision_start
            assert (sample["tokens"][start : start + rows] == image_token_id).all()
            assert sample["tokens"][start + rows] == vision_end
            # Loss masks refer to labels, so image targets are shifted one row left.
            assert not sample["loss_mask"][start - 2 : start + rows].any()
        samples[sid] = sample
        records[str(sid)] = dict(
            source=_sample_summary(prior, image_token_id),
            synthetic=_sample_summary(sample, image_token_id),
            image_spans=spans,
        )
    assert _describe(media) == image_description
    assert _describe(original) == source_description
    _assert_cpu((samples, media))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(dict(version=1, samples=samples, media=media), stream)
    restored = FrozenSourceDataset(output)
    repeated_samples, repeated_media = restored.build_global_batch(52, len(samples))
    assert _describe(repeated_samples) == _describe(samples)
    assert _describe(repeated_media) == image_description
    assert source_fingerprint(repeated_samples, repeated_media) == source_fingerprint(
        samples, media
    )
    assert not torch.cuda.is_initialized()
    metadata = dict(
        version=1,
        artificial_text=True,
        intended_use="Scheduling and performance only; no quality or convergence claims",
        source=str(Path(source).resolve()),
        source_file_sha256=_file_digest(source),
        source_fingerprint=source_fingerprint(original, media),
        output_file_sha256=_file_digest(output),
        output_fingerprint=source_fingerprint(samples, media),
        generator_sha256=_file_digest(__file__),
        lengths_file_sha256=_file_digest(lengths_json),
        hf_model=str(Path(hf_model).resolve()),
        repeated_text=_TEXT,
        repeated_token_ids=ordinary,
        supervision_policy="Target positions >=32 except complete vision delimiter/image spans",
        unsupervised_prefix_tokens=_PREFIX,
        padding_alignment=_ALIGNMENT,
        image_token_id=image_token_id,
        source_samples=len(samples),
        source_images=len(media),
        image_contents_and_identities_unchanged=True,
        sample_media_mapping_unchanged=True,
        frozen_dataset_roundtrip_exact=True,
        cpu_only=True,
        total_real_tokens=sum(sample["original_seq_len"] for sample in samples.values()),
        total_padded_tokens=sum(sample["padded_seq_len"] for sample in samples.values()),
        total_supervised_tokens=sum(int(sample["loss_mask"].sum()) for sample in samples.values()),
        samples=records,
        media=image_description,
    )
    with metadata_path.open("x") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(output), metadata=str(metadata_path), samples=len(samples))))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--lengths-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    create_fixture(args.source, args.hf_model, args.lengths_json, args.output)
