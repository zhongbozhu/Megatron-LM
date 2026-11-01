# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deterministic real image conversations for the native colocated MIMO entry point.

The HF processor owns chat formatting, image normalization, patch ordering and
placeholder expansion. Its Qwen position helper supplies the three RoPE axes;
no pretrained model is instantiated during preprocessing. Complete conversations
and reasoning are retained. Oversize conversations are excluded, never truncated.
"""

import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel


class Qwen35Dataset:
    """Build source global batches independently of the chosen decoder layout.

    ``step`` addresses the global sample stream directly, so resuming from a
    Megatron iteration yields the same samples without serializing an iterator.
    Document identity (or the first image hash) determines the heldout split.
    Media IDs are batch-local and distinct when an image is reused in a later step.
    """

    def __init__(
        self,
        manifest_path,
        hf_model_path,
        seq_length=131072,
        seed=1234,
        split="train",
        heldout_images=64,
        alignment=32,
        split_seed=None,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        max_length = seq_length
        self.max_length = max_length
        self.seed = seed
        self.split = "heldout" if split in ("valid", "validation") else split
        if self.split not in ("train", "heldout"):
            raise ValueError("split must be train or valid")
        self.alignment = alignment
        if max_length <= 0 or alignment <= 0 or max_length % alignment:
            raise ValueError("Maximum sequence length must be a positive multiple of alignment")
        manifest = json.loads(self.manifest_path.read_text())
        records_path = self.manifest_path.parent / manifest["records_file"]
        records = [json.loads(line) for line in records_path.read_text().splitlines()]
        self.processor = AutoProcessor.from_pretrained(hf_model_path, local_files_only=True)
        self.config = AutoConfig.from_pretrained(hf_model_path, local_files_only=True)
        self.image_token_id = self.config.image_token_id
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.processor.tokenizer.eos_token_id
        self.assistant_header = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self.message_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.position_helper = SimpleNamespace(config=self.config)
        self.position_helper.get_vision_position_ids = MethodType(
            Qwen3_5MoeModel.get_vision_position_ids, self.position_helper
        )

        def split_identity(item):
            return item.get("split_group", item["images"][0]["sha256"])

        identities = sorted({split_identity(item) for item in manifest["samples"]})
        if not 0 < heldout_images < len(identities):
            raise ValueError("Training and heldout image sets must both be nonempty")
        split_seed = seed if split_seed is None else split_seed
        identities.sort(
            key=lambda value: hashlib.sha256(f"{split_seed}:{value}".encode()).hexdigest()
        )
        heldout = set(identities[:heldout_images])
        self.examples = []
        self.indices = {"train": [], "heldout": []}
        for metadata in manifest["samples"]:
            if not metadata["image_paths"] or len(metadata["image_paths"]) != len(
                metadata["images"]
            ):
                raise ValueError(
                    "Each source conversation needs matching image paths and metadata"
                )
            record = records[metadata["record_index"]]
            identity = split_identity(metadata)
            image_paths = [self.manifest_path.parent / path for path in metadata["image_paths"]]
            for image_path in image_paths:
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
            messages = []
            image_index = 0
            for message in record["messages"]:
                content = message["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                normalized = []
                for item in content:
                    if isinstance(item, str):
                        normalized.append({"type": "text", "text": item})
                    elif item.get("type") == "image":
                        if image_index >= len(image_paths):
                            raise ValueError("More image markers than source image paths")
                        original_path = metadata["images"][image_index].get("path")
                        if (
                            len(image_paths) > 1
                            and original_path is not None
                            and item.get("image") != original_path
                        ):
                            raise ValueError("Source image markers and manifest paths disagree")
                        normalized.append(
                            {"type": "image", "image": str(image_paths[image_index])}
                        )
                        image_index += 1
                    elif item.get("type") == "text":
                        normalized.append({"type": "text", "text": item["text"]})
                    else:
                        raise ValueError(f"Unsupported image conversation content: {item}")
                messages.append({"role": message["role"], "content": normalized})
            if image_index != len(image_paths):
                raise ValueError("Source image paths must match image marker count")
            index = len(self.examples)
            self.examples.append(
                {"messages": messages, "paths": image_paths, "source_id": record["id"]}
            )
            self.indices["heldout" if identity in heldout else "train"].append(index)
        for split in self.indices:
            order_seed = split_seed + 1 if split == "heldout" else seed
            random.Random(order_seed).shuffle(self.indices[split])
        self._accepted_indices = {split: [] for split in self.indices}
        self._source_cursors = {split: 0 for split in self.indices}
        # Processed multi-page documents can occupy GBs each on CPU. This bounds
        # retained past documents; the current source batch still owns its inputs.
        cache_size = 8 if any(len(item["paths"]) > 1 for item in self.examples) else 512
        self._sample = lru_cache(maxsize=cache_size)(self._sample)

    def _sample(self, index):
        example = self.examples[index]
        # Qwen's inference template drops reasoning from historical assistant
        # turns. Render each complete QA pair with that same template, retaining
        # all real reasoning as training targets when joining the conversation.
        messages = example["messages"]
        if len(messages) % 2 or any(
            message["role"] != ("user" if index % 2 == 0 else "assistant")
            for index, message in enumerate(messages)
        ):
            raise ValueError("Image conversations must contain complete user/assistant pairs")
        rendered = "".join(
            self.processor.apply_chat_template(
                messages[index : index + 2], tokenize=False, add_generation_prompt=False
            )
            for index in range(0, len(messages), 2)
        )
        images = []
        try:
            for path in example["paths"]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            processed = self.processor(text=[rendered], images=images, return_tensors="pt")
        finally:
            for image in images:
                image.close()
        ids = processed["input_ids"][0]
        length = ids.numel() - 1
        if length > self.max_length:
            return None
        grid = processed["image_grid_thw"]
        image_rows = int(grid.prod(-1).sum()) // self.config.vision_config.spatial_merge_size**2
        if int((ids == self.image_token_id).sum()) != image_rows:
            raise ValueError("Image placeholders do not match the processor's merged feature grid")
        positions, _ = Qwen3_5MoeModel.get_rope_index(
            self.position_helper,
            input_ids=ids.unsqueeze(0),
            mm_token_type_ids=(ids == self.image_token_id).to(torch.int).unsqueeze(0),
            image_grid_thw=grid,
        )
        supervised = torch.zeros_like(ids, dtype=torch.float32)
        token_list = ids.tolist()
        header_length = len(self.assistant_header)
        cursor = 0
        while cursor < len(token_list):
            if token_list[cursor : cursor + header_length] != self.assistant_header:
                cursor += 1
                continue
            begin = cursor + header_length
            end = begin
            while end < len(token_list) and token_list[end] != self.message_end_id:
                end += 1
            if end == len(token_list):
                raise ValueError(
                    "A complete source conversation has an unterminated assistant turn"
                )
            supervised[begin : end + 1] = 1
            cursor = end + 1
        if not supervised[1:].any():
            raise ValueError("HF chat formatting produced no assistant supervision")
        sample = self._pad_sample(ids, supervised, positions[:, 0])
        if len(grid) != len(example["paths"]) or (grid[:, 0] != 1).any():
            raise ValueError("Each source image must produce one still-image grid")
        patch_size = self.config.vision_config.patch_size
        patch_counts = grid.prod(-1).tolist()
        pixels = processed["pixel_values"].split(patch_counts)
        media = []
        for index, (path, patches) in enumerate(zip(example["paths"], pixels)):
            _, height, width = grid[index].tolist()
            media.append(
                {
                    "path": str(path),
                    "size": (height * patch_size, width * patch_size),
                    "length": patch_counts[index]
                    // self.config.vision_config.spatial_merge_size**2,
                    "grid": grid[index : index + 1],
                    "pixel_values": patches,
                    "source_id": example["source_id"],
                }
            )
        return sample, media

    def _pad_sample(self, ids, supervised, positions):
        length = ids.numel() - 1
        padded = (length + self.alignment - 1) // self.alignment * self.alignment
        padding = padded - length
        return {
            "tokens": F.pad(ids[:-1], (0, padding), value=self.pad_token_id),
            "labels": F.pad(ids[1:], (0, padding), value=self.pad_token_id),
            "loss_mask": F.pad(supervised[1:], (0, padding)),
            "position_ids": F.pad(positions[:, :-1], (0, padding)),
            "original_seq_len": length,
            "padded_seq_len": padded,
        }

    def build_global_batch(self, step, global_batch_size, split=None):
        """Return ``(samples, media)`` using global-batch-local contiguous IDs."""
        split = self.split if split is None else split
        if step < 0 or global_batch_size <= 0 or split not in self.indices:
            raise ValueError("Invalid global batch address")
        indices = self.indices[split]
        samples, media = {}, []
        accepted = self._accepted_indices[split]
        rejected = 0
        begin = step * global_batch_size
        prepared = {}
        # Address the accepted stream, not the raw stream: excluding an oversize
        # record must not cause adjacent optimizer steps to overlap accidentally.
        while len(accepted) < (step + 1) * global_batch_size:
            source_index = indices[self._source_cursors[split] % len(indices)]
            self._source_cursors[split] += 1
            item = self._sample(source_index)
            if item is None:
                rejected += 1
                if rejected >= len(indices):
                    raise ValueError("No complete source conversations fit the context limit")
                continue
            accepted.append(source_index)
            if len(accepted) > begin:
                prepared[len(accepted) - 1] = item
            rejected = 0
        for position, source_index in enumerate(
            accepted[begin : begin + global_batch_size], begin
        ):
            sample, images = (
                prepared[position] if position in prepared else self._sample(source_index)
            )
            sample_id = len(samples)
            media_ids = list(range(len(media), len(media) + len(images)))
            samples[sample_id] = {**sample, "media_ids": media_ids}
            media.extend(
                {**image, "image_id": media_id} for image, media_id in zip(images, media_ids)
            )
        return samples, media


def build_vision_inputs(media, device):
    """Build the native Qwen vision encoder's THD patch stream."""
    if not media:
        return {}
    return {
        "images": {
            "qwen35": {
                "hidden_states": torch.cat([item["pixel_values"] for item in media]).to(
                    device=device, dtype=torch.bfloat16
                ),
                "grid_thw": torch.cat([item["grid"] for item in media]).to(device=device),
            }
        }
    }


def prepare_source(source, *, producer_ranks, rank, device, per_round=False):
    """Assign images for whole-step encoding, or materialize them lazily on demand.

    Lazy mode retains source patches on CPU and returns a callable that packs
    requested local media IDs in order. Both modes use the same producer
    assignment, projected-row offsets and image preprocessing.
    """
    from megatron.core.models.mimo.planning import MediaTask, assign_media_to_producers

    media = source["media"]
    tasks = []
    for item in media:
        height, width = item["size"]
        if min(height, width) <= 0:
            raise ValueError("Image dimensions must be positive")
        tasks.append(MediaTask(item["image_id"], item["length"], (height * width) ** 2))
    assignments, slices = assign_media_to_producers(tasks, producer_ranks)
    by_id = {item["image_id"]: item for item in media}
    local = {task.media_id: by_id[task.media_id] for task in assignments.get(rank, ())}

    def encoder_inputs(media_ids):
        if any(media_id not in local for media_id in media_ids):
            raise ValueError("Requested media must belong to this encoder producer")
        return build_vision_inputs([local[media_id] for media_id in media_ids], device)

    inputs = encoder_inputs if per_round else encoder_inputs(list(local))
    return source["samples"], slices, inputs


def train_valid_test_datasets_provider(*_args, **_kwargs):
    """Resume native train/heldout source streams using their own global batch sizes."""
    from examples.mimo.data.source_batch import SourceBatchIterator
    from megatron.training import get_args
    from megatron.training.global_vars import get_run_config, get_train_state

    args, config, state = get_args(), get_run_config(), get_train_state()
    if args.dataloader_type != "external":
        raise ValueError("Native MIMO source batches require --dataloader-type external")
    if config.validation.full_validation:
        raise ValueError("The infinite CLEVR source iterator requires --eval-iters")

    def iterator(split, global_batch_size, consumed):
        dataset = Qwen35Dataset(
            args.mimo_manifest,
            args.mimo_hf_model,
            args.seq_length,
            seed=config.rng.seed,
            split=split,
            heldout_images=args.mimo_heldout_images,
            split_seed=args.mimo_data_split_seed,
        )
        return SourceBatchIterator(dataset, global_batch_size, consumed)

    return (
        iterator("train", config.train.global_batch_size, state.consumed_train_samples),
        (
            iterator(
                "valid",
                config.validation.eval_global_batch_size,
                0 if config.validation.skip_train else state.consumed_valid_samples,
            )
            if config.validation.eval_iters
            else None
        ),
        None,
    )
