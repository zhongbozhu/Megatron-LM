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

from examples.mimo.data.source_batch import SourceBatchIterator
from megatron.core.models.mimo.planning import MediaTask, assign_media_to_producers


class Qwen35Dataset:
    """Build source global batches independently of the chosen decoder layout.

    ``step`` addresses the global sample stream directly, so resuming from a
    Megatron iteration yields the same samples without serializing an iterator.
    Image identity determines the heldout split. Media IDs are batch-local and
    distinct even when a source image is reused in a later optimizer step.
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
        identities = sorted({item["images"][0]["sha256"] for item in manifest["samples"]})
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
            if len(metadata["image_paths"]) != 1:
                raise ValueError("The CLEVR adapter expects one image per source conversation")
            record = records[metadata["record_index"]]
            identity = metadata["images"][0]["sha256"]
            image_path = self.manifest_path.parent / metadata["image_paths"][0]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            messages = []
            for message in record["messages"]:
                content = message["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                normalized = []
                for item in content:
                    if isinstance(item, str):
                        normalized.append({"type": "text", "text": item})
                    elif item.get("type") == "image":
                        normalized.append({"type": "image", "image": str(image_path)})
                    elif item.get("type") == "text":
                        normalized.append({"type": "text", "text": item["text"]})
                    else:
                        raise ValueError(f"Unsupported CLEVR content: {item}")
                messages.append({"role": message["role"], "content": normalized})
            index = len(self.examples)
            self.examples.append(
                {"messages": messages, "path": str(image_path), "source_id": record["id"]}
            )
            self.indices["heldout" if identity in heldout else "train"].append(index)
        for split in self.indices:
            order_seed = split_seed + 1 if split == "heldout" else seed
            random.Random(order_seed).shuffle(self.indices[split])
        self._accepted_indices = {split: [] for split in self.indices}
        self._source_cursors = {split: 0 for split in self.indices}

    @lru_cache(maxsize=512)
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
            raise ValueError("CLEVR conversations must contain complete user/assistant pairs")
        rendered = "".join(
            self.processor.apply_chat_template(
                messages[index : index + 2], tokenize=False, add_generation_prompt=False
            )
            for index in range(0, len(messages), 2)
        )
        with Image.open(example["path"]) as image:
            processed = self.processor(
                text=[rendered], images=[image.convert("RGB")], return_tensors="pt"
            )
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
        temporal, height, width = grid[0].tolist()
        if temporal != 1 or len(grid) != 1:
            raise ValueError("This CLEVR dataset contains still images only")
        patch_size = self.config.vision_config.patch_size
        media = {
            "path": example["path"],
            "size": (height * patch_size, width * patch_size),
            "length": image_rows,
            "grid": grid,
            "pixel_values": processed["pixel_values"],
            "source_id": example["source_id"],
        }
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
        # Address the accepted stream, not the raw stream: excluding an oversize
        # record must not cause adjacent optimizer steps to overlap accidentally.
        while len(accepted) < (step + 1) * global_batch_size:
            source_index = indices[self._source_cursors[split] % len(indices)]
            self._source_cursors[split] += 1
            if self._sample(source_index) is None:
                rejected += 1
                if rejected >= len(indices):
                    raise ValueError("No complete source conversations fit the context limit")
                continue
            accepted.append(source_index)
            rejected = 0
        begin = step * global_batch_size
        for source_index in accepted[begin : begin + global_batch_size]:
            sample, image = self._sample(source_index)
            sample_id = len(samples)
            samples[sample_id] = {**sample, "media_ids": [sample_id]}
            media.append({**image, "image_id": sample_id})
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


def prepare_disagg_source(source, *, args, topology):
    """Keep image assignment independent of the disjoint decoder rank domain."""
    producers = sorted(
        rank for line in topology.grids["images"].get_rank_enum("pp") for rank in line
    )
    return prepare_source(
        source,
        producer_ranks=producers,
        rank=torch.distributed.get_rank(),
        device=torch.device("cuda", torch.cuda.current_device()),
        per_round=args.mimo_dcp_encoder_schedule == "per-round",
    )


def train_valid_test_datasets_provider(*_args, **_kwargs):
    """Resume native train/heldout source streams using their own global batch sizes."""
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
