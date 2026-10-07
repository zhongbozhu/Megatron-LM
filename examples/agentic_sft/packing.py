# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU offline packing for the agentic SFT example, without a Bridge runtime.

The segment-tree first-fit algorithm is adapted from NVIDIA Megatron Bridge,
``src/megatron/bridge/data/packing/algorithms.py`` (Apache-2.0). Unlike the
Bridge histogram/fill pipeline, this implementation packs sample IDs using
their aligned runtime lengths and preserves unpadded token streams on disk.
"""

import hashlib
import json
import multiprocessing
import os
import random
import tempfile
from pathlib import Path

from megatron.training.datasets.jsonl_rows import JsonlRows
from megatron.training.datasets.packed_sft_dataset import (
    PACKED_SFT_CONTRACT,
    PACKED_SFT_METADATA_KEY,
)

_WORKER_ROWS = None
_WORKER_TOKENIZER = None
_WORKER_SEQUENCE_LENGTH = None
_WORKER_LOSS_MODE = None
_WORKER_TOKENIZER_CONFIG = None


class _SegmentTree:
    """Index remaining bin capacities for O(log N) first-fit placement."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = [0] * (4 * capacity)

    def update(self, index, value, node=1, start=0, end=None):
        end = self.capacity - 1 if end is None else end
        if start == end:
            self.tree[node] = value
            return
        middle = (start + end) // 2
        if index <= middle:
            self.update(index, value, 2 * node, start, middle)
        else:
            self.update(index, value, 2 * node + 1, middle + 1, end)
        self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])

    def first_fit(self, need, node=1, start=0, end=None):
        end = self.capacity - 1 if end is None else end
        if self.tree[node] < need:
            return -1
        if start == end:
            return start
        middle = (start + end) // 2
        left = self.first_fit(need, 2 * node, start, middle)
        if left != -1:
            return left
        return self.first_fit(need, 2 * node + 1, middle + 1, end)


def first_fit_decreasing(lengths, pack_size):
    """Return stable bins of sample IDs, preserving every positive length once."""
    if not lengths:
        raise ValueError("Cannot pack an empty dataset")
    if pack_size < 1 or any(length < 1 or length > pack_size for length in lengths):
        raise ValueError("Every aligned sample length must be between one and the pack size")
    tree = _SegmentTree(len(lengths))
    bins, remaining = [], []
    for sample_id in sorted(range(len(lengths)), key=lambda index: (-lengths[index], index)):
        length = lengths[sample_id]
        bin_id = tree.first_fit(length)
        if bin_id < 0:
            bin_id = len(bins)
            bins.append([])
            remaining.append(pack_size)
        bins[bin_id].append(sample_id)
        remaining[bin_id] -= length
        tree.update(bin_id, remaining[bin_id])
    return bins


def shuffle_packed_rows(assignments, seed):
    """Shuffle row order reproducibly, retaining each bin's constituent order.

    The single-pass training loader reads rows sequentially. Leaving FFD's
    longest-first ordering intact would hide multi-sample packs until late in
    the epoch. Use an isolated RNG without modifying the caller's assignments.
    """
    shuffled = list(assignments)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def make_packed_row(samples):
    """Combine unshifted samples, converting token-aligned targets exactly once."""
    ids, mask, starts = [], [], []
    for sample in samples:
        tokens, targets = sample["input_ids"], sample["targets"]
        if len(tokens) < 2 or len(tokens) != len(targets):
            raise ValueError("Packing requires equally sized ids/targets and at least two tokens")
        starts.append(len(ids))
        ids.extend(tokens)
        # Bridge artifacts align loss_mask to labels at packing time. The final
        # stored position is never a runtime token and always has zero loss.
        mask.extend([int(target != -100) for target in targets[1:]])
        mask.append(0)
    return {"input_ids": ids, "loss_mask": mask, "seq_start_id": starts}


def _initialize_worker(rows, tokenizer_path, sequence_length, loss_mode):
    global _WORKER_ROWS, _WORKER_TOKENIZER, _WORKER_SEQUENCE_LENGTH, _WORKER_LOSS_MODE
    global _WORKER_TOKENIZER_CONFIG
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    _WORKER_ROWS = rows
    _WORKER_TOKENIZER = None
    _WORKER_TOKENIZER_CONFIG = dict(
        tokenizer_path=str(tokenizer_path), prompt_format="default", loss_mode=loss_mode
    )
    _WORKER_SEQUENCE_LENGTH = sequence_length
    _WORKER_LOSS_MODE = loss_mode


def _tokenize_row(index):
    from megatron.core.tokenizers.text.libraries.chat_sft import truncate_chat
    from megatron.core.tokenizers.text.libraries.sft_tokenizer import SFTTokenizer

    global _WORKER_TOKENIZER
    try:
        # Initialize inside a task so errors reach the parent, instead of Pool
        # repeatedly respawning failed initializer processes.
        if _WORKER_TOKENIZER is None:
            _WORKER_TOKENIZER = SFTTokenizer(**_WORKER_TOKENIZER_CONFIG)
        row = _WORKER_ROWS[index]
        tokens, targets = _WORKER_TOKENIZER.tokenize_conversation(
            row["messages"],
            return_target=True,
            add_generation_prompt=False,
            tools=row.get("tools"),
            chat_template_kwargs=row.get("chat_template_kwargs"),
        )
        original_length = len(tokens)
        tokens, targets = truncate_chat(tokens, targets, _WORKER_SEQUENCE_LENGTH, _WORKER_LOSS_MODE)
        return {
            "input_ids": tokens.tolist(),
            "targets": targets.tolist(),
            "truncated": original_length > len(tokens),
        }
    except Exception as exc:
        raise ValueError(
            f"SFT tokenization failed at row {index} in {_WORKER_ROWS.path}: {exc}"
        ) from exc


def sha256_file(path):
    """Hash a local artifact without reading it all into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pack_split(
    input_path,
    output_path,
    *,
    tokenizer_path,
    sequence_length,
    padding_divisor,
    num_workers,
    seed=1234,
    loss_mode="assistant",
    max_samples=None,
):
    """Tokenize in CPU processes, then bin-pack/write one bounded row-group at a time.

    Only the scalar length index stays in memory across the whole split.
    Tokenized records are spooled to an indexed temporary JSONL, avoiding a
    multi-gigabyte list of token arrays in the parent process.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if num_workers < 1 or sequence_length < 1 or padding_divisor < 1:
        raise ValueError("Workers, sequence length and padding divisor must be positive")
    if sequence_length % padding_divisor:
        raise ValueError("Sequence length must be divisible by the topology's padding divisor")
    rows = JsonlRows(input_path)
    count = len(rows) if max_samples is None else min(len(rows), max_samples)
    if count < 1:
        raise ValueError("No source samples selected for packing")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("input_ids", pa.list_(pa.int64())),
            ("loss_mask", pa.list_(pa.int8())),
            ("seq_start_id", pa.list_(pa.int64())),
        ],
        metadata={PACKED_SFT_METADATA_KEY: json.dumps(PACKED_SFT_CONTRACT).encode()},
    )
    length_bounds = sorted({8192, 16384, 32768, 65536, sequence_length})
    stats = {
        "source": str(Path(input_path).resolve()),
        "source_sha256": sha256_file(input_path),
        "source_rows": len(rows),
        "logical_samples": count,
        "runtime_tokens": 0,
        "supervised_tokens": 0,
        "aligned_tokens": 0,
        "truncated_samples": 0,
        "runtime_length_counts": {f"le_{bound}": 0 for bound in length_bounds},
    }
    worker_args = (rows, tokenizer_path, sequence_length, loss_mode)
    with tempfile.TemporaryDirectory(prefix=".pack_", dir=output_path.parent) as temporary:
        cache_path = Path(temporary) / "tokenized.jsonl"
        lengths = []

        def spool(results):
            with cache_path.open("w") as cache:
                for sample in results:
                    real_length = len(sample["input_ids"]) - 1
                    aligned_length = real_length + (-real_length % padding_divisor)
                    lengths.append(aligned_length)
                    stats["runtime_tokens"] += real_length
                    stats["aligned_tokens"] += aligned_length
                    stats["supervised_tokens"] += sum(t != -100 for t in sample["targets"][1:])
                    stats["truncated_samples"] += int(sample.pop("truncated"))
                    for bound in length_bounds:
                        stats["runtime_length_counts"][f"le_{bound}"] += int(real_length <= bound)
                    cache.write(json.dumps(sample, separators=(",", ":")) + "\n")

        if num_workers == 1:
            _initialize_worker(*worker_args)
            spool(map(_tokenize_row, range(count)))
        else:
            # Spawn keeps tokenizer/native-library state out of forked children.
            context = multiprocessing.get_context("spawn")
            pool = context.Pool(
                min(num_workers, count), initializer=_initialize_worker, initargs=worker_args
            )
            try:

                def bounded_results():
                    # Bound outstanding results even when an early long sample
                    # is slower than later ones. imap alone feeds the entire
                    # input eagerly and can retain many completed token arrays.
                    window = max(32, num_workers * 4)
                    for start in range(0, count, window):
                        yield from pool.imap(
                            _tokenize_row, range(start, min(start + window, count)), chunksize=4
                        )

                spool(bounded_results())
            except BaseException:
                pool.terminate()
                raise
            else:
                pool.close()
            finally:
                pool.join()

        bins = shuffle_packed_rows(first_fit_decreasing(lengths, sequence_length), seed)
        cached_rows = JsonlRows(cache_path)
        packed_path = Path(temporary) / "packed.parquet"
        with pq.ParquetWriter(packed_path, schema, compression="zstd") as writer:
            pending = []
            for assignment in bins:
                pending.append(make_packed_row(cached_rows[index] for index in assignment))
                if len(pending) == 128:
                    writer.write_table(pa.Table.from_pylist(pending, schema=schema))
                    pending.clear()
            if pending:
                writer.write_table(pa.Table.from_pylist(pending, schema=schema))
        os.replace(packed_path, output_path)

    stats.update(
        packed_rows=len(bins),
        multi_sample_rows=sum(len(assignment) > 1 for assignment in bins),
        max_samples_per_row=max(map(len, bins)),
        packing_efficiency=stats["aligned_tokens"] / (len(bins) * sequence_length),
        output=str(output_path.resolve()),
        output_sha256=sha256_file(output_path),
    )
    return stats
