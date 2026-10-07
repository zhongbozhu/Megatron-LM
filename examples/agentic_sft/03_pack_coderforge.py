#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Offline-pack a materialized CoderForge dataset on CPUs."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.agentic_sft.packing import pack_split, sha256_file
from megatron.training.datasets.packed_sft_dataset import (
    PACKED_SFT_CONTRACT,
    get_sft_padding_divisor,
)

WORKSPACE = ROOT.parent / "debug-codex-workspace" / "agentic_sft"


def source_fingerprints():
    """Record local implementation changes that a git revision alone misses."""
    paths = (
        "examples/agentic_sft/03_pack_coderforge.py",
        "examples/agentic_sft/packing.py",
        "megatron/core/tokenizers/text/libraries/chat_sft.py",
        "megatron/core/tokenizers/text/libraries/sft_tokenizer.py",
        "megatron/training/datasets/jsonl_rows.py",
        "megatron/training/datasets/packed_sft_dataset.py",
    )
    return {path: sha256_file(ROOT / path) for path in paths}


def template_fingerprints(tokenizer_path):
    """Fingerprint the training templates loaded from the prepared tokenizer."""
    import hashlib

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    templates = tokenizer.chat_template
    if isinstance(templates, str):
        templates = {"default": templates}
    if not isinstance(templates, dict) or not templates:
        raise ValueError("The pinned tokenizer must provide a chat template")
    return {
        name: {"sha256": hashlib.sha256(template.encode()).hexdigest()}
        for name, template in templates.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Prepared training/validation JSONL directory"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--tokenizer", type=Path, default=WORKSPACE / "models/qwen35_35B_1node/tokenizer"
    )
    parser.add_argument("--seq-length", type=int, default=65536)
    parser.add_argument("--num-tokenizer-workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234, help="Seed for shuffling packed rows")
    parser.add_argument("--context-parallel-size", type=int, default=4)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--sequence-parallel-size", type=int, default=1)
    parser.add_argument(
        "--static-cp", action="store_true", help="Budget for fixed CP instead of DCP"
    )
    parser.add_argument("--loss-mode", choices=("assistant", "full"), default="assistant")
    parser.add_argument(
        "--max-samples", type=int, help="Optional per-split cap for a preparation smoke test"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.tokenizer.is_dir():
        parser.error("The pinned local tokenizer is missing; run script 01 first")
    if any(
        value < 1
        for value in (
            args.context_parallel_size,
            args.data_parallel_size,
            args.sequence_parallel_size,
            args.num_tokenizer_workers,
            args.seq_length,
        )
    ):
        parser.error("Parallel sizes, worker count, and sequence length must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    # Include the runtime context length so a new run cannot silently consume
    # the differently truncated samples from an earlier packing configuration.
    output_dir = args.output_dir or args.data_dir / f"packed_{args.seq_length}"
    if output_dir.resolve() == args.data_dir.resolve():
        parser.error("Choose a separate output directory to preserve the preparation manifest")
    paths = [
        (args.data_dir / f"{split}.jsonl", output_dir / f"{split}.parquet")
        for split in ("training", "validation")
    ]
    for source, output in paths:
        if not source.is_file():
            parser.error(f"Missing input {source}; run script 02 first")
        if output.exists() and not args.overwrite:
            parser.error(f"Output exists: {output}; use --overwrite to rebuild")
    divisor = get_sft_padding_divisor(
        context_parallel_size=args.context_parallel_size,
        data_parallel_size=args.data_parallel_size,
        sequence_parallel_size=args.sequence_parallel_size,
        dynamic_context_parallel=not args.static_cp,
    )
    if args.seq_length % divisor:
        parser.error(f"Sequence length must be divisible by the padding divisor ({divisor})")
    manifest = {
        "format": PACKED_SFT_CONTRACT,
        "sequence_length": args.seq_length,
        "padding_divisor": divisor,
        "tokenizer": str(args.tokenizer.resolve()),
        "tokenizer_files_sha256": {
            str(path.relative_to(args.tokenizer)): sha256_file(path)
            for path in sorted(args.tokenizer.rglob("*"))
            if path.is_file() and path.suffix in (".json", ".jinja", ".txt", ".model")
        },
        "loss_mode": args.loss_mode,
        "chat_templates": template_fingerprints(args.tokenizer),
        "code_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_files_sha256": source_fingerprints(),
        "num_tokenizer_workers": args.num_tokenizer_workers,
        "algorithm": "first_fit_decreasing",
        "row_order": "shuffled",
        "seed": args.seed,
        "splits": {},
    }
    preparation_manifest = args.data_dir / "manifest.json"
    if preparation_manifest.is_file():
        manifest["preparation_manifest"] = str(preparation_manifest.resolve())
        manifest["preparation_manifest_sha256"] = sha256_file(preparation_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    # An interrupted rebuild must not leave a completion manifest for mixed versions.
    if args.overwrite:
        (output_dir / "manifest.json").unlink(missing_ok=True)
    for source, output in paths:
        print(
            f"Packing {source.name}: workers={args.num_tokenizer_workers}, alignment={divisor}",
            flush=True,
        )
        stats = pack_split(
            source,
            output,
            tokenizer_path=args.tokenizer,
            sequence_length=args.seq_length,
            padding_divisor=divisor,
            num_workers=args.num_tokenizer_workers,
            seed=args.seed,
            loss_mode=args.loss_mode,
            max_samples=args.max_samples,
        )
        manifest["splits"][source.stem] = stats
        print(json.dumps(stats, indent=2), flush=True)
    if source_fingerprints() != manifest["source_files_sha256"]:
        raise RuntimeError(
            "Packing source changed while workers ran; rebuild for consistent provenance"
        )
    if "preparation_manifest_sha256" in manifest and (
        sha256_file(preparation_manifest) != manifest["preparation_manifest_sha256"]
    ):
        raise RuntimeError(
            "Preparation manifest changed while packing; rebuild for consistent provenance"
        )
    temporary = output_dir / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, output_dir / "manifest.json")


if __name__ == "__main__":
    main()
