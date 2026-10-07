#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Prepare pinned tokenizer assets and the full 40-layer Qwen3.5 text config.

No model weights are downloaded or converted. The vocabulary stays unchanged;
a maintained training template supplies native HF assistant generation masks.
"""

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = REPO_ROOT.parent / "debug-codex-workspace" / "agentic_sft"
MODEL_NAME = "qwen35_35B_1node"
HF_REPO = "Qwen/Qwen3.5-35B-A3B"
HF_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
TRAINING_TEMPLATE = Path(__file__).resolve().parent / "templates/qwen3_5_sft_v1.jinja"
ASSET_PATTERNS = (
    "tokenizer* chat_template* vocab.json merges.txt "
    "special_tokens_map.json added_tokens.json generation_config.json"
).split()


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def proxy_config(config, num_layers):
    """Retain full 35B-A3B layer shapes and complete 3-GDN/1-attention cycles."""
    text = dict(config.get("text_config", config))
    expected = {
        "model_type": "qwen3_5_moe_text",
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "num_experts": 256,
        "moe_intermediate_size": 512,
    }
    if any(text.get(key) != value for key, value in expected.items()):
        raise ValueError("Expected the original Qwen3.5-35B-A3B config")
    if num_layers < 4 or num_layers > 40 or num_layers % 4:
        raise ValueError("--num-layers must be a multiple of 4 between 4 and 40")
    pattern = ["linear_attention"] * 3 + ["full_attention"]
    if text.get("layer_types", pattern * 10) != pattern * 10:
        raise ValueError("Expected a repeating 3-GDN/1-attention layer pattern")
    text["architectures"] = ["Qwen3_5MoeForCausalLM"]
    text["num_hidden_layers"] = num_layers
    text["layer_types"] = pattern * (num_layers // 4)
    for field in ("mtp_num_hidden_layers", "num_nextn_predict_layers", "mtp_num_layers"):
        text[field] = 1
    for field in ("bos_token_id", "eos_token_id", "torch_dtype", "dtype", "tie_word_embeddings"):
        if field in config:
            text[field] = config[field]
    # AutoTokenizer uses this metadata for version-dependent compatibility fixes.
    if "transformers_version" in config:
        text["transformers_version"] = config["transformers_version"]
    return text


def prepare_tokenizer(source, output, *, num_layers=40, revision=None):
    """Preserve vocabulary bytes and install one explicit versioned SFT template."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output must not overlap the source directory")
    config = proxy_config(json.loads((source / "config.json").read_text()), num_layers)
    assets = set()
    for pattern in ASSET_PATTERNS:
        for path in source.glob(pattern):
            assets.update(p for p in (path.rglob("*") if path.is_dir() else [path]) if p.is_file())
    if source / "tokenizer_config.json" not in assets or not any(
        source / name in assets for name in ("tokenizer.json", "tokenizer.model", "vocab.json")
    ):
        raise ValueError("Source must contain tokenizer_config.json and a tokenizer vocabulary")
    source_checksums = {str(path.relative_to(source)): sha256(path) for path in sorted(assets)}
    tokenizer_config = json.loads((source / "tokenizer_config.json").read_text())
    original_templates = {
        name: digest
        for name, digest in source_checksums.items()
        if Path(name).parts[0].startswith("chat_template")
    }
    embedded = tokenizer_config.get("chat_template")
    if embedded is not None:
        value = embedded if isinstance(embedded, str) else json.dumps(embedded, sort_keys=True)
        original_templates["tokenizer_config.json:chat_template"] = hashlib.sha256(
            value.encode()
        ).hexdigest()
    template = TRAINING_TEMPLATE.read_text()
    tokenizer_config["chat_template"] = template
    # HF versions differ in precedence between embedded and external templates.
    # Install the same source in both locations and omit stale named templates.
    replacements = {
        "chat_template.jinja": template.encode(),
        "tokenizer_config.json": (
            json.dumps(tokenizer_config, indent=2, ensure_ascii=False) + "\n"
        ).encode(),
        "config.json": (json.dumps(config, indent=2) + "\n").encode(),
    }
    preserved = {
        name: digest
        for name, digest in source_checksums.items()
        if not Path(name).parts[0].startswith("chat_template") and name != "tokenizer_config.json"
    }
    expected_files = {
        **preserved,
        **{name: hashlib.sha256(value).hexdigest() for name, value in replacements.items()},
    }
    identity = {
        "format_version": 2,
        "source_repo": HF_REPO,
        "source_revision": revision,
        "source_config_sha256": sha256(source / "config.json"),
        "retained_decoder_layers": list(range(num_layers)),
        "initialization": "scratch",
        "training_chat_template": TRAINING_TEMPLATE.stem,
        "effective_chat_template_sha256": sha256(TRAINING_TEMPLATE),
        "original_chat_templates_sha256": original_templates,
    }
    if output.exists():
        manifest = json.loads((output / "proxy_manifest.json").read_text())
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise ValueError(
                f"Existing artifact has different provenance/config: {output}; "
                "choose a new --output-dir or --workspace"
            )
        if manifest.get("source_files_sha256") != source_checksums:
            raise ValueError("Existing tokenizer differs from the selected source")
        actual_files = {
            str(path.relative_to(output)): sha256(path)
            for path in output.rglob("*")
            if path.is_file() and path != output / "proxy_manifest.json"
        }
        if manifest.get("files_sha256") != expected_files or actual_files != expected_files:
            raise ValueError(
                f"Existing artifact is corrupt or has unexpected template assets: {output}"
            )
        print(f"Reusing verified tokenizer/config: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=output.parent))
    try:
        for name in preserved:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        for name, value in replacements.items():
            (staging / name).write_bytes(value)
        actual_files = {
            str(path.relative_to(staging)): sha256(path)
            for path in staging.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files:
            raise ValueError("Source assets changed while preparing the tokenizer")
        manifest = {
            **identity,
            "source_path": str(source),
            "source_files_sha256": source_checksums,
            "files_sha256": expected_files,
        }
        (staging / "proxy_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging)
        raise
    print(f"Prepared tokenizer/config: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New tokenizer directory; existing artifacts are never overwritten",
    )
    parser.add_argument("--source-hf", type=Path, help="Local tokenizer/config snapshot")
    parser.add_argument("--revision", default=HF_REVISION, help="Immutable HF commit SHA")
    parser.add_argument(
        "--num-layers", type=int, default=40, help="Decoder layers: default full 40; multiple of 4"
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be an immutable 40-character lowercase commit SHA")
    if args.revision != HF_REVISION:
        parser.error(f"This training template is maintained for source revision {HF_REVISION}")
    cache = REPO_ROOT.parent / "hf_home/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots"
    source = args.source_hf or cache / args.revision
    if not source.is_dir():
        if args.source_hf:
            parser.error(f"Source directory does not exist: {source}")
        from huggingface_hub import snapshot_download

        source = Path(
            snapshot_download(
                repo_id=HF_REPO,
                revision=args.revision,
                cache_dir=args.workspace / "hf_cache",
                allow_patterns=["config.json", *ASSET_PATTERNS],
            )
        )
    revision = args.revision if args.source_hf is None or source.name == args.revision else None
    output = args.output_dir or args.workspace / "models" / MODEL_NAME / "tokenizer"
    prepare_tokenizer(source, output, num_layers=args.num_layers, revision=revision)


if __name__ == "__main__":
    main()
