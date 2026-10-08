# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Download bounded CLEVR-2 prefixes and retain complete image conversations.

This is a data preparation CLI, independent of torch and training. Explicit byte
budgets bound downloads; the manifest records content hashes and source revision.
"""

import argparse
import hashlib
import io
import json
import re
import struct
import tarfile
import zlib
from pathlib import Path

import requests

REPOSITORY = "nvidia/Nemotron-Image-Training-v3"
REVISION = "7656391d4d4cb11ec3722b34f10d499435de0460"
BASE = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}"
TARGET = 16
BYTE_BUDGET = 32 * 1024 * 1024


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch_prefix(path, limit):
    """Refuse full-file responses before reading their bodies."""
    with requests.get(
        f"{BASE}/{path}",
        params={"download": "true", "fixture_prefix_bytes": str(limit)},
        headers={"Range": f"bytes=0-{limit - 1}", "Accept-Encoding": "identity"},
        stream=True,
        timeout=(15, 90),
    ) as response:
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError(f"Server ignored Range for {path}; stopped before reading body")
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes 0-(\d+)/(\d+)", content_range)
        if not match or int(match[1]) + 1 != limit:
            raise RuntimeError(f"Unexpected bounded response: {content_range!r}")
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise RuntimeError("Unexpected compressed HTTP response")
        payload = bytearray()
        progress = 0
        for chunk in response.iter_content(chunk_size=65536):
            if len(payload) + len(chunk) > limit:
                raise RuntimeError("Range response exceeded requested byte count")
            payload.extend(chunk)
            if len(payload) - progress >= 64 * 1024**2:
                progress = len(payload)
                print(f"{path}: {progress // 1024**2}/{limit // 1024**2} MiB", flush=True)
        if len(payload) != limit:
            raise RuntimeError("Incomplete HTTP range response")
        payload = bytes(payload)
        return payload, {
            "path": path,
            "status": 206,
            "content_range": content_range,
            "bytes_read": len(payload),
            "sha256": sha256(payload),
        }


def png_info(payload):
    """Validate every PNG chunk CRC and decompress all standard RGB scanlines."""
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("PNG signature mismatch")
    offset, compressed, ihdr, ended = 8, [], None, False
    while offset < len(payload):
        length = struct.unpack_from(">I", payload, offset)[0]
        kind = payload[offset + 4 : offset + 8]
        data = payload[offset + 8 : offset + 8 + length]
        crc = struct.unpack_from(">I", payload, offset + 8 + length)[0]
        if zlib.crc32(kind + data) & 0xFFFFFFFF != crc:
            raise ValueError(f"PNG CRC mismatch in {kind!r}")
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", data)
        elif kind == b"IDAT":
            compressed.append(data)
        elif kind == b"IEND":
            ended = True
        offset += 12 + length
    if not ended or offset != len(payload) or ihdr is None:
        raise ValueError("PNG is truncated")
    width, height, bits, color, compression, filtering, interlace = ihdr
    if (bits, compression, filtering, interlace) != (8, 0, 0, 0) or color not in (2, 6):
        raise ValueError(f"Unexpected PNG format: {ihdr}")
    channels = 3 if color == 2 else 4
    rows = zlib.decompress(b"".join(compressed))
    stride = width * channels + 1
    if len(rows) != height * stride or any(rows[i * stride] > 4 for i in range(height)):
        raise ValueError("Invalid PNG decoded scanlines")
    return {
        "width": width,
        "height": height,
        "channels": channels,
        "png_crc_and_deflate_valid": True,
    }


def image_references(record):
    return [
        item["image"]
        for message in record["messages"]
        for item in message["content"]
        if isinstance(item, dict) and item.get("type") == "image"
    ]


def first_text(record, role):
    for message in record["messages"]:
        if message["role"] != role:
            continue
        content = message["content"]
        if isinstance(content, str):
            return content
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item["text"]
            if isinstance(item, str):
                return item
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--annotation-mib", type=int, default=2)
    parser.add_argument("--media-mib", type=int, default=8)
    parser.add_argument("--budget-mib", type=int, default=BYTE_BUDGET // 1024**2)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Preserving existing fixture: {output}")
    annotation_limit, media_limit = args.annotation_mib * 1024**2, args.media_mib * 1024**2
    budget = args.budget_mib * 1024**2
    if (
        min(annotation_limit, media_limit, args.target) <= 0
        or annotation_limit + media_limit > budget
    ):
        raise ValueError("Requested prefixes must fit the positive download budget")
    annotations, annotation_report = fetch_prefix("clevr_2/clevr_2.jsonl", annotation_limit)
    records = [json.loads(line) for line in annotations.rsplit(b"\n", 1)[0].splitlines()]
    media, media_report = fetch_prefix("clevr_2/media/shard_000000.tar", media_limit)
    images = {}
    with tarfile.open(fileobj=io.BytesIO(media), mode="r:") as archive:
        while True:
            member = archive.next()
            if member is None or member.offset_data + member.size > len(media):
                break
            if member.isfile() and member.name.endswith(".png"):
                payload = media[member.offset_data : member.offset_data + member.size]
                images[member.name] = (payload, png_info(payload), member.offset_data)

    selected, used_images = [], set()
    for source_line, record in enumerate(records):
        refs = image_references(record)
        if refs and all(ref in images for ref in refs) and not used_images.intersection(refs):
            selected.append((source_line, record, refs))
            used_images.update(refs)
            if len(selected) == args.target:
                break
    if len(selected) < args.target:
        raise RuntimeError(
            f"Only {len(selected)} complete records in bounded prefixes; expected {args.target}"
        )

    (output / "images").mkdir(parents=True, exist_ok=True)
    saved_records, samples = [], []
    for source_line, record, refs in selected:
        image_paths, image_meta = [], []
        for ref in refs:
            if Path(ref).name != ref:
                raise ValueError("Unexpected image member path")
            payload, info, tar_offset = images[ref]
            relative = f"images/{ref}"
            (output / relative).write_bytes(payload)
            image_paths.append(relative)
            image_meta.append(
                {
                    "source_member": ref,
                    "local_path": relative,
                    "bytes": len(payload),
                    "sha256": sha256(payload),
                    "tar_data_offset": tar_offset,
                    **info,
                }
            )
        serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        saved_records.append(serialized)
        samples.append(
            {
                "sample_id": record["id"],
                "source_record_line": source_line,
                "record_index": len(saved_records) - 1,
                "record_path": "records.jsonl",
                "record_sha256": sha256(serialized.encode()),
                "image_paths": image_paths,
                "images": image_meta,
                "question": first_text(record, "user"),
                "answer": first_text(record, "assistant"),
                "message_count": len(record["messages"]),
            }
        )
    records_bytes = ("\n".join(saved_records) + "\n").encode()
    (output / "records.jsonl").write_bytes(records_bytes)
    manifest = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "subset": "clevr_2",
        "samples": samples,
        "sample_count": len(samples),
        "unique_image_count": len(used_images),
        "records_file": "records.jsonl",
        "records_sha256": sha256(records_bytes),
        "downloads": [annotation_report, media_report],
        "download_payload_bytes": len(annotations) + len(media),
        "payload_budget_bytes": budget,
        "annotation_records_in_prefix": len(records),
        "complete_images_in_prefix": len(images),
        "validation": "All references match complete tar members; PNG chunk CRCs, DEFLATE and scanline sizes checked.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "samples"}, indent=2))
    print(f"Manifest: {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
