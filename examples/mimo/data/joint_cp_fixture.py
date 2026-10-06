# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Frozen numerical fixture: real CLEVR pixels, controlled multi-image THD positions.

The text is artificial and must not be used for convergence or quality claims.
The actual DCP scheduler, vision model and native decoder/optimizer are unchanged.
Generate once on CPU, then supply --mimo-diagnostic-source-batch to train.py.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel

from examples.mimo.data.qwen35_native import Qwen35Dataset


class FrozenSourceDataset:
    """Repeat one source batch while native consumed-sample accounting advances."""

    def __init__(self, path):
        self.payload = torch.load(path, map_location='cpu', weights_only=True)
        if self.payload.get('version') != 1:
            raise ValueError('Unsupported frozen source batch version')

    def build_global_batch(self, step, global_batch_size):
        samples = self.payload['samples']
        if len(samples) != global_batch_size:
            raise ValueError('Frozen source batch must match --global-batch-size')
        return samples, self.payload['media']


def create_fixture(manifest, hf_model, output):
    dataset = Qwen35Dataset(manifest, hf_model, seq_length=131072)
    # Image ID controls the production greedy tie-break. Images in A/B therefore
    # come from producers both inside and outside their eventual CP8 group.
    image_ids = ((8, 12, 16), (11, 15), (0, 1, 2, 3, 4, 5), (6, 7, 9, 10, 13, 14))
    starts = (
        (200, 376, 568),
        (72, 184),
        (64, 160, 256, 352, 448, 544),
        (64, 160, 256, 352, 448, 544),
    )
    lengths = (3072, 1024, 2048, 2048)
    media = []
    for image_id, example_index in enumerate(dataset.indices['train'][:17]):
        example = dataset.examples[example_index]
        with Image.open(example['path']) as image:
            processed = dataset.processor.image_processor(
                images=[image.convert('RGB')],
                size={'shortest_edge': 128 * 128, 'longest_edge': 128 * 128},
                return_tensors='pt',
            )
        grid = processed['image_grid_thw']
        rows = int(grid.prod(-1).sum()) // dataset.config.vision_config.spatial_merge_size**2
        # A non-square image can be snapped to 96x160 while preserving aspect;
        # insist on the known shape instead of silently weakening ownership.
        if rows != 16:
            with Image.open(example['path']) as image:
                processed = dataset.processor.image_processor(
                    images=[image.convert('RGB').resize((128, 128))],
                    size={'shortest_edge': 128 * 128, 'longest_edge': 128 * 128},
                    return_tensors='pt',
                )
            grid = processed['image_grid_thw']
            rows = int(grid.prod(-1).sum()) // dataset.config.vision_config.spatial_merge_size**2
        if rows != 16 or grid.shape != (1, 3):
            raise ValueError(f'Expected a single 16-row image, got {grid.tolist()}')
        patch = dataset.config.vision_config.patch_size
        media.append(
            dict(
                image_id=image_id,
                source_id=example['source_id'],
                path=example['path'],
                size=(int(grid[0, 1]) * patch, int(grid[0, 2]) * patch),
                length=rows,
                grid=grid,
                pixel_values=processed['pixel_values'],
            )
        )
    samples = {}
    ordinary = dataset.processor.tokenizer.encode(
        'Describe the colored shapes. ', add_special_tokens=False
    )
    vision_start = dataset.config.vision_start_token_id
    vision_end = dataset.processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    for sid, (physical, images, positions) in enumerate(zip(lengths, image_ids, starts)):
        real = physical - 13
        ids = torch.tensor((ordinary * ((real + 1) // len(ordinary) + 1))[: real + 1])
        supervised = torch.ones(real + 1)
        supervised[0] = 0
        for image_id, start in zip(images, positions):
            rows = media[image_id]['length']
            ids[start - 1] = vision_start
            ids[start : start + rows] = dataset.image_token_id
            ids[start + rows] = vision_end
            supervised[start - 1 : start + rows + 1] = 0
        grids = torch.cat([media[image_id]['grid'] for image_id in images])
        rope, _ = Qwen3_5MoeModel.get_rope_index(
            dataset.position_helper,
            input_ids=ids[None],
            mm_token_type_ids=(ids == dataset.image_token_id).int()[None],
            image_grid_thw=grids,
        )
        sample = dataset._pad_sample(ids, supervised, rope[:, 0])
        sample['media_ids'] = list(images)
        sample['source_ids'] = tuple(media[image_id]['source_id'] for image_id in images)
        assert sample['padded_seq_len'] == physical
        samples[sid] = sample
    from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler

    scheduler = DefaultDynamicCPScheduler(512, 8, 2, None)
    assignments = scheduler.get_groups_and_subsamples(list(enumerate(lengths)))
    expected = [[[0, 1]] * 8 + [[2]] * 4 + [[3]] * 4]
    if assignments != expected:
        raise ValueError(f'Fixture no longer produces the required actual DCP plan: {assignments}')
    payload = dict(version=1, samples=samples, media=media)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        torch.save(payload, stream)
    summary = dict(
        artificial_text=True,
        real_clevr_images=17,
        image_rows=16,
        lengths=lengths,
        sample_image_ids=image_ids,
        image_token_starts=starts,
        assignments=assignments,
        local_capacity=512,
        global_batch_size=4,
    )
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--hf-model', required=True)
    parser.add_argument('--output', required=True)
    options = parser.parse_args()
    create_fixture(options.manifest, options.hf_model, options.output)
