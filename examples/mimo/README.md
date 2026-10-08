# Native colocated Qwen3.5 vision training with packed CP

`train.py --model-provider qwen35_native --dataset-provider qwen35_native` runs
the full Qwen3.5-35B-A3B vision-language model through Megatron's native
`pretrain()` loop, DDP, distributed optimizer, and checkpointing. It supports
colocated vision encoders with a packed decoder using either static CP or
Dynamic CP. The other `train.py` providers keep their existing behavior;
`pretrain_mimo.py` remains the separate heterogeneous training entrypoint.

## Inputs and environment

Use a Megatron-compatible environment with Transformer Engine, Qwen3.5-capable
Transformers, the GDN kernels, and HybridEP for the configuration below. Set these
paths before launching:

- `HF_MODEL`: local Qwen3.5-35B-A3B Hugging Face config, tokenizer, and processor
  directory. This supplies text/image preprocessing; it does not load model weights.
- `INITIAL_CHECKPOINT`: a converted full VL Bridge `torch_dist` **iteration
  directory**, containing `metadata.json`, language-model/MTP weights, and
  `vision_model.*` weights. `--mimo-pretrained-checkpoint` imports these weights
  strictly into the native MIMO namespaces. Conversion from HF is a separate step.
- `MANIFEST`: a CLEVR conversation manifest; `SAVE_DIR`: the native checkpoint root.

The current dataset adapter reads one still image per complete source
conversation. A manifest contains `records_file` and `samples`; each sample has
`record_index`, `image_paths` (one relative image path), and
`images: [{"sha256": "image-content-hash"}]`. The referenced JSONL records have
`id` and alternating `user`/`assistant` `messages`, with text/image content items
in Hugging Face chat format. Assistant text, including reasoning, is supervised.
The adapter reserves 64 distinct image identities for validation, so the manifest
must contain more than 64 distinct images. Oversized conversations are excluded
as complete units, rather than truncated.

## Example: four nodes, sixteen GPUs

Run the following on each of four nodes with the same `MASTER_ADDR` and a
different `NODE_RANK` in `0..3`. Launch from the repository root. The architecture
flags are required: the provider builds the decoder from CLI configuration, not
from the HF config file.

```bash
uv run python -m torch.distributed.run --nnodes=4 --nproc-per-node=4 \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port=29500 \
  examples/mimo/train.py \
  --model-provider qwen35_native --dataset-provider qwen35_native \
  --mimo-manifest "$MANIFEST" --mimo-hf-model "$HF_MODEL" \
  --mimo-pretrained-checkpoint "$INITIAL_CHECKPOINT" --mimo-vision-recompute \
  --num-layers 40 --hidden-size 2048 --ffn-hidden-size 8192 \
  --num-attention-heads 16 --group-query-attention --num-query-groups 2 --kv-channels 256 \
  --experimental-attention-variant gated_delta_net --linear-attention-freq 4 \
  --linear-num-key-heads 16 --linear-num-value-heads 32 \
  --linear-key-head-dim 128 --linear-value-head-dim 128 --linear-conv-kernel-dim 4 \
  --normalization RMSNorm --norm-epsilon 1e-6 --apply-layernorm-1p \
  --swiglu --disable-bias-linear --qk-layernorm --attention-output-gate \
  --num-experts 256 --moe-ffn-hidden-size 512 --moe-router-topk 8 \
  --moe-shared-expert-intermediate-size 512 --moe-shared-expert-gate \
  --moe-router-load-balancing-type global_aux_loss --moe-aux-loss-coeff 0.001 \
  --moe-router-dtype fp32 --padded-vocab-size 248320 \
  --position-embedding-type mrope --mrope-section 11 11 10 \
  --rotary-base 10000000 --rotary-percent 0.25 --no-rope-fusion \
  --untie-embeddings-and-output-weights --init-method-std 0.02 \
  --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1 \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$HF_MODEL" \
  --bf16 --transformer-impl transformer_engine --attention-backend auto \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --context-parallel-size 8 --expert-model-parallel-size 16 --expert-tensor-parallel-size 1 \
  --seq-length 131072 --max-position-embeddings 131072 \
  --use-varlen-dataset --sequence-packing-scheduler default_dynamic_cp \
  --dynamic-context-parallel --min-dynamic-context-parallel-size 1 \
  --max-seqlen-per-dp-cp-rank 8192 --pad-packed-seq-alignment 32 \
  --calculate-per-token-loss --dataloader-type external \
  --no-create-attention-mask-in-dataloader --num-workers 0 \
  --micro-batch-size 1 --global-batch-size 64 \
  --train-iters 52 --eval-interval 25 --eval-iters 1 \
  --eval-global-batch-size 64 --eval-micro-batch-size 1 \
  --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep \
  --moe-flex-dispatcher-num-sms 32 --moe-hybridep-pad-uneven-dispatch-inputs \
  --moe-grouped-gemm --moe-permute-fusion --moe-router-fusion \
  --use-distributed-optimizer --accumulate-allreduce-grads-in-fp32 \
  --optimizer adam --lr 2e-6 --min-lr 2e-7 --lr-warmup-iters 5 \
  --lr-decay-iters 300000 --lr-decay-style cosine \
  --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 --clip-grad 1.0 --weight-decay 0.033 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --hidden-dropout 0 --attention-dropout 0 --rerun-mode disabled --seed 1234 \
  --distributed-timeout-minutes 30 \
  --ckpt-format torch_dist --save "$SAVE_DIR" --save-interval 25 \
  --manual-gc --manual-gc-interval 20 \
  --log-interval 1 --log-throughput --log-num-zeros-in-grad
```

Here the encoder uses TP1/CP1 and DP16. The decoder starts with TP1/PP1/CP8/DP2;
Dynamic CP selects runtime CP groups within the same sixteen-rank DP×CP domain.
EP remains 16. A per-rank capacity of 8192 is an upper bound, not a request to
pad every rank to 8192 tokens. The 32-token packing alignment can leave different
dispatch sizes across ranks, so HybridEP uneven-input padding remains enabled.

For static CP, remove the two Dynamic CP flags and select
`--sequence-packing-scheduler dp_balanced`. The declared `--seq-length` must fit
`CP * max-seqlen-per-dp-cp-rank`: CP8 with capacity8192 permits 65536; CP1 needs
capacity at least its declared sequence length. Keep `--max-position-embeddings`
unchanged when comparing layouts, and verify that the same complete source
samples are selected. Large TP1 packs also increase vocabulary-logit/CE memory;
encoder or transformer recomputation does not remove that allocation.

`--mimo-vision-recompute` checkpoints the vision transformer one layer at a time.
Decoder recomputation is configured separately; the example uses full recompute.

To resume, replace `--mimo-pretrained-checkpoint "$INITIAL_CHECKPOINT"` with
`--load "$SAVE_DIR" --exit-on-missing-checkpoint`. Native checkpoints include the
vision/merger, decoder/MTP, optimizer, scheduler, RNG, and consumed-sample counts.
Keep the dataset, seed, global batch size, and preprocessing unchanged for
continuation of the same source stream.

## Data and gradient flow

`NativeMimoStep` imports the existing core packing scheduler and THD pack builder.
It assigns complete images independently to encoder ranks, then routes their
projected feature rows according to each scheduled decoder pack:

1. Producers send that pack's features to its CP leader with P2P.
2. The leader broadcasts the complete feature table inside that pack's runtime
   CP group. MIMO inserts features into placeholder positions before CP slicing.
3. Backward sums consumer feature gradients to the leader and returns the saved
   source slices to their producers. After all decoder rounds, the accumulated
   gradients backpropagate through the retained encoder graph.
4. Megatron performs native parameter-gradient reduction, token normalization,
   clipping, and optimizer updates.

The boundary uses explicit gradient return; ordinary P2P is not treated as an
autograd operation. There is no global feature all-gather, but complete pack
features are replicated within their CP group. Sending only locally consumed
visual rows is a separate optimization.

MoE auxiliary losses and MTP use Megatron's native statistics, normalization,
and logging. Changing pack composition can change these auxiliary objectives;
the adapter does not add a statistics forward or require equality across packing layouts.

## Supported scope and tests

The colocated adapter currently requires TP=PP=1, per-token loss, external
source batches, disabled rerun, and non-overlapped DDP gradient/parameter
communication. It supports static packed CP and Dynamic CP, including empty
encoder ranks and image features spanning CP shards. Independent source samples
retain separate THD boundaries. Native LM/MTP/auxiliary-loss, grad-norm and
checkpoint logging remain available through ordinary Megatron flags.

`tests/unit_tests/models/mimo/test_multisample_multiimage_cp8.py` exercises the
real scheduler, THD layout, bridge and backward for multiple multi-image samples
sharing a CP8 pack. Run it through `python -m torch.distributed.run` on **16 ranks**;
it skips with fewer ranks. The other focused model/transport/objective tests are
under `tests/unit_tests/models/mimo`, `tests/unit_tests/models/test_qwen35_vit.py`
and `tests/unit_tests/transformer`.
