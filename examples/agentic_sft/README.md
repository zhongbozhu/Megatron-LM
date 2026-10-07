# Agentic SFT: full Qwen3.5-35B-A3B on 4×GB200

Native `pretrain_gpt.py` with plain `torchrun` in a prepared four-GPU container.
No Megatron Bridge dependency. Tools are conversation data, never executed.

Run the commands below from the repository root. `DRY_RUN=1` prints a training
command; all generated assets stay in the workspace outside the repository.
Scripts 01 and 04 retain their historical `_proxy` filenames; both now default
to the full 40-layer decoder. No model weights are downloaded by preparation.

## Model and modes

The default model has **40 decoder layers**: ten cycles of
3 GDN + 1 attention, plus **MTP1 (coefficient 0.1)**. It preserves hidden 2048,
16 attention heads / 2 query groups, 256 top-8 experts (FFN 512), gated shared
expert, vocabulary 248320, and Qwen norm/RoPE settings. CPU optimizer offload
keeps FP32 master parameters and Adam moments in host memory; compute uses
MXFP8 with BF16 parameter storage/gather and full activation recomputation.
The full model has 35,505,251,456 unique trainable parameters including MTP.
All six trials below completed 75 optimizer updates, final evaluation and a
weights-only checkpoint on one four-GB200 node per run, with all four ranks
audited against the scheduler replay. These are scratch performance runs.

All modes use **64K context, 16384 tokens/rank, TP1/PP1/base CP4/EP4/ETP1, MBS1**.

| Mode | Scheduler | Runtime CP | Default GBS |
| --- | --- | --- | --- |
| `online_static_cp` | `dp_balanced` | 4 | 32 trajectories |
| `online_dynamic_cp` | `default_dynamic_cp` | 1, 2, 4 | 32 trajectories |
| `offline_dynamic_cp` | `default_dynamic_cp` | 1, 2, 4 | 8 packed rows |

Base DP is 1; static mode does not measure multi-DP balancing. DCP unpacks offline
rows before scheduling. The GBS defaults represent different workloads.

Default `INIT_MODE=scratch` randomly initializes decoder/MTP. To load an already
converted, shape-compatible Megatron checkpoint, set `INIT_MODE=checkpoint` and
`PRETRAINED_CHECKPOINT=/path/to/checkpoint`. Conversion stays outside this example.
`RESUME=1` requires a checkpoint containing optimizer/RNG state to restore the
complete training state. The benchmark command below saves weights only;
fresh runs refuse to overwrite existing checkpoints.

## Prepare natural and mixed data

Download **38,584 candidates (50% of pinned CoderForge SWE_Rebench)** once.
Script 02 defaults to 5%; this example requests 50% to supply enough unique parents.
Sampling uses seed-shuffled shards and a bounded prefix, not uniform row sampling.
See the [dataset card/licenses](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview).

- `natural`: retain all trainable trajectories, preserving the long-heavy distribution.
- `mixed`: target **5,120 training / 256 validation samples**, with **50% ≤16K,
  30% at 16–32K and 20% above 32K**. Prefer complete samples, then fill short/medium
  shortages with real prefixes ending at complete assistant messages, including tool calls.
  Each original trajectory contributes at most one output, within its original split.

```bash
EXAMPLE=examples/agentic_sft
WORKSPACE=../debug-codex-workspace/agentic_sft
TOKENIZER="${WORKSPACE}/models/qwen35_35B_1node/tokenizer"
DATA="${WORKSPACE}/data"

python "${EXAMPLE}/01_prepare_qwen35_35B_1node_proxy.py" \
    --output-dir "${TOKENIZER}"
python "${EXAMPLE}/02_prepare_coderforge.py" \
    --percentage 50 \
    --output-dir "${DATA}/coderforge_candidates_50pct"
for distribution in natural mixed; do
    python "${EXAMPLE}/02_5_filter_coderforge.py" \
        --data-dir "${DATA}/coderforge_candidates_50pct" \
        --output-dir "${DATA}/coderforge_${distribution}_v2_full40" \
        --distribution "${distribution}" \
        --tokenizer "${TOKENIZER}"
    python "${EXAMPLE}/03_pack_coderforge.py" \
        --data-dir "${DATA}/coderforge_${distribution}_v2_full40" \
        --tokenizer "${TOKENIZER}"
done
```

02_5 and 03 require `--data-dir`; both default to 32 tokenizer processes and 64K.
Mixed-only overrides are `--num-train-samples 5120`, `--num-validation-samples 256`
and `--length-bucket-ratios 0.5 0.3 0.2`. These specify **sample counts**, not token
shares: train targets are 2560/1536/1024; validation targets are 128/77/51.
Insufficient unique parents or usable prefixes fail explicitly. The manifest records
actual lengths/token shares, prefix endpoints and source IDs; mixed order is seeded.
The measured mixed preparation reached all 5,120/256 targets. Training token
shares were 19.16% / 36.01% / 44.83% across the three length buckets; 2,513
training outputs were prefixes. Source IDs are unique; one inherited pair
has identical rendered tokens/masks under different source IDs, and no repeated content
occurs within either measured 75-update input stream.

Natural retains complete JSONL conversations and applies the 65537-token cap plus
one shift during encoding. Prefixes preserve original messages/metadata and are
re-encoded to verify their bucket without cutting inside messages. Packing preserves
masks/boundaries using attributed Bridge FFD code without importing Bridge.
Use fresh output directories when changing selection, supervision, or
tokenizer/config provenance. The new `_v2_full40` directories avoid reusing
caches prepared with the earlier reduced-model tokenizer manifest.

Choose either dataset; **the same mode uses the same training configuration**:

```bash
export TOKENIZER_PATH="${TOKENIZER}"
DISTRIBUTION=natural  # Set mixed for the other distribution.
export DATA_DIR="${DATA}/coderforge_${DISTRIBUTION}_v2_full40"
for mode in online_static_cp online_dynamic_cp offline_dynamic_cp; do
    TRAIN_ITERS=75 \
    LR_WARMUP_ITERS=2 \
    EVAL_INTERVAL=1000 \
    EVAL_ITERS=1 \
    SAVE_DIR="${WORKSPACE}/runs/full40_v2_${DISTRIBUTION}_${mode}" \
        bash "${EXAMPLE}/04_train_qwen35_35B_1node_proxy.sh" "${mode}" \
            --log-timers-to-tensorboard \
            --no-save-optim \
            --no-save-rng
done
```

At online GBS32, 75 updates consume 2,400 selected conversations. Offline GBS8
consumes 600 packed rows; count their actual constituents/tokens separately. Verify
that the prepared dataset is large enough to avoid wrapping during your run.
Evaluate once and save weights after update 75; measure all steps 16–75.

## Supervision and performance settings

`--sft-loss-mode assistant` uses the training template installed by 01: generation
blocks supervise assistant reasoning, tool calls, body and existing endings, while
headers/other roles provide context. Historical reasoning is retained. No boundary
or profile flags are needed. `full` supervises every real next-token target; omitting
the flag retains legacy behavior. Online/offline share the [same encoder/contract](../../megatron/training/datasets/README.md).

| Performance switch | Default behavior |
| --- | --- |
| `PRECISION=mxfp8` | MXFP8 compute + TE fuser + GroupedTensor + CuTeDSL; `bf16` disables this compute bundle |
| `OPTIMIZER_CPU_OFFLOAD=1` | CPU Adam, FP32 masters/gradients/moments, BF16 parameter gather; `0` keeps optimizer work on GPU and enables FP8 gather/buffer reuse with MXFP8 |
| `HYBRID_EP=1` | HybridEP with uneven-input padding; `0` selects all-to-all |
| `FULL_RECOMPUTE=1` | Uniform full recompute; `0` selects GDN norm/MoE recompute |
| `DDP_OVERLAP=1` | Gradient reduction and parameter gathering overlap |

Both precision modes use `--bf16` as the base precision.
CPU offload defaults to 12 OpenMP threads per rank and one OpenBLAS thread;
GPU-only optimization defaults to one OpenMP thread. Environment overrides are
preserved. Launches requested 96 CPUs per node. Final sampled aggregate
peak RSS across the benchmark allocations was approximately 845–873 GiB.
This spans initialization, training, evaluation and checkpointing, may
double-count shared pages, and is not a minimum physical-RAM requirement.
Plan host capacity accordingly.
With the tested full-model 64K configuration, GPU-only FP32 Adam ran forward
and backward but OOMed at the first optimizer step despite full recompute.
CPU offload is required by this validated recipe.
FP8 parameter gather and MXFP8 gradient-buffer reuse are incompatible with this
CPU-offload path and are omitted when it is enabled. CPU Adam and host/device
transfer time remain included in measured iteration time.
Per-token LM loss, FP32 router/gradient accumulation, native fused CE, zero dropout
and MoE aux coefficient 0.001 remain enabled. The container must provide compatible
TE/CuTeDSL and DeepEP HybridEP. Microbatch order/grouping can change global MoE
auxiliary and MTP gradients; matching LM supervision does not imply full-gradient parity.
For explicit reproduction of the earlier reduced model, use `NUM_LAYERS=8
OPTIMIZER_CPU_OFFLOAD=0`; it is not the default full-model configuration. Preparation
also accepts `--num-layers 8` with a fresh output directory. Existing tokenizer
assets and preparation caches are never migrated in place.

## Measured DCP performance

With base DP1, static CP4 already spreads each trajectory over all four ranks.
Natural is long-heavy; its >32K samples contributed 91.8% of tokens in the pinned
50% preparation. New mixed increases short/medium inputs, giving DCP more CP1/CP2
opportunities. Sample ratios do not determine token shares or runtime CP choices.
Prefixes also favor earlier conversation turns, so this is a performance workload,
not a model-quality comparison.

Use online GBS32 / offline GBS8 for **both distributions**, with identical model,
LR, precision and other performance settings. Compare static/DCP online on the same
ordered inputs. Offline processes different work; lower step time alone is not a
packing speedup. Report actual tokens/s and nominal TFLOP/s/GPU, which excludes
recompute and retains approximate MTP accounting.

DCP's greedy packing does not guarantee fewer microbatches than static CP. Linear
GDN/MoE/logit work, kernel shapes and communication can offset scheduling gains.
All trials used the full 40-layer model + MTP1, MXFP8 compute, full recompute
and FP32 CPU optimizer offload. The window includes every step 16–75
(60 TensorBoard float timings), including slow steps. Rates include CPU
optimizer work and transfers; TFLOP/s is the duration-weighted native
nominal model estimate, not a hardware counter, and excludes recomputation.

| Distribution | Mode | Mean seconds/step | Nominal TFLOP/s/GPU | Runtime tokens/s (4-GPU aggregate) |
| --- | --- | ---: | ---: | ---: |
| natural | Online static CP4 | 95.951 | 138.800 | 16,169 |
| natural | Online DCP | 102.886 | 129.444 | 15,079 |
| natural | Offline DCP | 32.277 | 119.716 | 13,986 |
| mixed | Online static CP4 | 62.957 | 90.412 | 11,894 |
| mixed | Online DCP | 52.214 | 109.015 | 14,342 |
| mixed | Offline DCP | 34.858 | 112.391 | 14,925 |

On identical ordered online inputs and the same node, mixed DCP throughput
changed by **+20.57%** relative to static CP4; natural changed by
**-6.74%**. Each is one sequential pair, so these are measured
results rather than an order-independent speedup guarantee. Offline uses
different inputs/work and GBS; its lower seconds/step is not a comparable
elapsed-speedup claim. Losses/gradients remained finite with zero skipped/NaN
updates; this does not establish gradient parity or model quality.

Detailed data identities, CP counts, variability, memory results and audit
artifacts are in workspace
`validation_64k/real_length_study_mixed_v2_full40/RESULTS.md`. The frozen
`PROTOCOL.md` and execution record retain the original plan and the documented
replacement natural pair; the earlier successful natural static trial is
supplementary only. A documented stdout parser adapter handles one interleaved
warning in mixed static without changing any measurements. Historical results
under `validation_64k/real_length_study_generation_v1/RESULTS.md` used a different
mixed selection, reduced model and batch configuration.


## Validation

Run the preparation, filtering and launch-command contracts without a GPU:

```bash
python -m pytest --noconftest --import-mode=importlib -q \
  examples/agentic_sft/tests/test_prepare_proxy.py \
  examples/agentic_sft/tests/test_training_chat_template.py \
  examples/agentic_sft/tests/test_training_launcher.py \
  examples/agentic_sft/tests/test_agentic_preparation.py \
  examples/agentic_sft/tests/test_agentic_filtering.py
```

The template tests require Transformers and the pinned local tokenizer; set
`QWEN35_SOURCE_TOKENIZER` when its snapshot is outside the default cache.
`tests/test_packing.py` additionally requires the Megatron/PyTorch environment
and checks the example producer against the core packed-row reader.

The separate BF16 numerical test is a diagnostic with strict gradient/update
absolute-error bounds. Earlier runs failed those bounds; this is not a passing
full-model gradient-equivalence check. Its thresholds are retained unchanged.
Run it explicitly with one GPU in the training environment:

```bash
python -m torch.distributed.run --nproc-per-node=1 -m pytest \
  --noconftest --import-mode=importlib -q \
  examples/agentic_sft/tests/test_agentic_sft_numerics.py
```
