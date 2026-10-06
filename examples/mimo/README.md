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

`--mimo-long-sample-min-tokens 120000` optionally joins distinct complete source
conversations into long, multi-image sequences and recomputes their multimodal
positions. This is a dataset option, not required for Dynamic CP.

## Example: four nodes, sixteen GPUs

Run the following on each of four nodes with the same `MASTER_ADDR` and a
different `NODE_RANK` in `0..3`. Launch from the repository root. The architecture
flags are required: the provider builds the decoder from CLI configuration, not
from the HF config file.

```bash
torchrun --nnodes=4 --nproc-per-node=4 \
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
  --recompute-granularity selective --recompute-modules gdn_norm_out moe \
  --cross-entropy-loss-fusion --cross-entropy-fusion-impl native \
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

`--mimo-vision-recompute` checkpoints the vision transformer uniformly, one layer
at a time. Decoder recomputation is independent. To use full decoder recompute,
replace the selective recompute options with
`--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`.

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

Global MoE auxiliary loss uses an additional decoder statistics forward over the
source optimizer step, including MTP routers. It uses training grad mode to match
compiler specialization, skips vocabulary heads, and discards each temporary
decoder graph while retaining only detached statistics. Training/recompute uses the fixed global counts;
`mimo global aux loss` is the corresponding step-wide weighted metric.

## Current scope and diagnostics

The fixed expert-ID study completed on 2026-10-02 uses the full Qwen3.5-35B VL
model on 16 GB200 GPUs: colocated encoder DP16, decoder TP1/PP1/EP16, BF16,
HybridEP with variable-length padding, full recompute, MTP1, and global MoE
auxiliary loss. CP1 and DCP each completed 50 native optimizer updates with no
skipped or NaN iterations. DCP actually used CP1/2/4/8/16. Every step matched
the source batch, supervision counts, and fixed expert choices, including
training and recomputation dispatch. Probabilities remain differentiable and
are recomputed from current logits; this diagnostic uses unfused router scoring.

These are trajectories conditioned on the evolving CP1 teacher's expert choices:
the two models update their weights independently. They are not natural-routing
convergence tests or same-weight error measurements after the first update.
The dataset pool contains 448 training and 64 disjoint heldout images; the source
stream filters overlength conversations and repeats. The 50-step runs use a 40K
source limit, not 50 steps of 128K sequences. Across the first/last five updates,
mean training LM loss fell from 0.433375 to 0.125491 for CP1 and 0.433399 to
0.124368 for DCP; unscaled MTP loss fell from 0.555680 to 0.263207 and from
0.555680 to 0.262435, respectively. Native gradient norms remained finite.

Heldout evaluation used the same 64-image batch throughout:

| Optimizer iteration | CP1 LM loss | DCP LM loss |
| --- | ---: | ---: |
| Initial pretrained weights | 0.436398 | Not separately evaluated |
| 25 | 0.395013 | 0.395482 |
| 50 | 0.467032 | 0.468310 |

Both trajectories overfit this small pool: heldout loss worsened after step 25
and exceeded the initial CP1 result at step 50. Their step-50 heldout difference
was 0.274%, but training trajectories were not identical: the largest absolute
LM difference was 0.007661 at step 47, and the largest relative difference was
6.16% at step 50. Final evaluation repeated step 50 exactly in each layout.
These runs check finite gradients and sampled diagnostics, not every gradient or
optimizer-update element at every step. Checkpoints at steps 25 and 50 were
saved; this pair did not test checkpoint resume.

Strict full-gradient acceptance remains incomplete even with fixed expert IDs.
Separate one-step controls reload identical model and native Adam states from
step 52 and use the same source batch. Relative L2 differences from that warm
CP1 reference are:

| Layout control | Complete gradient | Actual FP32 master-weight update |
| --- | ---: | ---: |
| Repeat CP1 | 0.00000153% | 0% |
| Static CP8, same packs | 39.100% | 14.790% |
| CP1, changed packing | 9.138% | 3.222% |
| DCP | 36.150% | 13.423% |

The static CP8 result shows that dynamic scheduling is not required for the
large discrepancy. It still changes CP execution, concurrent packs within EP16,
expert GEMM shapes, and accumulation order, so it does not isolate one kernel.
The warm reference gradient norm is 0.276193; CP8's absolute gradient difference
is 0.107991. Neither a similar gradient norm nor close heldout losses establish
that these vector differences are normal rounding. No new numerical tolerance
is inferred from the trajectory. Local GDN companion-perturbation tests isolated
the real tokens in two captured cold samples, but their strict zero-padding VJP
check failed, with residues up to 1.44e-8 in both CP1 and CP4. The cause and impact
remain unresolved; the corresponding warm sample has not had this isolation test.

A separate near-128K static-CP16/DCP pair processed 8,010,892 real tokens in
64 constructed, joined conversations (longest 130,743 tokens) containing 572
image instances. All packs used CP16 in both layouts, with different execution
order. Complete visual features and returned feature gradients matched exactly
over 175,718,400 elements, including 20 images whose visual rows crossed CP rank
boundaries. Row ownership, actual embedding splice, and the reverse transport
adjoint checks passed, including ranks without local visual rows. Full-parameter
gradient relative L2 was 8.40e-8 and actual master-update relative L2 was 5.35e-6;
LM and MTP losses were identical. This validates the tested transport and
same-CP16 cases; it does not resolve the cross-CP/mixed-pack failures above.

Cluster evidence is retained under `analysis/` in the experiment workspace:
`fixed_route_train_7605971_seed1234_r1_trajectory.json`, its `paired_curves_r2`
PNG/PDF and provenance, `fixed_route_7605971_warm_controls_matrix_final.json`,
and the `fixed_route_long_7605971_dynamic_vs_fixed_route_long_7605971_cp16_step0_long_boundary_audit_r1.json`
boundary report. The reports distinguish mechanical checks from numerical
acceptance and retain unsuccessful controls.

Earlier natural-routing checkpoint loading and source-step replay work, but the
original cold resume
with nondeterministic TE algorithms enabled failed its gradient-norm tolerance
and one step's auxiliary replay. Two independent cold resumes using
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` had exact auxiliary replay and identical
logged scalars and all sampled gradient-projection vectors to each other. They
still differed from the uninterrupted TE1 run's gradient norm by 2.99%, above 2%.
An additional matched TE0 continuous/resume control reproduced two post-resume
steps exactly in logged scalars and sampled gradient diagnostics. This is not an
elementwise comparison of every gradient or optimizer state. TE0 alone also did
not eliminate within-step router-assignment changes in a separate CP1 repacking
run: stronger per-token diagnostics detected differences between statistics,
original training, and recompute passes. The full vision tower's BF16 comparison
against Hugging Face also remains outside its strict feature/gradient tolerances. These unresolved
differences are not assumed to be harmless rounding; training progress and
checkpoint loading do not establish numerical equivalence.

- This provider requires TP1/PP1, dropout0, packed per-token loss, and
  `--rerun-mode disabled`. Leave gradient-reduction and parameter-gather overlap
  disabled because encoder backward completes at the native finalization boundary.
- The delivered dataset adapter is for still-image conversations. It loads the
  source global batch on each rank before scheduling; CPU loading/preprocessing
  and text staging are not yet sharded for throughput.
- The recipe uses MTP1. Reusing one MTP router across multiple prediction depths
  is unsupported by the global auxiliary-loss scope. The provider is a training
  path and does not implement inference-cache execution.
- A DCP capacity of 8192 is an upper bound, not a requirement to pad every rank
  to 8192 tokens. The validation recipe keeps HybridEP variable-length padding
  enabled for unequal rank-local lengths.
- `--mimo-metrics-dir DIR` writes per-step LM/MTP/auxiliary metrics, runtime CP
  sizes, token counts, and peak allocated memory. Native logs report optimizer
  gradient norms. Optional `--mimo-gradient-diagnostics` adds sampled diagnostics;
  `--mimo-gradient-reference DIR` plus `--mimo-write-gradient-reference` writes
  complete normalized gradient shards for a baseline, and omitting the write flag
  compares against them. These comparisons require the same world/TP/PP/EP
  ownership and source batch; the reference files are not training checkpoints.
  The diagnostics flag also records `aux_replay`: detached per-router/per-round
  expert histograms and probability sums from the statistics and training/recompute
  passes. Matching histograms do not prove identical per-token assignments.
- With `--mimo-gradient-diagnostics`, `--mimo-audit-token-assignments`
  additionally compares exact token-to-expert
  sets, excluding padding, and distinguishes original training from observed
  recompute calls. This audit does not compare ordered top-k entries or prove
  equality of router probabilities.
- `--mimo-execution-diagnostics-dir DIR --mimo-diagnostic-steps STEP ...`
  records sampled decoder activations plus complete visual-feature routing and
  gradient boundaries. `--mimo-diagnostic-gdn-layers INDEX ...` selects GDN layers
  for full input/output/cotangent capture and recompute checks (default: layer 0).
  These opt-in diagnostics can produce large CPU snapshots and change execution
  timing; failure to reproduce an earlier discrepancy is not evidence of a fix.
- `--mimo-boundary-reference DIR` records or compares complete encoder feature
  rows and their normalized FP32 gradients before encoder backward. It uses the
  same write-reference flag and verifies image identity, order, and producer
  ownership. This helps separate decoder/transport differences from the vision
  backward pass.
- `--mimo-diagnostic-gemm-workspace` requests a 1024-byte TE GEMM workspace for
  investigating arithmetic changes across pack shapes. It is a process-scoped
  diagnostic, separate from full batch-invariant mode; layout equivalence still
  needs to be measured.
- `--mimo-fixed-routing-record DIR` records ordered expert IDs during auxiliary
  statistics, then uses them for training and recomputation. A separate run uses
  `--mimo-fixed-routing-replay DIR` with the same source steps. Records are keyed
  by original sample, token position, and router (including MTP), with source/image
  fingerprints and complete coverage checks. Probabilities are recomputed from
  live logits using the existing unfused helper, retaining router gradients; aux
  counts use the same IDs. Padding is excluded from auxiliary counts. Recording
  and replay support runtime zigzag CP, TP1/PP1, dropless top-k and distinct MTP
  routers. CP>1 recording exchanges compact coverage metadata, then merges CPU
  expert-ID parts through shared files into the existing full-table format; it
  introduces no feature collective. Existing CP1 references remain readable.
  Metrics include the validated source fingerprint and actual recording CP sizes.
  This tests an objective conditioned on fixed
  expert choices; it does not establish natural-routing numerical equivalence.
- `--mimo-fixed-packing-record DIR` records the scheduler's intact CP1 packs.
  `--mimo-fixed-packing-replay DIR` redistributes those packs to static CP groups
  while requiring identical logical and physical boundaries. Replaying under DCP
  is rejected: the DCP experiment must use the actual scheduler. For BF16 TP1
  controls with dataset padding to 32 and CP up to 16, overriding
  `--pad-packed-seq-alignment 2` avoids CP-dependent trailing padding. Production
  padding defaults are unchanged. Route and pack records separate training and
  evaluation source steps and refuse to overwrite an existing reference.

The model provider, dataset adapter, coordinator, transport, auxiliary-loss scope,
and validation programs all live in this repository. Cluster launch scripts and
downloaded datasets are external artifacts; no external training implementation
is imported.
