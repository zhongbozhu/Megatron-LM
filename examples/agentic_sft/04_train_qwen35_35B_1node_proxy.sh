#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Run inside a prepared GB200 container with four visible GPUs.
set -euo pipefail

# Paths and run selection. The last arguments are ordinary Megatron overrides.
MEGATRON_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE=${WORKSPACE:-${MEGATRON_ROOT}/../debug-codex-workspace/agentic_sft}
TOKENIZER_PATH=${TOKENIZER_PATH:-${WORKSPACE}/models/qwen35_35B_1node/tokenizer}
DATA_DIR=${DATA_DIR:-${WORKSPACE}/data/coderforge}
INIT_MODE=${INIT_MODE:-scratch}
MODE=${1:-online_dynamic_cp}
if (( $# > 0 )); then
    shift
fi
DRY_RUN=${DRY_RUN:-0}

# Model: full 35B-A3B decoder, ten GDN/GDN/GDN/attention blocks, plus MTP1.
NUM_LAYERS=${NUM_LAYERS:-40}
MODEL_ARGS=(
    --num-layers "${NUM_LAYERS}"
    --hidden-size 2048
    --ffn-hidden-size 8192
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 2
    --kv-channels 256
    --experimental-attention-variant gated_delta_net
    --linear-attention-freq 4
    --linear-num-key-heads 16
    --linear-num-value-heads 32
    --linear-key-head-dim 128
    --linear-value-head-dim 128
    --linear-conv-kernel-dim 4
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --apply-layernorm-1p
    --swiglu
    --disable-bias-linear
    --qk-layernorm
    --attention-output-gate
    --num-experts 256
    --moe-ffn-hidden-size 512
    --moe-router-topk 8
    --moe-shared-expert-intermediate-size 512
    --moe-shared-expert-gate
    --moe-router-load-balancing-type global_aux_loss
    --moe-aux-loss-coeff 0.001
    --moe-router-dtype fp32
    --padded-vocab-size 248320
    --position-embedding-type rope
    --rotary-base 10000000
    --rotary-percent 0.25
    --untie-embeddings-and-output-weights
    --init-method-std 0.02
    --hidden-dropout 0
    --attention-dropout 0
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
)

# Four physical ranks. DCP can repartition them into CP1, CP2, or CP4 groups.
PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 4
    --expert-model-parallel-size 4
    --expert-tensor-parallel-size 1
)

# Dataset and packing. One online item is a selected conversation (a complete
# trajectory or a real prefix); one offline item is a packed row. Never enable both
# --sft and --use-varlen-dataset. The scheduler unpacks offline rows first.
SEQ_LENGTH=${SEQ_LENGTH:-65536}
MAX_SEQLEN_PER_RANK=${MAX_SEQLEN_PER_RANK:-$((SEQ_LENGTH / 4))}
NUM_WORKERS=${NUM_WORKERS:-2}
DATASET_ARGS=(
    --tokenizer-type SFTTokenizer
    --tokenizer-model "${TOKENIZER_PATH}"
    --sft-tokenizer-prompt-format default
    --sft-loss-mode assistant
    --seq-length "${SEQ_LENGTH}"
    --max-position-embeddings "${SEQ_LENGTH}"
    --max-seqlen-per-dp-cp-rank "${MAX_SEQLEN_PER_RANK}"
    --pad-packed-seq-alignment 32
    --dataloader-type single
    --no-create-attention-mask-in-dataloader
    --num-workers "${NUM_WORKERS}"
)
case "${MODE}" in
    online_static_cp)
        GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
        BATCH_UNIT=trajectories
        DATASET_ARGS+=(
            --use-varlen-dataset
            --sequence-packing-scheduler dp_balanced
        )
        ;;
    online_dynamic_cp)
        GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
        BATCH_UNIT=trajectories
        DATASET_ARGS+=(
            --use-varlen-dataset
            --sequence-packing-scheduler default_dynamic_cp
            --dynamic-context-parallel
            --min-dynamic-context-parallel-size 1
        )
        ;;
    offline_dynamic_cp)
        GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
        BATCH_UNIT=packed_rows
        DATA_DIR=${PACKED_DATA_DIR:-${DATA_DIR}/packed_${SEQ_LENGTH}}
        DATASET_ARGS+=(
            --sft
            --sequence-packing-scheduler default_dynamic_cp
            --dynamic-context-parallel
            --min-dynamic-context-parallel-size 1
        )
        ;;
    *)
        echo "Usage: $0 {online_static_cp|online_dynamic_cp|offline_dynamic_cp} [Megatron arguments]" >&2
        exit 2
        ;;
esac
if [[ ${MODE} == offline_dynamic_cp ]]; then
    TRAIN_DATA=${TRAIN_DATA:-${DATA_DIR}/training.parquet}
    VALID_DATA=${VALID_DATA:-${DATA_DIR}/validation.parquet}
else
    TRAIN_DATA=${TRAIN_DATA:-${DATA_DIR}/training.jsonl}
    VALID_DATA=${VALID_DATA:-${DATA_DIR}/validation.jsonl}
fi
DATASET_ARGS+=(
    --train-data-path "${TRAIN_DATA}"
    --valid-data-path "${VALID_DATA}"
)

# Training numerics. Normalize by supervised tokens, never by packed-row count.
TRAIN_ITERS=${TRAIN_ITERS:-100}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-$((TRAIN_ITERS / 10))}
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --train-iters "${TRAIN_ITERS}"
    --calculate-per-token-loss
    --bf16
    --accumulate-allreduce-grads-in-fp32
    --optimizer adam
    --main-params-dtype fp32
    --main-grads-dtype fp32
    --exp-avg-dtype fp32
    --exp-avg-sq-dtype fp32
    --lr 2e-5
    --min-lr 2e-6
    --lr-warmup-iters "${LR_WARMUP_ITERS}"
    --lr-decay-iters "${TRAIN_ITERS}"
    --lr-decay-style cosine
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --clip-grad 1.0
    --weight-decay 0.033
    --seed 1234
)

# Performance toggles. Keep backend choices separate from model/data semantics.
HYBRID_EP=${HYBRID_EP:-1}
PRECISION=${PRECISION:-mxfp8}
FULL_RECOMPUTE=${FULL_RECOMPUTE:-1}
DDP_OVERLAP=${DDP_OVERLAP:-1}
OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD:-1}
for toggle in HYBRID_EP FULL_RECOMPUTE DDP_OVERLAP OPTIMIZER_CPU_OFFLOAD; do
    if [[ ${!toggle} != 0 && ${!toggle} != 1 ]]; then
        echo "${toggle} must be 0 or 1" >&2
        exit 2
    fi
done
PERFORMANCE_ARGS=(
    --transformer-impl transformer_engine
    --attention-backend auto
    --no-rope-fusion
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-router-fusion
    --use-distributed-optimizer
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl native
)
if [[ ${OPTIMIZER_CPU_OFFLOAD} == 1 ]]; then
    PERFORMANCE_ARGS+=(
        --optimizer-cpu-offload
        --optimizer-offload-fraction 1.0
        --use-precision-aware-optimizer
    )
fi
if [[ ${HYBRID_EP} == 1 ]]; then
    PERFORMANCE_ARGS+=(
        --moe-token-dispatcher-type flex
        --moe-flex-dispatcher-backend hybridep
        --moe-flex-dispatcher-num-sms 32
        --moe-hybridep-pad-uneven-dispatch-inputs
    )
    export USE_MNNVL=1
    export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=4
    export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
else
    PERFORMANCE_ARGS+=(
        --moe-token-dispatcher-type alltoall
    )
fi
# MXFP8 compute retains BF16 parameter storage/gather with CPU optimizer offload.
# GPU-only optimization also enables FP8 parameter gather and gradient-buffer reuse.
case "${PRECISION}" in
    bf16)
        export NVTE_CUTEDSL_FUSED_GROUPED_MLP=0
        export CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL=0
        ;;
    mxfp8)
        PERFORMANCE_ARGS+=(
            --fp8-format e4m3
            --fp8-recipe mxfp8
            --moe-use-grouped-tensor
            --use-transformer-engine-op-fuser
            --moe-mlp-glu-interleave-size 32
        )
        if [[ ${OPTIMIZER_CPU_OFFLOAD} == 0 ]]; then
            PERFORMANCE_ARGS+=(
                --fp8-param-gather
                --reuse-grad-buf-for-mxfp8-param-ag
            )
        fi
        export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
        export CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL=1
        ;;
    *)
        echo "PRECISION must be bf16 or mxfp8" >&2
        exit 2
        ;;
esac
if [[ ${FULL_RECOMPUTE} == 1 ]]; then
    PERFORMANCE_ARGS+=(
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers 1
    )
else
    PERFORMANCE_ARGS+=(
        --recompute-granularity selective
        --recompute-modules gdn_norm_out moe
    )
fi
if [[ ${DDP_OVERLAP} == 1 ]]; then
    PERFORMANCE_ARGS+=(
        --overlap-grad-reduce
        --overlap-param-gather
    )
fi
export NVTE_GROUPED_LINEAR_SINGLE_PARAM=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

# Checkpoints: initialize from scratch or a compatible checkpoint, or resume a run.
# Keep mode/precision-specific directories so unlike experiments cannot resume
# each other's optimizer state by accident.
RUN_NAME=qwen35_35B_1node_${MODE}_${INIT_MODE}_${PRECISION}_offload${OPTIMIZER_CPU_OFFLOAD}_l${NUM_LAYERS}_s${SEQ_LENGTH}_cap${MAX_SEQLEN_PER_RANK}_gbs${GLOBAL_BATCH_SIZE}
SAVE_DIR=${SAVE_DIR:-${WORKSPACE}/runs/${RUN_NAME}}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-${WORKSPACE}/models/qwen35_35B_1node/megatron}
RESUME=${RESUME:-0}
CHECKPOINT_ARGS=(
    --ckpt-format torch_dist
    --save "${SAVE_DIR}"
    --save-interval "${TRAIN_ITERS}"
)
if [[ ${RESUME} != 0 && ${RESUME} != 1 ]]; then
    echo "RESUME must be 0 or 1" >&2
    exit 2
fi
if [[ ${RESUME} == 1 ]]; then
    CHECKPOINT_ARGS+=(
        --load "${SAVE_DIR}"
        --exit-on-missing-checkpoint
        --use-checkpoint-opt-param-scheduler
    )
elif [[ -f ${SAVE_DIR}/latest_checkpointed_iteration.txt ]]; then
    echo "Checkpoint exists: use RESUME=1 or select a fresh SAVE_DIR" >&2
    exit 2
elif [[ ${INIT_MODE} == checkpoint ]]; then
    CHECKPOINT_ARGS+=(
        --load "${PRETRAINED_CHECKPOINT}"
        --exit-on-missing-checkpoint
        --finetune
        --no-load-optim
        --no-load-rng
    )
    # Initialize FP32 optimizer masters directly from the BF16 checkpoint,
    # avoiding a lossy detour through runtime MXFP8 model parameters.
    if [[ ${PRECISION} == mxfp8 && ${OPTIMIZER_CPU_OFFLOAD} == 0 ]]; then
        CHECKPOINT_ARGS+=(
            --load-main-params-from-ckpt
        )
    fi
elif [[ ${INIT_MODE} != scratch ]]; then
    echo "INIT_MODE must be scratch or checkpoint" >&2
    exit 2
fi

# Evaluation and logging use an independently prepared validation split.
EVAL_INTERVAL=${EVAL_INTERVAL:-25}
EVAL_ITERS=${EVAL_ITERS:-2}
EVAL_AND_LOGGING_ARGS=(
    --eval-interval "${EVAL_INTERVAL}"
    --eval-iters "${EVAL_ITERS}"
    --eval-global-batch-size "${GLOBAL_BATCH_SIZE}"
    --eval-micro-batch-size 1
    --log-interval 1
    --log-throughput
    --tensorboard-dir "${SAVE_DIR}/tensorboard"
    --distributed-timeout-minutes 30
)

# Use the selected container interpreter and this source checkout.
export PYTHONPATH=${MEGATRON_ROOT}
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
if [[ ${OPTIMIZER_CPU_OFFLOAD} == 1 ]]; then
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12}
else
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
fi
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
TORCHRUN_ARGS=(
    --standalone
    --nnodes 1
    --nproc-per-node 4
)
RUN_ARGS=(
    "${MODEL_ARGS[@]}"
    "${PARALLEL_ARGS[@]}"
    "${DATASET_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${PERFORMANCE_ARGS[@]}"
    "${CHECKPOINT_ARGS[@]}"
    "${EVAL_AND_LOGGING_ARGS[@]}"
    "$@"
)
echo "${MODE}: GBS=${GLOBAL_BATCH_SIZE} ${BATCH_UNIT}, MBS=1, TP1/PP1/base CP4/EP4"
echo "Layers=${NUM_LAYERS}+MTP1; context=${SEQ_LENGTH}; output=${SAVE_DIR}"
echo "Optimizer CPU offload=${OPTIMIZER_CPU_OFFLOAD}; OMP threads/rank=${OMP_NUM_THREADS}"
if [[ ${DRY_RUN} == 1 ]]; then
    printf '%q ' torchrun "${TORCHRUN_ARGS[@]}" "${MEGATRON_ROOT}/pretrain_gpt.py" "${RUN_ARGS[@]}"
    printf '\n'
    exit 0
fi
mkdir -p "${SAVE_DIR}"
cd "${MEGATRON_ROOT}"
exec torchrun "${TORCHRUN_ARGS[@]}" pretrain_gpt.py "${RUN_ARGS[@]}"
