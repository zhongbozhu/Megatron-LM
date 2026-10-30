#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Run once per node in an existing environment with TE, HybridEP and Qwen3.5 support.
set -euo pipefail

: "${MIMO_MANIFEST:?Set MIMO_MANIFEST to the image-conversation manifest}"
: "${MIMO_HF_MODEL:?Set MIMO_HF_MODEL to the local Qwen3.5 tokenizer/processor snapshot}"
: "${MIMO_OUTPUT_DIR:?Set MIMO_OUTPUT_DIR to the native checkpoint directory}"

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
layout=${MIMO_LAYOUT:-dcp}
cp_size=${CP_SIZE:-8}
source_length=${SEQ_LENGTH:-40960}
rank_capacity=${MAX_SEQLEN_PER_RANK:-8192}
layout_args=()
case "${layout}" in
    dcp)
        layout_args=(--dynamic-context-parallel --min-dynamic-context-parallel-size 1
                     --sequence-packing-scheduler default_dynamic_cp)
        ;;
    static_cp) layout_args=(--sequence-packing-scheduler dp_balanced) ;;
    cp1)
        cp_size=1
        rank_capacity=${MAX_SEQLEN_PER_RANK:-${source_length}}
        layout_args=(--sequence-packing-scheduler dp_balanced)
        ;;
    *) echo "MIMO_LAYOUT must be dcp, static_cp, or cp1" >&2; exit 2 ;;
esac

entrypoint=examples/mimo/train.py
placement_args=(--dataset-provider qwen35_native)
case "${MIMO_PLACEMENT:-colocated}" in
    colocated) ;;
    disagg)
        entrypoint=examples/mimo/pretrain_mimo.py
        encoder_dp=${ENCODER_DP:-4}
        decoder_ranks=$(( ${NNODES:-4} * ${GPUS_PER_NODE:-4} - encoder_dp ))
        if (( encoder_dp <= 0 || decoder_ranks <= 0 || cp_size <= 0 || decoder_ranks % cp_size )); then
            echo "Disagg requires nonempty E/D pools and decoder ranks divisible by CP_SIZE" >&2
            exit 2
        fi
        placement_args=(--mimo-packed --mimo-encoder-tp 1 --mimo-encoder-dp "${encoder_dp}"
                        --mimo-llm-offset "${encoder_dp}" --mimo-llm-tp 1 --mimo-llm-pp 1
                        --mimo-llm-cp "${cp_size}" --mimo-llm-dp "$((decoder_ranks / cp_size))"
                        --mimo-llm-ep "${EP_SIZE:-16}" --mimo-llm-expt-tp 1)
        ;;
    *) echo "MIMO_PLACEMENT must be colocated or disagg" >&2; exit 2 ;;
esac

checkpoint_args=(--save "${MIMO_OUTPUT_DIR}" --save-interval "${SAVE_INTERVAL:-25}")
if [[ -n ${MIMO_LOAD:-} ]]; then
    checkpoint_args+=(--load "${MIMO_LOAD}" --exit-on-missing-checkpoint
                      --use-checkpoint-opt-param-scheduler)
else
    : "${MIMO_PRETRAINED_CHECKPOINT:?Set the initial Bridge torch_dist iteration directory}"
    checkpoint_args+=(--mimo-pretrained-checkpoint "${MIMO_PRETRAINED_CHECKPOINT}")
fi
if [[ -n ${EXIT_INTERVAL:-} ]]; then
    checkpoint_args+=(--exit-interval "${EXIT_INTERVAL}")
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${repo_root}"
exec "${PYTHON:-python}" -u -m torch.distributed.run \
    --nnodes "${NNODES:-4}" --nproc-per-node "${GPUS_PER_NODE:-4}" \
    --node-rank "${NODE_RANK:-0}" --master-addr "${MASTER_ADDR:-localhost}" \
    --master-port "${MASTER_PORT:-29500}" --max-restarts 0 \
    "${entrypoint}" --model-provider qwen35_native "${placement_args[@]}" \
    --mimo-manifest "${MIMO_MANIFEST}" --mimo-hf-model "${MIMO_HF_MODEL}" \
    --mimo-heldout-images "${HELDOUT_IMAGES:-512}" \
    --mimo-data-split-seed "${SPLIT_SEED:-20261002}" --mimo-vision-recompute \
    --num-layers 40 --hidden-size 2048 --ffn-hidden-size 8192 \
    --num-attention-heads 16 --group-query-attention --num-query-groups 2 --kv-channels 256 \
    --experimental-attention-variant gdn --linear-attention-freq 4 \
    --linear-num-key-heads 16 --linear-num-value-heads 32 \
    --linear-key-head-dim 128 --linear-value-head-dim 128 --linear-conv-kernel-dim 4 \
    --linear-cp-mode headwise --linear-cp-layout zigzag --attention-cp-layout zigzag \
    --normalization RMSNorm --norm-epsilon 1e-6 --apply-layernorm-1p \
    --swiglu --disable-bias-linear --qk-layernorm --attention-output-gate \
    --num-experts 256 --moe-ffn-hidden-size 512 --moe-router-topk 8 \
    --moe-shared-expert-intermediate-size 512 --moe-shared-expert-gate \
    --moe-router-load-balancing-type global_aux_loss --moe-aux-loss-coeff 0.001 \
    --moe-router-dtype fp32 --moe-token-dispatcher-type flex \
    --moe-flex-dispatcher-backend hybridep --moe-flex-dispatcher-num-sms 32 \
    --moe-hybridep-pad-uneven-dispatch-inputs --moe-grouped-gemm \
    --moe-permute-fusion --moe-router-fusion \
    --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1 \
    --padded-vocab-size 248320 --position-embedding-type mrope --mrope-section 11 11 10 \
    --rotary-base 10000000 --rotary-percent 0.25 --no-rope-fusion \
    --untie-embeddings-and-output-weights --init-method-std 0.02 \
    --hidden-dropout 0 --attention-dropout 0 \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model "${MIMO_HF_MODEL}" \
    --bf16 --transformer-impl transformer_engine --attention-backend auto \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 --expert-model-parallel-size "${EP_SIZE:-16}" \
    --context-parallel-size "${cp_size}" --max-seqlen-per-dp-cp-rank "${rank_capacity}" \
    --seq-length "${source_length}" --max-position-embeddings 131072 \
    --use-varlen-dataset --pad-packed-seq-alignment 32 --calculate-per-token-loss \
    --dataloader-type external --no-create-attention-mask-in-dataloader \
    --micro-batch-size 1 --global-batch-size "${GLOBAL_BATCH_SIZE:-64}" \
    --train-iters "${TRAIN_ITERS:-50}" --eval-iters "${EVAL_ITERS:-0}" \
    --eval-interval 1000 --eval-global-batch-size "${GLOBAL_BATCH_SIZE:-64}" \
    --eval-micro-batch-size 1 --num-workers 0 \
    --use-distributed-optimizer --accumulate-allreduce-grads-in-fp32 --optimizer adam \
    --lr 2e-5 --min-lr 2e-6 --lr-warmup-iters 5 --lr-decay-iters 300000 \
    --lr-decay-style cosine --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
    --clip-grad 1.0 --weight-decay 0.033 --seed "${SEED:-1234}" --rerun-mode disabled \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --manual-gc --manual-gc-interval 20 --empty-unused-memory-level 1 \
    --ckpt-format torch_dist --log-interval 1 --log-throughput --log-num-zeros-in-grad \
    --distributed-timeout-minutes 30 --exit-signal-handler \
    "${layout_args[@]}" "${checkpoint_args[@]}" "$@"
