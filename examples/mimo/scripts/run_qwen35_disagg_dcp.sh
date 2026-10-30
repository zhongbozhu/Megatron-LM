#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Reuse the colocated model/data configuration with disjoint E4/D16 pools.
set -euo pipefail
export MIMO_PLACEMENT=disagg
export NNODES=${NNODES:-5}
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_qwen35_colocated_dcp.sh" "$@"
