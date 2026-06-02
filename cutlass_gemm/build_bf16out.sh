#!/usr/bin/env bash
# Standalone SM120 NVFP4xNVFP4 GEMM with BF16 output (no SFD generation).
# Usage: ./build_bf16out.sh TILE_M TILE_N TILE_K SCHED [STAGES]
#   -> binary nvfp4_gemm_bf16out_<TM>x<TN>x<TK>_s<SCHED>_st<STAGES>
# SCHED: 0=Pingpong 1=Cooperative 2=Auto. STAGES: 0=AutoCarveout, >0=fixed StageCount<N>.
# Cluster is fixed 1x1x1 (GeForce SM120 constraint).
set -euo pipefail
unset LD_PRELOAD || true
export CUDA_HOME=/usr/local/cuda-13
NVCC=/usr/local/cuda-13/bin/nvcc
CUT=/home/infatoshi/cuda/engines/cutlass
DIR=/home/infatoshi/experiments/_scratch/nvfp4-validate/cutlass_gemm
TM=${1:-128}; TN=${2:-128}; TK=${3:-128}; SC=${4:-0}; ST=${5:-0}
OUT=$DIR/nvfp4_gemm_bf16out_${TM}x${TN}x${TK}_s${SC}_st${ST}
cd "$DIR"
$NVCC nvfp4_gemm_bf16out.cu -o "$OUT" \
  -std=c++17 -O3 \
  -gencode arch=compute_120a,code=sm_120a \
  --expt-relaxed-constexpr --expt-extended-lambda \
  --threads 0 \
  -I$CUT/include \
  -I$CUT/tools/util/include \
  -I$CUT/examples/common \
  -DCFG_TILE_M=$TM -DCFG_TILE_N=$TN -DCFG_TILE_K=$TK -DCFG_SCHED=$SC -DCFG_STAGES=$ST
echo "BUILT $OUT"
