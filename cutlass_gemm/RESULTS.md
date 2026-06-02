# SM120 NVFP4xNVFP4 GEMM (CUTLASS 4.5.0) -- square shapes

GPU: RTX PRO 6000 Blackwell Workstation (sm_120, cc 12.0). Dense FP4 SOL = 2000 TFLOPS.
CUTLASS: /home/infatoshi/cuda/engines/cutlass (tag v4.5.0). Base example: 79b_blackwell_geforce_nvfp4_nvfp4_gemm.
ArchTag cutlass::arch::Sm120, OpClassBlockScaledTensorOp. NVFP4: e2m1 4-bit data, ue4m3 per-16 block scale, vector size 16.
GeForce SM120 constraints (hard): ClusterShape must be 1x1x1 (no TMA multicast); MMA atom is SM120_16x8x64 block-scaled.
Output D is FP4 (e2m1) with generated SFD scale factors (matches the example; this is a chained-GEMM style output).

Source : nvfp4_gemm.cu  (macro-parameterized tile + schedule; GEMM-only timing loop: 20-iter warmup, then N CUDA-event-timed gemm.run() calls, no per-iter initialize()).
Build  : ./build.sh TILE_M TILE_N TILE_K SCHED   (SCHED 0=Pingpong, 1=Cooperative)
Verify : exact bitwise TensorEquals vs CUTLASS host reference Gemm3x (block-scaled). All runs report Disposition: Passed.

## IMPORTANT compile finding
Only the 128x128x128 MMA tile compiles for this SM120 NVFP4 path. Every 256-dim tile
(256x*, *x256x*, *x*x256) and the Cooperative schedule FAIL with a hard static_assert in
MainloopSm120TmaWarpSpecializedBlockScaled (the SM120 block-scaled mainloop only supports the
128x128x128 tile + Pingpong here). So the winning config is forced, not chosen:
  tile 128x128x128, cluster 1x1x1, KernelTmaWarpSpecializedPingpong.

## Verified results (GPU idle, serialized, 100 iters, median via CUDA events)
  Shape     CUTLASS-FP4 TFLOPS   %SOL    cuBLAS-FP4 baseline   delta
  4096^3    1419.97              71.0%   1010.1                +40.6%
  8192^3    1594.81              79.7%   1110.9                +43.6%
  16384^3   1577.34              78.9%   1136.6                +38.8%

All three square shapes BEAT the cuBLAS FP4 baseline by ~39-44%.
8192 and 16384 are essentially AT the 80% SOL target (1600 TFLOPS); 4096 at 71%.
(Note: numbers under GPU contention drop sharply, e.g. a contended 16384 read 1148 TFLOPS.
 Always benchmark with the GPU idle: nvidia-smi should show 0% util and no compute-apps.)

## Reproduce
  cd /home/infatoshi/experiments/_scratch/nvfp4-validate/cutlass_gemm
  bash build.sh 128 128 128 0
  unset LD_PRELOAD; export CUDA_HOME=/usr/local/cuda-13
  ./nvfp4_gemm_128x128x128_s0 --m=16384 --n=16384 --k=16384 --iterations=100

## Next steps to push higher
- The 128x128x128 tile is the only legal MMA tile for SM120 NVFP4; to go further, tune
  StageCount (pipeline depth), epilogue schedule, and consider a cheaper epilogue (bf16 output,
  no SFD generation -> see 79a_blackwell_geforce_nvfp4_bf16_gemm) which removes the FP4+scale
  output cost and may lift 4096 toward 8192's efficiency.
- Try the CUTLASS profiler autotuner for SM120 blockscaled kernels.
