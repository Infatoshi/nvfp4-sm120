# DEVLOG

The journey of getting NVFP4 (4-bit float) training to work, and to run fast, on
an RTX PRO 6000 Blackwell Workstation (sm_120, compute cap 12.0) where NVIDIA's
Transformer Engine does not. Newest entries at the bottom. Numbers here were read
from commands that exited 0; anything not yet measured is marked PENDING.

Hardware: RTX PRO 6000 Blackwell Workstation 96GB, sm_120, 600W. Host "anvil-lan"
(Ryzen 9 9950X3D, 96GB DDR5, Ubuntu 24.04). torch 2.11.0+cu130, torchao 0.17.0,
triton 3.6.0, CUDA 13.2 toolkit (nvcc V13.2.51), driver 610.43.02. CUTLASS v4.5.0
at /home/infatoshi/cuda/engines/cutlass (external dependency, not vendored here).

---

## 1. The paper and the four stabilizers

Started from NVIDIA's NVFP4 pre-training recipe (arXiv 2509.25149). NVFP4 is E2M1
(1 sign / 2 exp / 1 mantissa, values +/- {0, .5, 1, 1.5, 2, 3, 4, 6}), block size 16
along K, a per-block E4M3 scale, and a per-tensor FP32 scale (two-level scaling).
Naive 4-bit training diverges; the paper's four load-bearing techniques are:

1. Selective high-precision layers (~15%, weighted to the end of the network).
2. Random Hadamard Transform (16x16) on the Wgrad cast, to spread outliers.
3. 2D weight scaling so forward and backward see consistent quantization.
4. Stochastic rounding on gradients (unbiased), instead of round-to-nearest.

## 2. Reference sim and the addition probe

Built a fake-quant NVFP4 simulator implementing all four techniques
(`nvfp4_validate.py`) as a correctness oracle, then a downscaled Nemotron-style
decoder trained on a 3-digit-addition char-LM with held-out pairs as a
generalization probe. Verified result:

| config | held-out acc | val loss |
|---|---|---|
| bf16 reference | 100% | 0.965 |
| NVFP4 full recipe (SR+RHT), all blocks | 100% | 0.965 |
| NVFP4 without SR/RHT (the only path TE runs on sm_120) | 1.9% (stalls) | 1.34 |

The stabilizers are not optional: drop SR+RHT and the model collapses. This is the
ablation that answers "is the match real or just an under-trained tie" - at this
scale the recipe is clearly the difference between converging and stalling.

## 3. Transformer Engine does not run on sm_120

TE 2.15's fused NVFP4 path crashes on sm_120. Two root causes, both found by
bisection:
- Its RHT/SR mega-kernel requests dynamic shared memory sized for the sm_100
  datacenter budget (~232 KB), exceeding sm_120's 101376-byte opt-in cap
  (`cudaFuncSetAttribute` returns "invalid argument").
- sm_120 has no hardware stochastic-rounding cast: `cvt.rs.*.e2m1` is rejected by
  ptxas ("Feature '.rs' not supported"). sm_120 DOES have the round-to-nearest
  cast `__nv_cvt_float2_to_fp4x2` and the native FP4 tensor-core GEMM.

Filed upstream as NVIDIA/TransformerEngine#3062. `te/nvfp4_sm120_degrade.patch`
makes TE auto-disable RHT/SR with a warning instead of crashing (degraded path only).

## 4. Decompose: own the quant, borrow the GEMM

Since TE will not run and sm_120 lacks the hardware SR cast, the approach is to
decompose the problem: write our own quantization (software SR + in-kernel RHT +
two-level scaling + nibble packing) and hand the quantized operands to a native
FP4 GEMM via `torch._scaled_mm` (cuBLASLt, BlockWise1x16 + SWIZZLE_32_4_4, through
torchao's `_addmm_nvfp4_dispatch`).

What is from scratch: the quantizers. Triton (`nvfp4_triton_quant.py`,
`_quant_pack_v2`) and a faster CUDA kernel (`nvfp4_cuda.py`,
`quant_kernel<DO_SR, DO_RHT>`) using the hardware E2M1 cast intrinsic plus a
wang-hash software SR (since the hardware SR cast is unavailable). Plus the
autograd `FP4Linear` (Fprop RNE / Dgrad SR / Wgrad RHT+SR) and weight-quant
amortization.

What is borrowed: the FP4 matmul itself (cuBLASLt via `torch._scaled_mm`) and the
hardware float->FP4 cast intrinsic. Writing a competitive FP4 GEMM was deferred
(see section 7).

Quantizer speed @ 8192x4096 (verified): the bottleneck was the per-tensor amax in
fp32 plus a `.item()` host sync (86% of quant time), not the rounding. Fixed with
bf16 amax and on-device scale.

| quantizer | time | bandwidth |
|---|---|---|
| pure torch | 3.7 ms | 23 GB/s |
| fused Triton | 0.45 ms | ~195 GB/s |
| hardware-cast CUDA | 0.12 ms | ~730 GB/s |

Gotcha worth remembering: torchao's `F.linear` / `torch.mm` on two NVFP4Tensors
silently takes a dequant path (correct numerics, NOT the FP4 tensor cores). You
must call `_addmm_nvfp4_dispatch` for the real GEMM. Separately, do NOT LD_PRELOAD
the system cuBLASLt (that is the TE workaround); it breaks `torch._scaled_mm` with
CUBLAS_STATUS_NOT_INITIALIZED.

## 5. Scaling-law study: NVFP4 generalizes like BF16 on real text

To answer "does this generalize beyond a toy task," ran a Chinchilla-style
iso-token sweep on OpenWebText (GPT-2 BPE, nanoGPT-style decoder: RMSNorm, GQA,
RoPE, squared-ReLU FFN, weight-tied head). Fitted L(N) = E + A * N^(-alpha).
Verified fits:
- BF16:  L = 5.53 + 745 * N^(-0.500)
- NVFP4: L = 5.52 + 823 * N^(-0.500)

Same exponent, indistinguishable curves across the sweep (figure in
`results/scaling_law.png`). Honest caveat recorded here too: this is small and
under-trained (~25M params, ~20-25M tokens/model). The decisive test is
convergence-scale, which has NOT been run. At this scale part of the "match" is
simply that neither model is trained far enough for FP4 error to dominate; what we
can claim is that the full recipe holds parity and the stabilizer ablation is
decisive.

## 6. The 80%-SOL goal: build a real CUTLASS FP4 GEMM for sm_120

Goal: push every NVFP4 GEMM call we make to 80%+ of the dense FP4 speed-of-light.

SOL definition: NVIDIA's datasheet lists 4000 AI TOPS for this card "using
sparsity" (2:4). Dense FP4 (no sparsity), which is what we run, is half of that:
2000 TFLOPS. That is the 100%-SOL figure. (A compute-clock derivation gives
188 SM x 3090 MHz x 4096 FP4 FLOP/SM/clk = 2379 TFLOPS as an upper bound assuming
sustained max boost; the datasheet-derived 2000 is the realistic dense ceiling.)

Baseline (cuBLAS FP4 via `torch._scaled_mm`, gemm-only, GPU idle, verified):
- 16384^3: 1136.6 TFLOPS = 56.8% SOL  (best case)
- 8192^3:  1110.9 = 55.5%
- 4096^3:  1010.1 = 50.5%
- The six real training shapes are far worse: 6.3% to 33.0% SOL (skinny/K-heavy).

So cuBLAS leaves a large gap, especially on the training shapes. Built a standalone
CUTLASS NVFP4xNVFP4 GEMM from example 79b_blackwell_geforce_nvfp4_nvfp4_gemm,
ArchTag cutlass::arch::Sm120, OpClassBlockScaledTensorOp, e2m1 data + e4m3 per-16
block scale. Source: `cutlass_gemm/nvfp4_gemm.cu`, build via
`cutlass_gemm/build.sh TM TN TK SCHED`. Every benchmarked config is bitwise-verified
against the CUTLASS host reference (Disposition: Passed) before its number counts.

Hard sm_120 constraints discovered the hard way:
- ClusterShape MUST be 1x1x1 (GeForce SM120 has no TMA multicast).
- ONLY the 128x128x128 MMA tile + Pingpong schedule COMPILES. Every 256-dim tile
  and the Cooperative schedule fail with a hard static_assert in
  MainloopSm120TmaWarpSpecializedBlockScaled. So the winning config is forced, not
  chosen.

Verified square results (GPU idle, 100 iters, median via CUDA events):

| shape | CUTLASS FP4 TFLOPS | %SOL | cuBLAS baseline | delta |
|---|---|---|---|---|
| 4096^3  | 1419.97 | 71.0% | 1010.1 | +40.6% |
| 8192^3  | 1594.81 | 79.7% | 1110.9 | +43.6% |
| 16384^3 | 1577.34 | 78.9% | 1136.6 | +38.8% |

All three beat cuBLAS by ~39-44%. 8192 and 16384 are essentially at the 80% target;
4096 trails at 71%. IMPORTANT measurement gotcha: numbers collapse under GPU
contention (a contended 16384 read 1148 TFLOPS), so benchmark only with nvidia-smi
showing 0% util and no compute apps.

## 7. In progress / PENDING (where to resume)

The goal is NOT yet met. Two threads were running and were stopped when the GPU was
needed for other work; their kernels are built and on disk but their numbers were
never captured to a file, so they are PENDING, not verified.

PENDING-A, push squares over 80% (cheaper epilogue): built a BF16-output variant
from example 79a_blackwell_geforce_nvfp4_bf16_gemm (NVFP4 inputs, BF16 output, no
SFD scale-factor generation) - cheaper epilogue, and more realistic for training
since the GEMM result feeds the next op in higher precision. Source:
`cutlass_gemm/nvfp4_gemm_bf16out.cu`, build via `build_bf16out.sh TM TN TK SCHED
[STAGES]`. Binaries built for stage counts 0/4/6/8 and Pingpong/Cooperative. NOT
yet benchmarked. Hypothesis: should lift 4096 toward 8192's efficiency and may push
8192/16384 over 80%.

PENDING-B, the six real training shapes (the actual point of the goal). These are
skinny/K-heavy and were NEVER tuned past the cuBLAS baseline. They are:

| M | N | K | produced by | cuBLAS %SOL |
|---|---|---|---|---|
| 16384 | 512  | 512   | q/k/v/o fprop+dgrad | 11.5% |
| 16384 | 512  | 2048  | down.fprop, up.dgrad | 33.0% |
| 16384 | 2048 | 512   | up.fprop, down.dgrad | 17.6% |
| 512   | 512  | 16384 | q/k/v/o wgrad | 6.3% |
| 512   | 2048 | 16384 | down.wgrad | 24.4% |
| 2048  | 512  | 16384 | up.wgrad | 24.7% |

CRITICAL open question for these: roofline first. Several are tiny (512x512x16384 is
only 8.6 GFLOP) and are likely bandwidth-bound, not compute-bound. For a
bandwidth-bound shape, 80% of the 2000-TFLOPS compute peak is physically impossible;
the honest ceiling is the memory-bandwidth roofline (~1.79 TB/s GDDR7) and the
target should be 80% of THAT, reported as %-of-BW-roofline. Do not chase a compute
SOL the roofline forbids. Skinny tiles (128x32, 128x64, 256x128) were built to try
to improve occupancy on these shapes but are NOT yet benchmarked. Split-K / stream-K
along the huge K=16384 dimension is the other untried lever.

Resume checklist:
1. `ssh anvil-lan`, confirm GPU idle (nvidia-smi 0% util). Note: GPU clock may be
   locked to 3090 MHz from a prior session; reset with `sudo nvidia-smi -rgc` if not
   benchmarking, or leave locked for stable numbers.
2. Benchmark PENDING-A (bf16out variants) at 4096/8192/16384, append a verified
   "## Square-shape push" section to `cutlass_gemm/RESULTS.md`.
3. Roofline each of the six training shapes, classify compute- vs bandwidth-bound,
   then benchmark/tune (skinny tiles, split-K), append "## Training shapes" to
   RESULTS.md with %-of-applicable-roofline clearly labeled.
4. Only count a number if its command exited 0 and Disposition: Passed.
