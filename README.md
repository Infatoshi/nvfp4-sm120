# nvfp4-sm120

Full **NVFP4 training** (NVIDIA's recipe, [arXiv 2509.25149](https://arxiv.org/abs/2509.25149)
— 2D weight scaling + Random Hadamard Transform + stochastic rounding) running on
**native FP4 tensor cores** on **sm_120** (RTX PRO 6000 Blackwell Workstation / GeForce
Blackwell), where NVIDIA's Transformer Engine cannot.

## Why this exists

On sm_120, TE 2.15's fused NVFP4 path crashes: its RHT/SR mega-kernel requests dynamic
shared memory sized for the sm_100 datacenter budget (~232 KB), exceeding sm_120's
101376-byte opt-in cap (`cudaFuncSetAttribute` → "invalid argument"). sm_120 also lacks
the hardware stochastic-rounding cast `cvt.rs.*.e2m1` (ptxas: *"Feature '.rs' not
supported"*). Filed upstream as [NVIDIA/TransformerEngine#3062](https://github.com/NVIDIA/TransformerEngine/issues/3062).

**Approach: decompose.** Own the quantization (software SR + RHT, in our own kernels) and
send the quantized operands to a native FP4 GEMM via `torch._scaled_mm`
(BlockWise1x16 + SWIZZLE_32_4_4, through torchao's `_addmm_nvfp4_dispatch`). sm_120 *does*
have the round-to-nearest E2M1 cast and the FP4 tensor-core GEMM — those are all we need.

## Layout

```
nvfp4_validate.py        Reference fake-quant NVFP4 sim (all 4 techniques) + addition dataset.
nvfp4_gemm.py            quant_nvfp4 (two-level E2M1/E4M3 quant) + fp4_matmul (torch._scaled_mm).
nvfp4_triton_quant.py    Fused Triton quantizer: scale + SR-round + pack + in-kernel RHT.
nvfp4_cuda.py            CUDA quantizer using the hardware E2M1 cast (fastest); + fp4_mm_preqB.
nvfp4_train.py           FP4Linear autograd (Fprop RNE / Dgrad SR / Wgrad RHT+SR) + char-LM trainer.
nvfp4_triton.py          Standalone Triton software-SR quantizer (the first correctness probe).
benchmarks/              Throughput + roofline profiling scripts.
tests/                   Numerical-correctness and convergence checks.
te/                      Transformer Engine comparison: TE trainer, benches, sm_120 degrade patch.
```

Backend for `nvfp4_train.py` is selected by env var (default: pure-torch quant):
`NVFP4_CUDA=1` (hardware-cast CUDA, fastest) · `NVFP4_FUSED=1` (fused Triton) ·
`NVFP4_AMORTIZE=1` (with `NVFP4_CUDA=1`, cache per-step weight quant).

## Quickstart

```bash
# sm_120 box with CUDA 13 toolkit (nvcc), torch+cu13, torchao, triton
python nvfp4_validate.py                 # reference sim sanity
NVFP4_CUDA=1 python nvfp4_train.py --steps 1500   # real FP4 training (JIT-builds the CUDA ext)
python tests/cuda_val.py                 # numerics: byte-match, SR unbiasedness, GEMM rel-err
python benchmarks/cuda_prof.py           # FP4 linear vs bf16 timing
```

## Results (RTX PRO 6000 Blackwell, sm_120; all read from green runs)

**Numerics.** Custom quant matches the reference bit-for-bit on round-to-nearest
(byte-match ~100%), the FP4 GEMM holds ~0.134 relative error vs fp32, and stochastic
rounding is unbiased (mean over 32 draws drops error 0.134 → ~0.038).

**GEMM throughput** (`benchmarks/nvfp4_bench.py`): native FP4 sustains ~1100 TFLOPS at
16384³ — **~3x bf16**, ~55% of the dense FP4 peak (the gap is sm_120 cuBLAS maturity).

**Training** (3.55M-param Nemotron-style Transformer, 3-digit-addition char-LM with
held-out pairs as a generalization probe). The full recipe on real FP4 matches bf16;
without SR/RHT it stalls — the stabilizers are load-bearing:

| config | held-out acc | val loss |
|---|---|---|
| bf16 reference | 100% | 0.965 |
| **NVFP4 full recipe (SR+RHT), all blocks** | **100%** | **0.965** |
| NVFP4 without SR/RHT (the only path TE runs on sm_120) | 1.9% (stalls) | 1.34 |

**Quantizer speed @ 8192×4096** (the wrapper bottleneck was the per-tensor amax in fp32
+ a `.item()` host sync — 86% of quant time — not the rounding; fixed with bf16 amax +
on-device scale):

| quantizer | time | bandwidth |
|---|---|---|
| pure torch | 3.7 ms | 23 GB/s |
| fused Triton | 0.45 ms | ~195 GB/s |
| **hardware-cast CUDA** | **0.12 ms** | **~730 GB/s** |

**FP4 linear vs bf16** (hardware-cast CUDA quant + in-kernel RHT, `benchmarks/cuda_prof.py`):
fprop 1.0–1.4x; wgrad (SR+RHT) 0.80–1.31x (≥bf16 at 8k). Both GEMMs meet-or-beat bf16 at
scale; the remaining per-call quant-wrapper overhead (amax + swizzle + alloc) is what keeps
the *full* layer from a universal win — fusing those is the next lever.

**Weight-quant amortization** (`NVFP4_AMORTIZE=1`): cache the weight's two quant
orientations (fprop contracts K = `quant(w)`; dgrad contracts N = `quant(w.t())`; both RNE,
so deterministic) once per optimizer step, keyed on a counter bumped by `mark_step()` after
`opt.step` (mandatory — a parity check confirms outputs go stale without it). On a
microbatched FFN step (d4096, h14336) this cut 437→122 ms at G=8 and 875→241 ms at G=16;
training still converges to bf16 parity. Note the large factor is partly because the
non-amortized baseline re-quantizes the weight on every fprop+dgrad across all G
microbatches — a real microbatched win, but G-coupled, not universal.

## Gotchas

- **Do not `LD_PRELOAD` the system cuBLASLt.** That is the *Transformer Engine* workaround
  (see `te/te_env.sh`); it breaks `torch._scaled_mm` with `CUBLAS_STATUS_NOT_INITIALIZED`.
  This pipeline needs no TE and no special environment.
- The CUDA extension (`nvfp4_cuda.py`) JIT-builds via `load_inline` and needs a CUDA 13
  toolkit: `CUDA_HOME=/usr/local/cuda-13`, gencode `compute_120a,code=sm_120a`, and
  `#include <ATen/cuda/CUDAContext.h>`. First build takes a couple minutes (cached after).
- torchao's `F.linear`/`torch.mm` on two `NVFP4Tensor`s silently takes a dequant path
  (correct numerics, *not* FP4 tensor cores). Call `_addmm_nvfp4_dispatch` for the real GEMM.

## Transformer Engine comparison (`te/`)

`te/train_te.py` runs the same task through TE's `NVFP4BlockScaling`. On sm_120 TE can only
run the degraded (no SR/RHT) path; `te/nvfp4_sm120_degrade.patch` makes TE auto-disable
RHT/SR with a warning instead of crashing. These scripts require Transformer Engine and
must be run with `te/te_env.sh` sourced (which is why they are isolated here — that env
breaks the main pipeline).
