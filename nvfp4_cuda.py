"""Hardware-cvt NVFP4 quantizer for sm_120 (CUDA extension).

Replaces the Triton tl.where E2M1 rounding chains (compute-bound, ~195 GB/s) with
the hardware E2M1 conversion intrinsic `__nv_cvt_float2_to_fp4x2` (RNE) and the
hardware E4M3 conversion for block scales. Software stochastic rounding is kept for
the gradient path (sm_120 has no hardware cvt.rs). Emits the exact torchao
NVFP4Tensor layout (packed float4_e2m1fn_x2 + E4M3 1x16 scales) so it drops into
the existing torch._scaled_mm GEMM.

Build: nvcc load_inline, arch sm_120a (cuda_fp4.h cvt instrs are arch-specific).
"""
import os, math, torch
from torch.utils.cpp_extension import load_inline
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor, hp_data_dims_to_swizzled_scale_dims_nvfp4, _addmm_nvfp4_dispatch)

F4_MAX, F8E4M3_MAX, E4M3_EPS, BLK = 6.0, 448.0, 0.015625, 16
_MM = torch.ops.aten.mm.default
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13")

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

// counter-based RNG (wang hash) -> uniform [0,1)
__device__ __forceinline__ float u01(unsigned int a, unsigned int b){
    unsigned int x = a * 0x9e3779b9u + b + 0x85ebca6bu;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return (x >> 8) * (1.0f / 16777216.0f);   // 24-bit mantissa -> [0,1)
}

// software stochastic-rounding E2M1 magnitude code (0..7) over non-uniform levels
__device__ __forceinline__ int e2m1_sr_code(float m, float u){
    float lo, hi; int loi;
    if(m < 0.5f){lo=0.f;hi=0.5f;loi=0;}
    else if(m<1.0f){lo=0.5f;hi=1.0f;loi=1;}
    else if(m<1.5f){lo=1.0f;hi=1.5f;loi=2;}
    else if(m<2.0f){lo=1.5f;hi=2.0f;loi=3;}
    else if(m<3.0f){lo=2.0f;hi=3.0f;loi=4;}
    else if(m<4.0f){lo=3.0f;hi=4.0f;loi=5;}
    else if(m<6.0f){lo=4.0f;hi=6.0f;loi=6;}
    else return 7;
    float p = (m - lo) / (hi - lo);
    return (u < p) ? (loi + 1) : loi;
}

template<bool DO_SR, bool DO_RHT>
__global__ void quant_kernel(const __nv_bfloat16* __restrict__ x,
                             uint8_t* __restrict__ data,
                             uint8_t* __restrict__ scale,
                             const float* __restrict__ pts_ptr,
                             const float* __restrict__ h_ptr,   // 16x16 Hadamard (row-major)
                             const float* __restrict__ sg_ptr,  // 16 signs
                             long NBLK, long NUMEL, unsigned int seed){
    long b = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if(b >= NBLK) return;
    long base = b * 16;
    const __nv_bfloat16* xb = x + base;
    float v[16];
    #pragma unroll
    for(int i=0;i<16;i++)
        v[i] = (base + i < NUMEL) ? __bfloat162float(xb[i]) : 0.f;

    if(DO_RHT){                          // v' = (v * signs) @ H16  (per 16-block)
        float t[16];
        #pragma unroll
        for(int i=0;i<16;i++) t[i] = v[i] * sg_ptr[i];
        #pragma unroll
        for(int j=0;j<16;j++){
            float acc = 0.f;
            #pragma unroll
            for(int i=0;i<16;i++) acc += t[i] * h_ptr[i*16 + j];
            v[j] = acc;
        }
    }

    float amax = 0.f;
    #pragma unroll
    for(int i=0;i<16;i++){ float a = fabsf(v[i]); amax = a>amax?a:amax; }

    float s_enc = 1.0f / pts_ptr[0];     // pts read on-device (no host sync)
    float sbs = fminf(fmaxf((amax / 6.0f) * s_enc, 0.015625f), 448.0f);
    __nv_fp8_e4m3 sbs8(sbs);
    scale[b] = *reinterpret_cast<uint8_t*>(&sbs8);
    float recip = s_enc / float(sbs8);            // (1/pts)/sbs_e4m3

    uint8_t* out = data + b * 8;
    #pragma unroll
    for(int j=0;j<8;j++){
        float d0 = fminf(fmaxf(v[2*j]   * recip, -6.f), 6.f);
        float d1 = fminf(fmaxf(v[2*j+1] * recip, -6.f), 6.f);
        if(DO_SR){
            float u0 = u01(seed, (unsigned)(b*16 + 2*j));
            float u1 = u01(seed, (unsigned)(b*16 + 2*j + 1));
            int c0 = e2m1_sr_code(fabsf(d0), u0) | (d0 < 0.f ? 8 : 0);
            int c1 = e2m1_sr_code(fabsf(d1), u1) | (d1 < 0.f ? 8 : 0);
            out[j] = (uint8_t)(c0 | (c1 << 4));
        } else {
            __nv_fp4x2_storage_t p = __nv_cvt_float2_to_fp4x2(make_float2(d0, d1),
                                                              __NV_E2M1, cudaRoundNearest);
            out[j] = (uint8_t)p;     // low nibble = d0, high nibble = d1
        }
    }
}

void quant_launch(torch::Tensor x, torch::Tensor data, torch::Tensor scale,
                  torch::Tensor pts, torch::Tensor H, torch::Tensor signs,
                  int64_t NBLK, int64_t seed, bool do_sr, bool do_rht){
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    int threads = 256;
    long blocks = (NBLK + threads - 1) / threads;
    long numel = x.numel();
    auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    auto dp = data.data_ptr<uint8_t>();
    auto sp = scale.data_ptr<uint8_t>();
    auto pp = pts.data_ptr<float>();
    auto hp = H.data_ptr<float>();
    auto gp = signs.data_ptr<float>();
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    if(do_sr && do_rht)
        quant_kernel<true,true><<<blocks,threads,0,s>>>(xp,dp,sp,pp,hp,gp,NBLK,numel,(unsigned)seed);
    else if(do_sr && !do_rht)
        quant_kernel<true,false><<<blocks,threads,0,s>>>(xp,dp,sp,pp,hp,gp,NBLK,numel,(unsigned)seed);
    else if(!do_sr && do_rht)
        quant_kernel<false,true><<<blocks,threads,0,s>>>(xp,dp,sp,pp,hp,gp,NBLK,numel,(unsigned)seed);
    else
        quant_kernel<false,false><<<blocks,threads,0,s>>>(xp,dp,sp,pp,hp,gp,NBLK,numel,(unsigned)seed);
}
'''

_CPP = ("void quant_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, int64_t, int64_t, bool, bool);")

_H16 = {}        # cached Hadamard per device
_ZERO16 = {}     # cached zero placeholders per device

_mod = None
def _ext():
    global _mod
    if _mod is None:
        _mod = load_inline(
            name="nvfp4_cuda_ext", cpp_sources=_CPP, cuda_sources=_CUDA_SRC,
            functions=["quant_launch"],
            extra_cuda_cflags=["-O3", "-gencode", "arch=compute_120a,code=sm_120a"],
            verbose=False)
    return _mod


_SEED = [0]
_scale_u8_view = lambda s: s.view(torch.uint8)


def quant_nvfp4_cuda(t, stochastic=False, rht=False, H=None, signs=None, seed=None):
    """t:[M,K] bf16/fp32, K%16==0 -> NVFP4Tensor. RHT (if any) is done IN-KERNEL."""
    if seed is None:
        seed = _SEED[0]
        if stochastic: _SEED[0] += 1
    M, K = t.shape
    assert K % BLK == 0, f"K={K} must be divisible by {BLK}"
    tf = t.to(torch.bfloat16).contiguous()
    dev = t.device
    # amax in bf16 (cheap), kept ON DEVICE (no .item() sync). RHT (orthonormal)
    # amplifies <= sqrt(16)=4x, so scale pts by 4 to avoid block-scale saturation;
    # pts cancels in per-element data scaling (E4M3 FP) -> accuracy-neutral.
    amax = tf.abs().amax().to(torch.float32).clamp_min(1e-12)
    pts = ((4.0 if rht else 1.0) * amax / (F4_MAX * F8E4M3_MAX)).reshape(1)
    if dev not in _ZERO16:
        _ZERO16[dev] = (torch.zeros(BLK, BLK, device=dev), torch.zeros(BLK, device=dev))
    if rht:
        Hd = (H if H is not None else _H16.setdefault(dev, hadamard16(dev))).contiguous().float()
        Sd = (signs if signs is not None else torch.ones(BLK, device=dev)).contiguous().float()
    else:
        Hd, Sd = _ZERO16[dev]
    NBLK = M * (K // BLK)
    data = torch.empty((M, K // 2), dtype=torch.uint8, device=dev)
    scale = torch.empty((M, K // BLK), dtype=torch.float8_e4m3fn, device=dev)
    _ext().quant_launch(tf, data, scale.view(torch.uint8), pts, Hd, Sd,
                        NBLK, int(seed), bool(stochastic), bool(rht))
    sM, sK = hp_data_dims_to_swizzled_scale_dims_nvfp4(M, K)
    sw = to_blocked(scale).flatten().view(sM, sK)
    return NVFP4Tensor(data, sw, BLK, t.dtype, pts.reshape(()),
                       None, True, False, None)


def fp4_matmul_cuda(A, B, sr_a=False, sr_b=False, rht=False, H=None, signs=None, seed=None):
    a = quant_nvfp4_cuda(A.contiguous(), sr_a, rht, H, signs, seed)
    bt = quant_nvfp4_cuda(B.t().contiguous(), sr_b, rht, H, signs, seed)
    return _addmm_nvfp4_dispatch(a, bt.t(), _MM)


def fp4_mm_preqB(A, Bq_t, sr_a=False, rht=False, H=None, signs=None, seed=None):
    """A @ B where B is already quantized as Bq_t = quant(B.t()) (an NVFP4Tensor).
    Only A is quantized here -> lets a constant weight be quantized once and reused."""
    a = quant_nvfp4_cuda(A.contiguous(), sr_a, rht, H, signs, seed)
    return _addmm_nvfp4_dispatch(a, Bq_t.t(), _MM)


def hadamard16(device):
    Hm = torch.ones(1, 1, device=device, dtype=torch.float32)
    while Hm.shape[0] < BLK:
        Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
    return Hm / math.sqrt(BLK)
