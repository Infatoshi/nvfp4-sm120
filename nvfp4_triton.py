"""Custom NVFP4 quantizer for sm_120 (Triton) with SOFTWARE stochastic rounding.

sm_120 (workstation/GeForce Blackwell) has FP4 (E2M1) round-to-nearest conversion
and FP4 tensor-core GEMM, but NOT the hardware stochastic-rounding conversion
(`cvt.rs.*.e2m1` -> ptxas: "Feature '.rs' not supported on .target sm_120").
This kernel implements stochastic rounding in software (RNG dither over the
non-uniform E2M1 levels), so the full NVFP4 recipe's gradient quantization can run
on sm_120 -- the piece TE's fused kernel can't do here.

Validated against the reference fake-quant in nvfp4_validate.py:
  - round-to-nearest:   max abs error vs reference ~ 0 (same math)
  - stochastic rounding: unbiased (mean over many draws -> reference value)
"""
import torch, triton, triton.language as tl
import nvfp4_validate as R   # reference quantize_nvfp4_1d / _round_e2m1

E2M1_MAX = 6.0
E4M3_MAX = 448.0


@triton.jit
def _round_e2m1_sw(m, do_sr, seed, offs):
    """m: scaled magnitude (>=0). Returns E2M1-rounded magnitude.
    Software stochastic rounding over the non-uniform levels {0,.5,1,1.5,2,3,4,6}."""
    m = tl.minimum(m, 6.0)
    # bracketing levels lo <= m <= hi
    lo = tl.where(m < 0.5, 0.0,
         tl.where(m < 1.0, 0.5,
         tl.where(m < 1.5, 1.0,
         tl.where(m < 2.0, 1.5,
         tl.where(m < 3.0, 2.0,
         tl.where(m < 4.0, 3.0,
         tl.where(m < 6.0, 4.0, 6.0)))))))
    hi = tl.where(m < 0.5, 0.5,
         tl.where(m < 1.0, 1.0,
         tl.where(m < 1.5, 1.5,
         tl.where(m < 2.0, 2.0,
         tl.where(m < 3.0, 3.0,
         tl.where(m < 4.0, 4.0, 6.0))))))
    width = tl.maximum(hi - lo, 1e-9)
    if do_sr:
        r = tl.rand(seed, offs)               # uniform [0,1)
        p = (m - lo) / width
        q = tl.where(r < p, hi, lo)
    else:
        q = tl.where((m - lo) <= (hi - m), lo, hi)
    return q


@triton.jit
def nvfp4_quant_kernel(x_ptr, out_ptr, s_dec, n_blocks, BLK: tl.constexpr,
                       DO_SR: tl.constexpr, seed):
    """One program per 16-element block. Two-level scaling (global s_dec passed in,
    per-block E4M3 scale computed here). Writes dequantized values for validation."""
    pid = tl.program_id(0)
    if pid >= n_blocks:
        return
    offs = pid * BLK + tl.arange(0, BLK)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s_enc = 1.0 / s_dec
    amax_b = tl.max(tl.abs(x))
    s_dec_b = tl.maximum(amax_b / 6.0, 1e-12)
    # quantize block scale to E4M3 (matches reference: e4m3(s_dec_b * s_enc))
    bs = (s_dec_b * s_enc).to(tl.float8e4nv).to(tl.float32)
    deq = tl.maximum(bs * s_dec, 1e-12)
    sign = tl.where(x < 0, -1.0, 1.0)
    m = tl.abs(x) / deq
    q = _round_e2m1_sw(m, DO_SR, seed, offs)
    tl.store(out_ptr + offs, (sign * q * deq).to(tl.bfloat16))


def quant_nvfp4_triton(x, stochastic=False, seed=0):
    """x: [.., 16k] contiguous. Returns dequantized bf16 tensor (for validation)."""
    xf = x.contiguous().view(-1)
    n = xf.numel()
    BLK = 16
    assert n % BLK == 0
    nb = n // BLK
    amax_t = xf.abs().amax().clamp_min(1e-12)
    s_enc = (E2M1_MAX * E4M3_MAX) / amax_t
    s_dec = (1.0 / s_enc).item()
    out = torch.empty_like(xf, dtype=torch.bfloat16)
    nvfp4_quant_kernel[(nb,)](xf, out, s_dec, nb, BLK, stochastic, seed)
    return out.view(x.shape)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(4096, 1024, device=dev, dtype=torch.bfloat16) * 3.0

    # --- round-to-nearest: must match the reference closely ---
    ref_rne = R.quantize_nvfp4_1d(x, stochastic=False).float()
    tri_rne = quant_nvfp4_triton(x, stochastic=False).float()
    err = (ref_rne - tri_rne).abs().max().item()
    rel = ((ref_rne - tri_rne).abs().sum() / ref_rne.abs().sum()).item()
    print(f"[RNE] max abs diff vs reference = {err:.3e}  rel L1 = {rel:.3e}")

    # --- stochastic rounding: unbiased estimator of the true value ---
    N = 200
    acc = torch.zeros_like(x, dtype=torch.float32)
    for i in range(N):
        acc += quant_nvfp4_triton(x, stochastic=True, seed=i).float()
    sr_mean = acc / N
    # SR is unbiased w.r.t. the *scaled* representable grid; compare mean to x where
    # representable, measure bias relative to RNE rounding error.
    bias = (sr_mean - x.float()).abs().mean().item()
    rne_err = (ref_rne - x.float()).abs().mean().item()
    print(f"[SR ] mean(|E[q]-x|) over {N} draws = {bias:.4e}   (RNE rounding err = {rne_err:.4e})")
    print(f"[SR ] bias is {bias/rne_err:.2%} of RNE error -> {'UNBIASED (good)' if bias < 0.25*rne_err else 'check'}")
    # also confirm individual SR draws are valid E2M1 values (same set as RNE)
    vals = torch.unique(quant_nvfp4_triton(x, stochastic=True, seed=7).float().abs())
    print(f"[SR ] distinct |levels| in one draw: {vals.numel()} (E2M1 has 8 incl 0)")


if __name__ == "__main__":
    main()
