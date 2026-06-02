# Roofline for the six NVFP4 training GEMM shapes on RTX PRO 6000 Blackwell (sm_120).
# Pure arithmetic, no GPU. Determines honest SOL ceiling per shape.
# Card specs (datasheet): dense FP4 compute peak = 2000 TFLOPS; GDDR7 BW = 1792 GB/s.
PEAK_FP4 = 2000e12      # dense FP4 FLOP/s (4000 AI TOPS sparse / 2)
BW = 1792e9             # bytes/s, GDDR7 1.79 TB/s
shapes = [
    (16384, 512, 512,  "q/k/v/o fprop+dgrad", 0.115),
    (16384, 512, 2048, "down.fprop, up.dgrad", 0.330),
    (16384, 2048,512,  "up.fprop, down.dgrad", 0.176),
    (512,   512, 16384,"q/k/v/o wgrad",        0.063),
    (512,   2048,16384,"down.wgrad",           0.244),
    (2048,  512, 16384,"up.wgrad",             0.247),
]
ridge = PEAK_FP4 / BW   # FLOP/byte where compute peak == BW roofline
print(f"Ridge point (arith intensity): {ridge:.1f} FLOP/byte")
print(f"{'M':>6}{'N':>6}{'K':>7}  {'GFLOP':>8} {'bytes(MB)':>10} {'AI':>7} {'bound':>6} "
      f"{'ceil_TFLOPS':>12} {'cuBLAS%2000':>11} {'cuBLAS%ceil':>11}")
for M,N,K,who,cublas_sol in shapes:
    flop = 2*M*N*K
    # NVFP4: A[M,K] + B[K,N] at 0.5 byte/elem + e4m3 block scales (1 byte per 16 elems).
    # Output bf16 = 2 bytes/elem (realistic training: result goes to next op in higher prec).
    a = M*K*0.5 + (M*K/16)      # data + scales
    b = K*N*0.5 + (K*N/16)
    d = M*N*2                    # bf16 output
    byts = a+b+d
    ai = flop/byts
    bound = "comp" if ai>=ridge else "BW"
    ceil = PEAK_FP4 if bound=="comp" else ai*BW   # achievable FLOP/s ceiling
    ceil_tf = ceil/1e12
    cub_tf = cublas_sol*2000
    print(f"{M:>6}{N:>6}{K:>7}  {flop/1e9:>8.1f} {byts/1e6:>10.1f} {ai:>7.1f} {bound:>6} "
          f"{ceil_tf:>12.0f} {cublas_sol*100:>10.1f}% {cub_tf/ceil_tf*100:>10.1f}%")
print()
print("ceil_TFLOPS = honest performance ceiling for that shape (compute peak if compute-bound,")
print("              else AI*BW). cuBLAS%ceil = how close cuBLAS already is to the REAL ceiling.")
