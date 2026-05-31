"""Fit and plot NVFP4 vs BF16 scaling law L(N) from scaling_results.jsonl.
Power-law fit L = E + A * N^(-alpha). Two panels: (left) full view with published
GPT-2/nanoGPT OWT anchors; (right) zoomed BF16-vs-NVFP4 comparison with fits."""
import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

RES = sys.argv[1] if len(sys.argv) > 1 else "/home/infatoshi/data/scaling_results.jsonl"
rows = [json.loads(l) for l in open(RES) if l.strip()]
by = {}
for r in rows: by.setdefault(r["backend"], []).append((r["N"], r["val_loss"]))
for k in by: by[k].sort()

def powlaw(N, E, A, al): return E + A * np.power(N, -al)
colors = {"bf16": "#1f77b4", "nvfp4": "#d62728"}
labels = {"bf16": "BF16 (baseline)", "nvfp4": "NVFP4 (full SR+RHT recipe)"}

fits = {}
for be, pts in by.items():
    N = np.array([p[0] for p in pts], float); L = np.array([p[1] for p in pts], float)
    try:
        popt, _ = curve_fit(powlaw, N, L, p0=[min(L)*0.8, 10.0, 0.07], maxfev=20000,
                            bounds=([0,0,0.01],[10,1e3,0.5]))
        fits[be] = popt
    except Exception as e:
        fits[be] = None
        print(f"{be} fit failed: {e}")
    print(f"{be}: " + ", ".join(f"N={n/1e6:.1f}M L={l:.3f}" for n,l in pts) +
          (f"  fit L={popt[0]:.2f}+{popt[1]:.0f}*N^-{popt[2]:.3f}" if fits[be] is not None else ""))

allN = np.array([p[0] for v in by.values() for p in v], float)
xs = np.logspace(np.log10(allN.min()*0.85), np.log10(allN.max()*1.3), 120)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))

# --- left: full view with published anchors ---
for be, pts in by.items():
    N = np.array([p[0] for p in pts]); L = np.array([p[1] for p in pts])
    axL.scatter(N, L, s=70, color=colors[be], zorder=3, label=labels[be])
    if fits[be] is not None:
        axL.plot(xs, powlaw(xs, *fits[be]), "--", color=colors[be], alpha=0.8)
for y, t, a in [(2.85, "nanoGPT GPT-2 124M, fully trained (~300B tok): 2.85", 1.0),
                (3.11, "GPT-2 124M zero-shot on OWT: 3.11", 0.6)]:
    axL.axhline(y, color="gray", ls=":", lw=1, alpha=a)
    axL.text(allN.max(), y+0.02, t, ha="right", va="bottom", fontsize=8, color="gray")
axL.set_xscale("log"); axL.set_xlabel("Non-embedding parameters N")
axL.set_ylabel("OpenWebText val loss (GPT-2 BPE)")
axL.set_title("Full view vs published GPT-2 / nanoGPT anchors")
axL.legend(); axL.grid(True, which="both", alpha=0.3)

# --- right: zoomed comparison + fit equations ---
for be, pts in by.items():
    N = np.array([p[0] for p in pts]); L = np.array([p[1] for p in pts])
    axR.scatter(N, L, s=80, color=colors[be], zorder=3, label=labels[be])
    if fits[be] is not None:
        E, A, al = fits[be]
        axR.plot(xs, powlaw(xs, *fits[be]), "--", color=colors[be], alpha=0.85,
                 label=f"  fit: L = {E:.2f} + {A:.0f}·N^-{al:.3f}")
axR.set_xscale("log"); axR.set_xlabel("Non-embedding parameters N")
axR.set_ylabel("OpenWebText val loss")
axR.set_title("Zoomed: NVFP4 tracks BF16 (Δval ≤ 0.07 across the sweep)")
axR.legend(fontsize=9); axR.grid(True, which="both", alpha=0.3)

fig.suptitle("NVFP4 vs BF16 scaling law on real text (OpenWebText) — RTX PRO 6000, sm_120\n"
             "nanoGPT-style decoder, iso-token budget ~20M tok/model", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
OUT = "/home/infatoshi/data/scaling_law.png"
fig.savefig(OUT, dpi=140)
print("SAVED", OUT)
