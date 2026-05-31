#!/bin/bash
cd /home/infatoshi/experiments/_scratch/nvfp4-validate/scaling
unset LD_PRELOAD
export CUDA_HOME=/usr/local/cuda-13
OUT=/home/infatoshi/data/scaling_results.jsonl
rm -f "$OUT"
# size: dim nl heads ; fixed token budget via steps (bs128 x T512 = 65536 tok/step)
# steps chosen so smaller models see more tok/param, larger fewer (iso-step -> iso-data-ish);
# use iso-TOKENS: ~20M tokens each => 305 steps.
STEPS=305
for cfg in "256 6 4" "384 6 6" "512 8 8" "640 10 10"; do
  read d l h <<< "$cfg"
  echo "=== BF16 dim=$d ==="
  python3 -u train_text.py --dim $d --nl $l --nh $h --nkv $h --T 512 --bs 128 \
    --steps $STEPS --warmup 30 --lr 6e-4 --tag bf16_$d --out "$OUT" 2>&1 | grep -E 'backend=|RESULT|val '
  echo "=== NVFP4 dim=$d ==="
  NVFP4_CUDA=1 python3 -u train_text.py --dim $d --nl $l --nh $h --nkv $h --T 512 --bs 128 \
    --steps $STEPS --warmup 30 --lr 6e-4 --tag nvfp4_$d --out "$OUT" 2>&1 | grep -E 'backend=|RESULT|val '
done
echo "=== SWEEP DONE ==="
cat "$OUT"
