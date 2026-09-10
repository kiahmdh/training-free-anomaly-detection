#!/usr/bin/env bash
# DN2 + Prov-GigaPath on Camelyon16 -- full pipeline.
# Run each STEP by hand, in tmux, checking the output before moving on.
# Do not run this file top-to-bottom unattended the first time.
set -euo pipefail

ROOT=/home/user01/dn2_gigapath
FOLD=/home/user01/camelyon16/single/fold1
TILECSV=/home/user01/camelyon16/tile_label.csv
REFCSV=/home/user01/camelyon16/reference.csv

MAN=$ROOT/manifests
FEAT=$ROOT/features
RES=$ROOT/results          # one subfolder per run + results/summary.tsv
mkdir -p "$MAN" "$FEAT" "$RES"

# ---------------------------------------------------------------- STEP 0
# conda activate anomalyclip
# pip install "timm>=1.0.3"          # gigapath needs this; nothing else new
# export HF_TOKEN=hf_your_NEW_token  # never hardcode it in a file

# ---------------------------------------------------------------- STEP 1
# Manifests. Seconds to a few minutes. READ manifests/manifest_report.txt.
step1() {
python "$ROOT/build_manifests.py" \
  --fold_root "$FOLD" \
  --tile_label_csv "$TILECSV" \
  --out_dir "$MAN" \
  --holdout_val_slides 8 \
  --holdout_tumor_slides 12 \
  --dev_holdout_patches 40000 \
  --dev_tumorslide_patches 60000
}

# ---------------------------------------------------------------- STEP 1b
# Preflight. ~20 s. Gates the long run: exits non-zero on any hard failure.
preflight() {
python "$ROOT/preflight.py" \
  --root "$ROOT" --fold_root "$FOLD" \
  --tile_label_csv "$TILECSV" --reference_csv "$REFCSV" \
  --auroc_script /home/user01/Aclip/AnomalyCLIP/auroc_from_csv.py
}

# ---------------------------------------------------------------- STEP 2
# PILOT: 20k bank + 20k test. ~5 min. Proves the whole chain before you
# commit 5 hours. Expect a rough but non-trivial AUROC (~0.7-0.9).
pilot() {
python "$ROOT/extract_features.py" --manifest "$MAN/bank.csv" \
  --out_dir "$FEAT/pilot_bank" --limit 20000 --batch_size 256 --num_workers 12
python "$ROOT/extract_features.py" --manifest "$MAN/test.csv" \
  --out_dir "$FEAT/pilot_test" --limit 20000 --batch_size 256 --num_workers 12
python "$ROOT/dn2_score.py" --bank_dir "$FEAT/pilot_bank" \
  --query_dir "$FEAT/pilot_test" --run_dir "$RES/pilot" --max_k 50 --k 2 \
  --reference_csv "$REFCSV" \
  --note "20k bank vs 20k random test patches"
}

# ---------------------------------------------------------------- STEP 3
# FULL extraction. ~3.5-6 h total on a 4090. Resumable: if it dies, rerun the
# exact same command and it picks up where it stopped. Run in tmux.
extract() {
python "$ROOT/extract_features.py" --manifest "$MAN/dev.csv" \
  --out_dir "$FEAT/dev"  --batch_size 256 --num_workers 12   # ~10 min
python "$ROOT/extract_features.py" --manifest "$MAN/bank.csv" \
  --out_dir "$FEAT/bank" --batch_size 256 --num_workers 12   # 1.30M patches
python "$ROOT/extract_features.py" --manifest "$MAN/test.csv" \
  --out_dir "$FEAT/test" --batch_size 256 --num_workers 12   # 1.10M patches
python "$ROOT/extract_features.py" --manifest "$MAN/bank_tumorslide_normals.csv" \
  --out_dir "$FEAT/bankB" --batch_size 256 --num_workers 12  # ~1.02M patches
}

# ---------------------------------------------------------------- STEP 4
# Choose k on DEV. The holdout slides are removed from the bank so held-out
# normals cannot match themselves at distance 0.
# Two bank definitions, same dev set. Pick the winner on evidence.
#   A   = normal slides only            (~1.30M vectors)
#   A+B = plus normal-labelled patches
#         from train/val tumour slides   (~2.32M vectors)
tune() {
echo "########## BANK A: normal slides only ##########"
python "$ROOT/dn2_score.py" \
  --bank_dir "$FEAT/bank" --query_dir "$FEAT/dev" \
  --exclude_slides "$MAN/holdout_slides.txt" \
  --run_dir "$RES/devA" --max_k 50 --k 2 \
  --reference_csv "$REFCSV" \
  --note "bank A = normal slides only"
python "$ROOT/sweep_k.py" --run_dir "$RES/devA" --query_dir "$FEAT/dev"

echo "########## BANK A+B: plus tumour-slide normals ##########"
python "$ROOT/dn2_score.py" \
  --bank_dir "$FEAT/bank" "$FEAT/bankB" --query_dir "$FEAT/dev" \
  --exclude_slides "$MAN/holdout_slides.txt" \
  --run_dir "$RES/devAB" --max_k 50 --k 2 \
  --reference_csv "$REFCSV" \
  --note "bank A+B = plus tumour-slide normals"
python "$ROOT/sweep_k.py" --run_dir "$RES/devAB" --query_dir "$FEAT/dev"
}

# ---------------------------------------------------------------- STEP 5
# TEST, once, with the k you picked. Pass K=<n> when calling.
# usage: test_run <k> <A|AB>
test_run() {
K=${1:?usage: test_run <k> <A|AB>}
BANK=${2:?usage: test_run <k> <A|AB>}
if [ "$BANK" = "AB" ]; then DIRS="$FEAT/bank $FEAT/bankB"; else DIRS="$FEAT/bank"; fi
RUN="$RES/test_${BANK}_k${K}"
python "$ROOT/dn2_score.py" \
  --bank_dir $DIRS --query_dir "$FEAT/test" \
  --run_dir "$RUN" --max_k 50 --k "$K" \
  --reference_csv "$REFCSV" \
  --note "FINAL test run, bank $BANK, k=$K"
# cross-check against the script you already have and trust; the patch/slide
# AUROC and AP must match metrics.txt to 2dp
python /home/user01/Aclip/AnomalyCLIP/auroc_from_csv.py \
  --csv "$RUN/scores.csv" --reference_csv "$REFCSV" \
  2>&1 | tee "$RUN/auroc_from_csv_crosscheck.txt"
python "$ROOT/check_stain.py" --run_dir "$RUN" --reference_csv "$REFCSV"
echo; echo "===== all runs so far ====="; { column -t -s "$(printf '\t')" "$RES/summary.tsv" 2>/dev/null || cat "$RES/summary.tsv"; }
}

"$@"