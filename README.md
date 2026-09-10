# Training-Free Anomaly Detection for Lymph-Node Metastasis

This repository implements a training-free anomaly detection framework for
metastatic lymph node detection in histopathology images.

Instead of optimizing a task-specific classifier, we use a frozen foundation
model encoder [Prov-GigaPath](https://www.nature.com/articles/s41586-024-07441-w) to extract patch-level representations and detect
tumor regions based on their distance from a reference bank of normal tissue
embeddings.

The goal is to investigate whether large-scale pathology foundation models can
support reliable anomaly detection without additional training or annotation.

**Nothing in this pipeline is trained.** The encoder is frozen, there are no
weights to fit, and detection is a $k$-nearest-neighbour distance to a reference
bank of healthy tissue embeddings. The only design decisions are *what goes into
the bank* and *the value of k* — and this project's main finding is that the
first matters enormously while the second barely matters at all.



## Pipeline

```
patches ──► extract_features.py ──► features.npy (fp16, L2-normalised, 1536-d)
                  (Prov-GigaPath, frozen — run once)          │
                                                              ▼
                                            dn2_score.py ──► scores.csv
                                            (exact kNN on GPU)  topk_dist2.npy
                                                              │
                              ┌───────────────────────────────┼──────────────┐
                              ▼                               ▼              ▼
                        sweep_k.py                    metrics_report.py  make_heatmaps.py
                     (every k, free)                  (AUROC/AP/oper.pt)  check_stain.py
```

Stage 1 (feature extraction) is GPU-bound and runs **once**, ~3.5–6 h on an
RTX 4090 for 3.5 M patches. Stage 2 reads only cached vectors, so a new bank or a
new $k$ costs minutes. That separation is what made the bank investigation
practical.

Because `dn2_score.py` saves the **top-50** sorted distances per query, every
$k \le 50$ is obtained by re-summing numbers already on disk. A 7-value sweep
costs one GPU pass, not seven.

### Scripts

| Script | Step | What it does |
|---|---|---|
| `build_manifests.py` | 1 | Camelyon16 manifests (bank / bank-B / dev / test) with **true per-patch** labels from `tile_label.csv`, plus leakage assertions |
| `build_manifest_c17.py` | 1b | Camelyon17 query manifest (node = slide unit) |
| `preflight.py` | 1c | ~20 s gate: env/ABI, GPU, HF cache, manifests, disk, leakage invariants. Exits non-zero on hard failure |
| `extract_features.py` | 2 | Frozen encoder → `features.npy`. Resumable, memmapped, crashes early on a bad environment instead of writing zeros |
| `dn2_score.py` | 3 | Exact kNN, no faiss. Bank in fp16 on GPU, queries chunked past it |
| `sweep_k.py` | 4 | Every $k$ from the cached distance matrix, no GPU |
| `metrics_report.py` | — | Patch + slide (max, top-1 %-mean) AUROC / AP / operating point. Importable and standalone |
| `make_heatmaps.py` | 5 | Whole-slide anomaly maps beside ground truth, on a **shared** colour scale |
| `check_stain.py` | 5b | Stain / centre confound check: does normal tissue score differently *purely* by slide? |


## Setup

```bash
conda create -n dn2 python=3.10 && conda activate dn2
pip install -r requirements.txt
export HF_TOKEN=hf_...      # Prov-GigaPath is a gated model; never commit this
```

`numpy < 2` is pinned deliberately: with numpy 2.x against a torch built for
1.x, `torch.from_numpy` raises inside the dataloader, every patch is flagged
unreadable, and extraction silently writes a file of zeros. `preflight.py` and
the extraction preflight both catch this in seconds.

### Data

- **Camelyon16** — [camelyon16.grand-challenge.org](https://camelyon16.grand-challenge.org/) ·  [Bejnordi et al., *JAMA* 2017](https://doi.org/10.1001/jama.2017.14585)
- **Camelyon17** — [camelyon17.grand-challenge.org](https://camelyon17.grand-challenge.org/) ·  [Bándi et al., *IEEE TMI* 2019](https://doi.org/10.1109/TMI.2018.2867350)

Both are public but require registration. Tiles are not redistributed here; only
the manifest-building code is. Expected layout:

```
<fold_root>/{train,validation,test}/{0_normal,1_tumor}/<slide>/<col>_<row>-<level>.jpeg
```

---

## Reproducing

Edit the paths at the top of `run_all.sh`, then run each step **by hand, in
tmux**, checking the output before moving on.

```bash
bash run_all.sh step1        # manifests   — then READ manifests/manifest_report.txt
bash run_all.sh preflight    #  gate
bash run_all.sh pilot        # 20k×20k end-to-end smoke test
bash run_all.sh extract      #  resumable
bash run_all.sh tune         # bank A vs A+B on dev + k sweep
bash run_all.sh test_run 5 A # the single final test run
```

`results/summary.tsv` accumulates one line per run so every configuration is
comparable at a glance. Each run also writes `config.json` with every argument.

### Reported configuration

| | |
|---|---|
| Encoder | `hf_hub:prov-gigapath/prov-gigapath`, frozen, 1536-d output |
| Preprocessing | `Resize(256, bicubic) → CenterCrop(224) → ImageNet mean/std` (official recipe; on 256×256 tiles the resize is a no-op) |
| Distance | squared L2 on unit vectors, i.e. $d^2 = 2 - 2\cos\theta$ |
| $k$ | 5 — chosen as a deliberately **untuned** value inside the flat region, not the sweep maximum |
| Bank A | 1,300,767 vectors / 159 slides (1,232,652 / 151 during model selection) |
| Bank A+B | 2,318,160 vectors (2,150,355 during model selection) |
| Slide aggregation | max patch score, and mean of the top 1 % |

On $k$: across a fiftyfold change, Bank A's AUROC moves by ~0.001. When a curve
is that flat, picking its maximum is fitting to noise, so we fixed $k=5$ in
advance and report it as untuned.
For Camelyon17 pass `--stride 256`: its filenames encode pixel coordinates,
whereas Camelyon16's are already grid indices.

