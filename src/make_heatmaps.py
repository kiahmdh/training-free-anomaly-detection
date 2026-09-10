#!/usr/bin/env python
"""
Whole-slide anomaly heatmaps from a DN2 per-patch score CSV.

Each patch filename encodes its grid position: "<col>_<row>-<dzlevel>.jpeg"
(e.g. 137_42-16.jpeg -> col=137, row=42). Every patch's Ascore is placed back at
[row, col] to rebuild the slide. No GPU, no model, no re-scoring -- scores.csv
already holds everything.

Per slide it writes a PNG with the predicted anomaly map on the left and the
ground-truth tumour map on the right (from the CSV `label` column), so you can
see at a glance whether the hot regions land where the metastasis actually is.

One deliberate difference from the AnomalyCLIP version: DN2 scores are summed
squared distances, not probabilities in [0,1]. The colour scale is therefore
taken from percentiles of the WHOLE csv and shared across every slide. With
per-slide autoscaling a tumour-free slide still renders a full red-to-blue
range, because matplotlib stretches whatever tiny variation exists -- it looks
exactly like a detection. A shared scale means red here and red there mean the
same thing. --per_slide_scale overrides this if you want it.

Also writes index.csv (per-slide summary, handy for choosing which slides to
show) and montage.png (the N most anomalous slides on one sheet).

Usage:
  python make_heatmaps.py \
    --csv results/test_A_k5/scores.csv \
    --reference_csv /home/user01/camelyon16/reference.csv \
    --out_dir results/test_A_k5/heatmaps
"""
import argparse
import csv
import math
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PATCH_RE = re.compile(r'^(\d+)_(\d+)-(\d+)\.jpe?g$', re.IGNORECASE)


def load_reference(path):
    """slide (lower, no extension) -> 'tumour' | 'normal'."""
    ref = {}
    if not path:
        return ref
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        cols = [h.strip().lower() for h in header] if header else []
        name_i = cols.index('image') if 'image' in cols else 0
        type_i = cols.index('type') if 'type' in cols else 1
        for row in reader:
            if len(row) <= type_i:
                continue
            key = os.path.splitext(row[name_i].strip())[0].lower()
            t = row[type_i].strip().lower()
            if t in ('tumor', 'tumour'):
                ref[key] = 'tumour'
            elif t == 'normal':
                ref[key] = 'normal'
    return ref


def build_grid(pts):
    """[(row, col, score, label)] -> (heat, gt) arrays, NaN where no patch."""
    rows = [p[0] for p in pts]
    cols = [p[1] for p in pts]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    heat = np.full((r1 - r0 + 1, c1 - c0 + 1), np.nan)
    gt = np.full_like(heat, np.nan)
    for row, col, score, lab in pts:
        heat[row - r0, col - c0] = score
        gt[row - r0, col - c0] = lab
    return heat, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True,
                    help='per-patch CSV: patch_name,slide,label,Ascore')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--slides', nargs='+', default=None,
                    help='which slides (default test_001..test_010)')
    ap.add_argument('--all_slides', action='store_true',
                    help='render every slide present in the CSV')
    ap.add_argument('--reference_csv', default=None,
                    help='optional, puts tumour/normal in each title')
    ap.add_argument('--cmap', default='jet',
                    help='matplotlib colormap (jet, inferno, viridis, ...)')
    ap.add_argument('--stride', type=int, default=1,
                    help='pixel stride between patches. Camelyon16 filenames '
                         'are already grid indices so leave at 1; Camelyon17 '
                         'uses pixel coordinates, so pass 256.')
    ap.add_argument('--vmin_pct', type=float, default=1.0)
    ap.add_argument('--vmax_pct', type=float, default=99.0)
    ap.add_argument('--vmin', type=float, default=None,
                    help='pin the colour scale to an absolute value. Use this '
                         'to render two runs on ONE scale -- otherwise each CSV '
                         'gets its own and the figures are not comparable.')
    ap.add_argument('--vmax', type=float, default=None)
    ap.add_argument('--rank', action='store_true',
                    help='plot each score as its percentile rank within this '
                         'CSV (0-100) instead of the raw distance. Best way to '
                         'compare WHERE two different banks light up, since it '
                         'removes the overall scale shift a bigger bank causes.')
    ap.add_argument('--per_slide_scale', action='store_true',
                    help='autoscale each slide separately (slides then are NOT '
                         'visually comparable -- see module docstring)')
    ap.add_argument('--montage', type=int, default=12,
                    help='also write montage.png of the N most anomalous '
                         'rendered slides; 0 disables')
    ap.add_argument('--dpi', type=int, default=130)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ref = load_reference(args.reference_csv)

    by_slide = defaultdict(list)
    all_scores, skipped = [], 0
    with open(args.csv, newline='') as f:
        for r in csv.DictReader(f):
            m = PATCH_RE.match(r['patch_name'].strip())
            if not m:
                skipped += 1
                continue
            try:
                score = float(r['Ascore'])
            except (ValueError, KeyError):
                skipped += 1
                continue
            if not math.isfinite(score):
                skipped += 1
                continue
            col = int(m.group(1)) // args.stride
            row = int(m.group(2)) // args.stride
            by_slide[r['slide']].append((row, col, score, int(r['label'])))
            all_scores.append(score)

    if skipped:
        print(f'skipped {skipped:,} rows (unparsable name, or non-finite score)')
    if not by_slide:
        raise SystemExit('no usable rows -- check --csv and the patch names')

    arr = np.asarray(all_scores, dtype=float)

    if args.rank:
        order = np.argsort(np.argsort(arr))
        pct = 100.0 * order / max(len(arr) - 1, 1)
        lut = {}
        for a, p_ in zip(arr, pct):
            lut[a] = p_
        for sl_ in by_slide:
            by_slide[sl_] = [(r, c, lut[v], l) for r, c, v, l in by_slide[sl_]]
        arr = pct
        gmin, gmax = 0.0, 100.0
        print(f'{len(arr):,} scores across {len(by_slide)} slides; '
              f'RANK mode -- colour is percentile within this CSV [0, 100]')
    else:
        gmin = (args.vmin if args.vmin is not None
                else float(np.percentile(arr, args.vmin_pct)))
        gmax = (args.vmax if args.vmax is not None
                else float(np.percentile(arr, args.vmax_pct)))
        how = ('pinned by --vmin/--vmax' if args.vmin is not None
               or args.vmax is not None
               else f'p{args.vmin_pct:g}-p{args.vmax_pct:g} of this CSV')
        print(f'{len(arr):,} scores across {len(by_slide)} slides; shared '
              f'colour scale [{gmin:.4f}, {gmax:.4f}] ({how})')
        print(f'  full range of this CSV: [{arr.min():.4f}, {arr.max():.4f}] '
              f'-- pass these to --vmin/--vmax on another run to compare them '
              f'on one scale')

    if args.all_slides:
        slides = sorted(by_slide)
    elif args.slides:
        slides = args.slides
    else:
        slides = [f'test_{i:03d}' for i in range(1, 11)]

    index = []
    for slide in slides:
        pts = by_slide.get(slide)
        if not pts:
            print(f'  {slide}: not present in the CSV, skipped')
            continue
        heat, gt = build_grid(pts)
        H, W = heat.shape
        n_tumour = int(np.nansum(gt == 1))
        vals = np.sort(np.asarray([p[2] for p in pts]))[::-1]
        top1 = float(vals[:max(1, len(vals) // 100)].mean())

        if args.per_slide_scale:
            vmin, vmax = float(np.nanmin(heat)), float(np.nanmax(heat))
            if vmax <= vmin:
                vmax = vmin + 1e-9
        else:
            vmin, vmax = gmin, gmax

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
        im0 = axes[0].imshow(heat, cmap=args.cmap, vmin=vmin, vmax=vmax,
                             interpolation='nearest')
        if args.rank:
            t0 = 'DN2 score percentile within run'
        else:
            t0 = ('DN2 anomaly score  ('
                  + ('per-slide' if args.per_slide_scale else 'shared')
                  + ' scale)')
        axes[0].set_title(t0)
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(gt, cmap='Reds', vmin=0, vmax=1,
                             interpolation='nearest')
        axes[1].set_title(f'ground-truth tumour patches (n={n_tumour:,})')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        for ax in axes:
            ax.set_xlabel('column')
            ax.set_ylabel('row')

        stype = ref.get(slide.lower(), '')
        title = slide + (f'  ({stype})' if stype else '')
        fig.suptitle(f'{title}   grid {H}x{W}, {len(pts):,} patches')
        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f'{slide}.png')
        fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)

        index.append(dict(slide=slide, type=stype or '?', patches=len(pts),
                          tumour_patches=n_tumour, grid=f'{H}x{W}',
                          max_score=f'{vals[0]:.6f}', top1pct_mean=f'{top1:.6f}'))
        print(f'  {slide}: {len(pts):,} patches, grid {H}x{W}, '
              f'tumour {n_tumour:,}, max {vals[0]:.4f} -> {out_path}')

    if not index:
        raise SystemExit('nothing rendered')

    with open(os.path.join(args.out_dir, 'index.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)

    if args.montage:
        order = sorted(index, key=lambda d: -float(d['max_score']))
        sel = order[:args.montage]
        n = len(sel)
        ncol = min(4, n)
        nrow = math.ceil(n / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow),
                                 squeeze=False)
        for ax in axes.ravel():
            ax.axis('off')
        for i, d in enumerate(sel):
            heat, _ = build_grid(by_slide[d['slide']])
            ax = axes[i // ncol][i % ncol]
            ax.imshow(heat, cmap=args.cmap, vmin=gmin, vmax=gmax,
                      interpolation='nearest')
            ax.set_title(f"{d['slide']} ({d['type']})", fontsize=9)
            ax.axis('off')
        fig.suptitle(f'{n} most anomalous slides by max patch score '
                     f'(shared colour scale)')
        fig.tight_layout()
        mp = os.path.join(args.out_dir, 'montage.png')
        fig.savefig(mp, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'  montage -> {mp}')

    print(f'\n-> {len(index)} slides in {args.out_dir}/  (+ index.csv)')


if __name__ == '__main__':
    main()