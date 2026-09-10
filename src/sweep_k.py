#!/usr/bin/env python
"""
DN2 / Prov-GigaPath on Camelyon16 -- Step 4: choose k on DEV.

Reads the top-K distance matrix saved by dn2_score.py and evaluates every k
without touching the GPU. Run this on dev only. Pick a k, then run dn2_score.py
on test once with that k.

Reporting both 'sum' and 'mean' aggregation is free: they give identical
rankings for a fixed k (mean = sum/k), so AUROC is identical -- mean is just
easier to read across different k. Any difference printed would mean a bug.
"""
import argparse
import csv
import os

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True,
                    help='results folder written by dn2_score.py')
    ap.add_argument('--query_dir', required=True,
                    help='feature dir with manifest.csv + ok.npy')
    ap.add_argument('--ks', type=int, nargs='+',
                    default=[1, 2, 3, 5, 10, 20, 50])
    args = ap.parse_args()

    topd = np.load(os.path.join(args.run_dir, 'topk_dist2.npy'))
    ok = np.load(os.path.join(args.query_dir, 'ok.npy')) == 1
    rows = list(csv.DictReader(open(os.path.join(args.query_dir, 'manifest.csv'),
                                    newline='')))
    lab = np.array([int(r['label']) for r in rows])
    slides = np.array([r['slide'] for r in rows])

    topd, lab, slides = topd[ok], lab[ok], slides[ok]
    run = os.path.basename(args.run_dir.rstrip('/'))
    out = [f'k sweep for run: {run}',
           f'query: {args.query_dir}',
           f'n={len(lab):,}  tumour={int((lab==1).sum()):,}  '
           f'normal={int((lab==0).sum()):,}',
           '',
           f'{"k":>4} {"patch-AUROC":>12} {"patch-AP":>10} '
           f'{"slide-AUROC(max)":>18}']

    uniq = np.unique(slides)
    sl_true = np.array([int((lab[slides == s] == 1).any()) for s in uniq])

    best = (-1.0, None)
    for k in args.ks:
        if k > topd.shape[1]:
            continue
        sc = topd[:, :k].mean(axis=1)
        auroc = roc_auc_score(lab, sc) * 100
        ap_ = average_precision_score(lab, sc) * 100
        line = f'{k:>4} {auroc:>12.2f} {ap_:>10.2f}'
        if sl_true.min() != sl_true.max():
            sl_score = np.array([sc[slides == x].max() for x in uniq])
            line += f' {roc_auc_score(sl_true, sl_score) * 100:>18.2f}'
        out.append(line)
        if auroc > best[0]:
            best = (auroc, k)
    out += ['', f'best patch-AUROC {best[0]:.2f} at k={best[1]}']

    txt = '\n'.join(out)
    open(os.path.join(args.run_dir, 'sweep_k.txt'), 'w').write(txt + '\n')
    print(txt)
    print(f'\n-> {args.run_dir}/sweep_k.txt')


if __name__ == '__main__':
    main()