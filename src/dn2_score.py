#!/usr/bin/env python
"""
DN2 / Prov-GigaPath on Camelyon16 -- Step 3: kNN scoring.

Exact search, no faiss, no approximation. Features are L2-normalised, so cosine
similarity is one matmul and squared-L2 distance = 2 - 2*cos. The bank sits on
the GPU and queries stream past it in chunks.

Everything for one run lands in --run_dir:

    scores.csv        patch_name, slide, label, Ascore   (feeds auroc_from_csv.py)
    topk_dist2.npy    (N_query, max_k) ascending distances -- k sweeps are free
    metrics.txt       human-readable result + full provenance
    config.json       every argument, so a run is reproducible

and one line is appended to <parent of run_dir>/summary.tsv so every run can be
compared at a glance.
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm


def load_split(d):
    feats = np.load(os.path.join(d, 'features.npy'), mmap_mode='r')
    ok = np.load(os.path.join(d, 'ok.npy'))
    rows = list(csv.DictReader(open(os.path.join(d, 'manifest.csv'), newline='')))
    n = feats.shape[0]
    assert len(rows) == n and ok.shape[0] == n, \
        f'row mismatch in {d}: feats={n} ok={ok.shape[0]} manifest={len(rows)}'
    prog = os.path.join(d, 'progress.json')
    if os.path.exists(prog):
        done = json.load(open(prog))['n_done']
        assert done == n, (f'{d} is INCOMPLETE: {done:,}/{n:,} rows extracted. '
                           f'Finish extraction before scoring.')
    return feats, ok, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank_dir', required=True, nargs='+',
                    help='one or more feature dirs, concatenated. '
                         'e.g. features/bank features/bankB')
    ap.add_argument('--query_dir', required=True)
    ap.add_argument('--run_dir', required=True,
                    help='results folder for this run, e.g. results/devA')
    ap.add_argument('--exclude_slides', default=None,
                    help='txt file of slide names to drop from the BANK '
                         '(use holdout_slides.txt when scoring dev)')
    ap.add_argument('--max_k', type=int, default=50)
    ap.add_argument('--k', type=int, default=2, help='k used for scores.csv')
    ap.add_argument('--query_chunk', type=int, default=8192)
    ap.add_argument('--bank_chunk', type=int, default=131072)
    ap.add_argument('--bank_per_slide_cap', type=int, default=0,
                    help='>0 : cap patches per bank slide so one huge slide '
                         'cannot dominate')
    ap.add_argument('--reference_csv', default=None,
                    help='slide-level truth for the reference.csv-based '
                         'slide metrics (optional; derived truth always shown)')
    ap.add_argument('--topk_frac', type=float, default=0.01,
                    help='top fraction of patches per slide for the '
                         'top-k%%-mean slide aggregation')
    ap.add_argument('--threshold', type=float, default=None,
                    help='fixed decision threshold for recall/precision; '
                         'default = Youden-optimal per level')
    ap.add_argument('--note', default='', help='free text stored in metrics.txt')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)
    run_name = os.path.basename(args.run_dir.rstrip('/'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_start = time.time()

    json.dump(vars(args), open(os.path.join(args.run_dir, 'config.json'), 'w'),
              indent=2)

    # ---------------------------------------------------------------- bank
    drop = set()
    if args.exclude_slides:
        drop = {l.strip() for l in open(args.exclude_slides) if l.strip()}
        print(f'excluding {len(drop)} slides from the bank '
              f'({args.exclude_slides})')

    parts, total, per_dir = [], 0, []
    for d in args.bank_dir:
        bf, bok, brows = load_split(d)
        keep = np.asarray(bok) == 1
        if drop:
            keep &= ~np.array([r['slide'] in drop for r in brows])
        idx = np.flatnonzero(keep)

        if args.bank_per_slide_cap > 0:
            by = {}
            for j in idx:
                by.setdefault(brows[j]['slide'], []).append(j)
            sel = []
            for _s, js in by.items():
                js = np.array(js)
                sel.append(rng.choice(js, args.bank_per_slide_cap, replace=False)
                           if len(js) > args.bank_per_slide_cap else js)
            idx = np.sort(np.concatenate(sel))

        n_sl = len({brows[j]['slide'] for j in idx})
        print(f'  {d}: {len(idx):,} vectors of {bf.shape[0]:,}, {n_sl} slides')
        per_dir.append((d, int(len(idx)), int(bf.shape[0]), n_sl))
        parts.append((bf, idx))
        total += len(idx)

    dim = parts[0][0].shape[1]
    print(f'bank: {total:,} vectors  ({total * dim * 2 / 1e9:.2f} GB fp16)')
    bank = torch.empty((total, dim), dtype=torch.float16, device=device)
    off = 0
    for bf, idx in tqdm(parts, desc='loading bank', unit='dir',
                        dynamic_ncols=True):
        for cs in range(0, len(idx), 200000):
            ce = min(cs + 200000, len(idx))
            blk = np.ascontiguousarray(bf[idx[cs:ce]]).copy()
            bank[off:off + (ce - cs)] = torch.from_numpy(blk).to(device)
            off += ce - cs
    assert off == total

    # --------------------------------------------------------------- query
    qf, qok, qrows = load_split(args.query_dir)
    nq, K = qf.shape[0], args.max_k
    topd = np.zeros((nq, K), dtype=np.float32)

    pbar = tqdm(total=nq, unit='patch', unit_scale=True, smoothing=0.05,
                dynamic_ncols=True, desc=f'knn {run_name}',
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                           '[{elapsed}<{remaining}, {rate_fmt}]')
    with torch.no_grad():
        for qs in range(0, nq, args.query_chunk):
            qe = min(qs + args.query_chunk, nq)
            Q = torch.from_numpy(np.ascontiguousarray(qf[qs:qe]).copy()).to(device)
            best = None
            for bs in range(0, bank.shape[0], args.bank_chunk):
                sims = Q @ bank[bs:bs + args.bank_chunk].T
                kk = min(K, sims.shape[1])
                top = torch.topk(sims.float(), kk, dim=1).values
                best = top if best is None else torch.topk(
                    torch.cat([best, top], dim=1), K, dim=1).values
                del sims, top
            topd[qs:qe] = (2.0 - 2.0 * best).clamp_(min=0.0).cpu().numpy()
            pbar.update(qe - qs)
    pbar.close()

    np.save(os.path.join(args.run_dir, 'topk_dist2.npy'), topd)

    scores = topd[:, :args.k].sum(axis=1)
    valid = np.asarray(qok) == 1
    with open(os.path.join(args.run_dir, 'scores.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['patch_name', 'slide', 'label', 'Ascore'])
        for i, r in enumerate(qrows):
            if valid[i]:
                w.writerow([r['patch_name'], r['slide'], r['label'],
                            f'{scores[i]:.6f}'])

    # -------------------------------------------------------------- metrics
    lab = np.array([int(r['label']) for r in qrows])
    sl = np.array([r['slide'] for r in qrows])
    lab_v, sl_v, s_v = lab[valid], sl[valid], scores[valid]

    L = [f'run           : {run_name}',
         f'timestamp     : {datetime.now():%Y-%m-%d %H:%M:%S}',
         f'wall clock    : {(time.time() - t_start) / 60:.1f} min',
         f'note          : {args.note or "-"}',
         '',
         'BANK']
    for d, kept, tot, n_sl in per_dir:
        L.append(f'  {d}: {kept:,} of {tot:,} vectors, {n_sl} slides')
    L += [f'  total        : {total:,} vectors '
          f'({total * dim * 2 / 1e9:.2f} GB fp16)',
          f'  excluded     : {len(drop)} slides'
          + (f' via {args.exclude_slides}' if drop else ''),
          f'  per-slide cap: {args.bank_per_slide_cap or "none"}',
          '',
          'QUERY',
          f'  dir          : {args.query_dir}',
          f'  patches      : {int(valid.sum()):,} scored, '
          f'{int((~valid).sum()):,} dropped as unreadable',
          f'  slides       : {len(set(sl_v))}',
          f'  k            : {args.k}  (top-{K} distances saved; sweep_k.py '
          f'tries every k <= {K} for free)',
          '',
          '=' * 62,
          'METRICS',
          '=' * 62]

    from metrics_report import build_report, load_reference
    ref = load_reference(args.reference_csv) if args.reference_csv else None
    rep, m = build_report(lab_v, s_v, sl_v, ref, args.topk_frac, args.threshold)
    L += rep

    txt = '\n'.join(L)
    open(os.path.join(args.run_dir, 'metrics.txt'), 'w').write(txt + '\n')
    print('\n' + txt)
    print(f'\n-> {args.run_dir}/  (scores.csv, topk_dist2.npy, metrics.txt, '
          f'config.json)')

    # ------------------------------------------------------------- summary
    def g(key, field):
        return m.get(key, {}).get(field, float('nan'))

    cols = [('run', run_name),
            ('timestamp', f'{datetime.now():%Y-%m-%d %H:%M}'),
            ('bank', '+'.join(os.path.basename(d.rstrip('/'))
                              for d in args.bank_dir)),
            ('bank_vectors', total),
            ('queries', int(valid.sum())),
            ('k', args.k),
            ('patch_AUROC', f"{g('patch','auroc'):.2f}"),
            ('patch_AP', f"{g('patch','ap'):.2f}"),
            ('patch_recall', f"{g('patch','recall'):.2f}"),
            ('patch_precision', f"{g('patch','precision'):.2f}"),
            ('patch_F1', f"{g('patch','f1'):.2f}"),
            ('slideMax_AUROC', f"{g('slide_max_der','auroc'):.2f}"),
            ('slideMax_AP', f"{g('slide_max_der','ap'):.2f}"),
            ('slideMax_recall', f"{g('slide_max_der','recall'):.2f}"),
            ('note', args.note)]
    summ = os.path.join(os.path.dirname(os.path.abspath(args.run_dir)),
                        'summary.tsv')
    new = not os.path.exists(summ)
    with open(summ, 'a') as f:
        if new:
            f.write('\t'.join(c for c, _ in cols) + '\n')
        f.write('\t'.join(str(v) for _, v in cols) + '\n')
    print(f'-> appended to {summ}')


if __name__ == '__main__':
    main()