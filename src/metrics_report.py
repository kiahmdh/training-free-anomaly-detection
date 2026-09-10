#!/usr/bin/env python
"""
Full metric report: patch level and slide level (max + top-k%% mean), each with
AUROC, AP, and an operating point (threshold, recall, precision, specificity,
F1, TP/FP/FN/TN).

Same definitions as the auroc_from_csv.py used elsewhere in this project, so
numbers are directly comparable. Imported by dn2_score.py; also runnable
standalone on any CSV with columns patch_name, slide, label, Ascore.

Recall needs a decision threshold, unlike AUROC/AP which are threshold-free.
Default is the Youden-optimal point (max of sensitivity + specificity - 1),
computed SEPARATELY per level because the score scales differ: a slide's max
over thousands of patches naturally sits much higher than one patch's score.
Override with --threshold / threshold=.

Slide truth comes from two independent sources when both are available:
  derived    -- a slide is positive if any of its patches is labelled tumour
  reference  -- the `type` column of reference.csv
They should agree on Camelyon16; a disagreement is reported, not hidden.
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


# ------------------------------------------------------------------ helpers
def load_reference(path):
    """slide (lower, no extension) -> 1 tumour / 0 normal."""
    ref = {}
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
                ref[key] = 1
            elif t == 'normal':
                ref[key] = 0
    return ref


def operating_point(y, s, threshold=None):
    """-> (threshold, recall, precision, specificity, f1, (tp, fp, fn, tn))"""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    if threshold is None:
        fpr, tpr, thr = roc_curve(y, s)
        threshold = float(thr[int(np.argmax(tpr - fpr))])
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    rec = tp / (tp + fn) if (tp + fn) else float('nan')
    prec = tp / (tp + fp) if (tp + fp) else float('nan')
    spec = tn / (tn + fp) if (tn + fp) else float('nan')
    f1 = (2 * prec * rec / (prec + rec)
          if (prec + rec) and np.isfinite(prec + rec) else float('nan'))
    return threshold, rec, prec, spec, f1, (tp, fp, fn, tn)


def _level_block(name, y, s, fixed_thr, out, stash, key):
    """AUROC/AP + operating point for one level. Appends lines to `out`."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    out.append(f'{name}: n={len(y):,}  tumour={int((y == 1).sum()):,}  '
               f'normal={int((y == 0).sum()):,}')
    if not ((y == 1).any() and (y == 0).any()):
        out.append('  only one class present -> metrics undefined')
        return
    auroc = roc_auc_score(y, s) * 100
    ap = average_precision_score(y, s) * 100
    thr, rec, prec, spec, f1, (tp, fp, fn, tn) = operating_point(y, s, fixed_thr)
    tag = 'fixed' if fixed_thr is not None else 'Youden-opt'
    out += [f'  AUROC       = {auroc:.2f}',
            f'  AP          = {ap:.2f}',
            f'  threshold   = {thr:.6f} ({tag})',
            f'  recall      = {rec * 100:.2f}   (sensitivity, TP/(TP+FN))',
            f'  precision   = {prec * 100:.2f}',
            f'  specificity = {spec * 100:.2f}',
            f'  F1          = {f1 * 100:.2f}',
            f'  TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}']
    stash[key] = dict(auroc=auroc, ap=ap, threshold=thr, recall=rec * 100,
                      precision=prec * 100, specificity=spec * 100, f1=f1 * 100,
                      tp=tp, fp=fp, fn=fn, tn=tn)


# ------------------------------------------------------------------- report
def build_report(labels, scores, slides, reference=None, topk_frac=0.01,
                 threshold=None):
    """-> (list_of_lines, dict_of_headline_numbers)"""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    sl = np.asarray(slides)
    out, stash = [], {}

    finite = np.isfinite(s)
    n_bad = int((~finite).sum())
    if n_bad:
        bad = sorted(set(sl[~finite].tolist()))
        out += [f'WARNING: {n_bad:,} / {len(s):,} patches had NaN/inf scores; '
                f'dropped from metrics.',
                f'  affected slides ({len(bad)}): {bad[:10]}'
                + (' ...' if len(bad) > 10 else ''), '']
        y, s, sl = y[finite], s[finite], sl[finite]

    _level_block('PATCH-LEVEL', y, s, threshold, out, stash, 'patch')

    # ------------------------------------------------------- slide level
    by = defaultdict(list)
    for k, v in zip(sl, s):
        by[k].append(float(v))
    derived = {k: int((y[sl == k] == 1).any()) for k in by}

    def top_frac_mean(a):
        a = np.sort(np.asarray(a, dtype=float))[::-1]
        n = max(1, int(np.ceil(len(a) * topk_frac)))
        return float(a[:n].mean())

    aggs = [('max', lambda a: float(np.max(a))),
            (f'top-{topk_frac * 100:g}%-mean', top_frac_mean)]

    truths = [('derived: slide positive if any patch is tumour', derived, 'der')]
    if reference:
        cov = {k: reference[k.lower()] for k in by if k.lower() in reference}
        miss = [k for k in by if k.lower() not in reference]
        out.append('')
        if cov:
            out.append(f'reference.csv covers {len(cov)}/{len(by)} slides'
                       + (f'; {len(miss)} missing, e.g. {sorted(miss)[:5]}'
                          if miss else ''))
            agree = sum(1 for k in cov if cov[k] == derived[k])
            identical = (agree == len(cov) == len(by))
            if agree != len(cov):
                # direction matters enormously
                ref_pos = [k for k in cov if cov[k] == 1 and derived[k] == 0]
                ref_neg = [k for k in cov if cov[k] == 0 and derived[k] == 1]
                out.append(f'  !! DISAGREEMENT on {len(cov) - agree}/{len(cov)} '
                           f'slides:')
                if ref_pos:
                    frac = [f'{k}({int((y[sl == k] == 1).sum())}/'
                            f'{int((sl == k).sum())})' for k in sorted(ref_pos)[:5]]
                    out += [f'    {len(ref_pos)} slides reference=tumour but no '
                            f'tumour patch present, e.g. {frac}',
                            '      (tumour_patches/patches_scored shown). BENIGN '
                            'if this query set is a subsample -- a small',
                            '      metastasis can be missed entirely by '
                            'sampling. On a FULL test run this must be 0;',
                            '      if it is not, the patch labels and '
                            'reference.csv genuinely conflict.']
                if ref_neg:
                    out += [f'    !!! {len(ref_neg)} slides reference=NORMAL but '
                            f'contain tumour-labelled patches: '
                            f'{[str(x) for x in sorted(ref_neg)[:5]]}',
                            '      This is NEVER benign. A slide the reference '
                            'calls normal cannot hold tumour tiles.',
                            '      Stop and check the label join before '
                            'reporting anything.']
                truths.append(('reference.csv `type` column', cov, 'ref'))
            elif identical:
                out.append('  derived and reference.csv agree on every slide '
                           '[OK] -- the reference-based blocks would be '
                           'identical, so they are omitted')
            else:
                out.append('  derived and reference.csv agree where they '
                           'overlap [OK]; reference-based blocks below cover '
                           'the subset')
                truths.append(('reference.csv `type` column', cov, 'ref'))
        else:
            out.append(f'reference.csv covers 0/{len(by)} slides -- using '
                       f'derived slide truth only (expected for dev runs: '
                       f'the official Camelyon16 reference.csv lists test '
                       f'slides only)')

    for tname, truth, tkey in truths:
        for aname, fn in aggs:
            ks = sorted(truth)
            out.append('')
            _level_block(f'SLIDE-LEVEL [{aname}] (truth = {tname})',
                         [truth[k] for k in ks], [fn(by[k]) for k in ks],
                         threshold, out, stash, f'slide_{aname}_{tkey}')
    return out, stash


# --------------------------------------------------------------- standalone
def read_scores(path):
    y, s, sl = [], [], []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            try:
                v = float(r['Ascore'])
            except (ValueError, KeyError):
                v = float('nan')
            y.append(int(r['label']))
            s.append(v)
            sl.append(r['slide'])
    return y, s, sl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='scores.csv from dn2_score.py')
    ap.add_argument('--reference_csv', default=None)
    ap.add_argument('--topk_frac', type=float, default=0.01)
    ap.add_argument('--threshold', type=float, default=None,
                    help='fixed decision threshold; default = Youden-optimal '
                         'per level')
    ap.add_argument('--out', default=None, help='also write the report here')
    args = ap.parse_args()

    y, s, sl = read_scores(args.csv)
    ref = load_reference(args.reference_csv) if args.reference_csv else None
    lines, _ = build_report(y, s, sl, ref, args.topk_frac, args.threshold)
    txt = '\n'.join(lines)
    print(txt)
    if args.out:
        open(args.out, 'w').write(txt + '\n')
        print(f'\n-> {args.out}')


if __name__ == '__main__':
    main()