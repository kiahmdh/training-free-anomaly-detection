#!/usr/bin/env python
"""
DN2 / Prov-GigaPath on Camelyon16 -- Step 5: stain / centre confound check.

A kNN detector measures "unlike anything in the bank". Camelyon16 comes from
two centres with visibly different H&E staining, so a normal test slide stained
unlike your bank can score high for reasons that have nothing to do with
metastasis. Run this BEFORE you believe an AUROC.

Method: look only at patches whose TRUE label is 0. In a clean run their median
score should be similar across slides. If slides separate into two clusters --
or into two groups by the `center` column of reference.csv -- your AUROC is
partly measuring stain, and you should add Macenko normalisation or per-slide
score standardisation before drawing conclusions.
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np


def load_reference(path):
    """slide (lower, no .tif) -> dict of that row's columns."""
    ref = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            k = list(r.keys())[0]
            name = str(r[k]).strip().lower()
            if name.endswith('.tif'):
                name = name[:-4]
            ref[name] = r
    return ref


_LOG = []


def print(*a, **k):          # tee to stdout and the report file
    import builtins
    builtins.print(*a, **k)
    _LOG.append(' '.join(str(x) for x in a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True,
                    help='results folder from dn2_score.py (reads scores.csv, '
                         'writes stain_report.txt)')
    ap.add_argument('--reference_csv', default=None)
    args = ap.parse_args()
    args.csv = os.path.join(args.run_dir, 'scores.csv')

    per = defaultdict(list)
    with open(args.csv, newline='') as f:
        for r in csv.DictReader(f):
            if int(r['label']) == 0:                 # normal patches only
                per[r['slide']].append(float(r['Ascore']))

    meds = {s: float(np.median(v)) for s, v in per.items() if len(v) >= 50}
    vals = np.array(list(meds.values()))
    print(f'slides with >=50 normal patches: {len(meds)}')
    print(f'median-of-normal-scores: min={vals.min():.4f} '
          f'p25={np.percentile(vals,25):.4f} med={np.median(vals):.4f} '
          f'p75={np.percentile(vals,75):.4f} max={vals.max():.4f}')
    print(f'spread ratio max/min = {vals.max()/max(vals.min(),1e-9):.2f}   '
          f'(a large ratio means normal tissue is scored very differently '
          f'depending only on which slide it came from)\n')

    worst = sorted(meds.items(), key=lambda kv: -kv[1])[:10]
    print('10 slides whose NORMAL tissue scores highest (prime stain suspects):')
    for s, m in worst:
        print(f'  {s:<20} median={m:.4f}  n={len(per[s]):,}')

    if args.reference_csv:
        ref = load_reference(args.reference_csv)
        key = None
        for r in ref.values():
            for c in r:
                if c.strip().lower() == 'center':
                    key = c
            break
        if key is None:
            print('\nno `center` column in reference.csv -- per-slide spread above '
                  'is still the check that matters')
            return
        by = defaultdict(list)
        for s, m in meds.items():
            row = ref.get(s.lower())
            if row:
                by[row[key]].append(m)
        print('\nmedian normal-patch score grouped by centre:')
        for c, v in sorted(by.items()):
            print(f'  centre {c}: n_slides={len(v):>3}  median={np.median(v):.4f}')
        print('If these differ a lot, the confound is real.')


if __name__ == '__main__':
    import sys
    _rd = None
    for i, a in enumerate(sys.argv):
        if a == '--run_dir':
            _rd = sys.argv[i + 1]
    try:
        main()
    finally:
        if _rd:
            with open(os.path.join(_rd, 'stain_report.txt'), 'w') as f:
                f.write('\n'.join(_LOG) + '\n')
            import builtins
            builtins.print(f'\n-> {_rd}/stain_report.txt')