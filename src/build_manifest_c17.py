#!/usr/bin/env python
"""
Build a Camelyon17 query manifest for the DN2 / Prov-GigaPath pipeline.

The bank is untouched -- it stays Camelyon16 normal tissue. This only builds the
set of patches to score, so the run measures transfer to an unseen cohort.

Layout (verified from the earlier Camelyon17 work):
    <fold_root>/<split>/{0_normal,1_tumor}/patient_XXX_node_Y/<x>_<y>-1.jpeg
The "slide" unit is the node, patient_XXX_node_Y.

Labels:
    0_normal patches -> 0 (the node contains no tumour anywhere)
    1_tumor  patches -> the true per-patch label from tile_label.csv,
                        keyed as "patient_XXX_node_Y/<file>.jpeg"

Camelyon17 grades nodes as negative / ITC / micro / macro. For the binary task,
negative -> normal and ITC/micro/macro -> tumour, which is what reference.csv's
`type` column already encodes. ITC lesions are sub-0.2 mm, so expect them to be
the misses; that is a known-hard category, not necessarily a model failure.

Usage:
  python build_manifest_c17.py \
    --fold_root /home/user01/camelyon17/clam/no_otsu/patches/stratified_60_20_20_rand42/fold_0 \
    --tile_label_csv /home/user01/camelyon17/clam/no_otsu/patches/tile_label.csv \
    --out_csv manifests/c17_test.csv --splits test
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

IMG_EXTS = ('.jpeg', '.jpg', '.png')


def load_tile_labels(path, report):
    """'patient_XXX_node_Y/<file>.jpeg' -> int label."""
    labels, conflicts, unparsed, n = {}, [], 0, 0
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        report.append(f'tile_label.csv header : {header}')
        for row in reader:
            if len(row) < 2:
                continue
            n += 1
            key = row[0].strip().replace('\\', '/')
            try:
                lab = int(float(row[1]))
            except ValueError:
                unparsed += 1
                continue
            if key in labels and labels[key] != lab:
                if len(conflicts) < 10:
                    conflicts.append(key)
            labels[key] = lab
    report.append(f'rows read             : {n:,}')
    report.append(f'unparsed              : {unparsed:,}')
    report.append(f'unique keys           : {len(labels):,}')
    pos = sum(1 for v in labels.values() if v == 1)
    report.append(f'tumour(1)={pos:,}  normal(0)={len(labels) - pos:,}')
    nodes = {k.split('/')[0] for k in labels}
    report.append(f'nodes in csv          : {len(nodes):,}')
    if conflicts:
        report.append(f'!! conflicting duplicate keys: {conflicts[:5]}')
    else:
        report.append('no conflicting duplicate keys')
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold_root', required=True)
    ap.add_argument('--tile_label_csv', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--splits', nargs='+', default=['test'],
                    help='which split folders to include. Default: test. '
                         'Pass "train test validation" to pool all three -- '
                         'legitimate here, since the bank never saw any '
                         'Camelyon17 data.')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or '.',
                exist_ok=True)
    report = ['=' * 62, 'CAMELYON17 QUERY MANIFEST', '=' * 62]
    labels = load_tile_labels(args.tile_label_csv, report)

    rows, missing, per_split = [], 0, defaultdict(int)
    n_pos = 0
    for split in args.splits:
        for cls in ('0_normal', '1_tumor'):
            root = os.path.join(args.fold_root, split, cls)
            if not os.path.isdir(root):
                report.append(f'!! missing directory: {root}')
                continue
            for node in sorted(os.listdir(root)):
                ndir = os.path.join(root, node)
                if not os.path.isdir(ndir):
                    continue
                for fn in sorted(os.listdir(ndir)):
                    if not fn.lower().endswith(IMG_EXTS):
                        continue
                    if cls == '0_normal':
                        lab = 0
                    else:
                        lab = labels.get(f'{node}/{fn}')
                        if lab is None:
                            missing += 1
                            continue
                    n_pos += (lab == 1)
                    rows.append((os.path.join(ndir, fn), fn, node, lab))
                    per_split[f'{split}/{cls}'] += 1

    report.append('')
    for k in sorted(per_split):
        report.append(f'{k:<24}: {per_split[k]:,}')
    nodes = {r[2] for r in rows}
    report.append('')
    report.append(f'total patches kept    : {len(rows):,}')
    report.append(f'  tumour              : {n_pos:,}')
    report.append(f'  normal              : {len(rows) - n_pos:,}'
                  f'  ({100 * n_pos / max(len(rows), 1):.2f}% prevalence)')
    report.append(f'nodes (slide unit)    : {len(nodes):,}')
    report.append(f'dropped, no csv label : {missing:,}')
    if missing > 0.01 * max(len(rows), 1):
        report.append('!! more than 1% dropped -- the join key is probably '
                      'wrong. Do not proceed.')

    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'patch_name', 'slide', 'label'])
        w.writerows(rows)

    txt = '\n'.join(report)
    print(txt)
    open(os.path.splitext(args.out_csv)[0] + '_report.txt', 'w').write(txt + '\n')
    print(f'\nwrote {len(rows):,} rows -> {args.out_csv}')
    if not rows:
        sys.exit('ABORT: no patches found -- check --fold_root and --splits')


if __name__ == '__main__':
    main()
