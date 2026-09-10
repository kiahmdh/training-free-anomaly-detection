#!/usr/bin/env python
"""
DN2 / Prov-GigaPath on Camelyon16 -- Step 1: build manifests.

Reads ONLY. Never writes anything into the Camelyon16 data folder.

Produces, in --out_dir:
    bank.csv            one-class feature bank: every patch under
                        train/0_normal and validation/0_normal
    test.csv            every patch under test/ with its TRUE per-patch label
    dev.csv             a held-out tuning set (never overlaps test):
                          - patches from train/validation 1_tumor slides
                            (both labels, from tile_label.csv)
                          - patches from a few validation normal slides that
                            are excluded from the bank at scoring time
    holdout_slides.txt  the normal slides to exclude from the bank when
                        scoring dev (prevents distance-0 self-matches)
    manifest_report.txt human-readable sanity report -- READ THIS

Label source
------------
The 0_normal / 1_tumor folders are SLIDE-level groupings, not patch labels.
Patches inside 1_tumor/<slide>/ are mostly normal tissue. True per-patch
labels come from tile_label.csv, whose keys look like:

    datasets/camelyon16/1_tumor/test_013.tif/WSI_temp_files/10/174_0-17.jpeg
                        ^class_dir ^slide.tif                  ^basename

We join on (slide, basename), which is unique across Camelyon16 because slide
names (normal_XXX / tumor_XXX / test_XXX) are globally unique. The script
detects and reports duplicate keys with conflicting labels rather than
silently picking one.
"""
import argparse
import csv
import os
import random
import sys
from collections import defaultdict

IMG_EXTS = ('.jpeg', '.jpg', '.png')


# ----------------------------------------------------------------- csv parse
def parse_csv_key(path):
    """
    '.../1_tumor/test_013.tif/WSI_temp_files/10/174_0-17.jpeg'
        -> ('1_tumor', 'test_013', '174_0-17.jpeg')
    Returns None if the row does not look like a patch path.
    """
    parts = path.replace('\\', '/').strip().split('/')
    if len(parts) < 2:
        return None
    tif_idx = None
    for i, p in enumerate(parts):
        low = p.lower()
        if low.endswith('.tif') or low.endswith('.tiff'):
            tif_idx = i
            break
    if tif_idx is None:
        return None
    slide = parts[tif_idx].rsplit('.', 1)[0]
    class_dir = parts[tif_idx - 1] if tif_idx >= 1 else ''
    basename = parts[-1]
    if not basename.lower().endswith(IMG_EXTS):
        return None
    return class_dir, slide, basename


def load_tile_labels(csv_path, report):
    """(slide, basename) -> int label. Reports conflicts and slide coverage."""
    labels = {}
    conflicts = []
    unparsed = 0
    n_rows = 0
    slides_seen = defaultdict(int)
    class_of_slide = {}

    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        report.append(f"tile_label.csv header: {header}")
        for row in reader:
            if len(row) < 2:
                continue
            n_rows += 1
            parsed = parse_csv_key(row[0])
            if parsed is None:
                unparsed += 1
                continue
            class_dir, slide, basename = parsed
            try:
                lab = int(float(row[1]))
            except ValueError:
                unparsed += 1
                continue
            key = (slide, basename)
            if key in labels and labels[key] != lab:
                if len(conflicts) < 20:
                    conflicts.append((key, labels[key], lab))
            else:
                labels[key] = lab
            slides_seen[slide] += 1
            class_of_slide.setdefault(slide, class_dir)

    report.append(f"rows read              : {n_rows:,}")
    report.append(f"rows unparsed/skipped  : {unparsed:,}")
    report.append(f"unique (slide,patch)   : {len(labels):,}")
    report.append(f"distinct slides in csv : {len(slides_seen):,}")

    fams = defaultdict(int)
    for s in slides_seen:
        fams[s.split('_')[0]] += 1
    report.append(f"slide name families    : {dict(fams)}")

    pos = sum(1 for v in labels.values() if v == 1)
    report.append(f"patch labels: tumor(1)={pos:,}  normal(0)={len(labels) - pos:,}")

    if conflicts:
        report.append(f"!! {len(conflicts)}+ CONFLICTING duplicate keys, e.g. {conflicts[:5]}")
    else:
        report.append("no conflicting duplicate keys")
    return labels, slides_seen


# ---------------------------------------------------------------- disk walk
def walk_patches(root):
    """<root>/<slide>/<patch>.jpeg -> list of (abs_path, slide, basename)."""
    out = []
    if not os.path.isdir(root):
        return out
    for slide in sorted(os.listdir(root)):
        sdir = os.path.join(root, slide)
        if not os.path.isdir(sdir):
            continue
        for fn in sorted(os.listdir(sdir)):
            if fn.lower().endswith(IMG_EXTS):
                out.append((os.path.join(sdir, fn), slide, fn))
    return out


def write_manifest(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'patch_name', 'slide', 'label'])
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold_root', required=True,
                    help='e.g. /home/user01/camelyon16/single/fold1')
    ap.add_argument('--tile_label_csv', required=True,
                    help='e.g. /home/user01/camelyon16/tile_label.csv')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--holdout_val_slides', type=int, default=8,
                    help='normal slides reserved as dev normals')
    ap.add_argument('--holdout_tumor_slides', type=int, default=12,
                    help='train/val TUMOUR slides reserved for dev; their '
                         'patches never enter bank_tumorslide_normals at '
                         'scoring time')
    ap.add_argument('--dev_holdout_patches', type=int, default=40000)
    ap.add_argument('--dev_tumorslide_patches', type=int, default=60000,
                    help='patches sampled from train/val 1_tumor slides')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    report = []

    report.append('=' * 62)
    report.append('TILE LABEL CSV')
    report.append('=' * 62)
    labels, csv_slides = load_tile_labels(args.tile_label_csv, report)

    # ------------------------------------------------------------- bank
    report.append('')
    report.append('=' * 62)
    report.append('BANK (one-class training set)')
    report.append('=' * 62)
    bank_src = []
    for phase in ('train', 'validation'):
        root = os.path.join(args.fold_root, phase, '0_normal')
        got = walk_patches(root)
        report.append(f"{phase}/0_normal : {len(got):,} patches, "
                      f"{len({s for _, s, _ in got})} slides")
        bank_src += got

    # verify: nothing labelled tumor may enter the bank
    bank_rows, bank_missing, bank_bad = [], 0, []
    for p, slide, base in bank_src:
        lab = labels.get((slide, base))
        if lab is None:
            bank_missing += 1
        elif lab != 0:
            bank_bad.append((slide, base))
        bank_rows.append((p, base, slide, 0))
    report.append(f"bank total             : {len(bank_rows):,}")
    report.append(f"not found in csv       : {bank_missing:,}"
                  f"  ({100.0 * bank_missing / max(len(bank_rows), 1):.2f}%)")
    if bank_bad:
        report.append(f"!! {len(bank_bad)} BANK PATCHES LABELLED TUMOR -- STOP. "
                      f"e.g. {bank_bad[:5]}")
    else:
        report.append("all csv-covered bank patches are label 0  [OK]")

    # ------------------------------------------------------------- test
    report.append('')
    report.append('=' * 62)
    report.append('TEST')
    report.append('=' * 62)
    test_rows, test_missing = [], 0
    n_pos = 0
    for class_dir in ('0_normal', '1_tumor'):
        root = os.path.join(args.fold_root, 'test', class_dir)
        got = walk_patches(root)
        for p, slide, base in got:
            if class_dir == '0_normal':
                lab = 0                       # normal slide -> all patches normal
            else:
                lab = labels.get((slide, base))
                if lab is None:
                    test_missing += 1
                    continue                  # cannot score without a true label
            n_pos += (lab == 1)
            test_rows.append((p, base, slide, lab))
        report.append(f"test/{class_dir} : {len(got):,} patches, "
                      f"{len({s for _, s, _ in got})} slides")
    report.append(f"test kept              : {len(test_rows):,}"
                  f"  tumor={n_pos:,}  normal={len(test_rows) - n_pos:,}")
    report.append(f"dropped (no csv label) : {test_missing:,}")
    if test_missing:
        report.append("!! dropped patches from 1_tumor slides lacking a csv label. "
                      "If this is large, the join key is wrong -- do not proceed.")

    # ------------------------------------ bank B: normals from tumour slides
    report.append('')
    report.append('=' * 62)
    report.append('BANK-B (normal-LABELLED patches inside train/val tumour slides)')
    report.append('=' * 62)
    ts_rows, ts_missing = [], 0
    for phase in ('train', 'validation'):
        got = walk_patches(os.path.join(args.fold_root, phase, '1_tumor'))
        report.append(f"{phase}/1_tumor : {len(got):,} patches, "
                      f"{len({s_ for _, s_, _ in got})} slides")
        for p, slide, base in got:
            lab = labels.get((slide, base))
            if lab is None:
                ts_missing += 1
            else:
                ts_rows.append((p, base, slide, lab))
    bankB_rows = [r for r in ts_rows if r[3] == 0]
    n_tum = sum(1 for r in ts_rows if r[3] == 1)
    report.append(f"tumour-slide patches   : {len(ts_rows):,} "
                  f"(tumour={n_tum:,}  normal={len(bankB_rows):,})")
    report.append(f"no csv label           : {ts_missing:,}")
    report.append("NOTE: these normals come from slides that DO contain tumour. "
                  "Boundary tiles may hold unannotated tumour cells, which "
                  "contaminates a one-class bank. Compare A vs A+B on dev.")

    # ------------------------------------------------------- dev / holdout
    report.append('')
    report.append('=' * 62)
    report.append('DEV (for choosing k and bank -- never touches test)')
    report.append('=' * 62)
    norm_slides = sorted({r[2] for r in bank_rows})
    random.shuffle(norm_slides)
    hold_norm = sorted(norm_slides[:args.holdout_val_slides])

    tum_slides = sorted({r[2] for r in ts_rows})
    random.shuffle(tum_slides)
    hold_tum = sorted(tum_slides[:args.holdout_tumor_slides])

    holdout = hold_norm + hold_tum
    with open(os.path.join(args.out_dir, 'holdout_slides.txt'), 'w') as f:
        f.write('\n'.join(holdout) + '\n')
    report.append(f"holdout normal slides  : {len(hold_norm)} -> {hold_norm}")
    report.append(f"holdout tumour slides  : {len(hold_tum)} -> {hold_tum}")

    dev_norm = [r for r in bank_rows if r[2] in set(hold_norm)]
    if len(dev_norm) > args.dev_holdout_patches:
        dev_norm = random.sample(dev_norm, args.dev_holdout_patches)

    hs = set(hold_tum)
    pos = [r for r in ts_rows if r[3] == 1 and r[2] in hs]
    neg = [r for r in ts_rows if r[3] == 0 and r[2] in hs]
    half = args.dev_tumorslide_patches // 2
    if len(pos) > half:
        pos = random.sample(pos, half)
    if len(neg) > half:
        neg = random.sample(neg, half)
    dev_rows = dev_norm + pos + neg
    random.shuffle(dev_rows)
    d_pos = sum(1 for r in dev_rows if r[3] == 1)
    report.append(f"dev from holdout tumour slides : tumour={len(pos):,} "
                  f"normal={len(neg):,}")
    report.append(f"dev from holdout normal slides : {len(dev_norm):,}")
    report.append(f"dev total              : {len(dev_rows):,}"
                  f"  tumour={d_pos:,}  normal={len(dev_rows) - d_pos:,}")

    write_manifest(os.path.join(args.out_dir, 'bank_tumorslide_normals.csv'),
                   bankB_rows)
    write_manifest(os.path.join(args.out_dir, 'bank.csv'), bank_rows)
    write_manifest(os.path.join(args.out_dir, 'test.csv'), test_rows)
    write_manifest(os.path.join(args.out_dir, 'dev.csv'), dev_rows)

    # leakage assertions -------------------------------------------------
    report.append('')
    report.append('=' * 62)
    report.append('LEAKAGE CHECKS')
    report.append('=' * 62)
    bank_slides = {r[2] for r in bank_rows} | {r[2] for r in bankB_rows}
    test_slides = {r[2] for r in test_rows}
    dev_slides = {r[2] for r in dev_rows}
    report.append(f"bank slides n test slides : {len(bank_slides & test_slides)} "
                  f"(must be 0)")
    report.append(f"bank-minus-holdout n dev  : "
                  f"{len((bank_slides - set(holdout)) & dev_slides)} (must be 0)")
    report.append(f"bankB slides n test slides: "
                  f"{len({r[2] for r in bankB_rows} & test_slides)} (must be 0)")

    txt = '\n'.join(report)
    with open(os.path.join(args.out_dir, 'manifest_report.txt'), 'w') as f:
        f.write(txt + '\n')
    print(txt)
    print(f"\nwrote bank.csv / bank_tumorslide_normals.csv / test.csv / "
          f"dev.csv to {args.out_dir}")

    if bank_bad:
        sys.exit("ABORT: tumor-labelled patches found in the bank.")


if __name__ == '__main__':
    main()
