#!/usr/bin/env python
"""
DN2 / Prov-GigaPath -- preflight. Run before the long extraction.

Checks the environment, the manifests, disk space, the model cache and the
leakage invariants. Exits non-zero if anything is a hard FAIL, so it can gate
the real run. Takes ~20 s.
"""
import argparse
import csv
import os
import shutil
import sys

FAILS = []
WARNS = []


def ok(msg):
    print(f'  [ OK ] {msg}')


def fail(msg):
    print(f'  [FAIL] {msg}')
    FAILS.append(msg)


def warn(msg):
    print(f'  [WARN] {msg}')
    WARNS.append(msg)


def head(t):
    print(f'\n== {t} ' + '=' * max(0, 58 - len(t)))


def slides_and_labels(path):
    """-> (n_rows, set_of_slides, n_tumour)"""
    n, sl, pos = 0, set(), 0
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            n += 1
            sl.add(r['slide'])
            pos += r['label'] == '1'
    return n, sl, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--fold_root', required=True)
    ap.add_argument('--tile_label_csv', required=True)
    ap.add_argument('--reference_csv', required=True)
    ap.add_argument('--auroc_script', default='')
    args = ap.parse_args()

    MAN = os.path.join(args.root, 'manifests')
    FEAT = os.path.join(args.root, 'features')

    # ------------------------------------------------------------ packages
    head('environment')
    try:
        import numpy as np
        import torch
        print(f'  numpy {np.__version__} | torch {torch.__version__} '
              f'| python {sys.version.split()[0]}')
        torch.from_numpy(np.zeros((2, 2), dtype=np.float32))
        ok('torch.from_numpy accepts this numpy (no ABI mismatch)')
    except Exception as e:
        fail(f'numpy/torch mismatch: {type(e).__name__}: {e}')
        print('\nFATAL -- fix this first, nothing else can work.')
        sys.exit(1)

    for mod, minver in [('timm', '1.0.3'), ('tqdm', None),
                        ('sklearn', None), ('scipy', None),
                        ('torchvision', None), ('PIL', None)]:
        try:
            m = __import__(mod)
            v = getattr(m, '__version__', '?')
            if mod == 'timm':
                try:
                    tup = tuple(int(x) for x in v.split('.')[:2])
                except ValueError:
                    tup = None
                if tup is None:
                    warn(f'timm version unreadable ({v!r}); need >= {minver}')
                elif tup < (1, 0):
                    fail(f'timm {v} is too old for Prov-GigaPath '
                         f'(need >= {minver}): pip install "timm>=1.0.3"')
                else:
                    ok(f'{mod} {v}')
            else:
                ok(f'{mod} {v}')
        except Exception as e:
            fail(f'cannot import {mod}: {e}')

    try:
        from sklearn.metrics import roc_auc_score
        roc_auc_score([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
        ok('sklearn.metrics works (no numpy ABI break)')
    except Exception as e:
        fail(f'sklearn ABI broken: {e}')

    # ----------------------------------------------------------------- gpu
    head('gpu')
    if not torch.cuda.is_available():
        fail('CUDA not available -- extraction would run on CPU for weeks')
    else:
        free, total = torch.cuda.mem_get_info()
        ok(f'{torch.cuda.get_device_name(0)}  '
           f'{total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free')
        if free < 12e9:
            warn(f'only {free / 1e9:.1f} GB free -- something else is using the '
                 f'card. Bank A+B scoring wants ~10 GB. Check nvidia-smi.')

    # --------------------------------------------------------- model cache
    head('prov-gigapath weights')
    hits = []
    for base in [os.environ.get('HF_HOME', ''),
                 os.path.expanduser('~/.cache/huggingface')]:
        if base and os.path.isdir(base):
            for dp, _, fns in os.walk(base):
                if 'prov-gigapath' in dp.replace('--', '/'):
                    for fn in fns:
                        p = os.path.join(dp, fn)
                        if os.path.isfile(p) and os.path.getsize(p) > 1e9:
                            hits.append((p, os.path.getsize(p)))
    if hits:
        ok(f'cached: {hits[0][1] / 1e9:.2f} GB (no re-download)')
    else:
        warn('checkpoint not found in the HF cache -- first run will download '
             '4.54 GB. Fine, just slow to start.')

    # ----------------------------------------------------------- manifests
    head('manifests')
    expect = ['bank.csv', 'bank_tumorslide_normals.csv', 'test.csv', 'dev.csv',
              'holdout_slides.txt']
    missing = [f for f in expect if not os.path.exists(os.path.join(MAN, f))]
    if missing:
        fail(f'missing {missing} -- run: bash run_all.sh step1')
        print('\nFATAL -- cannot continue without manifests.')
        sys.exit(1)

    info, grand = {}, 0
    for name in ['bank.csv', 'bank_tumorslide_normals.csv', 'test.csv', 'dev.csv']:
        n, sl, pos = slides_and_labels(os.path.join(MAN, name))
        info[name] = (n, sl, pos)
        grand += n
        ok(f'{name:<32} {n:>10,} patches  {len(sl):>4} slides  tumour={pos:,}')
    hold = {l.strip() for l in open(os.path.join(MAN, 'holdout_slides.txt'))
            if l.strip()}
    ok(f'holdout_slides.txt              {len(hold)} slides')

    # spot-check that the paths in a manifest still resolve
    with open(os.path.join(MAN, 'test.csv'), newline='') as f:
        rows = [r for _, r in zip(range(3), csv.DictReader(f))]
    bad = [r['path'] for r in rows if not os.path.exists(r['path'])]
    if bad:
        fail(f'manifest paths do not resolve, e.g. {bad[0]}')
    else:
        ok('sampled manifest paths exist on disk')

    for p, what in [(args.tile_label_csv, 'tile_label.csv'),
                    (args.reference_csv, 'reference.csv'),
                    (args.fold_root, 'fold1 root')]:
        (ok if os.path.exists(p) else fail)(f'{what}: {p}')
    if args.auroc_script:
        (ok if os.path.exists(args.auroc_script) else warn)(
            f'auroc_from_csv.py: {args.auroc_script}')

    # ------------------------------------------------------------- leakage
    head('leakage invariants')
    bankA = info['bank.csv'][1]
    bankB = info['bank_tumorslide_normals.csv'][1]
    test = info['test.csv'][1]
    dev = info['dev.csv'][1]
    checks = [('bank A slides n test slides', len(bankA & test)),
              ('bank B slides n test slides', len(bankB & test)),
              ('dev slides in bank after holdout removal',
               len(((bankA | bankB) - hold) & dev)),
              ('dev slides NOT in the holdout list', len(dev - hold))]
    for label, v in checks:
        (ok if v == 0 else fail)(f'{label}: {v} (must be 0)')

    # ---------------------------------------------------- extraction state
    head('feature dirs')
    plan = [('dev', 'dev.csv'), ('bank', 'bank.csv'), ('test', 'test.csv'),
            ('bankB', 'bank_tumorslide_normals.csv')]
    todo = 0
    for d, man in plan:
        n = info[man][0]
        fd = os.path.join(FEAT, d)
        pj = os.path.join(fd, 'progress.json')
        if os.path.exists(pj):
            import json
            done = json.load(open(pj))['n_done']
            if done >= n:
                ok(f'{d:<6} complete ({done:,}/{n:,})')
            else:
                todo += n - done
                ok(f'{d:<6} partial  ({done:,}/{n:,}) -- will resume')
        else:
            todo += n
            ok(f'{d:<6} not started ({n:,} patches)')

    # ---------------------------------------------------------------- disk
    head('disk & time')
    need = grand * 1536 * 2 + 3 * 1_100_000 * 50 * 4      # features + topk files
    free = shutil.disk_usage(args.root).free
    print(f'  features + score files need ~{need / 1e9:.1f} GB; '
          f'{free / 1e9:.0f} GB free')
    (ok if free > need * 2 else fail)('disk space')
    for rate in (90, 130):
        print(f'  {todo:,} patches left at {rate} img/s -> '
              f'{todo / rate / 3600:.1f} h')

    # -------------------------------------------------------------- verdict
    head('verdict')
    if FAILS:
        print(f'  {len(FAILS)} FAILURE(S) -- do not start the run:')
        for f_ in FAILS:
            print(f'    - {f_}')
        sys.exit(1)
    if WARNS:
        print(f'  {len(WARNS)} warning(s), none blocking:')
        for w in WARNS:
            print(f'    - {w}')
    print('  ALL CHECKS PASSED -- safe to run: bash run_all.sh extract')


if __name__ == '__main__':
    main()
