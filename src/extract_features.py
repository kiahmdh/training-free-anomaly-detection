#!/usr/bin/env python
"""
DN2 / Prov-GigaPath on Camelyon16 -- Step 2: extract features once, to disk.

Reads a manifest CSV (path,patch_name,slide,label) and writes, into --out_dir:

    features.npy   float16 (N, 1536), L2-normalised, row-aligned to the manifest
    ok.npy         uint8  (N,)  1 = decoded fine, 0 = unreadable/blank -> drop
    progress.json  rows completed, so an interrupted run resumes where it died
    manifest.csv   a copy of the input, so features can never drift from labels

Preprocessing is the OFFICIAL Prov-GigaPath recipe:
    Resize(256, bicubic) -> CenterCrop(224) -> ToTensor -> ImageNet mean/std
For 256x256 Camelyon16 patches the resize is a no-op, so this is exactly the
centre crop the tile encoder was trained on. Do NOT substitute Resize((224,224)):
that squashes the tile and shifts the feature distribution.

Run this twice (bank, test) plus once for dev. Always inside tmux.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
FEAT_DIM = 1536


def build_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


class PatchList(Dataset):
    """Returns (tensor, ok_flag). Never raises -- a bad JPEG yields zeros+ok=0."""

    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert('RGB')
            return self.transform(img), 1
        except (OSError, ValueError, SyntaxError):
            # genuinely unreadable / truncated JPEG -- skip it
            return torch.zeros(3, 224, 224), 0


def read_manifest(path):
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--num_workers', type=int, default=12)
    ap.add_argument('--limit', type=int, default=0,
                    help='>0 : a random N-row subset (use for the pilot). '
                         'Random, not the first N: the manifests are ordered '
                         'by slide, so the head of test.csv is all normal and '
                         'would give a single-class pilot with no AUROC.')
    ap.add_argument('--limit_seed', type=int, default=0,
                    help='fixed so --limit is reproducible and resumable')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_manifest(args.manifest)
    if args.limit and args.limit < len(rows):
        sel = np.sort(np.random.default_rng(args.limit_seed).choice(
            len(rows), args.limit, replace=False))
        rows = [rows[i] for i in sel]
    n = len(rows)
    paths = [r['path'] for r in rows]

    # keep labels welded to features
    mcopy = os.path.join(args.out_dir, 'manifest.csv')
    if not os.path.exists(mcopy):
        with open(mcopy, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['path', 'patch_name', 'slide', 'label'])
            w.writeheader()
            w.writerows(rows)

    feat_path = os.path.join(args.out_dir, 'features.npy')
    ok_path = os.path.join(args.out_dir, 'ok.npy')
    prog_path = os.path.join(args.out_dir, 'progress.json')

    if os.path.exists(feat_path):
        feats = np.lib.format.open_memmap(feat_path, mode='r+')
        assert feats.shape == (n, FEAT_DIM), \
            f'existing features.npy has shape {feats.shape}, expected {(n, FEAT_DIM)}'
        ok = np.lib.format.open_memmap(ok_path, mode='r+')
    else:
        feats = np.lib.format.open_memmap(feat_path, mode='w+',
                                          dtype=np.float16, shape=(n, FEAT_DIM))
        ok = np.lib.format.open_memmap(ok_path, mode='w+',
                                       dtype=np.uint8, shape=(n,))

    start = 0
    if os.path.exists(prog_path):
        start = json.load(open(prog_path))['n_done']
        print(f'resuming at row {start:,} / {n:,}')
    if start >= n:
        print('already complete')
        return

    # ---- PREFLIGHT: transform one real image with NO exception handling.
    # A version mismatch (e.g. numpy 2.x against a torch built for numpy 1.x)
    # dies here in two seconds with a real traceback, instead of silently
    # flagging every patch 'unreadable' and writing a file of zeros.
    _tf = build_transform()
    _t = _tf(Image.open(paths[start]).convert('RGB'))
    print(f'preflight OK: {paths[start]} -> {tuple(_t.shape)} {_t.dtype}',
          flush=True)
    import numpy as _np
    print(f'numpy {_np.__version__}   torch {torch.__version__}', flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    import timm
    model = timm.create_model('hf_hub:prov-gigapath/prov-gigapath', pretrained=True)
    model.eval().to(device)

    ds = PatchList(paths, build_transform())
    loader = DataLoader(Subset(ds, range(start, n)),
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        persistent_workers=args.num_workers > 0,
                        prefetch_factor=4 if args.num_workers > 0 else None)

    i = start
    aborted_check = [False]
    t0 = time.time()
    pbar = tqdm(total=n, initial=start, unit='img', unit_scale=True,
                smoothing=0.05, dynamic_ncols=True,
                desc=os.path.basename(args.out_dir.rstrip('/')),
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                           '[{elapsed}<{remaining}, {rate_fmt}]')
    with torch.no_grad():
        for imgs, okflag in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16,
                                enabled=device.type == 'cuda'):
                z = model(imgs)
            z = F.normalize(z.float(), dim=-1)
            b = z.shape[0]
            feats[i:i + b] = z.cpu().numpy().astype(np.float16)
            ok[i:i + b] = okflag.numpy().astype(np.uint8)
            i += b
            pbar.update(b)
            seen = i - start
            if seen >= 2000 and not aborted_check[0]:
                aborted_check[0] = True
                bad = float((np.asarray(ok[start:i]) == 0).mean())
                if bad > 0.5:
                    raise RuntimeError(
                        f'{bad:.0%} of the first {seen:,} patches failed to '
                        f'decode. This is an environment problem, not bad '
                        f'data. Fix it before rerunning; delete '
                        f'{args.out_dir} first.')
            if (i - start) % (args.batch_size * 20) < args.batch_size:
                feats.flush(); ok.flush()
                json.dump({'n_done': i}, open(prog_path, 'w'))
                nbad = int((np.asarray(ok[start:i]) == 0).sum())
                if nbad:
                    pbar.set_postfix(unreadable=nbad)

    pbar.close()
    feats.flush(); ok.flush()
    json.dump({'n_done': i}, open(prog_path, 'w'))
    bad = int((np.asarray(ok) == 0).sum())
    print(f'done: {i:,} rows in {(time.time()-t0)/3600:.2f}h, unreadable={bad:,}')
    if bad > 0.01 * max(i, 1):
        print(f'!! WARNING: {bad / i:.1%} unreadable. Above ~1% this is an '
              f'environment or path problem, not corrupt tiles. Investigate '
              f'before trusting these features.')


if __name__ == '__main__':
    main()