"""Run 2D image regression experiments and save results.

Examples
--------
# Experiment 1 (paper Table 3): all mappings, 5 seeds, paper's sigma and lr
python scripts/run_experiments.py --image data/images/natural.png --dataset natural \
    --mapping none basic positional gaussian --seeds 0 1 2 3 4

# Experiment 2 (paper Fig. 9/10): sigma sweep for Gaussian RFF
python scripts/run_experiments.py --image data/images/natural.png --dataset natural \
    --mapping gaussian --sigma 1 2 4 8 10 14 20 32 64 128 --seeds 0 1 2 3 4

Each run writes results/<image>/<mapping>_s<sigma>_seed<seed>.json.
Runs whose JSON already exists are skipped, so a disconnected Colab session can resume.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ffrepro import MAPPINGS, PAPER_DEFAULTS, load_image, make_splits, train_one  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--dataset", choices=["natural", "text"], default="natural",
                   help="picks the paper's default lr and sigma")
    p.add_argument("--mapping", nargs="+", default=list(MAPPINGS), choices=MAPPINGS)
    p.add_argument("--sigma", nargs="+", type=float, default=None,
                   help="override sigma (default: paper value per mapping)")
    p.add_argument("--lr", type=float, default=None, help="override lr (default: paper value)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--out", default="results")
    p.add_argument("--save-recon", action="store_true", help="save reconstruction PNG for every run")
    return p.parse_args()


def main():
    a = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    img = load_image(a.image)
    splits = make_splits(img, device=device)
    out_dir = Path(a.out) / Path(a.image).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    defaults = PAPER_DEFAULTS[a.dataset]

    for mapping in a.mapping:
        if mapping in ("none", "basic"):
            sigmas = [0.0]  # sigma is unused
        else:
            sigmas = a.sigma or [defaults["sigma"][mapping]]
        lr = a.lr or defaults["lr"][mapping]
        for sigma in sigmas:
            for seed in a.seeds:
                name = f"{mapping}_s{sigma:g}_seed{seed}"
                path = out_dir / f"{name}.json"
                if path.exists():
                    print(f"skip {name} (exists)")
                    continue
                r = train_one(splits, mapping, sigma, lr, seed, iterations=a.iterations, device=device)
                recon = r.pop("recon")
                r["image"] = Path(a.image).name
                path.write_text(json.dumps(r, indent=1))
                if a.save_recon or seed == a.seeds[0]:
                    Image.fromarray((np.clip(recon, 0, 1) * 255).astype(np.uint8)).save(out_dir / f"{name}.png")
                print(f"{name}: test PSNR {r['test_psnr']:.2f} dB, train PSNR {r['train_psnr']:.2f} dB ({r['seconds']}s)")

    summarize(out_dir)


def summarize(out_dir: Path):
    """Aggregate all JSONs in out_dir into summary.csv (mean +- std over seeds)."""
    groups = {}
    for f in out_dir.glob("*.json"):
        r = json.loads(f.read_text())
        groups.setdefault((r["mapping"], r["sigma"]), []).append(r)
    rows = []
    for (mapping, sigma), rs in sorted(groups.items()):
        te = np.array([r["test_psnr"] for r in rs])
        tr = np.array([r["train_psnr"] for r in rs])
        rows.append({"mapping": mapping, "sigma": sigma, "n_seeds": len(rs),
                     "test_psnr_mean": round(te.mean(), 3), "test_psnr_std": round(te.std(ddof=1) if len(rs) > 1 else 0.0, 3),
                     "train_psnr_mean": round(tr.mean(), 3), "train_psnr_std": round(tr.std(ddof=1) if len(rs) > 1 else 0.0, 3)})
    if not rows:
        return
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsummary -> {out_dir / 'summary.csv'}")
    for r in rows:
        print(f"  {r['mapping']:<11} sigma={r['sigma']:<6g} test {r['test_psnr_mean']:.2f} ± {r['test_psnr_std']:.2f} dB  (n={r['n_seeds']})")


if __name__ == "__main__":
    main()
