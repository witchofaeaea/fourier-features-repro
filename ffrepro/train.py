"""One training run = one (image, mapping, sigma, seed) configuration."""
import random
import time

import numpy as np
import torch

from .mappings import InputMapping
from .metrics import mse, psnr
from .model import FourierFeatureMLP

# Settings from the paper, Appendix E.1 (2D image task).
PAPER_DEFAULTS = {
    "natural": {
        "lr": {"none": 1e-2, "basic": 1e-2, "positional": 1e-3, "gaussian": 1e-3},
        "sigma": {"positional": 6.0, "gaussian": 10.0},
    },
    "text": {
        "lr": {"none": 1e-3, "basic": 1e-3, "positional": 1e-3, "gaussian": 1e-3},
        "sigma": {"positional": 5.0, "gaussian": 14.0},
    },
}
ITERATIONS = 2000
NUM_FREQS = 256


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(splits: dict, mapping: str, sigma: float, lr: float, seed: int,
              iterations: int = ITERATIONS, num_freqs: int = NUM_FREQS,
              log_every: int = 25, device: str = "cpu", verbose: bool = True) -> dict:
    """Full-batch training with Adam and L2 loss. Returns metrics, curves and the full reconstruction."""
    set_seed(seed)
    gamma = InputMapping(mapping, in_dim=2, num_freqs=num_freqs, sigma=sigma, seed=seed)
    model = FourierFeatureMLP(gamma).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    x_tr, y_tr = splits["train"]
    x_te, y_te = splits["test"]
    curve = {"iter": [], "train_psnr": [], "test_psnr": []}

    t0 = time.time()
    for it in range(1, iterations + 1):
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((model(x_tr) - y_tr) ** 2)
        loss.backward()
        opt.step()
        if it % log_every == 0 or it == iterations:
            with torch.no_grad():
                curve["iter"].append(it)
                curve["train_psnr"].append(psnr(model(x_tr), y_tr))
                curve["test_psnr"].append(psnr(model(x_te), y_te))
            if verbose and (it % 100 == 0 or it == iterations):
                print(f"  [{mapping} seed={seed}] iter {it}/{iterations}  "
                      f"train {curve['train_psnr'][-1]:.2f} dB  test {curve['test_psnr'][-1]:.2f} dB  "
                      f"({time.time() - t0:.0f}s)", flush=True)

    with torch.no_grad():
        p_tr, p_te = model(x_tr), model(x_te)
        x_all, _ = splits["full"]
        s = splits["size"]
        recon = model(x_all).reshape(s, s, 3).cpu().numpy()

    return {
        "mapping": mapping, "sigma": sigma, "lr": lr, "seed": seed,
        "iterations": iterations, "num_freqs": num_freqs,
        "train_mse": mse(p_tr, y_tr), "test_mse": mse(p_te, y_te),
        "train_psnr": psnr(p_tr, y_tr), "test_psnr": psnr(p_te, y_te),
        "seconds": round(time.time() - t0, 2),
        "curve": curve,
        "recon": recon,  # (H, W, 3) numpy, not saved to JSON
    }
