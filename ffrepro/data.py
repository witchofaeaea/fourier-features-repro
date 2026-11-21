"""Image loading and train/test grids (paper Appendix E.1).

A 512x512 image is split into two 256x256 grids:
  train = pixels with even (row, col)  -> img[0::2, 0::2]
  test  = the offset grid             -> img[1::2, 1::2]
Coordinates are normalized to [0, 1)^2.
"""
import numpy as np
import torch
from PIL import Image


def load_image(path: str, size: int = 512) -> np.ndarray:
    """Center-crop to a square, resize to size x size, return float32 in [0, 1], shape (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def coord_grid(size: int) -> np.ndarray:
    """(size, size, 2) grid of (x, y) coordinates in [0, 1)."""
    lin = np.arange(size, dtype=np.float32) / size
    yy, xx = np.meshgrid(lin, lin, indexing="ij")
    return np.stack([xx, yy], axis=-1)


def make_splits(img: np.ndarray, device="cpu") -> dict:
    """Return flattened tensors for train / test / full image."""
    size = img.shape[0]
    coords = coord_grid(size)

    def flat(c, p):
        return (torch.from_numpy(c.reshape(-1, 2)).to(device),
                torch.from_numpy(p.reshape(-1, 3)).to(device))

    x_tr, y_tr = flat(coords[0::2, 0::2], img[0::2, 0::2])
    x_te, y_te = flat(coords[1::2, 1::2], img[1::2, 1::2])
    x_all, y_all = flat(coords, img)
    return {"train": (x_tr, y_tr), "test": (x_te, y_te), "full": (x_all, y_all), "size": size}
