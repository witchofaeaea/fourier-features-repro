"""Quick sanity checks. Run with: pytest -q"""
import math

import numpy as np
import torch

from ffrepro import InputMapping, FourierFeatureMLP, make_splits


def test_output_dims():
    assert InputMapping("none").out_dim == 2
    assert InputMapping("basic").out_dim == 4
    assert InputMapping("positional", num_freqs=256).out_dim == 512
    assert InputMapping("gaussian", num_freqs=256).out_dim == 512


def test_kernel_is_stationary():
    """Paper Eq. 6: gamma(v1)^T gamma(v2) depends only on v1 - v2."""
    g = InputMapping("gaussian", sigma=10.0, seed=0)
    v1, v2 = torch.tensor([[0.1, 0.2]]), torch.tensor([[0.3, 0.25]])
    shift = torch.tensor([[0.37, -0.11]])
    k = (g(v1) * g(v2)).sum()
    k_shifted = (g(v1 + shift) * g(v2 + shift)).sum()
    assert torch.allclose(k, k_shifted, atol=1e-3)


def test_same_seed_same_B():
    a = InputMapping("gaussian", seed=3).B
    b = InputMapping("gaussian", seed=3).B
    assert torch.equal(a, b)


def test_model_output_range():
    m = FourierFeatureMLP(InputMapping("gaussian"))
    y = m(torch.rand(100, 2))
    assert y.shape == (100, 3) and y.min() >= 0 and y.max() <= 1


def test_splits():
    img = np.random.rand(512, 512, 3).astype(np.float32)
    s = make_splits(img)
    assert s["train"][0].shape == (256 * 256, 2)
    assert s["test"][0].shape == (256 * 256, 2)
    assert s["train"][0].max() < 1 and s["train"][0].min() >= 0
    # train and test grids must not overlap
    assert not torch.isclose(s["train"][0][0], s["test"][0][0]).all()
