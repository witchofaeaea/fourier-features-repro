# Fourier Features — Reproduction (CMP4509 Term Project)

Reproduction of the 2D image regression results of
**Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains", NeurIPS 2020** ([arXiv:2006.10739](https://arxiv.org/abs/2006.10739)) in PyTorch.

Team: _name 1, name 2, name 3_

## Claims we test

1. A coordinate MLP with no input mapping fits only low frequencies (blurry output). Fourier feature mappings fix this, and Gaussian RFF performs best (paper Table 3).
2. The scale σ of the Gaussian features controls underfitting (σ too small → oversmoothed) vs. overfitting (σ too large → noisy) (paper Fig. 9, 10).

## Setup (paper Appendix E.1)

| | |
|---|---|
| Image | 512×512, train = 256×256 grid `img[0::2,0::2]`, test = offset grid `img[1::2,1::2]` |
| Coordinates | normalized to [0, 1)² |
| Model | MLP, 4 linear layers, 256 channels, ReLU, sigmoid output |
| Mappings | none, basic, positional (σ^(j/m), axis-aligned), Gaussian (B ~ N(0, σ²)), 256 frequencies |
| Training | full batch, Adam (β=(0.9, 0.999), ε=1e-8), L2 loss, 2000 iterations |
| LR (natural) | 1e-3 Gaussian/positional, 1e-2 none/basic |
| LR (text) | 1e-3 for all |
| σ | natural: σ_g = 10, σ_p = 6 · text: σ_g = 14, σ_p = 5 |

## Repository layout

```
ffrepro/          model, mappings, data, metrics, training loop
scripts/          run_experiments.py, make_text_image.py
notebooks/        reproduce.ipynb (runs top to bottom on Colab / Kaggle)
data/images/      target images (committed)
results/          generated, not committed
tests/            sanity checks (pytest)
```

## Run locally (PyCharm)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
python scripts/make_text_image.py --out data/images/text.png
python scripts/run_experiments.py --image data/images/text.png --dataset text --seeds 0 --iterations 200   # quick smoke test
```

## Experiments

```bash
# Exp 1: Table 3 (all mappings, 5 seeds)
python scripts/run_experiments.py --image data/images/natural.png --dataset natural --seeds 0 1 2 3 4
python scripts/run_experiments.py --image data/images/text.png    --dataset text    --seeds 0 1 2 3 4

# Exp 2: sigma sweep (Fig. 9 / 10)
python scripts/run_experiments.py --image data/images/natural.png --dataset natural \
    --mapping gaussian --sigma 1 2 4 8 10 14 20 32 64 128 --seeds 0 1 2 3 4
```

Each run saves a JSON (metrics + PSNR curve); `summary.csv` holds mean ± std over seeds. Existing runs are skipped, so an interrupted session can be resumed.

## Differences from the paper

- The paper reports mean ± std over **16 images**; we report mean ± std over **seeds** (as required by the course) on _N_ images.
- _(add every other change here)_
