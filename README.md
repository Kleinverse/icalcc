# icalcc

Locally centered cyclic contrasts for scikit-learn FastICA.

Drop-in replacement for `sklearn.decomposition.FastICA` with LCC contrast functions of order 4, 6, and 8, plus bounded LCC-tanh and LCC-exp variants.

```python
from icalcc import ICALCC
ica = ICALCC(n_components=4, K=6)
S_hat = ica.fit_transform(X)
```

## See Also

- [gpuicalcc](https://github.com/Kleinverse/gpuicalcc) — GPU-accelerated version
- [Kleinverse Open Research Repository (KORR)](https://github.com/Kleinverse/research/lcc) — reseach and experiment code

## Installation

```bash
pip install icalcc
```

## Background

FastICA maximizes the excess kurtosis $\kappa_4$, truncating the cumulant generating function at order $k=4$. Naive substitution of higher-order centered moments introduces a permanent bias bounded below by the Itakura--Saito divergence. The locally centered cyclic (LCC) kernel eliminates this bias through cyclic centering, yielding a nondegenerate V-statistic at every even order.

LCC outperforms FastICA on super-Gaussian sources of moderate kurtosis. Extreme kurtosis marks the boundary where estimator variance overtakes the bias correction.

## Usage

```python
from icalcc import ICALCC

# LCC order 6 (recommended for super-Gaussian sources)
ica = ICALCC(n_components=4, K=6, random_state=0)
S_hat = ica.fit_transform(X)

# LCC order 8 (best for symmetric, low-kurtosis sources)
ica = ICALCC(n_components=4, K=8, random_state=0)
S_hat = ica.fit_transform(X)

# Bounded LCC-tanh (robust to extreme kurtosis)
ica = ICALCC(n_components=4, K='ltanh', random_state=0)
S_hat = ica.fit_transform(X)

# Classical FastICA baseline
ica = ICALCC(n_components=4, K='tanh', random_state=0)
S_hat = ica.fit_transform(X)
```

## Supported K Values

| K | Description |
|---|---|
| `4` | LCC polynomial order 4 (equivalent to FastICA(4)) |
| `6` | LCC polynomial order 6, exploits m₃, m₄, m₆ |
| `8` | LCC polynomial order 8, uses moments up to order 8 |
| `'fast4'` | FastICA(4), g(y) = y³ |
| `'fast6'` | FastICA(6), g(y) = y⁵ |
| `'fast8'` | FastICA(8), g(y) = y⁷ |
| `'tanh'` | logcosh contrast |
| `'exp'` | Gaussian contrast |
| `'ltanh'` | Locally centered tanh |
| `'lexp'` | Locally centered exp |
| `'skew'` | Skewness contrast (experimental) |

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.24
- scikit-learn ≥ 1.3

## Citation

```bibtex
@misc{saito2026lcc,
  author    = {Saito, Tetsuya},
  title     = {Locally Centered Cyclic Kernels for Higher-Order Independent Component Analysis},
  year      = {2026},
  publisher = {TechRxiv},
  doi       = {10.36227/techrxiv.XXXXXXX}
}
```

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
