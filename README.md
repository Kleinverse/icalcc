# icalcc
Locally centered contrast functions for scikit-learn FastICA.
Drop-in replacement for `sklearn.decomposition.FastICA` with
bounded LCC-tanh and LCC-exp contrasts, plus polynomial LCC
contrasts of order 4, 6, and 8.
```python
from icalcc import ICALCC
ica = ICALCC(n_components=4, K='ltanh', random_state=0)
S_hat = ica.fit_transform(X)
```

## Installation
```bash
pip install icalcc
```

## Supported K Values
| K | Description |
|---|---|
| `'ltanh'` | Bounded LCC-tanh (robust to heavy tails and skewness) |
| `'lexp'` | Bounded LCC-exp (maximizes Rényi-2 entropy) |
| `4` | Polynomial LCC order 4 |
| `6` | Polynomial LCC order 6, couples m₃, m₄, m₆ |
| `8` | Polynomial LCC order 8, couples moments up to order 8 |
| `'tanh'` | Classical logcosh contrast (scikit-learn default) |
| `'exp'` | Classical Gaussian contrast |
| `'skew'` | Classical cube contrast |

## Usage
```python
from icalcc import ICALCC

# Bounded LCC-tanh (recommended for heavy-tailed or skewed sources)
ica = ICALCC(n_components=4, K='ltanh', random_state=0)
S_hat = ica.fit_transform(X)

# Bounded LCC-exp (Rényi-2 entropy interpretation)
ica = ICALCC(n_components=4, K='lexp', random_state=0)
S_hat = ica.fit_transform(X)

# Polynomial LCC order 8 (near-Gaussian sources)
ica = ICALCC(n_components=4, K=8, random_state=0)
S_hat = ica.fit_transform(X)

# Classical FastICA baseline
ica = ICALCC(n_components=4, K='tanh', random_state=0)
S_hat = ica.fit_transform(X)
```

## Convergence tracking
```python
ica.fit(X)
print(ica.converged_)  # True if all components converged
```

## See Also
- [gpuicalcc](https://github.com/Kleinverse/gpuicalcc) —
  PyTorch GPU-accelerated extension (40–48× speedup for bounded contrasts)
- [Experiment code](https://github.com/Kleinverse/research/tree/main/icalcc)

## Requirements
- Python ≥ 3.9
- numpy ≥ 1.24
- scikit-learn ≥ 1.3

## Citation
```bibtex
@article{saito2026icalcc,
  author  = {Saito, Tetsuya},
  title   = {{ICALCC}: Locally Centered Contrast Functions for
             {FastICA} with {GPU} Acceleration},
  journal = {IEEE Signal Processing Letters},
  year    = {2026},
  note    = {submitted}
}
```

## License
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
