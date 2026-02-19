# ICALCC

**Higher-order cumulant contrasts for scikit-learn's FastICA.**

ICALCC extends `sklearn.decomposition.FastICA` with the Locally Centered
Cyclic (LCC) kernel, a V-statistic contrast that reaches sixth- and
eighth-order cumulant structure through a single parameter `K`.
Scikit-learn's built-in nonlinearities (logcosh, exp, cube) are limited
to fourth-order statistics. ICALCC fills this gap with a drop-in subclass
that preserves the full scikit-learn API.

```python
from icalcc import ICALCC

ica = ICALCC(n_components=4, K=6, random_state=0)
S_hat = ica.fit_transform(X)
```

## Installation

```bash
pip install icalcc
```

Or install from source:

```bash
git clone https://github.com/Kleinverse/icalcc.git
cd icalcc
pip install .
```

**Requirements:** Python >= 3.9, NumPy >= 1.21, scikit-learn >= 1.2

## Why ICALCC?

Scikit-learn's `FastICA` accepts a custom contrast function `fun(y) -> (g, g')`,
but ships only three: logcosh (tanh), exp (Gaussian), and cube (kurtosis).
All exploit at most fourth-order cumulant structure.

The LCC V-statistic of order `k` on whitened data expands into closed-form
moment polynomials.  For `K=6`:

    V_6 = (145/3888) m_3^2 + (115/2592) m_4 - (5/7776) m_6 - 125/648

The gradient dV_6/dw jointly weights m_3^2, m_4, and m_6.  No existing
scikit-learn nonlinearity couples multiple cumulant orders this way.

### When does it matter?

When source distributions are near-Gaussian (small excess kurtosis),
fourth-order contrasts lose statistical efficiency and often fail to converge.
ICALCC remains robust:

| Source (Gamma) | Kurtosis | tanh (logcosh) | ICALCC K=6 | ICALCC K=8 |
|----------------|----------|----------------|------------|------------|
| alpha = 5      | 1.2      | 3.71           | 1.55       | 1.36       |
| alpha = 8      | 0.8      | 7.11           | 2.18       | 1.50       |
| alpha = 12     | 0.5      | 10.95 (15% fail) | 2.37     | 2.19       |
| alpha = 20     | 0.3      | 21.70 (55% fail) | 3.45     | 2.55       |
| alpha = 50     | 0.1      | 34.84 (65% fail) | 7.99     | 5.15       |

*Amari index x 10^-2, d=4, N=10000, 20 trials.  Lower is better.
"fail" = fraction of trials where FastICA did not converge.*

## Usage

### LCC contrasts (the extension)

```python
from icalcc import ICALCC

ica6 = ICALCC(n_components=4, K=6)   # sixth-order cumulants
ica8 = ICALCC(n_components=4, K=8)   # eighth-order cumulants
ica4 = ICALCC(n_components=4, K=4)   # equivalent to sklearn cube
```

### Classical baselines through the same interface

```python
ica_tanh = ICALCC(n_components=4, K='tanh')   # logcosh
ica_exp  = ICALCC(n_components=4, K='exp')    # Gaussian
ica_skew = ICALCC(n_components=4, K='skew')   # skewness |y|*y
```

### Convergence diagnostics

Scikit-learn emits a `ConvergenceWarning` that is easy to miss in batch
experiments.  ICALCC intercepts it and exposes a boolean attribute:

```python
ica = ICALCC(n_components=4, K=6)
ica.fit(X)
if not ica.converged_:
    print("ICA did not converge")
```

### Pipeline compatibility

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('ica', ICALCC(n_components=10, K=6)),
])
S_hat = pipe.fit_transform(X)
```

### Upgrading existing code

```python
# Before
from sklearn.decomposition import FastICA
ica = FastICA(n_components=4, fun='logcosh', random_state=0)

# After (one import change)
from icalcc import ICALCC
ica = ICALCC(n_components=4, K=6, random_state=0)
```

## API Reference

### `ICALCC(n_components=None, *, K=6, algorithm='parallel', whiten='unit-variance', max_iter=200, tol=1e-4, w_init=None, whiten_solver='svd', random_state=None)`

**Parameters**

| Parameter | Description |
|-----------|-------------|
| `K` | Contrast order. `4`, `6`, `8` for LCC V-statistic; `'tanh'`, `'exp'`, `'skew'` for classical baselines. Default: `6`. |

All other parameters are identical to `sklearn.decomposition.FastICA`.

**Attributes** (after `fit`)

All attributes from `FastICA` (`components_`, `mixing_`, `mean_`, `n_iter_`,
`whitening_`), plus:

| Attribute | Description |
|-----------|-------------|
| `converged_` | `True` if the iteration converged within `max_iter`, `False` otherwise. |

**Methods**

`fit(X)`, `transform(X)`, `fit_transform(X)`, `inverse_transform(S)`,
`get_params()`, `set_params()` -- all inherited from `FastICA`.

## How it works

The LCC kernel of order k is kappa_k = prod_{i=1}^{k} (y_i - y_bar_k).
Its V-statistic V_k = E[kappa_k] on whitened data (zero mean, unit variance)
reduces to a polynomial in raw moments m_r = E[y^r].  The partial derivatives
dV_k / dm_r yield the nonlinearity h(y) and its derivative h'(y):

    h(y) = sum_r (dV_k / dm_r) * r * y^{r-1}

These slot into scikit-learn's Newton fixed-point iteration:

    w <- E[z * h(w'z)] - E[h'(w'z)] * w

The moment polynomial gradient has bounded derivatives for bounded data,
preventing the divergence that afflicts transcendental nonlinearities
(tanh, exp) when source kurtosis is near zero.

### V-statistic formulas (whitened data)

**V_4:**
`V_4 = 21/64 - 3*m_4/64`

**V_6:**
`V_6 = 145*m_3^2/3888 + 115*m_4/2592 - 5*m_6/7776 - 125/648`

**V_8:**
`V_8 = -7665*m_3^2/131072 + 497*m_3*m_5/262144 + 2765*m_4^2/2097152 - 18795*m_4/524288 + 329*m_6/524288 - 7*m_8/2097152 + 117705/1048576`

## Experiments

The `experiments/` directory contains benchmark scripts for the companion
paper. Each table script is standalone and can also be run through the
master `experiment.py`:

| Script | Description | Standalone |
|--------|-------------|------------|
| `exp_table1.py` | Table I: Amari index across distributions | `python exp_table1.py --quick` |
| `exp_table2.py` | Table II: Gamma(alpha) scan, convergence failures | `python exp_table2.py --quick` |
| `exp_table3.py` | Table III: ARE vs empirical efficiency ratio | `python exp_table3.py --quick` |
| `exp_table4.py` | Table IV: V-statistic sample mean and t-statistic | `python exp_table4.py --quick` |
| `exp_sanity.py` | Quick sanity check across all methods | `python exp_sanity.py` |
| `exp_utils.py`  | Shared utilities: distributions, algorithms, trial runner | (imported) |

Via the master script:

```bash
python experiments/experiment.py                    # sanity check
python experiments/experiment.py --table1 --quick   # Table I (20 trials)
python experiments/experiment.py --table2 --quick   # Table II (20 trials)
python experiments/experiment.py --all              # all tables (200 trials)
python experiments/experiment.py --test             # unit tests
```

Append `--quick` to any command for 20 trials instead of 200.

## Testing

```bash
python experiments/experiment.py --test
```

## Citation

If you use ICALCC in your research, please cite:

```bibtex
@article{saito2025lcc,
  author  = {Saito, Tetsuya},
  title   = {Locally Centered Cyclic Kernels for Higher-Order
             Independent Component Analysis},
  journal = {TechRxiv},
  year    = {2025},
  doi     = {}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
