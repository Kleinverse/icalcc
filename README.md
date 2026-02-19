# ICALCC

**Higher-order cumulant contrasts for scikit-learn's FastICA.**

ICALCC extends `sklearn.decomposition.FastICA` with the Locally Centered
Cyclic (LCC) kernel, a V-statistic contrast that reaches sixth- and
eighth-order cumulant structure through a single parameter `K`.
Scikit-learn's built-in nonlinearities (`logcosh`, `exp`, `cube`) are limited
to fourth-order statistics. ICALCC fills this gap with a drop-in subclass
that preserves the full scikit-learn API.

## Installation

```bash
pip install icalcc
```

**Requirements:** Python >= 3.9, NumPy >= 1.21, scikit-learn >= 1.2


## Quick start

```python
from icalcc import ICALCC

ica = ICALCC(n_components=4, K=6, random_state=0)
S_hat = ica.fit_transform(X)
```

### Choosing the contrast order

```python
ica6 = ICALCC(n_components=4, K=6)   # sixth-order cumulants (default)
ica8 = ICALCC(n_components=4, K=8)   # eighth-order cumulants
ica4 = ICALCC(n_components=4, K=4)   # LCC V_4 (locally centered, not sklearn cube)
```

### Classical baselines through the same interface

```python
ica_tanh = ICALCC(n_components=4, K='tanh')   # logcosh
ica_exp  = ICALCC(n_components=4, K='exp')    # Gaussian
ica_skew = ICALCC(n_components=4, K='skew')   # skewness |y|*y
```

### Convergence diagnostics

Scikit-learn emits a `ConvergenceWarning` that is easy to miss in batch
experiments. ICALCC intercepts it and exposes a boolean attribute:

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


## When does it matter?

When sources are near-Gaussian (small excess kurtosis), fourth-order
contrasts lose power and often fail to converge. ICALCC remains robust.
The table below shows the Amari index ($\times 10^{-2}$; lower is better)
for $d = 4$, $N = 10{,}000$, averaged over 20 trials on Gamma sources
with excess kurtosis $6/\alpha$.

| Source (Gamma) | Kurtosis | tanh (logcosh) | ICALCC K=6 | ICALCC K=8 |
|----------------|----------|----------------|------------|------------|
| $\alpha = 5$   | 1.2      | 3.71           | 1.55       | 1.36       |
| $\alpha = 8$   | 0.8      | 7.11           | 2.18       | 1.50       |
| $\alpha = 12$  | 0.5      | 10.95 (15% fail) | 2.37     | 2.19       |
| $\alpha = 20$  | 0.3      | 21.70 (55% fail) | 3.45     | 2.55       |
| $\alpha = 50$  | 0.1      | 34.84 (65% fail) | 7.99     | 5.15       |

"fail" = fraction of trials where FastICA did not converge within 200
iterations. ICALCC converged in all trials.


## API Reference

### `ICALCC(n_components=None, *, K=6, algorithm='parallel', whiten='unit-variance', max_iter=200, tol=1e-4, w_init=None, whiten_solver='svd', random_state=None)`

**Parameters.**
`K` sets the contrast order: `4`, `6`, or `8` for LCC V-statistic contrasts;
`'tanh'`, `'exp'`, or `'skew'` for classical baselines. Default is `6`.
All other parameters are identical to `sklearn.decomposition.FastICA`.

**Attributes** (after `fit`).
All attributes from `FastICA` (`components_`, `mixing_`, `mean_`, `n_iter_`,
`whitening_`), plus `converged_` (`True` if the iteration converged within
`max_iter`).

**Methods.**
`fit(X)`, `transform(X)`, `fit_transform(X)`, `inverse_transform(S)`,
`get_params()`, `set_params()` are all inherited from `FastICA`.


## How it works

### Why higher-order contrasts?

FastICA maximizes a contrast function whose gradient drives a Newton
fixed-point iteration. The three contrasts shipped with scikit-learn
exploit at most fourth-order cumulant information: `logcosh` approximates
negentropy via $\tanh$, `exp` uses a Gaussian kernel, and `cube` directly
targets kurtosis $\kappa_4$. When $\kappa_4 \approx 0$ these contrasts
lose statistical power. Reaching into $\kappa_6$ and $\kappa_8$ restores
discriminability.

### Why cyclic kernels?

The full $k$th-order structure requires all $\binom{k}{2}$ pairwise
interactions (28 terms for $k = 8$). The cyclic kernel visits all $k$
nodes with only $k$ edges:

$$C_k \;=\; \prod_{\ell=1}^{k} \delta_{\ell,\,\ell+1}, \qquad \delta_{k,\,k+1} \equiv \delta_{k,1}$$

where $\delta_{ij} = z_i - z_j$. This keeps computation linear in $k$.

### Local centering

The raw cyclic kernel $C_k$ is degenerate: $\mathbb{E}[C_k] = 0$ for odd
$k$, and $\mathbb{E}[C_k]$ depends only on $\sigma^2$ for even $k$. The
locally centered variant $\kappa_k = \prod_{i=1}^{k}(y_i - \bar{y}_k)$
breaks this degeneracy. Its V-statistic $V_k = \mathbb{E}[\kappa_k]$ on
whitened data reduces to a polynomial in raw moments $m_r = \mathbb{E}[y^r]$.

### V-statistic formulas (whitened data)

**Order 4** (locally centered; differs from `cube` by local vs global centering):

$$V_4 \;=\; \frac{21}{64} \;-\; \frac{3}{64}\,m_4$$

**Order 6:**

$$V_6 \;=\; \frac{145}{3888}\,m_3^2 \;+\; \frac{115}{2592}\,m_4 \;-\; \frac{5}{7776}\,m_6 \;-\; \frac{125}{648}$$

**Order 8:**

$$V_8 \;=\; -\frac{7665}{131072}\,m_3^2 \;+\; \frac{497}{262144}\,m_3\,m_5 \;+\; \frac{2765}{2097152}\,m_4^2 \;-\; \frac{18795}{524288}\,m_4 \;+\; \frac{329}{524288}\,m_6 \;-\; \frac{7}{2097152}\,m_8 \;+\; \frac{117705}{1048576}$$

The gradient $\partial V_k / \partial m_r$ yields the nonlinearity $h(y)$
and $h^\prime(y)$ for scikit-learn's Newton update:

$$\mathbf{w} \;\leftarrow\; \mathbb{E}[\mathbf{z}\,h(\mathbf{w}^\top\mathbf{z})] \;-\; \mathbb{E}[h^\prime(\mathbf{w}^\top\mathbf{z})]\,\mathbf{w}$$

Because $h(y)$ is a polynomial, the iteration avoids the divergence that
afflicts transcendental nonlinearities when source kurtosis is near zero.


## Replication

The [`experiments/`](experiments/) directory contains benchmark scripts.
See [`experiments/README.md`](experiments/README.md) for details.


## Citation

```bibtex
@article{saito2026lcc,
  author  = {Saito, Tetsuya},
  title   = {Locally Centered Cyclic Kernels for Higher-Order
             Independent Component Analysis},
  journal = {TechRxiv},
  year    = {2026}
}
```


## License

MIT License. See [LICENSE](LICENSE) for details.
