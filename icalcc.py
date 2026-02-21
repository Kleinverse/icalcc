"""ICALCC: Locally centered cyclic contrasts for scikit-learn FastICA.

Drop-in replacement for sklearn.decomposition.FastICA.
Just replace the import and set K.

    from icalcc import ICALCC
    est = ICALCC(K=6)
    est.fit(X)

Supported K values: 4, 6, 8, 'fast4', 'fast6', 'fast8',
                    'tanh', 'exp', 'ltanh', 'lexp', 'skew'.

Reference: T. Saito, "Locally Centered Cyclic Kernels for Higher-Order
Independent Component Analysis," TechRxiv, 2026.
https://doi.org/10.36227/techrxiv.XXXXXXX
"""

import numpy as np
import warnings
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning


def _lcc_h_gprime(y, k):
    """LCC nonlinearity g(y) and per-sample g'(y).

    Parameters
    ----------
    y : ndarray, shape (N,)
        Whitened projection.
    k : int
        Cumulant order (4, 6, or 8).

    Returns
    -------
    gy : ndarray, shape (N,)
        Contrast nonlinearity evaluated per sample.
    gpy : ndarray, shape (N,)
        Derivative of g evaluated per sample.
    """
    y2 = y * y
    y3 = y2 * y

    if k == 4:
        gy = (-3.0 / 16) * y3
        gpy = (-9.0 / 16) * y2
        return gy, gpy

    m3 = np.mean(y3)
    m4 = np.mean(y2 * y2)

    if k == 6:
        dJ3 = 145 * m3 / 1944.0
        dJ4 = 115.0 / 2592
        dJ6 = -5.0 / 7776
        y4 = y2 * y2
        gy = dJ3 * 3 * y2 + dJ4 * 4 * y3 + dJ6 * 6 * (y4 * y)
        gpy = dJ3 * 6 * y + dJ4 * 12 * y2 + dJ6 * 30 * y4
        return gy, gpy

    # k == 8
    m5 = np.mean(y2 * y3)
    m6 = np.mean(y3 * y3)
    dJ3 = -7665 * m3 / 65536.0 + 497 * m5 / 262144.0
    dJ4 = 2765 * m4 / 1048576.0 - 18795.0 / 524288
    dJ5 = 497 * m3 / 262144.0
    dJ6 = 329.0 / 524288
    dJ8 = -7.0 / 2097152
    y4 = y2 * y2
    gy = (dJ3 * 3 * y2 + dJ4 * 4 * y3
          + dJ5 * 5 * y4 + dJ6 * 6 * (y4 * y)
          + dJ8 * 8 * (y4 * y3))
    gpy = (dJ3 * 6 * y + dJ4 * 12 * y2
           + dJ5 * 20 * y3 + dJ6 * 30 * y4
           + dJ8 * 56 * (y3 * y3))
    return gy, gpy


def _fast_h_gprime(y, k):
    """FastICA(k) nonlinearity: g(y) = y^{k-1}, g'(y) = (k-1)*y^{k-2}.

    Maximises |m_k| = |E[y^k]| using global-mean centering.
    Provided for matched-order comparison with LCC(k).

    Parameters
    ----------
    y : ndarray, shape (N,)
    k : int (4, 6, or 8)

    Returns
    -------
    gy, gpy : ndarray, shape (N,)
    """
    gy = y ** (k - 1)
    gpy = (k - 1) * y ** (k - 2)
    return gy, gpy


def _fast_contrast(x, K=4):
    """FastICA(k) contrast callable for sklearn's ``fun`` parameter."""
    if x.ndim == 1:
        return _fast_h_gprime(x, K)
    p, N = x.shape
    gY = np.empty_like(x)
    gpy = np.empty(p, dtype=x.dtype)
    for i in range(p):
        gi, gpi = _fast_h_gprime(x[i], K)
        gY[i] = gi
        gpy[i] = gpi.mean()
    return gY, gpy


def _lcc_contrast(x, K=6):
    """LCC contrast callable for sklearn's ``fun`` parameter.

    Handles both deflation (1-D input) and parallel (2-D input)
    algorithms.
    """
    if x.ndim == 1:
        return _lcc_h_gprime(x, K)
    p, N = x.shape
    gY = np.empty_like(x)
    gpy = np.empty(p, dtype=x.dtype)
    for i in range(p):
        gi, gpi = _lcc_h_gprime(x[i], K)
        gY[i] = gi
        gpy[i] = gpi.mean()
    return gY, gpy


def _lcc_bounded_h_gprime(y, G="tanh", batch_size=500):
    """Locally centered bounded nonlinearity.

    Computes the locally centered gradient with bounded kernel G:
      g(y_i) = (1/B) sum_j G'(y_i - y_j)
      g'(y_i) = (1/B) sum_j G''(y_i - y_j)

    where j is a random subsample of size B for O(NB) cost.

    For G = 'tanh':  G(u) = log cosh(u),  G'(u) = tanh(u)
    For G = 'exp':   G(u) = -exp(-u^2/2), G'(u) = u exp(-u^2/2)

    Parameters
    ----------
    y : ndarray, shape (N,)
    G : {'tanh', 'exp'}
    batch_size : int

    Returns
    -------
    gy, gpy : ndarray, shape (N,)
    """
    N = len(y)
    B = min(batch_size, N)
    step = max(1, N // B)
    idx = np.arange(0, N, step)[:B]
    y_batch = y[idx]
    diff = y[:, None] - y_batch[None, :]

    if G == "tanh":
        t = np.tanh(diff)
        gy = t.mean(axis=1)
        gpy = (1.0 - t * t).mean(axis=1)
    elif G == "exp":
        e = np.exp(-0.5 * diff * diff)
        gy = (diff * e).mean(axis=1)
        gpy = ((1.0 - diff * diff) * e).mean(axis=1)
    else:
        raise ValueError(f"Unknown G: {G}")

    return gy, gpy


def _lcc_bounded_contrast(x, G="tanh", batch_size=500):
    """LCC bounded contrast callable for sklearn's ``fun`` parameter."""
    if x.ndim == 1:
        return _lcc_bounded_h_gprime(x, G=G, batch_size=batch_size)
    p, N = x.shape
    gY = np.empty_like(x)
    gpy = np.empty(p, dtype=x.dtype)
    for i in range(p):
        gi, gpi = _lcc_bounded_h_gprime(x[i], G=G, batch_size=batch_size)
        gY[i] = gi
        gpy[i] = gpi.mean()
    return gY, gpy


def _skew_contrast(x):
    """Skewness contrast: g(y) = y|y|, g'(y) = 2|y|.

    Experimental. Useful for strongly asymmetric source distributions.
    Not analyzed in the paper.
    """
    ay = np.abs(x)
    gy = x * ay
    if x.ndim == 1:
        return gy, 2.0 * ay
    return gy, np.mean(2.0 * ay, axis=1)


class ICALCC(FastICA):
    """Independent Component Analysis with LCC contrast.

    Drop-in replacement for ``sklearn.decomposition.FastICA``
    that provides locally centered cyclic contrast functions:
    polynomial contrasts of order 4, 6, 8 and bounded contrasts
    LCC-tanh and LCC-exp. Classical nonlinearities (tanh, exp)
    are also available for comparison.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of components to extract.
    K : {4, 6, 8, 'fast4', 'fast6', 'fast8', 'tanh', 'exp',
         'ltanh', 'lexp', 'skew'}, default=6
        Contrast function.

        - ``4``: LCC polynomial order 4 (equivalent to FastICA(4)).
        - ``6``: LCC polynomial order 6, exploits m_3, m_4, m_6.
          Recommended for super-Gaussian sources of moderate kurtosis.
        - ``8``: LCC polynomial order 8, uses moments up to order 8.
          Best for symmetric sources with low kurtosis and finite m_8.
        - ``'fast4'``: FastICA(4), g(y) = y^3.
        - ``'fast6'``: FastICA(6), g(y) = y^5.
        - ``'fast8'``: FastICA(8), g(y) = y^7.
        - ``'tanh'``: logcosh contrast, g(y) = tanh(y).
        - ``'exp'``: Gaussian contrast, g(y) = y exp(-y^2/2).
        - ``'ltanh'``: Locally centered tanh.
        - ``'lexp'``: Locally centered exp.
        - ``'skew'``: Skewness contrast, g(y) = y|y|. Experimental.
    algorithm : {'parallel', 'deflation'}, default='parallel'
        FastICA algorithm variant.
    whiten : str or False, default='unit-variance'
        Whitening strategy.
    max_iter : int, default=200
        Maximum number of iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    w_init : array-like or None, default=None
        Initial unmixing matrix.
    whiten_solver : {'svd', 'eigh'}, default='svd'
        Solver for whitening.
    random_state : int, RandomState or None, default=None
        Random seed.

    Attributes
    ----------
    components_ : ndarray, shape (n_components, n_features)
        Unmixing matrix.
    mixing_ : ndarray, shape (n_features, n_components)
        Mixing matrix.
    converged_ : bool
        True if FastICA converged within max_iter iterations.
    n_iter_ : int
        Number of iterations used.

    Examples
    --------
    >>> import numpy as np
    >>> from icalcc import ICALCC
    >>> rng = np.random.RandomState(42)
    >>> S = rng.laplace(size=(4, 10000))
    >>> A = rng.randn(4, 4)
    >>> X = (A @ S).T
    >>> ica = ICALCC(n_components=4, K=6, random_state=0)
    >>> S_hat = ica.fit_transform(X)

    References
    ----------
    T. Saito, "Locally Centered Cyclic Kernels for Higher-Order
    Independent Component Analysis," TechRxiv, 2026.
    https://doi.org/10.36227/techrxiv.XXXXXXX

    See Also
    --------
    sklearn.decomposition.FastICA : Base class.
    """

    _VALID_K = (4, 6, 8, "fast4", "fast6", "fast8",
                "tanh", "exp", "ltanh", "lexp", "skew")
    _SKLEARN_MAP = {"tanh": "logcosh", "exp": "exp"}
    _FAST_MAP = {"fast4": 4, "fast6": 6, "fast8": 8}
    _LCC_BOUNDED_MAP = {"ltanh": "tanh", "lexp": "exp"}

    def __init__(
        self,
        n_components=None,
        *,
        K=6,
        algorithm="parallel",
        whiten="unit-variance",
        max_iter=200,
        tol=1e-4,
        w_init=None,
        whiten_solver="svd",
        random_state=None,
    ):
        if K not in self._VALID_K:
            raise ValueError(
                f"K must be one of {self._VALID_K}, got {K}")
        self.K = K

        if K in self._SKLEARN_MAP:
            fun = self._SKLEARN_MAP[K]
            fun_args = {}
        elif K == "skew":
            fun = _skew_contrast
            fun_args = {}
        elif K in self._FAST_MAP:
            fun = _fast_contrast
            fun_args = dict(K=self._FAST_MAP[K])
        elif K in self._LCC_BOUNDED_MAP:
            fun = _lcc_bounded_contrast
            fun_args = dict(G=self._LCC_BOUNDED_MAP[K])
        else:
            fun = _lcc_contrast
            fun_args = dict(K=K)

        super().__init__(
            n_components=n_components,
            algorithm=algorithm,
            whiten=whiten,
            fun=fun,
            fun_args=fun_args,
            max_iter=max_iter,
            tol=tol,
            w_init=w_init,
            whiten_solver=whiten_solver,
            random_state=random_state,
        )

    def fit(self, X, y=None):
        """Fit the model, tracking convergence."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            super().fit(X, y)
        self.converged_ = not any(
            issubclass(w.category, ConvergenceWarning)
            for w in caught)
        return self

    def fit_transform(self, X, y=None):
        """Fit and transform, tracking convergence."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            result = super().fit_transform(X, y)
        self.converged_ = not any(
            issubclass(w.category, ConvergenceWarning)
            for w in caught)
        return result
