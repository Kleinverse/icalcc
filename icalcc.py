"""ICALCC: Higher-order cumulant contrasts for scikit-learn FastICA.

This module implements the Locally Centered Cyclic (LCC) kernel
as a scikit-learn compatible ICA estimator. ICALCC extends
sklearn.decomposition.FastICA with V-statistic contrasts of
order k = 4, 6, 8 that reach beyond fourth-order cumulant
structure.

V_k formulas (whitened: m1=0, m2=1):

  V_4 = 21/64 - 3*m4/64

  V_6 = 145*m3^2/3888 + 115*m4/2592
        - 5*m6/7776 - 125/648

  V_8 = -7665*m3^2/131072 + 497*m3*m5/262144
        + 2765*m4^2/2097152 - 18795*m4/524288
        + 329*m6/524288 - 7*m8/2097152
        + 117705/1048576

References
----------
T. Saito, "Locally Centered Cyclic Kernels for Higher-Order
Independent Component Analysis," TechRxiv.
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


def _lcc_contrast(x, K=6):
    """LCC contrast callable for sklearn's ``fun`` parameter.

    Handles both deflation (1-D input) and parallel (2-D input)
    algorithms.

    Parameters
    ----------
    x : ndarray, shape (N,) or (p, N)
    K : int
        Cumulant order.

    Returns
    -------
    gy : ndarray, same shape as x
    gpy : ndarray, shape (N,) or (p,)
        For 2-D input, returns row-wise means of g'.
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


def _skew_contrast(x):
    """Skewness contrast: g(y) = y|y|, g'(y) = 2|y|.

    Useful for asymmetric source distributions.

    Parameters
    ----------
    x : ndarray, shape (N,) or (p, N)

    Returns
    -------
    gy : ndarray, same shape as x
    gpy : ndarray, shape (N,) or (p,)
    """
    ay = np.abs(x)
    gy = x * ay
    if x.ndim == 1:
        return gy, 2.0 * ay
    return gy, np.mean(2.0 * ay, axis=1)


class ICALCC(FastICA):
    """Independent Component Analysis with LCC contrast.

    Drop-in replacement for ``sklearn.decomposition.FastICA``
    that targets higher-order cumulants via the Locally Centered
    Cyclic (LCC) V-statistic kernel, with optional classical
    nonlinearities for comparison.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of components to extract.
    K : {4, 6, 8, 'tanh', 'exp', 'skew'}, default=6
        Contrast function.

        - ``4``: LCC V_4, equivalent to ``fun='cube'`` (kurtosis).
        - ``6``: LCC V_6, jointly exploits m_3, m_4, m_6.
          Recommended for most applications.
        - ``8``: LCC V_8, uses moments up to order 8.
          Best for symmetric sources with finite m_8.
        - ``'tanh'``: logcosh contrast, g(y) = tanh(y).
          Robust general-purpose baseline.
        - ``'exp'``: Gaussian contrast, g(y) = y exp(-y^2/2).
          Good for super-Gaussian sources.
        - ``'skew'``: Skewness contrast, g(y) = y|y|.
          Exploits asymmetry in skewed sources.
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
        Mixing matrix (pseudo-inverse of components_).
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
    >>> # Classical tanh baseline for comparison
    >>> ica_tanh = ICALCC(n_components=4, K='tanh', random_state=0)
    >>> S_tanh = ica_tanh.fit_transform(X)

    References
    ----------
    T. Saito, "Locally Centered Cyclic Kernels for Higher-Order
    Independent Component Analysis," IEEE Signal Processing Letters,
    2025.

    See Also
    --------
    sklearn.decomposition.FastICA : Base class.
    """

    _VALID_K = (4, 6, 8, "tanh", "exp", "skew")

    # sklearn built-in contrasts
    _SKLEARN_MAP = {
        "tanh": "logcosh",
        "exp": "exp",
    }

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
        """Fit the model, tracking convergence.

        Sets ``self.converged_`` to True if FastICA converged
        within ``max_iter`` iterations, False otherwise.
        All ConvergenceWarnings are suppressed.
        """
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
