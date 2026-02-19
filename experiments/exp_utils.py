#!/usr/bin/env python3
"""
Shared utilities for ICA experiments.

Provides: distributions, mixing/whitening, Amari index,
standalone FastICA variants, JADE, LCC V_k formulas,
ICALCC wrapper, and generic trial runner.

Author: Dr. Tetsuya Saito
"""

import numpy as np
import warnings

warnings.filterwarnings("ignore", module="sklearn")

from icalcc import ICALCC


# ======================================================================
# Distributions (unit variance, zero mean)
# ======================================================================

class Laplace:
    name = "Laplace"
    true_kappa4 = 3.0
    @staticmethod
    def sample(N):
        return np.random.laplace(0, 1.0 / np.sqrt(2), N)

class Uniform:
    name = "Uniform"
    true_kappa4 = -1.2
    @staticmethod
    def sample(N):
        return np.random.uniform(-np.sqrt(3), np.sqrt(3), N)

class Exponential:
    name = "Exponential"
    true_kappa4 = 6.0
    @staticmethod
    def sample(N):
        return np.random.exponential(1.0, N) - 1.0

class StudentT15:
    name = "Student-t15"
    true_kappa4 = 12.0 / (15 - 4)  # 12/11
    @staticmethod
    def sample(N):
        x = np.random.standard_t(15, N)
        return x / np.std(x)

class Logistic:
    name = "Logistic"
    true_kappa4 = 1.2
    @staticmethod
    def sample(N):
        x = np.random.logistic(0, 1, N)
        return x / np.std(x)


DISTS = [Uniform, Laplace, Exponential, StudentT15, Logistic]
DISTS_T1 = [Laplace, Logistic, Uniform, StudentT15]
DISTS_T4 = [Uniform, Laplace, Exponential]


# ======================================================================
# ICA utilities
# ======================================================================

def random_mixing(d):
    H = np.random.randn(d, d)
    Q, R = np.linalg.qr(H)
    return Q * np.sign(np.diag(R))


def whiten(X):
    Xc = X - X.mean(axis=1, keepdims=True)
    C = Xc @ Xc.T / Xc.shape[1]
    vals, vecs = np.linalg.eigh(C)
    D = np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12)))
    W = D @ vecs.T
    return W @ Xc, W


def amari_index(W_est, A_true):
    P = np.abs(W_est @ A_true)
    d = P.shape[0]
    row_max = P.max(axis=1, keepdims=True)
    col_max = P.max(axis=0, keepdims=True)
    return (np.sum(P / row_max) - d + np.sum(P / col_max) - d) / (2*d*(d-1))


# ======================================================================
# FastICA single-order (k=4,6,8)
# ======================================================================

def fastica_single(Z, k=4, max_iter=200, tol=1e-7):
    d, N = Z.shape
    W = np.zeros((d, d))
    for p in range(d):
        w = np.random.randn(d)
        w /= np.linalg.norm(w)
        for _ in range(max_iter):
            y = w @ Z
            ykm1 = y ** (k - 1)
            beta = (k - 1) * np.mean(y ** (k - 2))
            w_new = (Z * ykm1).mean(axis=1) - beta * w
            for j in range(p):
                w_new -= (w_new @ W[j]) * W[j]
            w_new /= np.linalg.norm(w_new) + 1e-12
            if abs(abs(w_new @ w) - 1.0) < tol:
                w = w_new
                break
            w = w_new
        W[p] = w
    return W


def fastica_kurtosis(Z, max_iter=200, tol=1e-7):
    return fastica_single(Z, k=4, max_iter=max_iter, tol=tol)


# ======================================================================
# FastICA with general nonlinearity g(y)
# ======================================================================

def fastica_general(Z, g_func, max_iter=200, tol=1e-7):
    """FastICA deflation with arbitrary nonlinearity.

    Parameters
    ----------
    g_func : callable
        Takes y (N,) -> (gy, gpy) where gy (N,) and gpy scalar.
    """
    d, N = Z.shape
    W = np.zeros((d, d))
    for p in range(d):
        w = np.random.randn(d)
        w /= np.linalg.norm(w)
        for _ in range(max_iter):
            y = w @ Z
            gy, gpy = g_func(y)
            w_new = (Z * gy).mean(axis=1) - gpy * w
            for j in range(p):
                w_new -= (w_new @ W[j]) * W[j]
            w_new /= np.linalg.norm(w_new) + 1e-12
            if abs(abs(w_new @ w) - 1.0) < tol:
                w = w_new
                break
            w = w_new
        W[p] = w
    return W


def _g_tanh(y):
    """logcosh contrast: g(y) = tanh(y), g'(y) = 1 - tanh(y)^2."""
    t = np.tanh(y)
    return t, np.mean(1.0 - t * t)


def _g_exp(y):
    """Gaussian contrast: g(y) = y*exp(-y^2/2),
    g'(y) = (1-y^2)*exp(-y^2/2)."""
    ey2 = np.exp(-0.5 * y * y)
    return y * ey2, np.mean((1.0 - y * y) * ey2)


def _g_abs(y):
    """Skewness contrast: g(y) = y*|y|, g'(y) = 2*|y|."""
    s = np.sign(y)
    return y * np.abs(y), np.mean(2.0 * np.abs(y))


# ======================================================================
# LCC V_k formulas and Newton fixed-point
# ======================================================================

def _lcc_Vk(y, k):
    m3 = np.mean(y**3)
    m4 = np.mean(y**4)
    if k == 4:
        return 21.0/64 - 3*m4/64
    m5 = np.mean(y**5)
    m6 = np.mean(y**6)
    if k == 6:
        return 145*m3**2/3888 + 115*m4/2592 - 5*m6/7776 - 125.0/648
    m8 = np.mean(y**8)
    return (-7665*m3**2/131072 + 497*m3*m5/262144
            + 2765*m4**2/2097152 - 18795*m4/524288
            + 329*m6/524288 - 7*m8/2097152 + 117705.0/1048576)


def _lcc_h_beta(y, k):
    y2 = y * y
    y3 = y2 * y

    if k == 4:
        return (-3.0/16) * y3, -9.0/16

    m3 = np.mean(y3)
    m4 = np.mean(y2 * y2)

    if k == 6:
        dJ3 = 145*m3/1944.0
        dJ4 = 115.0/2592
        dJ6 = -5.0/7776
        y4 = y2 * y2
        h_y = dJ3*3*y2 + dJ4*4*y3 + dJ6*6*(y4*y)
        beta = 12*dJ4 + 30*dJ6*m4
        return h_y, beta

    # k == 8
    m5 = np.mean(y2 * y3)
    m6 = np.mean(y3 * y3)
    dJ3 = -7665*m3/65536.0 + 497*m5/262144.0
    dJ4 = 2765*m4/1048576.0 - 18795.0/524288
    dJ5 = 497*m3/262144.0
    dJ6 = 329.0/524288
    dJ8 = -7.0/2097152
    y4 = y2 * y2
    h_y = (dJ3*3*y2 + dJ4*4*y3 + dJ5*5*y4
           + dJ6*6*(y4*y) + dJ8*8*(y4*y3))
    beta = 12*dJ4 + 20*dJ5*m3 + 30*dJ6*m4 + 56*dJ8*m6
    return h_y, beta


def fastica_lcc(Z, k=4, max_iter=200, tol=1e-7,
                n_restarts=5, warm_W=None, damping=1.0):
    d, N = Z.shape
    W = np.zeros((d, d))

    for p in range(d):
        best_w = None
        best_Vk = -np.inf

        for restart in range(n_restarts):
            if restart == 0 and warm_W is not None:
                w = warm_W[p].copy()
                w /= np.linalg.norm(w)
            else:
                w = np.random.randn(d)
                w /= np.linalg.norm(w)

            for it in range(max_iter):
                y = w @ Z
                h_y, beta = _lcc_h_beta(y, k)

                w_newton = (Z * h_y).mean(axis=1) - beta * w
                w_new = (1 - damping) * w + damping * w_newton

                for j in range(p):
                    w_new -= (w_new @ W[j]) * W[j]
                w_new /= np.linalg.norm(w_new) + 1e-12

                if abs(abs(w_new @ w) - 1.0) < tol:
                    w = w_new
                    break
                w = w_new

            Vk = abs(_lcc_Vk(w @ Z, k))
            if Vk > best_Vk:
                best_Vk = Vk
                best_w = w.copy()

        W[p] = best_w
    return W


# ======================================================================
# JADE (Cardoso & Souloumiac, jadeR v1.8)
# ======================================================================

def jade(Z):
    d, N = Z.shape
    m = d
    T = float(N)
    X = Z.T.copy()

    nbcm = int(m * (m + 1) / 2)
    CM = np.zeros((m, m * nbcm))
    R = np.eye(m)

    Range = np.arange(m)
    for im in range(m):
        Xim = X[:, im]
        Xijm = Xim * Xim
        Qij = (Xijm[:, np.newaxis] * X).T @ X / T \
              - R - 2.0 * np.outer(R[:, im], R[:, im])
        CM[:, Range] = Qij
        Range = Range + m
        for jm in range(im):
            Xijm = Xim * X[:, jm]
            Qij = np.sqrt(2.0) * (Xijm[:, np.newaxis] * X).T @ X / T \
                  - np.outer(R[:, im], R[:, jm]) \
                  - np.outer(R[:, jm], R[:, im])
            CM[:, Range] = Qij
            Range = Range + m

    V = np.eye(m)
    seuil = 1.0e-6 / np.sqrt(T)
    encore = True
    sweep = 0

    while encore:
        encore = False
        sweep += 1
        if sweep > 200:
            break
        for p in range(m - 1):
            for q in range(p + 1, m):
                Ip = np.arange(p, m * nbcm, m)
                Iq = np.arange(q, m * nbcm, m)
                g = np.vstack([CM[p, Ip] - CM[q, Iq],
                               CM[p, Iq] + CM[q, Ip]])
                gg = g @ g.T
                ton = gg[0, 0] - gg[1, 1]
                toff = gg[0, 1] + gg[1, 0]
                theta = 0.5 * np.arctan2(
                    toff, ton + np.sqrt(ton * ton + toff * toff))

                if abs(theta) > seuil:
                    encore = True
                    c = np.cos(theta)
                    s = np.sin(theta)
                    rp = CM[p, :].copy()
                    rq = CM[q, :].copy()
                    CM[p, :] = c * rp + s * rq
                    CM[q, :] = -s * rp + c * rq
                    cp = CM[:, Ip].copy()
                    cq = CM[:, Iq].copy()
                    CM[:, Ip] = c * cp + s * cq
                    CM[:, Iq] = -s * cp + c * cq
                    vp = V[:, p].copy()
                    vq = V[:, q].copy()
                    V[:, p] = c * vp + s * vq
                    V[:, q] = -s * vp + c * vq

    return V.T


# ======================================================================
# ICALCC wrapper for run_trials interface
# ======================================================================

def icalcc_algo(K, max_iter=200, tol=1e-7, algorithm='deflation'):
    """Return algo_fn(Z) -> (W, converged) using ICALCC on whitened data."""
    def fn(Z):
        d = Z.shape[0]
        ica = ICALCC(
            n_components=d, K=K, algorithm=algorithm, whiten=False,
            max_iter=max_iter, tol=tol, random_state=None)
        ica.fit(Z.T)
        return ica.components_, ica.converged_
    fn._icalcc = True
    return fn


# ======================================================================
# Generic trial runner
# ======================================================================

def run_trials(Dist, d, N, algo_fn, n_trials, seed_base):
    """Returns (ais_array, n_failures)."""
    ais = []
    n_fail = 0
    is_icalcc = getattr(algo_fn, '_icalcc', False)
    for trial in range(n_trials):
        np.random.seed(seed_base + trial)
        S = np.vstack([Dist.sample(N) for _ in range(d)])
        A_mix = random_mixing(d)
        X = A_mix @ S
        Z, Ww = whiten(X)
        try:
            if is_icalcc:
                Ws, converged = algo_fn(Z)
                if not converged:
                    n_fail += 1
            else:
                Ws = algo_fn(Z)
            ais.append(amari_index(Ws @ Ww, A_mix))
        except Exception:
            ais.append(1.0)
            n_fail += 1
    return np.array(ais), n_fail
