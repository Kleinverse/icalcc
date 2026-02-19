#!/usr/bin/env python3
"""
Table III: ARE vs empirical (AF/AL)^2
  ASV(g) = (E[g^2] - (E[sg])^2) / (E[g'] - E[sg])^2
  ARE(k) = ASV_FastICA(k) / ASV_LCC(k)
  Empirical: (Amari_FastICA / Amari_LCC)^2

Usage:
  python exp_table3.py              # 200 trials
  python exp_table3.py --quick      # 20 trials
"""

import sys
import time
import numpy as np
from exp_utils import (
    Laplace, Logistic, Uniform,
    random_mixing, whiten, amari_index,
    fastica_single, fastica_lcc, _lcc_h_beta,
)


def _compute_asv_fastica(s, k):
    """ASV for FastICA with g(u)=u^{k-1}."""
    g = s ** (k - 1)
    gp = (k - 1) * s ** (k - 2)
    Eg2 = np.mean(g**2)
    Esg = np.mean(s * g)
    Egp = np.mean(gp)
    return (Eg2 - Esg**2) / (Egp - Esg)**2


def _compute_asv_lcc(s, k):
    """ASV for LCC with g = dV_k/dw projected nonlinearity."""
    h_y, _ = _lcc_h_beta(s, k)
    Eh2 = np.mean(h_y**2)
    Esh = np.mean(s * h_y)
    eps = 1e-5
    hp, _ = _lcc_h_beta(s + eps, k)
    hm, _ = _lcc_h_beta(s - eps, k)
    h_prime = (hp - hm) / (2 * eps)
    Ehp = np.mean(h_prime)
    denom = (Ehp - Esh)**2
    if abs(denom) < 1e-30:
        return np.inf
    return (Eh2 - Esh**2) / denom


def table3(n_trials=200, d=4, N=100000):
    print("=" * 72)
    print(f"TABLE III: ARE vs empirical (d={d}, N={N}, {n_trials} trials)")
    print("=" * 72)

    dists_t3 = [Laplace, Logistic, Uniform]
    ks = [6, 8]

    N_th = 1000000
    print(f"\n{'':8s}", end="")
    for Dist in dists_t3:
        print(f"  {Dist.name:>16s}", end="")
    print()
    print(f"{'':8s}", end="")
    for _ in dists_t3:
        print(f"  {'ARE':>7s} {'Emp.':>7s}", end="")
    print()
    print("-" * (8 + 16 * len(dists_t3)))

    for k in ks:
        print(f"  k = {k}  ", end="", flush=True)
        for di, Dist in enumerate(dists_t3):
            np.random.seed(99999 + di * 100 + k)
            s_th = Dist.sample(N_th)
            asv_f = _compute_asv_fastica(s_th, k)
            asv_l = _compute_asv_lcc(s_th, k)
            are = asv_f / asv_l if asv_l > 0 else np.inf

            ai_f_list = []
            ai_l_list = []
            nf_f, nf_l = 0, 0
            for trial in range(n_trials):
                np.random.seed(di * 1000000 + k * 100000 + trial)
                S = np.vstack([Dist.sample(N) for _ in range(d)])
                A_mix = random_mixing(d)
                X = A_mix @ S
                Z, Ww = whiten(X)

                np.random.seed(di * 1000000 + k * 100000 + trial + 50000)
                try:
                    Wf = fastica_single(Z, k=k)
                    ai_f_list.append(amari_index(Wf @ Ww, A_mix))
                except Exception:
                    ai_f_list.append(1.0)
                    nf_f += 1

                np.random.seed(di * 1000000 + k * 100000 + trial + 60000)
                try:
                    Wl = fastica_lcc(Z, k=k)
                    ai_l_list.append(amari_index(Wl @ Ww, A_mix))
                except Exception:
                    ai_l_list.append(1.0)
                    nf_l += 1

            af = np.mean(ai_f_list)
            al = np.mean(ai_l_list)
            emp = (af / al)**2 if al > 0 else np.inf
            tag = ""
            if nf_f + nf_l > 0:
                tag = f" f:{nf_f}/{nf_l}"
            print(f"  {are:7.2f} {emp:7.2f}{tag}", end="", flush=True)
        print()


if __name__ == "__main__":
    n = 20 if "--quick" in sys.argv else 200
    t0 = time.time()
    table3(n_trials=n)
    print(f"\nCompleted in {time.time() - t0:.1f}s")
