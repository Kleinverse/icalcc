#!/usr/bin/env python3
"""
Experiment: Synthetic ICA -- Tables I-IV for IEEE SPL
=========================================================

Master script that delegates to individual experiment modules.
Each table module is also standalone (python exp_table1.py --quick).

Usage:
  python experiment.py                 # sanity check (20 trials)
  python experiment.py --table1        # Table I (200 trials)
  python experiment.py --table2        # Table II (200 trials)
  python experiment.py --table3        # Table III (200 trials)
  python experiment.py --table4        # Table IV
  python experiment.py --all           # all tables (200 trials)
  python experiment.py --test          # unit tests (14 tests)
  Append --quick to any for 20 trials.

Author: Dr. Tetsuya Saito
"""

import sys
import time
import numpy as np

from exp_table1 import table1
from exp_table2 import table2
from exp_table3 import table3
from exp_table4 import table4
from exp_sanity import sanity
from exp_utils import (
    DISTS,
    Uniform, Laplace, Exponential,
    random_mixing, whiten, amari_index,
    fastica_kurtosis, fastica_single, jade,
    icalcc_algo, _lcc_Vk,
)


# ======================================================================
# Test suite (14 tests)
# ======================================================================

def run_tests():
    np.random.seed(123)
    n_pass, n_fail = 0, 0

    def check(name, cond, detail=""):
        nonlocal n_pass, n_fail
        if cond:
            n_pass += 1
            print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
        else:
            n_fail += 1
            print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))

    print("=" * 70)
    print("TEST SUITE")
    print("=" * 70)

    print("\nT1. Distribution parameters (N=500000)")
    N = 500000
    for Dist, want_var, want_k4 in [
        (Uniform, 1.0, -1.2), (Laplace, 1.0, 3.0), (Exponential, 1.0, 6.0)]:
        x = Dist.sample(N)
        mu = np.mean(x); var = np.var(x)
        m4c = np.mean((x - mu)**4); k4 = m4c - 3.0 * var**2
        check(f"{Dist.name} mean~0", abs(mu) < 0.02, f"got {mu:.4f}")
        check(f"{Dist.name} var~{want_var}", abs(var - want_var) < 0.02, f"got {var:.4f}")
        check(f"{Dist.name} kappa4~{want_k4}", abs(k4 - want_k4) < 0.2, f"got {k4:.3f}")

    print("\nT2. Whitening")
    d, N = 4, 10000
    S = np.vstack([Uniform.sample(N) for _ in range(d)])
    A = random_mixing(d); Z, Ww = whiten(A @ S)
    cov = Z @ Z.T / N
    check("diag(Cov) ~ 1", np.allclose(np.diag(cov), 1.0, atol=0.05))
    off = cov - np.diag(np.diag(cov))
    check("off-diag ~ 0", np.max(np.abs(off)) < 0.05)

    print("\nT3. m_2 = 1 for unit w on whitened data")
    d, N = 4, 50000
    S = np.vstack([Laplace.sample(N) for _ in range(d)])
    Z, _ = whiten(random_mixing(d) @ S)
    for trial in range(5):
        w = np.random.randn(d); w /= np.linalg.norm(w)
        m2 = np.mean((w @ Z)**2)
        check(f"trial {trial+1}", abs(m2 - 1.0) < 0.03, f"m_2 = {m2:.4f}")

    print("\nT4. Amari index")
    I4 = np.eye(4)
    check("Amari(I, I) = 0", abs(amari_index(I4, I4)) < 1e-10)
    P = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]], dtype=float)
    check("Amari(perm, I) = 0", abs(amari_index(P, I4)) < 1e-10)
    D = np.diag([2.0, -1.0, 0.5, 3.0])
    check("Amari(scale*perm, I) = 0", abs(amari_index(D @ P, I4)) < 1e-10)

    print("\nT5. LCC V_4")
    N = 300000
    for Dist in DISTS:
        x = Dist.sample(N)
        v4 = 21.0/64.0 - 3.0*np.mean(x**4)/64.0
        v4_theory = 12.0/64.0 - 3.0 * Dist.true_kappa4 / 64.0
        check(f"{Dist.name}", abs(v4 - v4_theory) < 0.02,
              f"sample={v4:.4f}, theory={v4_theory:.4f}")

    print("\nT6. JADE cumulant structure")
    np.random.seed(600)
    d, N = 4, 50000
    S = np.vstack([Laplace.sample(N) for _ in range(d)])
    Z, Ww = whiten(S); W = jade(Z)
    ai = amari_index(W @ Ww, np.eye(d))
    check("JADE on independent sources", ai < 0.05, f"Amari = {ai*100:.2f}%")

    print("\nT7. JADE 2D")
    np.random.seed(600)
    d, N = 2, 30000
    S = np.vstack([Uniform.sample(N), Laplace.sample(N)])
    theta = np.pi / 5
    A2 = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    Z, Ww = whiten(A2 @ S); W = jade(Z)
    ai = amari_index(W @ Ww, A2)
    check("JADE 2D Amari < 5%", ai < 0.05, f"Amari = {ai*100:.2f}%")

    print("\nT8. FastICA 2D")
    np.random.seed(700)
    Z, Ww = whiten(A2 @ S); W = fastica_kurtosis(Z)
    ai = amari_index(W @ Ww, A2)
    check("FastICA 2D Amari < 2%", ai < 0.02, f"Amari = {ai*100:.2f}%")

    print("\nT9. 4D separation (30 trials)")
    np.random.seed(800)
    d, N, n_tr = 4, 10000, 30
    for Dist in DISTS:
        for name, algo_fn in [("FastICA", fastica_kurtosis),
                               ("JADE", jade),
                               ("LCC-k4", icalcc_algo(4))]:
            is_ic = getattr(algo_fn, '_icalcc', False)
            ais = []
            for _ in range(n_tr):
                S = np.vstack([Dist.sample(N) for _ in range(d)])
                A = random_mixing(d); Z, Ww = whiten(A @ S)
                try:
                    r = algo_fn(Z)
                    W = r[0] if is_ic else r
                    ais.append(amari_index(W @ Ww, A))
                except Exception:
                    ais.append(1.0)
            mean_ai = np.mean(ais) * 100
            check(f"{Dist.name} {name} < 10%", mean_ai < 10, f"Amari = {mean_ai:.2f}%")

    print("\nT10. JADE monotonicity (30 trials)")
    np.random.seed(900)
    d, n_tr = 4, 30
    for Dist in DISTS:
        results = {}
        for N in [2000, 10000, 50000]:
            ais = []
            for _ in range(n_tr):
                S = np.vstack([Dist.sample(N) for _ in range(d)])
                A = random_mixing(d); Z, Ww = whiten(A @ S)
                try:
                    W = jade(Z); ais.append(amari_index(W @ Ww, A))
                except Exception:
                    ais.append(1.0)
            results[N] = np.mean(ais) * 100
        check(f"{Dist.name} 10k < 2k", results[10000] < results[2000] + 0.5,
              f"{results[2000]:.2f} -> {results[10000]:.2f}")
        check(f"{Dist.name} 50k < 10k", results[50000] < results[10000] + 0.5,
              f"{results[10000]:.2f} -> {results[50000]:.2f}")

    print("\nT11. ICALCC(4) ~ FastICA at k=4 (parallel vs deflation)")
    np.random.seed(1000)
    d, N, n_tr = 4, 10000, 30
    for Dist in DISTS:
        diffs = []
        for _ in range(n_tr):
            S = np.vstack([Dist.sample(N) for _ in range(d)])
            A = random_mixing(d); Z, Ww = whiten(A @ S)
            state = np.random.get_state()
            np.random.set_state(state)
            ai_f = amari_index(fastica_kurtosis(Z) @ Ww, A)
            np.random.set_state(state)
            ai_l = amari_index(icalcc_algo(4)(Z)[0] @ Ww, A)
            diffs.append(abs(ai_f - ai_l))
        md = np.mean(diffs) * 100
        check(f"{Dist.name} |diff| < 1%", md < 1.0, f"mean |diff| = {md:.4f}%")

    print("\nT12. V_6 analytic formula")
    np.random.seed(1200)
    N = 200
    for Dist in DISTS:
        y = Dist.sample(N)
        n_mc = 300000
        idx = np.random.randint(0, N, (n_mc, 6))
        yi = y[idx]; ybar = yi.mean(axis=1, keepdims=True)
        v6_mc = np.mean(np.prod(yi - ybar, axis=1))
        m1 = np.mean(y); m2 = np.mean(y**2); m3 = np.mean(y**3)
        m4 = np.mean(y**4); m6 = np.mean(y**6)
        v6_fm = (155*m1**6/324 - 155*m1**4*m2/108 + 55*m1**3*m3/162
                 + 85*m1**2*m2**2/72 - 35*m1**2*m4/648
                 - 65*m1*m2*m3/162 + 5*m1*np.mean(y**5)/1296
                 - 125*m2**3/648 + 115*m2*m4/2592
                 + 145*m3**2/3888 - 5*m6/7776)
        check(f"{Dist.name} V_6 MC~formula", abs(v6_mc - v6_fm) < 0.015,
              f"MC={v6_mc:.5f}, formula={v6_fm:.5f}")

    print("\nT13. LCC single-order 2D")
    np.random.seed(1300)
    d, N = 2, 20000
    S = np.vstack([Exponential.sample(N), Laplace.sample(N)])
    theta13 = np.pi / 5
    A13 = np.array([[np.cos(theta13), -np.sin(theta13)],
                     [np.sin(theta13), np.cos(theta13)]])
    Z, Ww = whiten(A13 @ S)
    for kk in [4, 6, 8]:
        np.random.seed(1300 + kk)
        W, _ = icalcc_algo(kk)(Z)
        ai = amari_index(W @ Ww, A13)
        check(f"LCC k={kk} Amari < 10%", ai < 0.10, f"Amari = {ai*100:.2f}%")

    print("\nT14. FastICA single-order 2D")
    for kk in [4, 6, 8]:
        np.random.seed(1400 + kk)
        W = fastica_single(Z, k=kk)
        ai = amari_index(W @ Ww, A13)
        check(f"Fast k={kk} Amari < 10%", ai < 0.10, f"Amari = {ai*100:.2f}%")

    print("\n" + "=" * 70)
    total = n_pass + n_fail
    print(f"RESULTS: {n_pass}/{total} passed, {n_fail} failed")
    if n_fail == 0:
        print("ALL TESTS PASSED")
    print("=" * 70)
    return n_fail == 0


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    quick = "--quick" in sys.argv
    n = 20 if quick else 200

    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)

    t0 = time.time()

    if "--table1" in sys.argv:
        table1(n_trials=n)
    elif "--table2" in sys.argv:
        table2(n_trials=n)
    elif "--table3" in sys.argv:
        table3(n_trials=n)
    elif "--table4" in sys.argv:
        table4(n_trials=n)
    elif "--all" in sys.argv:
        table1(n_trials=n)
        print()
        table2(n_trials=n)
        print()
        table3(n_trials=n)
        print()
        table4(n_trials=n)
    else:
        sanity(n_trials=n)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")
