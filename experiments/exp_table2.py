#!/usr/bin/env python3
"""
Table II: Gamma(alpha) scan, Amari x10^-2
  alpha = 0.5, 1, 2, 3, 5, 8, 12, 20, 50
  d=4, N=10000;  Methods: Fast(k), tanh, exp, LCC(k)

Usage:
  python exp_table2.py              # 200 trials
  python exp_table2.py --quick      # 20 trials
"""

import sys
import time
import numpy as np
from exp_utils import fastica_single, icalcc_algo, run_trials


def table2(n_trials=200, d=4, N=10000):
    print("=" * 72)
    print(f"TABLE II: Gamma(alpha) scan, Amari x10^-2 (d={d}, N={N}, {n_trials} trials)")
    print("=" * 72)

    alphas = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0]

    algos = [
        ("Fast(4)", lambda Z: fastica_single(Z, k=4)),
        ("tanh",    icalcc_algo("tanh")),
        ("exp",     icalcc_algo("exp")),
        ("Fast(6)", lambda Z: fastica_single(Z, k=6)),
        ("Fast(8)", lambda Z: fastica_single(Z, k=8)),
        ("LCC(4)",  icalcc_algo(4)),
        ("LCC(6)",  icalcc_algo(6)),
        ("LCC(8)",  icalcc_algo(8)),
    ]

    print(f"\n  {'alpha':>5s}  {'m3':>5s}  {'m4':>5s}", end="")
    for aname, _ in algos:
        print(f"  {aname:>7s}", end="")
    print()
    print("-" * (20 + 9 * len(algos)))

    for alpha in alphas:
        m3_th = 2.0 / np.sqrt(alpha)
        m4_th = 6.0 / alpha

        class GammaDist:
            name = f"Gamma({alpha})"
            @staticmethod
            def sample(N, a=alpha):
                x = np.random.gamma(a, 1.0, N)
                return (x - a) / np.sqrt(a)

        print(f"  {alpha:5.1f}  {m3_th:5.2f}  {m4_th:5.1f}", end="", flush=True)

        for ai, (aname, afn) in enumerate(algos):
            seed_base = int(alpha * 1000) + ai * 100000
            ais, nf = run_trials(GammaDist, d, N, afn, n_trials, seed_base)
            cell = f"{np.mean(ais)*100:.2f}"
            if nf > 0:
                cell += f"({nf})"
            print(f"  {cell:>7s}", end="", flush=True)
        print()


if __name__ == "__main__":
    n = 20 if "--quick" in sys.argv else 200
    t0 = time.time()
    table2(n_trials=n)
    print(f"\nCompleted in {time.time() - t0:.1f}s")
