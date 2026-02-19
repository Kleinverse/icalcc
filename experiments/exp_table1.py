#!/usr/bin/env python3
"""
Table I: Mean Amari index (x10^-2)
  Distributions: Laplace, Logistic, Uniform, Student-t15
  N = 10000, 100000;  Methods: Fast(k), tanh, exp, LCC(k)

Usage:
  python exp_table1.py              # 200 trials
  python exp_table1.py --quick      # 20 trials
"""

import sys
import time
import numpy as np
from exp_utils import (
    DISTS_T1, fastica_single, icalcc_algo, run_trials,
)


def table1(n_trials=200, d=4):
    print("=" * 72)
    print(f"TABLE I: Mean Amari index x10^-2 (d={d}, {n_trials} trials)")
    print("=" * 72)

    Ns = [10000, 100000]
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

    print(f"\n{'':14s}", end="")
    for Dist in DISTS_T1:
        print(f"  {Dist.name:>16s}", end="")
    print()
    print(f"{'':14s}", end="")
    for _ in DISTS_T1:
        print(f"  {'10k':>7s} {'100k':>7s}", end="")
    print()
    print("-" * (14 + 16 * len(DISTS_T1)))

    for ai, (aname, afn) in enumerate(algos):
        print(f"  {aname:<12s}", end="", flush=True)
        for di, Dist in enumerate(DISTS_T1):
            for ni, N in enumerate(Ns):
                seed_base = di * 1000000 + ni * 100000 + ai * 10000
                ais, nf = run_trials(Dist, d, N, afn, n_trials, seed_base)
                cell = f"{np.mean(ais)*100:.2f}"
                if nf > 0:
                    cell += f"({nf})"
                print(f"  {cell:>7s}", end="", flush=True)
        print()
        if ai == 4:
            print()


if __name__ == "__main__":
    n = 20 if "--quick" in sys.argv else 200
    t0 = time.time()
    table1(n_trials=n)
    print(f"\nCompleted in {time.time() - t0:.1f}s")
