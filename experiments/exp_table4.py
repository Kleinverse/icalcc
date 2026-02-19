#!/usr/bin/env python3
"""
Table IV: V-statistic sample mean (t-statistic)
  Distributions: Uniform, Laplace, Exponential
  k = 2, 4, 6, 8;  N = 10000

Usage:
  python exp_table4.py              # 200 trials
  python exp_table4.py --quick      # 20 trials
"""

import sys
import time
import numpy as np
from exp_utils import DISTS_T4, _lcc_Vk


def _lcc_V2(y):
    """V_2 for whitened data: V_2 = -m2/4 = -1/4."""
    return -np.mean(y**2) / 4.0


def table4(n_trials=200, N=10000):
    print("=" * 72)
    print(f"TABLE IV: V-statistic mean (t-stat) (N={N}, {n_trials} trials)")
    print("=" * 72)

    ks = [2, 4, 6, 8]

    print(f"\n{'':14s}", end="")
    for k in ks:
        print(f"  {'k='+str(k):>16s}", end="")
    print()
    print("-" * (14 + 18 * len(ks)))

    for Dist in DISTS_T4:
        print(f"  {Dist.name:<12s}", end="", flush=True)
        for k in ks:
            vals = []
            for trial in range(n_trials):
                np.random.seed(7777 + trial)
                y = Dist.sample(N)
                if k == 2:
                    vals.append(_lcc_V2(y))
                else:
                    vals.append(_lcc_Vk(y, k))
            vals = np.array(vals)
            mn = np.mean(vals)
            se = np.std(vals, ddof=1) / np.sqrt(n_trials)
            t_stat = mn / se if se > 0 else np.inf
            print(f"  {mn:7.3f} ({t_stat:5.0f})", end="")
        print()


if __name__ == "__main__":
    n = 20 if "--quick" in sys.argv else 200
    t0 = time.time()
    table4(n_trials=n)
    print(f"\nCompleted in {time.time() - t0:.1f}s")
