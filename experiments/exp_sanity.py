#!/usr/bin/env python3
"""
Sanity check: quick Amari index verification across distributions.
  d=4, N=2000 and N=10000, 20 trials

Usage:
  python exp_sanity.py
  python exp_sanity.py --quick      # same (20 trials is default)
"""

import sys
import time
import numpy as np
from exp_utils import (
    DISTS_T1,
    random_mixing, whiten, amari_index,
    fastica_kurtosis, fastica_single, jade,
    icalcc_algo,
)


def sanity(n_trials=20, d=4):
    print("=" * 72)
    print(f"SANITY CHECK: Amari x10^-2 (d={d}, {n_trials} trials)")
    print("=" * 72)

    Ns = [2000, 10000]
    algos = [
        ("FastICA",  fastica_kurtosis),
        ("tanh",     icalcc_algo("tanh")),
        ("exp",      icalcc_algo("exp")),
        ("JADE",     jade),
        ("Fast-k6",  lambda Z: fastica_single(Z, k=6)),
        ("LCC-k4",   icalcc_algo(4)),
        ("LCC-k6",   icalcc_algo(6)),
    ]

    for di, Dist in enumerate(DISTS_T1):
        print(f"\n  {Dist.name}")
        print(f"  {'':12s}", end="")
        for N in Ns:
            print(f"  {'N='+str(N):>8s}", end="")
        print()

        for ai, (name, algo_fn) in enumerate(algos):
            is_icalcc = getattr(algo_fn, '_icalcc', False)
            print(f"  {name:<12s}", end="", flush=True)
            for ni, N in enumerate(Ns):
                ais = []
                nf = 0
                for trial in range(n_trials):
                    np.random.seed(di * 1000000 + ni * 10000 + trial)
                    S = np.vstack([Dist.sample(N) for _ in range(d)])
                    A_mix = random_mixing(d)
                    X = A_mix @ S
                    Z, Ww = whiten(X)
                    np.random.seed(di*1000000 + ni*10000 + trial + (ai+1)*100000)
                    try:
                        if is_icalcc:
                            Ws, conv = algo_fn(Z)
                            if not conv:
                                nf += 1
                        else:
                            Ws = algo_fn(Z)
                        ais.append(amari_index(Ws @ Ww, A_mix))
                    except Exception:
                        ais.append(1.0)
                        nf += 1
                cell = f"{np.mean(ais)*100:.2f}"
                if nf > 0:
                    cell += f"({nf})"
                print(f"  {cell:>8s}", end="", flush=True)
            print()


if __name__ == "__main__":
    n = 20 if "--quick" in sys.argv else 20
    t0 = time.time()
    sanity(n_trials=n)
    print(f"\nCompleted in {time.time() - t0:.1f}s")
