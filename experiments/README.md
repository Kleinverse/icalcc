# Experiments

Benchmark scripts for the ICALCC TechRxiv software paper.

## Design

Baseline: sklearn `FastICA` with its three built-in contrasts (`logcosh`, `exp`, `cube`).
Package: `ICALCC` with `K=4`, `K=6`, `K=8`.

Both use the same sklearn fixed-point iteration engine.
The contrast function is the only variable.

Distributions include both symmetric (Laplace, Logistic, Uniform,
Student-t15) and asymmetric (Exponential, Gamma) sources.
The asymmetric sources activate the m3^2 term in V6.

## Requirements

```bash
pip install numpy scikit-learn icalcc
```

## Quick start

```bash
python experiment.py                     # sanity check
python experiment.py --table1 --quick    # Table I, 20 trials
python experiment.py --all               # all tables, 200 trials
python experiment.py --all --quick       # all tables, 20 trials
python experiment.py --benchmark         # runtime comparison
python experiment.py --benchmark --quick # runtime, 5 trials
python experiment.py --test              # unit tests
```

## Tables

| Script | Description |
|--------|-------------|
| `exp_table1.py` | Amari index: 5 distributions (incl. Exponential), N=10k/100k |
| `exp_table2.py` | Gamma(alpha) scan, alpha=0.5 to 50, max_iter=500 |
| `exp_table3.py` | ARE: ASV(logcosh)/ASV(LCC(k)), 4 distributions (incl. Exponential) |
| `exp_table4.py` | V-statistic sample mean and t-statistic |
| `exp_benchmark.py` | Runtime: sklearn vs ICALCC wall-clock per fit() |
| `exp_sanity.py` | Quick sanity check |
