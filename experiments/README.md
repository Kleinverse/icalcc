# Experiments

Benchmark scripts for the ICALCC software paper.
The underlying theory is described in Saito (2026, TechRxiv).

All scripts use the actual `sklearn.decomposition.FastICA` and
`icalcc.ICALCC` packages.  Results are directly reproducible.

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
python experiment.py --benchmark         # runtime: sklearn vs ICALCC
python experiment.py --test              # unit tests
```

## Scripts

| Script | Description |
|--------|-------------|
| `exp_table1.py` | Amari index across distributions |
| `exp_table2.py` | Gamma(alpha) scan, convergence failures |
| `exp_table3.py` | ARE vs empirical efficiency ratio |
| `exp_table4.py` | V-statistic sample mean and t-statistic |
| `exp_benchmark.py` | Runtime: sklearn FastICA vs ICALCC |
| `exp_sanity.py` | Quick sanity check across all methods |
| `exp_utils.py`  | Shared utilities |
