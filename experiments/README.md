# Experiments

Benchmark scripts for the ICALCC TechRxiv software paper,
reproducing the tables from the IEEE SPL paper (Saito 2026).

## Design

All experiments compare **FastICA(k)** vs **LCC(k)** at matched order k:

- **FastICA(k):** maximises |m_k| via g(y) = y^{k-1} (custom callable
  passed to sklearn's FastICA)
- **LCC(k):** maximises |V_k| via the V-statistic gradient (ICALCC package)

Both use the same sklearn fixed-point iteration engine.
The contrast function is the only variable.

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
python experiment.py --test              # unit tests
```

## Tables

| Script | Paper Table | Description |
|--------|------------|-------------|
| `exp_table1.py` | Table I | Amari index across 5 distributions, N=10k/100k |
| `exp_table2.py` | Table II | Gamma(alpha) scan, alpha=0.5 to 50 |
| `exp_table3.py` | Table III | Theoretical ARE vs empirical ratio |
| `exp_table4.py` | Table IV | V-statistic sample mean and t-statistic |
| `exp_benchmark.py` | -- | Runtime: FastICA(k) vs LCC(k) wall-clock |
| `exp_sanity.py` | -- | Quick sanity check |
