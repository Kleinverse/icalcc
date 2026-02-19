# Experiments

Benchmark scripts for the ICALCC TechRxiv software paper.

## Design

sklearn built-in FastICA (logcosh, exp, cube) vs ICALCC (K=4, 6, 8).

## Requirements

```bash
pip install numpy scikit-learn icalcc
```

## Usage

```bash
python experiment.py                     # sanity check
python experiment.py --table1 --quick    # Table I, 20 trials
python experiment.py --all               # all tables, 200 trials
python experiment.py --all --quick       # all tables, 20 trials
python experiment.py --test              # unit tests
```

## Tables

| Script | Description |
|--------|-------------|
| `exp_table1.py` | Amari index across 5 distributions, N=10k/100k |
| `exp_table2.py` | Gamma(alpha) scan, alpha=0.5 to 50 |
| `exp_table3.py` | Runtime: wall-clock per fit() |
| `exp_table4.py` | V-statistic sample mean and t-statistic |
