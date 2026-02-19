# Experiments

Benchmark scripts for the ICALCC software paper.
The underlying theory is described in Saito (2026, TechRxiv).

## Quick start

```bash
python experiments/experiment.py                    # sanity check
python experiments/experiment.py --table1 --quick   # Table I (20 trials)
python experiments/experiment.py --table2 --quick   # Table II (20 trials)
python experiments/experiment.py --all              # all tables (200 trials)
python experiments/experiment.py --test             # unit tests
```

Append `--quick` to any command for 20 trials instead of 200.

## Scripts

| Script | Description | Standalone |
|--------|-------------|------------|
| `exp_table1.py` | Amari index across distributions | `python exp_table1.py --quick` |
| `exp_table2.py` | Gamma(alpha) scan, convergence failures | `python exp_table2.py --quick` |
| `exp_table3.py` | ARE vs empirical efficiency ratio | `python exp_table3.py --quick` |
| `exp_table4.py` | V-statistic sample mean and t-statistic | `python exp_table4.py --quick` |
| `exp_sanity.py` | Quick sanity check across all methods | `python exp_sanity.py` |
| `exp_utils.py`  | Shared utilities (imported) | |

## Testing

```bash
python experiments/experiment.py --test
```
