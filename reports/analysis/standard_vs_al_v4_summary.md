# Standard/passive baseline vs v4 AL summary
Source of truth for AL: `results/al_v4/v4_primary_fixed_member` final metric row at N=550. No AL reruns are included here.

| Architecture | Method | Labels | Energy MAE (meV) | Force MAE (meV/Å) | Total training time (h) |
|---|---:|---:|---:|---:|---:|
| mace | QBC AL | 550 | 1.85 ± 0.27 | 9.05 ± 0.71 | 14.03 ± 0.26 |
| mace | RANDOM AL | 550 | 1.87 ± 0.15 | 9.66 ± 0.28 | 2.92 ± 0.08 |
| nequip | QBC AL | 550 | 4.40 ± 1.95 | 10.07 ± 0.56 | 33.72 ± 0.31 |
| nequip | RANDOM AL | 550 | 3.31 ± 0.88 | 11.03 ± 0.34 | 8.04 ± 0.13 |
| nequip | Standard/passive Phase 2 | 950 train + 50 valid | 1.30 | 6.79 | not recorded in `results/phase2_metrics.json` |

Interpretation:
- NequIP Phase 2 standard/passive 950-label baseline is stronger than v4 NequIP AL at 550 labels on both force and energy MAE. It supports the statement that roughly 950 passive labels were needed in the older standard run to reach or exceed the 550-label AL force-accuracy range.
- For v4 fixed-member runs, NequIP-QBC improves force MAE over NequIP-random at N=550 (10.07 vs 11.03 meV/Å) but costs much more total training time (33.72 vs 8.04 h).
- For v4 fixed-member runs, MACE-QBC improves force MAE over MACE-random at N=550 (9.05 vs 9.66 meV/Å) but costs much more total training time (14.03 vs 2.92 h).
- A directly comparable MACE standard/passive 950-label baseline was not found in existing results. Do not substitute the interrupted quick 2-epoch smoke run for a thesis claim.
