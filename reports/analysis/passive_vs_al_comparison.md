# Passive 550-label baselines vs active-learning results

This comparison uses completed passive 550-label runs on the frozen v4 split and existing AL artifacts. No AL reruns are included. Metrics are reported on held-out test structures in meV and meV/Å.

## Direct v4 comparison

| Architecture | Method | Labels | Seeds | Energy MAE (meV) | Force MAE (meV/Å) | Training time (h) | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| mace | AL QBC (v4 fixed-member) | 550 | 3 | 1.85 ± 0.27 | 9.05 ± 0.71 | 14.03 ± 0.26 | `results/al_v4/v4_primary_fixed_member/mace_qbc_seed*/metrics.json` |
| mace | AL RANDOM (v4 fixed-member) | 550 | 3 | 1.87 ± 0.15 | 9.66 ± 0.28 | 2.92 ± 0.08 | `results/al_v4/v4_primary_fixed_member/mace_random_seed*/metrics.json` |
| mace | Passive single model (v4 split) | 550 | 3 | 1.97 ± 0.07 | 10.31 ± 0.10 | 0.47 ± 0.00 | `results/passive_v4_550/mace_passive_seed*/metrics.json` |
| nequip | AL QBC (v4 fixed-member) | 550 | 3 | 4.40 ± 1.95 | 10.07 ± 0.56 | 33.72 ± 0.31 | `results/al_v4/v4_primary_fixed_member/nequip_qbc_seed*/metrics.json` |
| nequip | AL RANDOM (v4 fixed-member) | 550 | 3 | 3.31 ± 0.88 | 11.03 ± 0.34 | 8.04 ± 0.13 | `results/al_v4/v4_primary_fixed_member/nequip_random_seed*/metrics.json` |
| nequip | Passive single model (v4 split) | 550 | 3 | 3.21 ± 0.59 | 10.65 ± 0.50 | 1.13 ± 0.06 | `results/passive_v4_550/nequip_passive_seed*/metrics.json` |

## Context baselines mentioned in dissertation materials

| Architecture | Method | Labels | Seeds | Energy MAE (meV) | Force MAE (meV/Å) | Training time (h) | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| nequip | Passive single model (Phase 2 950 train + 50 valid) | 950 | 1 | 1.30 | 6.79 | n/a | `results/phase2_metrics.json` |
| mace | AL MHC (historical, not v4) | 550 | 3 | 2.07 ± 0.01 | 10.07 ± 0.11 | 5.98 ± 0.08 | `results/al/mace_mhc_seed*/metrics.json` |
| mace | AL QBC (historical) | 550 | 3 | 1.63 ± 0.03 | 8.60 ± 0.12 | 14.02 ± 0.27 | `results/al/mace_qbc_seed*/metrics.json` |
| mace | AL Random (historical) | 550 | 4 | 1.74 ± 0.10 | 9.31 ± 0.38 | 3.84 ± 0.48 | `results/al/random_seed*/metrics.json` |
| nequip | AL QBC (historical) | 550 | 3 | 3.68 ± 2.05 | 7.71 ± 0.14 | 16.62 ± 0.60 | `results/al/nequip_qbc_seed*/metrics.json` |
| nequip | AL Random (historical) | 550 | 3 | 4.47 ± 1.96 | 9.23 ± 0.31 | 5.31 ± 0.32 | `results/al/nequip_random_seed*/metrics.json` |

## Peer-review interpretation

A peer reviewer would not accept these results as evidence that active learning clearly dominates passive sampling on rMD17 ethanol. The passive 550-label baselines are too competitive, the seed count is small, and the QBC wall-clock cost is much higher. The strongest defensible result is narrower: QBC gives the lowest mean force MAE at the fixed 550-label budget, especially for MACE, but the improvement is modest and not matched by a consistent energy-MAE advantage.

For MACE, QBC reduces force MAE from 10.31 ± 0.10 meV/Å for passive training to 9.05 ± 0.71 meV/Å, a 12.2% mean reduction. Random active learning is intermediate at 9.66 ± 0.28 meV/Å. This is the clearest support for uncertainty-guided selection, but it should be described as a modest force-MAE improvement, not as a large data-efficiency breakthrough.

For NequIP, QBC reduces force MAE from 10.65 ± 0.50 meV/Å for passive training to 10.07 ± 0.56 meV/Å, only a 5.4% mean reduction. Passive NequIP also has the best mean energy MAE among the three NequIP methods. This weakens any architecture-general claim and shows that the acquisition rule helps forces more reliably than energies.

## Dissertation framing

The passive controls strengthen the dissertation only if they are used to limit the claim. They show critical evaluation under the marking form's Results analysis and Interpretation criteria: the thesis tests QBC against a harder control, recognises that dense rMD17 ethanol makes passive/random sampling strong, and avoids pretending that AL is automatically cost-effective.

Recommended thesis claim:

> Under the frozen v4 split at N = 550, QBC gives the lowest mean force MAE for both MACE and NequIP, but the passive baselines are close and much cheaper. The result supports QBC as a force-oriented selection strategy, not as a universally superior or automatically cost-saving training method.

Avoid these claims:

- AL clearly outperforms passive learning.
- QBC is more computationally efficient in wall-clock terms.
- The NequIP energy results support QBC.
- The rMD17 ethanol result proves transfer to larger or reactive systems.
