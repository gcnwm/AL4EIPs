# FYP Active Learning for Interatomic Potentials

This repository is the production-code companion to a larger research workspace.
It contains the reusable package, experiment runners, dataset fetch helper, and
smoke tests needed to reproduce the active-learning workflows for MACE and
NequIP on the rMD17 ethanol benchmark.

## Included contents

- `src/fyp_al/`: backend abstractions plus MACE and NequIP integrations
- `scripts/experiments/run_al_experiment.py`: backend-agnostic AL runner
- `scripts/experiments/smoke_test_mace.py`: minimal MACE smoke test
- `scripts/experiments/smoke_test_nequip.py`: minimal NequIP smoke test
- `scripts/setup/01_fetch_md17.py`: reproducible rMD17 ethanol fetch + extract helper
- `configs/`: NequIP config presets retained from the thesis workspace
- `tests/test_smoke.py`: import and geometry smoke coverage

## Quick start

```bash
pixi install
pixi run python scripts/setup/01_fetch_md17.py --force
pixi run smoke-mace
pixi run python scripts/experiments/run_al_experiment.py --backend mace_qbc --seed 1
```

For NequIP workflows:

```bash
pixi install -e nequip
pixi run -e nequip smoke-nequip
pixi run -e nequip python scripts/experiments/run_al_experiment.py --backend nequip_qbc --seed 1
```

## Notes

- This release intentionally excludes notebooks, reports, poster assets, cached
  outputs, and other non-production artifacts from the thesis workspace.
- Model checkpoints are published separately on Hugging Face: `companion Hugging Face model repository`.
- Licensing is intentionally left for a manual maintainer decision before wider
  redistribution.
