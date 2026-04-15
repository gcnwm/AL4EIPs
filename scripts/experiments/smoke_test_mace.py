#!/usr/bin/env python
# pyright: reportMissingImports=false
"""Local smoke test for MACE backend on a small ethanol subset.

Trains a tiny MACE model (20 epochs, 100 structures) on the local GPU,
verifies the .model file is produced, and checks force MAE < 500 meV/Å.
Run on RTX 4060 Ti — should complete in < 5 minutes.

Usage::

    pixi run python scripts/experiments/smoke_test_mace.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

_src_dir = str(PROJECT_ROOT / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from fyp_al.geometry import KCAL_TO_EV  # noqa: E402
from fyp_al.mace_backend import MACEMHCBackend, MACEQBCBackend  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

NPZ_PATH = PROJECT_ROOT / "data" / "rmd17_ethanol.npz"
SMOKE_DIR = PROJECT_ROOT / "results" / "smoke_test_mace"
MAX_EPOCHS = 20
N_TRAIN = 100
N_VAL = 50
N_TEST = 50
FORCE_MAE_THRESHOLD = 0.500  # eV/Å


def build_smoke_xyz() -> tuple[Path, Path, list, np.ndarray, list[np.ndarray]]:
    from ase import Atoms
    from ase.io import write as ase_write

    data = np.load(NPZ_PATH)
    nuclear_charges = data["nuclear_charges"]
    coords = data["coords"]
    energies = data["energies"]
    forces = data["forces"]

    rng = np.random.default_rng(99)
    indices = rng.permutation(len(energies))[: N_TRAIN + N_VAL + N_TEST]
    train_idx = indices[:N_TRAIN]
    val_idx = indices[N_TRAIN : N_TRAIN + N_VAL]
    test_idx = indices[N_TRAIN + N_VAL :]

    def to_atoms(i: int) -> Atoms:
        a = Atoms(
            numbers=nuclear_charges,
            positions=coords[i],
            info={"energy": float(energies[i] * KCAL_TO_EV)},
        )
        a.arrays["forces"] = forces[i] * KCAL_TO_EV
        return a

    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    train_xyz = SMOKE_DIR / "smoke_train.xyz"
    val_xyz = SMOKE_DIR / "smoke_val.xyz"

    ase_write(str(train_xyz), [to_atoms(int(i)) for i in train_idx])
    ase_write(str(val_xyz), [to_atoms(int(i)) for i in val_idx])

    test_atoms = [to_atoms(int(i)) for i in test_idx]
    test_energies = np.array([energies[i] * KCAL_TO_EV for i in test_idx])
    test_forces = [forces[i] * KCAL_TO_EV for i in test_idx]

    return train_xyz, val_xyz, test_atoms, test_energies, test_forces


def test_qbc_backend(
    train_xyz: Path,
    val_xyz: Path,
    test_atoms: list,
    test_energies: np.ndarray,
    test_forces: list[np.ndarray],
) -> None:
    log.info("=" * 60)
    log.info("SMOKE TEST: MACEQBCBackend (2 independent models)")
    log.info("=" * 60)

    backend = MACEQBCBackend(project_root=PROJECT_ROOT)
    qbc_dir = SMOKE_DIR / "qbc"

    t0 = time.time()
    result = backend.train_committee(
        train_xyz=train_xyz,
        val_xyz=val_xyz,
        output_dir=qbc_dir,
        seeds=[0, 1],
        max_epochs=MAX_EPOCHS,
    )
    elapsed = time.time() - t0

    for seed, ckpt in result.checkpoints.items():
        assert ckpt.exists(), f"Missing checkpoint for seed={seed}: {ckpt}"
        log.info(
            "  ✓ seed=%d model: %s (%.1f MB)", seed, ckpt, ckpt.stat().st_size / 1e6
        )

    metrics = backend.evaluate(
        result.best_checkpoint, test_atoms, test_energies, test_forces
    )
    log.info("  QBC evaluation: %s", metrics)
    assert metrics["forces_mae"] < FORCE_MAE_THRESHOLD, (
        f"Force MAE {metrics['forces_mae']:.4f} exceeds {FORCE_MAE_THRESHOLD} eV/Å"
    )
    log.info(
        "  ✓ Force MAE %.4f < %.3f eV/Å threshold",
        metrics["forces_mae"],
        FORCE_MAE_THRESHOLD,
    )

    disagreements = backend.compute_committee_disagreement(result, test_atoms)
    assert len(disagreements) == len(test_atoms)
    assert np.all(disagreements >= 0)
    log.info(
        "  ✓ Disagreement computed — mean=%.6f max=%.6f",
        disagreements.mean(),
        disagreements.max(),
    )
    log.info("  QBC smoke test PASSED in %.0fs", elapsed)


def test_mhc_backend(
    train_xyz: Path,
    val_xyz: Path,
    test_atoms: list,
    test_energies: np.ndarray,
    test_forces: list[np.ndarray],
) -> None:
    log.info("=" * 60)
    log.info("SMOKE TEST: MACEMHCBackend (1 model, 2 heads)")
    log.info("=" * 60)

    backend = MACEMHCBackend(project_root=PROJECT_ROOT)
    mhc_dir = SMOKE_DIR / "mhc"

    t0 = time.time()
    result = backend.train_committee(
        train_xyz=train_xyz,
        val_xyz=val_xyz,
        output_dir=mhc_dir,
        seeds=[0, 1],
        max_epochs=MAX_EPOCHS,
    )
    elapsed = time.time() - t0

    assert result.best_checkpoint.exists(), f"Missing model: {result.best_checkpoint}"
    log.info(
        "  ✓ MHC model: %s (%.1f MB)",
        result.best_checkpoint,
        result.best_checkpoint.stat().st_size / 1e6,
    )
    assert result.extra.get("n_heads") == 2

    metrics = backend.evaluate(
        result.best_checkpoint, test_atoms, test_energies, test_forces
    )
    log.info("  MHC evaluation: %s", metrics)
    assert metrics["forces_mae"] < FORCE_MAE_THRESHOLD, (
        f"Force MAE {metrics['forces_mae']:.4f} exceeds {FORCE_MAE_THRESHOLD} eV/Å"
    )
    log.info(
        "  ✓ Force MAE %.4f < %.3f eV/Å threshold",
        metrics["forces_mae"],
        FORCE_MAE_THRESHOLD,
    )

    disagreements = backend.compute_committee_disagreement(result, test_atoms)
    assert len(disagreements) == len(test_atoms)
    assert np.all(disagreements >= 0)
    log.info(
        "  ✓ Disagreement computed — mean=%.6f max=%.6f",
        disagreements.mean(),
        disagreements.max(),
    )
    log.info("  MHC smoke test PASSED in %.0fs", elapsed)


def main() -> None:
    log.info("Building smoke test XYZ files from %s …", NPZ_PATH)
    train_xyz, val_xyz, test_atoms, test_energies, test_forces = build_smoke_xyz()
    log.info(
        "  train=%s  val=%s  test=%d structures", train_xyz, val_xyz, len(test_atoms)
    )

    test_qbc_backend(train_xyz, val_xyz, test_atoms, test_energies, test_forces)
    test_mhc_backend(train_xyz, val_xyz, test_atoms, test_energies, test_forces)

    log.info("=" * 60)
    log.info("ALL SMOKE TESTS PASSED")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
