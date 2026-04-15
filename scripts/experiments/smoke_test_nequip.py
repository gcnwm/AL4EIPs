#!/usr/bin/env python
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
"""Local smoke test for NequIP backend — QBC and Random strategies.

Trains tiny NequIP models (5 epochs, 20 structures) on the local GPU,
verifies checkpoints are produced, runs evaluation, and checks force
disagreement computation.

Usage::

    PYTHONPATH=src:$PYTHONPATH pixi run -e nequip python scripts/experiments/smoke_test_nequip.py
"""

from __future__ import annotations

import logging
import shutil
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
from fyp_al.model_backend import TrainingResult  # noqa: E402
from fyp_al.nequip_backend import NequIPBackend  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

NPZ_PATH = PROJECT_ROOT / "data" / "rmd17_ethanol.npz"
SMOKE_DIR = PROJECT_ROOT / "results" / "smoke_test_nequip"
MAX_EPOCHS = 5
N_TRAIN = 20
N_VAL = 10
N_TEST = 10
FORCE_MAE_THRESHOLD = 1.0  # eV/Å — loose threshold for 5-epoch smoke test


def build_smoke_xyz() -> tuple[Path, Path, list, np.ndarray, list[np.ndarray]]:
    """Create tiny train/val/test splits from the rMD17 ethanol dataset."""
    from ase import Atoms
    from ase.io import write as ase_write

    data = np.load(NPZ_PATH)
    nuclear_charges = data["nuclear_charges"]
    coords = data["coords"]
    energies = data["energies"]
    forces = data["forces"]

    rng = np.random.default_rng(42)
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


def test_qbc(
    backend: NequIPBackend,
    train_xyz: Path,
    val_xyz: Path,
    test_atoms: list,
    test_energies: np.ndarray,
    test_forces: list[np.ndarray],
) -> None:
    """QBC smoke test: train 2-member committee, evaluate, compute disagreement."""
    log.info("=" * 60)
    log.info("SMOKE TEST: NequIP QBC (2 independent models)")
    log.info("=" * 60)

    qbc_dir = SMOKE_DIR / "qbc"
    if qbc_dir.exists():
        shutil.rmtree(qbc_dir)

    t0 = time.time()
    result = backend.train_committee(
        train_xyz=train_xyz,
        val_xyz=val_xyz,
        output_dir=qbc_dir,
        seeds=[0, 1],
        max_epochs=MAX_EPOCHS,
    )
    elapsed = time.time() - t0

    # --- verify checkpoints ---
    for seed, ckpt in result.checkpoints.items():
        assert ckpt.exists(), f"Missing checkpoint for seed={seed}: {ckpt}"
        log.info(
            "  ✓ seed=%d checkpoint: %s (%.1f MB)",
            seed,
            ckpt,
            ckpt.stat().st_size / 1e6,
        )

    # --- evaluate best model ---
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

    # --- compute committee disagreement ---
    disagreements = backend.compute_committee_disagreement(
        result, test_atoms, max_eval=N_TEST
    )
    assert len(disagreements) == len(test_atoms)
    assert np.all(disagreements >= 0)
    log.info(
        "  ✓ Disagreement computed — mean=%.6f max=%.6f eV/Å",
        disagreements.mean(),
        disagreements.max(),
    )
    log.info("  NequIP QBC smoke test PASSED in %.0fs", elapsed)


def test_random(
    backend: NequIPBackend,
    train_xyz: Path,
    val_xyz: Path,
    test_atoms: list,
    test_energies: np.ndarray,
    test_forces: list[np.ndarray],
) -> TrainingResult:
    """Random baseline smoke test: train single model, evaluate."""
    log.info("=" * 60)
    log.info("SMOKE TEST: NequIP Random (single model)")
    log.info("=" * 60)

    random_dir = SMOKE_DIR / "random"
    if random_dir.exists():
        shutil.rmtree(random_dir)

    t0 = time.time()
    result = backend.train_single(
        train_xyz=train_xyz,
        val_xyz=val_xyz,
        output_dir=random_dir,
        seed=42,
        max_epochs=MAX_EPOCHS,
    )
    elapsed = time.time() - t0

    # --- verify checkpoint ---
    assert result.checkpoint_path.exists(), (
        f"Missing checkpoint: {result.checkpoint_path}"
    )
    log.info(
        "  ✓ checkpoint: %s (%.1f MB)",
        result.checkpoint_path,
        result.checkpoint_path.stat().st_size / 1e6,
    )
    log.info(
        "  ✓ training_time=%.1fs  best_val_metric=%.6f",
        result.training_time,
        result.best_val_metric,
    )

    # --- evaluate ---
    metrics = backend.evaluate(
        result.checkpoint_path, test_atoms, test_energies, test_forces
    )
    log.info("  Random evaluation: %s", metrics)
    assert metrics["forces_mae"] < FORCE_MAE_THRESHOLD, (
        f"Force MAE {metrics['forces_mae']:.4f} exceeds {FORCE_MAE_THRESHOLD} eV/Å"
    )
    log.info(
        "  ✓ Force MAE %.4f < %.3f eV/Å threshold",
        metrics["forces_mae"],
        FORCE_MAE_THRESHOLD,
    )
    log.info("  NequIP Random smoke test PASSED in %.0fs", elapsed)
    return result


def test_warm_start(
    backend: NequIPBackend,
    train_xyz: Path,
    val_xyz: Path,
    pretrained_ckpt: Path,
) -> None:
    """Warm-start smoke test: resume training from a pretrained checkpoint."""
    log.info("=" * 60)
    log.info("SMOKE TEST: NequIP Warm Start (from pretrained checkpoint)")
    log.info("=" * 60)

    warm_dir = SMOKE_DIR / "warm_start"
    if warm_dir.exists():
        shutil.rmtree(warm_dir)

    t0 = time.time()
    result = backend.train_single(
        train_xyz=train_xyz,
        val_xyz=val_xyz,
        output_dir=warm_dir,
        seed=99,
        max_epochs=MAX_EPOCHS,
        pretrained_ckpt=pretrained_ckpt,
    )
    elapsed = time.time() - t0

    assert result.checkpoint_path.exists(), (
        f"Missing checkpoint: {result.checkpoint_path}"
    )
    log.info(
        "  ✓ warm-start checkpoint: %s (%.1f MB)",
        result.checkpoint_path,
        result.checkpoint_path.stat().st_size / 1e6,
    )
    log.info("  NequIP Warm Start smoke test PASSED in %.0fs", elapsed)


def main() -> None:
    log.info("Building smoke test XYZ files from %s …", NPZ_PATH)
    train_xyz, val_xyz, test_atoms, test_energies, test_forces = build_smoke_xyz()
    log.info("  train=%d  val=%d  test=%d structures", N_TRAIN, N_VAL, N_TEST)

    config_dir = SMOKE_DIR / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    backend = NequIPBackend(
        project_root=PROJECT_ROOT,
        config_dir=config_dir,
    )

    random_result = test_random(
        backend, train_xyz, val_xyz, test_atoms, test_energies, test_forces
    )
    test_qbc(backend, train_xyz, val_xyz, test_atoms, test_energies, test_forces)
    test_warm_start(backend, train_xyz, val_xyz, random_result.checkpoint_path)

    log.info("=" * 60)
    log.info("ALL NEQUIP SMOKE TESTS PASSED")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
