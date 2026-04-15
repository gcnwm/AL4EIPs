#!/usr/bin/env python
# pyright: reportMissingImports=false
"""Backend-agnostic active learning experiment runner.

Implements the shared AL protocol from the hybrid comparison plan (v2.2):
fixed shuffle → test/val/pool split → paired initial sets → iterative
train-evaluate-acquire loop.  Works with any ``ModelBackend`` subclass.

Usage::

    # Single arm
    python run_al_experiment.py --backend mace_mhc --seed 1

    # All arms × all seeds
    python run_al_experiment.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from ase.io import write as ase_write

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

_src_dir = str(PROJECT_ROOT / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
existing = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{_src_dir}:{existing}" if existing else _src_dir

from fyp_al.geometry import KCAL_TO_EV, npz_to_ase_atoms  # noqa: E402
from fyp_al.model_backend import ModelBackend  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AL protocol constants (plan §3.3)
# ---------------------------------------------------------------------------

SHUFFLE_SEED = 42
N_TEST = 1000
N_VAL = 500
N_INITIAL = 100
N_QUERY = 50
N_ITERATIONS = 10
AL_SEEDS = [1, 2, 3]

DATA_DIR = PROJECT_ROOT / "data"
NPZ_PATH = DATA_DIR / "rmd17_ethanol.npz"
RESULTS_DIR = PROJECT_ROOT / "results" / "al"
AL_DATA_DIR = DATA_DIR / "al"


# ---------------------------------------------------------------------------
# Data loading and splitting (plan §3.3)
# ---------------------------------------------------------------------------


def load_npz() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load rMD17 ethanol.  Returns (nuclear_charges, coords, energies, forces)."""
    data = np.load(NPZ_PATH)
    return (
        data["nuclear_charges"],
        data["coords"],
        data["energies"],
        data["forces"],
    )


def build_splits() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed-shuffle split per plan §3.3.

    Returns (test_indices, val_indices, pool_indices) into the original NPZ.
    """
    n_total = int(np.load(NPZ_PATH)["energies"].shape[0])
    rng = np.random.default_rng(seed=SHUFFLE_SEED)
    shuffled = rng.permutation(n_total)

    test_idx = shuffled[:N_TEST]
    val_idx = shuffled[N_TEST : N_TEST + N_VAL]
    pool_idx = shuffled[N_TEST + N_VAL :]
    return test_idx, val_idx, pool_idx


def save_shuffle_indices() -> Path:
    """Persist the shuffled index order for reproducibility."""
    out = AL_DATA_DIR / "shuffle_indices.npy"
    if out.exists():
        return out
    n_total = int(np.load(NPZ_PATH)["energies"].shape[0])
    rng = np.random.default_rng(seed=SHUFFLE_SEED)
    shuffled = rng.permutation(n_total)
    AL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(out, shuffled)
    log.info("Saved shuffle indices to %s", out)
    return out


def write_xyz_subset(
    indices: np.ndarray,
    nuclear_charges: np.ndarray,
    coords: np.ndarray,
    energies: np.ndarray,
    forces: np.ndarray,
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    structures = npz_to_ase_atoms(coords, nuclear_charges, energies, forces, indices)
    ase_write(out_path, structures)
    return out_path


# ---------------------------------------------------------------------------
# Pool manager (adapted from existing, now backend-agnostic)
# ---------------------------------------------------------------------------


class ALPoolManager:
    """Tracks labeled / unlabeled splits and simulates oracle queries."""

    def __init__(self, pool_indices: np.ndarray, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.all_pool = pool_indices.copy()
        self.labeled: np.ndarray = np.array([], dtype=np.intp)
        self.unlabeled: np.ndarray = pool_indices.copy()

    def initialize(self, n_initial: int) -> None:
        chosen = self.rng.choice(len(self.unlabeled), size=n_initial, replace=False)
        self.labeled = self.unlabeled[chosen]
        self.unlabeled = np.setdiff1d(self.unlabeled, self.labeled)

    def query(self, indices: np.ndarray) -> None:
        assert np.all(np.isin(indices, self.unlabeled))
        self.labeled = np.concatenate([self.labeled, indices])
        self.unlabeled = np.setdiff1d(self.unlabeled, indices)

    def select_top_k(self, scores: np.ndarray, k: int) -> np.ndarray:
        """Return pool indices for the *k* highest-scoring structures."""
        top_k = np.argsort(scores)[-k:]
        return self.unlabeled[top_k]

    def state_dict(self) -> dict:
        return {"labeled": self.labeled, "unlabeled": self.unlabeled}

    def load_state_dict(self, state: dict) -> None:
        self.labeled = state["labeled"]
        self.unlabeled = state["unlabeled"]


# ---------------------------------------------------------------------------
# Iteration metrics
# ---------------------------------------------------------------------------


@dataclass
class ALIterationMetrics:
    iteration: int
    n_labeled: int
    energy_mae: float
    forces_mae: float
    energy_rmse: float
    forces_rmse: float
    mean_disagreement: float
    max_disagreement: float
    training_time: float


# ---------------------------------------------------------------------------
# Core AL loop
# ---------------------------------------------------------------------------


def run_al_experiment(
    backend: ModelBackend,
    arm_name: str,
    seed: int,
    nuclear_charges: np.ndarray,
    coords: np.ndarray,
    energies: np.ndarray,
    forces: np.ndarray,
    test_indices: np.ndarray,
    val_indices: np.ndarray,
    pool_indices: np.ndarray,
    n_initial: int = N_INITIAL,
    n_query: int = N_QUERY,
    n_iterations: int = N_ITERATIONS,
    max_epochs: int = 200,
    committee_seeds: list[int] | None = None,
) -> list[ALIterationMetrics]:
    """Run one complete AL experiment for a given backend/arm/seed."""
    if committee_seeds is None:
        committee_seeds = [0, 1, 2, 3]

    run_dir = RESULTS_DIR / f"{arm_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "al_checkpoint.pkl"

    # Prepare fixed test/val XYZ
    val_xyz = AL_DATA_DIR / "val_set_v2.xyz"
    test_xyz = AL_DATA_DIR / "test_set_v2.xyz"
    if not val_xyz.exists():
        write_xyz_subset(
            val_indices, nuclear_charges, coords, energies, forces, val_xyz
        )
    if not test_xyz.exists():
        write_xyz_subset(
            test_indices, nuclear_charges, coords, energies, forces, test_xyz
        )

    # Pre-build test atoms/energies/forces for evaluation
    test_atoms = npz_to_ase_atoms(
        coords, nuclear_charges, energies, forces, test_indices
    )
    test_e = np.array([energies[i] * KCAL_TO_EV for i in test_indices])
    test_f = [forces[i] * KCAL_TO_EV for i in test_indices]

    pool = ALPoolManager(pool_indices, seed=seed)
    metrics_history: list[ALIterationMetrics] = []
    warm_ckpts: dict[int, Path] | None = None
    start_iter = 0

    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        pool.load_state_dict(ckpt["pool_state"])
        metrics_history = [ALIterationMetrics(**m) for m in ckpt["metrics"]]
        start_iter = ckpt["iteration"] + 1
        warm_ckpts = ckpt.get("warm_ckpts")
        log.info("Resuming %s seed=%d from iteration %d", arm_name, seed, start_iter)
    else:
        pool.initialize(n_initial)

    log.info(
        "Starting %s seed=%d — %d labeled, %d pool, %d iterations",
        arm_name,
        seed,
        len(pool.labeled),
        len(pool.unlabeled),
        n_iterations - start_iter,
    )

    for iteration in range(start_iter, n_iterations):
        log.info(
            "=== %s seed=%d iter %d/%d — %d labeled ===",
            arm_name,
            seed,
            iteration + 1,
            n_iterations,
            len(pool.labeled),
        )
        t0 = time.time()

        # Write current training set
        train_xyz = run_dir / f"iter{iteration:02d}_train.xyz"
        write_xyz_subset(
            pool.labeled, nuclear_charges, coords, energies, forces, train_xyz
        )

        # Train committee
        committee_dir = run_dir / f"iter{iteration:02d}_committee"
        committee_result = backend.train_committee(
            train_xyz=train_xyz,
            val_xyz=val_xyz,
            output_dir=committee_dir,
            seeds=committee_seeds,
            max_epochs=max_epochs,
            warm_start_ckpts=warm_ckpts,
        )

        # Evaluate best model on test set
        test_metrics = backend.evaluate(
            committee_result.best_checkpoint, test_atoms, test_e, test_f
        )

        # Acquire new samples (skip on last iteration)
        mean_dis = 0.0
        max_dis = 0.0
        if iteration < n_iterations - 1:
            pool_atoms = npz_to_ase_atoms(
                coords,
                nuclear_charges,
                energies,
                forces,
                pool.unlabeled,
            )
            disagreements = backend.compute_committee_disagreement(
                committee_result, pool_atoms
            )
            # disagreements may be shorter than pool.unlabeled if max_eval < len(pool)
            n_eval = len(disagreements)
            top_k = np.argsort(disagreements)[-n_query:]
            new_indices = pool.unlabeled[:n_eval][top_k]
            pool.query(new_indices)
            mean_dis = float(disagreements.mean())
            max_dis = float(disagreements.max())

        elapsed = time.time() - t0
        metrics = ALIterationMetrics(
            iteration=iteration,
            n_labeled=len(pool.labeled),
            energy_mae=test_metrics["energy_mae"],
            forces_mae=test_metrics["forces_mae"],
            energy_rmse=test_metrics["energy_rmse"],
            forces_rmse=test_metrics["forces_rmse"],
            mean_disagreement=mean_dis,
            max_disagreement=max_dis,
            training_time=elapsed,
        )
        metrics_history.append(metrics)

        log.info(
            "  forces_mae=%.4f eV/Å  energy_mae=%.6f eV  time=%.0fs",
            metrics.forces_mae,
            metrics.energy_mae,
            metrics.training_time,
        )

        # Warm-start from this iteration's committee
        warm_ckpts = dict(committee_result.checkpoints)

        # Checkpoint
        with open(checkpoint_path, "wb") as f:
            pickle.dump(
                {
                    "iteration": iteration,
                    "pool_state": pool.state_dict(),
                    "metrics": [asdict(m) for m in metrics_history],
                    "warm_ckpts": warm_ckpts,
                },
                f,
            )

    # Save final metrics JSON
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump([asdict(m) for m in metrics_history], f, indent=2)
    log.info("Saved metrics to %s", metrics_file)

    return metrics_history


# ---------------------------------------------------------------------------
# Random baseline (no committee — single model, random acquisition)
# ---------------------------------------------------------------------------


def run_random_baseline(
    backend: ModelBackend,
    seed: int,
    nuclear_charges: np.ndarray,
    coords: np.ndarray,
    energies: np.ndarray,
    forces: np.ndarray,
    test_indices: np.ndarray,
    val_indices: np.ndarray,
    pool_indices: np.ndarray,
    n_initial: int = N_INITIAL,
    n_query: int = N_QUERY,
    n_iterations: int = N_ITERATIONS,
    max_epochs: int = 200,
    model_seed: int = 0,
    arm_name: str = "random",
) -> list[ALIterationMetrics]:
    """Random acquisition baseline — same budget, no uncertainty guidance."""
    run_dir = RESULTS_DIR / f"{arm_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "al_checkpoint.pkl"

    val_xyz = AL_DATA_DIR / "val_set_v2.xyz"
    test_xyz = AL_DATA_DIR / "test_set_v2.xyz"
    if not val_xyz.exists():
        write_xyz_subset(
            val_indices, nuclear_charges, coords, energies, forces, val_xyz
        )
    if not test_xyz.exists():
        write_xyz_subset(
            test_indices, nuclear_charges, coords, energies, forces, test_xyz
        )

    test_atoms = npz_to_ase_atoms(
        coords, nuclear_charges, energies, forces, test_indices
    )
    test_e = np.array([energies[i] * KCAL_TO_EV for i in test_indices])
    test_f = [forces[i] * KCAL_TO_EV for i in test_indices]

    pool = ALPoolManager(pool_indices, seed=seed)
    metrics_history: list[ALIterationMetrics] = []
    warm_ckpt: Path | None = None
    start_iter = 0

    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        pool.load_state_dict(ckpt["pool_state"])
        metrics_history = [ALIterationMetrics(**m) for m in ckpt["metrics"]]
        start_iter = ckpt["iteration"] + 1
        warm_ckpt = ckpt.get("warm_ckpt")
        log.info("Resuming %s seed=%d from iteration %d", arm_name, seed, start_iter)
    else:
        pool.initialize(n_initial)

    for iteration in range(start_iter, n_iterations):
        log.info(
            "=== %s seed=%d iter %d/%d — %d labeled ===",
            arm_name,
            seed,
            iteration + 1,
            n_iterations,
            len(pool.labeled),
        )
        t0 = time.time()

        train_xyz = run_dir / f"iter{iteration:02d}_train.xyz"
        write_xyz_subset(
            pool.labeled, nuclear_charges, coords, energies, forces, train_xyz
        )

        model_dir = run_dir / f"iter{iteration:02d}_model"
        result = backend.train_single(
            train_xyz=train_xyz,
            val_xyz=val_xyz,
            output_dir=model_dir,
            seed=model_seed,
            max_epochs=max_epochs,
            pretrained_ckpt=warm_ckpt,
        )

        test_metrics = backend.evaluate(
            result.checkpoint_path, test_atoms, test_e, test_f
        )

        if iteration < n_iterations - 1:
            rng = np.random.default_rng(seed * 1000 + iteration)
            random_scores = rng.random(len(pool.unlabeled))
            new_indices = pool.select_top_k(random_scores, n_query)
            pool.query(new_indices)

        elapsed = time.time() - t0
        metrics = ALIterationMetrics(
            iteration=iteration,
            n_labeled=len(pool.labeled),
            energy_mae=test_metrics["energy_mae"],
            forces_mae=test_metrics["forces_mae"],
            energy_rmse=test_metrics["energy_rmse"],
            forces_rmse=test_metrics["forces_rmse"],
            mean_disagreement=0.0,
            max_disagreement=0.0,
            training_time=elapsed,
        )
        metrics_history.append(metrics)
        log.info("  forces_mae=%.4f eV/Å  time=%.0fs", metrics.forces_mae, elapsed)

        warm_ckpt = result.checkpoint_path
        with open(checkpoint_path, "wb") as f:
            pickle.dump(
                {
                    "iteration": iteration,
                    "pool_state": pool.state_dict(),
                    "metrics": [asdict(m) for m in metrics_history],
                    "warm_ckpt": warm_ckpt,
                },
                f,
            )

    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump([asdict(m) for m in metrics_history], f, indent=2)
    log.info("Saved metrics to %s", metrics_file)
    return metrics_history


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def make_backend(name: str) -> ModelBackend:
    """Instantiate a backend by short name."""
    if name == "nequip_qbc":
        from fyp_al.nequip_backend import NequIPBackend

        return NequIPBackend(project_root=PROJECT_ROOT)

    if name == "mace_qbc":
        from fyp_al.mace_backend import MACEQBCBackend

        return MACEQBCBackend(project_root=PROJECT_ROOT)

    if name == "mace_mhc":
        from fyp_al.mace_backend import MACEMHCBackend

        return MACEMHCBackend(project_root=PROJECT_ROOT)

    msg = f"Unknown backend: {name}"
    raise ValueError(msg)


COMMITTEE_SEEDS_MAP: dict[str, list[int]] = {
    "nequip_qbc": [0, 1, 2],
    "mace_qbc": [0, 1, 2, 3],
    "mace_mhc": [0, 1, 2, 3],
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backend-agnostic AL experiment runner"
    )
    parser.add_argument(
        "--backend",
        choices=["nequip_qbc", "mace_qbc", "mace_mhc"],
        help="Backend / arm to run",
    )
    parser.add_argument("--seed", type=int, choices=AL_SEEDS, help="AL seed")
    parser.add_argument(
        "--random", action="store_true", help="Run random baseline instead"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all arms × all seeds (including random)",
    )
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument(
        "--random-arm-name",
        default="random",
        help="Directory prefix for random baseline (e.g. nequip_random)",
    )
    args = parser.parse_args()

    nuclear_charges, coords, energies, forces_arr = load_npz()
    test_idx, val_idx, pool_idx = build_splits()
    save_shuffle_indices()

    if args.all:
        for backend_name in ["mace_mhc", "mace_qbc", "nequip_qbc"]:
            backend = make_backend(backend_name)
            for s in AL_SEEDS:
                run_al_experiment(
                    backend=backend,
                    arm_name=backend_name,
                    seed=s,
                    nuclear_charges=nuclear_charges,
                    coords=coords,
                    energies=energies,
                    forces=forces_arr,
                    test_indices=test_idx,
                    val_indices=val_idx,
                    pool_indices=pool_idx,
                    committee_seeds=COMMITTEE_SEEDS_MAP[backend_name],
                    max_epochs=args.max_epochs,
                )
        # Random baseline uses MACE single model
        mace_backend = make_backend("mace_qbc")
        for s in AL_SEEDS:
            run_random_baseline(
                backend=mace_backend,
                seed=s,
                nuclear_charges=nuclear_charges,
                coords=coords,
                energies=energies,
                forces=forces_arr,
                test_indices=test_idx,
                val_indices=val_idx,
                pool_indices=pool_idx,
                max_epochs=args.max_epochs,
            )
        return

    if args.random:
        backend_name = args.backend or "mace_qbc"
        backend = make_backend(backend_name)
        seeds = [args.seed] if args.seed else AL_SEEDS
        for s in seeds:
            run_random_baseline(
                backend=backend,
                seed=s,
                nuclear_charges=nuclear_charges,
                coords=coords,
                energies=energies,
                forces=forces_arr,
                test_indices=test_idx,
                val_indices=val_idx,
                pool_indices=pool_idx,
                max_epochs=args.max_epochs,
                arm_name=args.random_arm_name,
            )
        return

    if not args.backend:
        parser.error("--backend is required (or use --all)")

    backend = make_backend(args.backend)
    seeds = [args.seed] if args.seed else AL_SEEDS
    for s in seeds:
        run_al_experiment(
            backend=backend,
            arm_name=args.backend,
            seed=s,
            nuclear_charges=nuclear_charges,
            coords=coords,
            energies=energies,
            forces=forces_arr,
            test_indices=test_idx,
            val_indices=val_idx,
            pool_indices=pool_idx,
            committee_seeds=COMMITTEE_SEEDS_MAP[args.backend],
            max_epochs=args.max_epochs,
        )


if __name__ == "__main__":
    main()
