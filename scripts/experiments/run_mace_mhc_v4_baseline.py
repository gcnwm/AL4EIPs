#!/usr/bin/env python
# pyright: reportMissingImports=false
"""MACE-MHC 550-label runs on frozen v4 passive datasets.

Uses the same train/validation/test files as the passive v4 550 baseline when
available. This does not run active learning; it trains a multi-head committee
model on each 550-label passive dataset for direct method comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from ase.io import write as ase_write

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

_src_dir = str(PROJECT_ROOT / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from fyp_al.geometry import KCAL_TO_EV, npz_to_ase_atoms  # noqa: E402
from fyp_al.mace_backend import MACEMHCBackend  # noqa: E402

NPZ_PATH = PROJECT_ROOT / "data" / "rmd17_ethanol.npz"
V4_ROOT = PROJECT_ROOT / "results" / "al_v4" / "v4_primary_fixed_member"
PASSIVE_ROOT = PROJECT_ROOT / "results" / "passive_v4_550"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "mace_mhc_v4_550"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MHCMetric:
    architecture: str
    method: str
    seed: int
    n_heads: int
    n_train: int
    n_valid: int
    n_test: int
    evaluation_policy: str
    energy_mae: float
    forces_mae: float
    energy_rmse: float
    forces_rmse: float
    train_eval_time_s: float
    checkpoint_path: str


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_npz() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(NPZ_PATH)
    return data["nuclear_charges"], data["coords"], data["energies"], data["forces"]


def _load_v4_indices() -> dict[str, np.ndarray]:
    data = np.load(V4_ROOT / "split_indices.npz")
    return {key: np.asarray(data[key], dtype=np.intp) for key in data.files}


def _write_xyz(path: Path, atoms: list[Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(path), atoms)
    return path


def _ensure_train_val_files(seed: int, n_train: int) -> tuple[Path, Path]:
    passive_dir = PASSIVE_ROOT / f"mace_passive_seed{seed}"
    train_xyz = passive_dir / f"train_passive_{n_train}.xyz"
    val_xyz = passive_dir / "valid_v4_500.xyz"
    if train_xyz.exists() and val_xyz.exists():
        return train_xyz, val_xyz

    nuclear_charges, coords, energies, forces = _load_npz()
    split = _load_v4_indices()
    rng = np.random.default_rng(seed)
    train_idx = split["pool_indices"][
        rng.choice(len(split["pool_indices"]), size=n_train, replace=False)
    ]
    val_idx = split["val_indices"]
    train_atoms = npz_to_ase_atoms(coords, nuclear_charges, energies, forces, train_idx)
    val_atoms = npz_to_ase_atoms(coords, nuclear_charges, energies, forces, val_idx)
    return _write_xyz(train_xyz, train_atoms), _write_xyz(val_xyz, val_atoms)


def _test_data() -> tuple[list[Any], np.ndarray, list[np.ndarray]]:
    nuclear_charges, coords, energies, forces = _load_npz()
    split = _load_v4_indices()
    test_idx = split["test_indices"]
    atoms = npz_to_ase_atoms(coords, nuclear_charges, energies, forces, test_idx)
    test_e = np.asarray([energies[i] * KCAL_TO_EV for i in test_idx], dtype=float)
    test_f = [forces[i] * KCAL_TO_EV for i in test_idx]
    return atoms, test_e, test_f


def run_one(
    *,
    seed: int,
    n_train: int,
    n_heads: int,
    max_epochs: int,
    results_root: Path,
    evaluation_policy: str,
) -> MHCMetric:
    train_xyz, val_xyz = _ensure_train_val_files(seed, n_train)
    test_atoms, test_e, test_f = _test_data()
    output_dir = results_root / f"mace_mhc_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "run_metadata.json",
        {
            "architecture": "mace",
            "method": "mace_mhc_passive_dataset",
            "seed": seed,
            "n_train": n_train,
            "n_heads": n_heads,
            "max_epochs": max_epochs,
            "train_xyz": str(train_xyz),
            "val_xyz": str(val_xyz),
            "v4_root": str(V4_ROOT),
            "evaluation_policy": evaluation_policy,
        },
    )
    backend = MACEMHCBackend(project_root=PROJECT_ROOT)
    t0 = time.time()
    log.info(
        "Training MACE-MHC seed=%d n=%d heads=%d epochs=%d",
        seed,
        n_train,
        n_heads,
        max_epochs,
    )
    result = backend.train_committee(
        train_xyz=train_xyz,
        val_xyz=val_xyz,
        output_dir=output_dir / "model",
        seeds=list(range(n_heads)),
        max_epochs=max_epochs,
    )
    metrics = backend.evaluate_committee(
        result,
        test_atoms,
        test_e,
        test_f,
        policy=evaluation_policy,
    )
    row = MHCMetric(
        architecture="mace",
        method="mace_mhc_passive_dataset",
        seed=seed,
        n_heads=n_heads,
        n_train=n_train,
        n_valid=500,
        n_test=len(test_atoms),
        evaluation_policy=evaluation_policy,
        energy_mae=float(metrics["energy_mae"]),
        forces_mae=float(metrics["forces_mae"]),
        energy_rmse=float(metrics["energy_rmse"]),
        forces_rmse=float(metrics["forces_rmse"]),
        train_eval_time_s=time.time() - t0,
        checkpoint_path=str(result.best_checkpoint),
    )
    _write_json(output_dir / "metrics.json", asdict(row))
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--n-train", type=int, default=550)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument(
        "--evaluation-policy",
        choices=["fixed_member", "ensemble_mean"],
        default="fixed_member",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    args = parse_args()
    rows = [
        run_one(
            seed=seed,
            n_train=args.n_train,
            n_heads=args.n_heads,
            max_epochs=args.max_epochs,
            results_root=args.results_root,
            evaluation_policy=args.evaluation_policy,
        )
        for seed in args.seeds
    ]
    _write_json(
        args.results_root / "mace_mhc_summary.json", [asdict(row) for row in rows]
    )
    for row in rows:
        log.info(
            "MHC seed=%d E_MAE=%.6f eV F_MAE=%.6f eV/A time=%.1fs",
            row.seed,
            row.energy_mae,
            row.forces_mae,
            row.train_eval_time_s,
        )


if __name__ == "__main__":
    main()
