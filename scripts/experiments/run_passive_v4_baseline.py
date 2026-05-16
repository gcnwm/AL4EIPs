#!/usr/bin/env python
# pyright: reportMissingImports=false
"""Passive 550-label v4 baselines for MACE or NequIP.

This script intentionally does not run active learning. It trains a single
passive/random model on 550 labels sampled from the frozen v4 pool and evaluates
on the frozen v4 audit test set, so results can be compared directly with
``results/al_v4/v4_primary_fixed_member``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Literal

import numpy as np
from ase.io import write as ase_write

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

_src_dir = str(PROJECT_ROOT / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from fyp_al.geometry import KCAL_TO_EV, npz_to_ase_atoms  # noqa: E402
from fyp_al.model_backend import ModelBackend  # noqa: E402

Architecture = Literal["mace", "nequip"]

NPZ_PATH = PROJECT_ROOT / "data" / "rmd17_ethanol.npz"
V4_ROOT = PROJECT_ROOT / "results" / "al_v4" / "v4_primary_fixed_member"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "passive_v4_550"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PassiveMetrics:
    architecture: str
    method: str
    seed: int
    n_train: int
    n_valid: int
    n_test: int
    energy_mae: float
    forces_mae: float
    energy_rmse: float
    forces_rmse: float
    train_eval_time_s: float
    train_indices_sha256: str
    checkpoint_path: str


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _array_sha256(values: np.ndarray) -> str:
    import hashlib

    arr = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode())
    digest.update(str(arr.dtype).encode())
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def _load_npz() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(NPZ_PATH)
    return data["nuclear_charges"], data["coords"], data["energies"], data["forces"]


def _load_v4_indices() -> dict[str, np.ndarray]:
    path = V4_ROOT / "split_indices.npz"
    if not path.exists():
        msg = f"Missing frozen v4 split indices: {path}"
        raise FileNotFoundError(msg)
    data = np.load(path)
    return {key: np.asarray(data[key], dtype=np.intp) for key in data.files}


def _sample_passive_train(
    pool_indices: np.ndarray, *, seed: int, n_train: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(pool_indices), size=n_train, replace=False)
    return np.asarray(pool_indices[positions], dtype=np.intp)


def _write_xyz(
    path: Path,
    indices: np.ndarray,
    nuclear_charges: np.ndarray,
    coords: np.ndarray,
    energies: np.ndarray,
    forces: np.ndarray,
) -> Path:
    atoms = npz_to_ase_atoms(coords, nuclear_charges, energies, forces, indices)
    path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(path), atoms)
    return path


def _test_data(
    indices: np.ndarray,
    nuclear_charges: np.ndarray,
    coords: np.ndarray,
    energies: np.ndarray,
    forces: np.ndarray,
) -> tuple[list[Any], np.ndarray, list[np.ndarray]]:
    atoms = npz_to_ase_atoms(coords, nuclear_charges, energies, forces, indices)
    test_e = np.asarray([energies[i] * KCAL_TO_EV for i in indices], dtype=float)
    test_f = [forces[i] * KCAL_TO_EV for i in indices]
    return atoms, test_e, test_f


def _make_backend(architecture: Architecture, run_dir: Path) -> ModelBackend:
    if architecture == "mace":
        from fyp_al.mace_backend import MACEQBCBackend

        return MACEQBCBackend(project_root=PROJECT_ROOT)
    if architecture == "nequip":
        from fyp_al.nequip_backend import NequIPBackend

        return NequIPBackend(project_root=PROJECT_ROOT, config_dir=run_dir / "configs")
    msg = f"Unknown architecture: {architecture}"
    raise ValueError(msg)


def run_one(
    *,
    architecture: Architecture,
    seed: int,
    n_train: int,
    max_epochs: int,
    results_root: Path,
) -> PassiveMetrics:
    nuclear_charges, coords, energies, forces = _load_npz()
    split = _load_v4_indices()
    train_indices = _sample_passive_train(
        split["pool_indices"], seed=seed, n_train=n_train
    )
    val_indices = split["val_indices"]
    test_indices = split["test_indices"]

    run_dir = results_root / f"{architecture}_passive_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_xyz = _write_xyz(
        run_dir / f"train_passive_{n_train}.xyz",
        train_indices,
        nuclear_charges,
        coords,
        energies,
        forces,
    )
    val_xyz = _write_xyz(
        run_dir / f"valid_v4_{len(val_indices)}.xyz",
        val_indices,
        nuclear_charges,
        coords,
        energies,
        forces,
    )
    test_atoms, test_e, test_f = _test_data(
        test_indices, nuclear_charges, coords, energies, forces
    )

    _write_json(
        run_dir / "run_metadata.json",
        {
            "architecture": architecture,
            "method": "passive_single_model",
            "seed": seed,
            "n_train": n_train,
            "n_valid": int(len(val_indices)),
            "n_test": int(len(test_indices)),
            "split_root": str(V4_ROOT),
            "pool_indices_sha256": _array_sha256(split["pool_indices"]),
            "train_indices_sha256": _array_sha256(train_indices),
            "max_epochs": max_epochs,
        },
    )

    backend = _make_backend(architecture, run_dir)
    log.info(
        "Training %s passive baseline seed=%d n=%d epochs=%d",
        architecture,
        seed,
        n_train,
        max_epochs,
    )
    t0 = time.time()
    result = backend.train_single(
        train_xyz=train_xyz,
        val_xyz=val_xyz,
        output_dir=run_dir / "model",
        seed=seed,
        max_epochs=max_epochs,
    )
    metrics = backend.evaluate(result.checkpoint_path, test_atoms, test_e, test_f)
    elapsed = time.time() - t0
    row = PassiveMetrics(
        architecture=architecture,
        method="passive_single_model",
        seed=seed,
        n_train=n_train,
        n_valid=int(len(val_indices)),
        n_test=int(len(test_indices)),
        energy_mae=float(metrics["energy_mae"]),
        forces_mae=float(metrics["forces_mae"]),
        energy_rmse=float(metrics["energy_rmse"]),
        forces_rmse=float(metrics["forces_rmse"]),
        train_eval_time_s=elapsed,
        train_indices_sha256=_array_sha256(train_indices),
        checkpoint_path=str(result.checkpoint_path),
    )
    _write_json(run_dir / "metrics.json", asdict(row))
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=["mace", "nequip"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--n-train", type=int, default=550)
    parser.add_argument("--max-epochs", type=int, default=200)
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
            architecture=args.architecture,
            seed=seed,
            n_train=args.n_train,
            max_epochs=args.max_epochs,
            results_root=args.results_root,
        )
        for seed in args.seeds
    ]
    summary_path = args.results_root / f"{args.architecture}_passive_summary.json"
    _write_json(summary_path, [asdict(row) for row in rows])
    log.info("Wrote %s", summary_path)
    for row in rows:
        log.info(
            "%s seed=%d E_MAE=%.6f eV F_MAE=%.6f eV/A time=%.1fs",
            row.architecture,
            row.seed,
            row.energy_mae,
            row.forces_mae,
            row.train_eval_time_s,
        )


if __name__ == "__main__":
    main()
