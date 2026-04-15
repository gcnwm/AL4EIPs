"""Shared geometry utilities for molecular analysis.

Provides vectorised dihedral angle computation and ethanol-specific
coordinate extraction used across analysis scripts.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from ase import Atoms


KCAL_TO_EV = 0.0433641153
"""Conversion factor from kcal/mol to eV."""


def compute_dihedral(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> np.ndarray:
    """Compute dihedral angle (radians) for arrays of 4 points.

    Parameters
    ----------
    p0, p1, p2, p3 : np.ndarray
        Each is shape (N, 3) or (1, 3).

    Returns
    -------
    np.ndarray
        Shape (N,), dihedral angles in [-π, π].
    """
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1 = n1 / np.maximum(np.linalg.norm(n1, axis=-1, keepdims=True), 1e-10)
    n2 = n2 / np.maximum(np.linalg.norm(n2, axis=-1, keepdims=True), 1e-10)
    b2_unit = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-10)
    m1 = np.cross(n1, b2_unit)
    x = np.sum(n1 * n2, axis=-1)
    y = np.sum(m1 * n2, axis=-1)
    return np.arctan2(y, x)


def get_hocc_dihedral(
    coords: np.ndarray,
    indices: np.ndarray,
    c_indices: list[int],
    o_index: int,
    h_indices: list[int],
) -> np.ndarray:
    """Compute H-O-C-C dihedrals for selected conformations.

    The hydroxyl H is identified for each structure as the hydrogen nearest to O.
    Carbon ordering follows ``c_indices[0], c_indices[1]``.
    """
    dihedrals = np.zeros(len(indices))
    for i, idx in enumerate(indices):
        pos = coords[idx]
        o_pos = pos[o_index]
        h_dists = np.linalg.norm(pos[h_indices] - o_pos, axis=-1)
        hydroxyl_h = h_indices[int(np.argmin(h_dists))]
        dihedrals[i] = compute_dihedral(
            pos[hydroxyl_h : hydroxyl_h + 1],
            pos[o_index : o_index + 1],
            pos[c_indices[0] : c_indices[0] + 1],
            pos[c_indices[1] : c_indices[1] + 1],
        )[0]
    return dihedrals


def load_al_selected_indices(results_dir: Path, method: str, seed: int) -> np.ndarray:
    """Load labeled indices from an AL checkpoint."""
    ckpt_path = results_dir / f"{method}_seed{seed}" / "al_checkpoint.pkl"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    return ckpt["pool_state"]["labeled_indices"]


def npz_to_ase_atoms(
    coords: np.ndarray,
    nuclear_charges: np.ndarray,
    energies: np.ndarray | None = None,
    forces: np.ndarray | None = None,
    indices: np.ndarray | list[int] | None = None,
) -> list[Atoms]:
    """Convert NPZ data to a list of ASE Atoms objects."""
    if indices is None:
        selected_indices = np.arange(len(coords), dtype=int)
    else:
        selected_indices = np.asarray(indices, dtype=int)

    structures: list[Atoms] = []
    for idx in selected_indices:
        atoms = Atoms(numbers=nuclear_charges, positions=coords[idx])
        if energies is not None:
            atoms.info["energy"] = float(energies[idx] * KCAL_TO_EV)
        if forces is not None:
            atoms.arrays["forces"] = forces[idx] * KCAL_TO_EV
        structures.append(atoms)
    return structures


__all__ = [
    "KCAL_TO_EV",
    "compute_dihedral",
    "get_hocc_dihedral",
    "load_al_selected_indices",
    "npz_to_ase_atoms",
]
