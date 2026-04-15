"""Smoke tests for the fyp_al package.

These tests verify that the package is importable and that core
utilities work correctly without requiring GPU, checkpoints, or
large data files.
"""

from __future__ import annotations

import numpy as np

from fyp_al.geometry import (
    KCAL_TO_EV,
    compute_dihedral,
    get_hocc_dihedral,
    npz_to_ase_atoms,
)


class TestKcalToEv:
    """Verify the unit conversion constant."""

    def test_value(self) -> None:
        assert abs(KCAL_TO_EV - 0.0433641153) < 1e-10

    def test_roundtrip(self) -> None:
        energy_kcal = 10.0
        energy_ev = energy_kcal * KCAL_TO_EV
        assert abs(energy_ev - 0.433641153) < 1e-8


class TestComputeDihedral:
    """Test dihedral angle computation."""

    def test_trans_dihedral(self) -> None:
        """Trans configuration should give ~π."""
        p0 = np.array([[1.0, 0.0, 0.0]])
        p1 = np.array([[0.0, 0.0, 0.0]])
        p2 = np.array([[0.0, 1.0, 0.0]])
        p3 = np.array([[-1.0, 1.0, 0.0]])
        angle = compute_dihedral(p0, p1, p2, p3)
        assert abs(abs(angle[0]) - np.pi) < 0.01

    def test_cis_dihedral(self) -> None:
        """Cis configuration should give ~0."""
        p0 = np.array([[1.0, 0.0, 0.0]])
        p1 = np.array([[0.0, 0.0, 0.0]])
        p2 = np.array([[0.0, 1.0, 0.0]])
        p3 = np.array([[1.0, 1.0, 0.0]])
        angle = compute_dihedral(p0, p1, p2, p3)
        assert abs(angle[0]) < 0.01

    def test_batch(self) -> None:
        """Multiple dihedrals at once."""
        n = 5
        p0 = np.random.default_rng(42).standard_normal((n, 3))
        p1 = np.random.default_rng(43).standard_normal((n, 3))
        p2 = np.random.default_rng(44).standard_normal((n, 3))
        p3 = np.random.default_rng(45).standard_normal((n, 3))
        angles = compute_dihedral(p0, p1, p2, p3)
        assert angles.shape == (n,)
        assert np.all(np.abs(angles) <= np.pi + 1e-6)

    def test_output_range(self) -> None:
        """Angles should be in [-π, π]."""
        rng = np.random.default_rng(99)
        for _ in range(10):
            pts = [rng.standard_normal((1, 3)) for _ in range(4)]
            angle = compute_dihedral(*pts)
            assert -np.pi - 1e-6 <= angle[0] <= np.pi + 1e-6


class TestNpzToAseAtoms:
    """Test NPZ data to ASE Atoms conversion."""

    def test_basic_conversion(self) -> None:
        """Convert minimal data to Atoms list."""
        coords = np.random.default_rng(0).standard_normal((3, 9, 3))
        charges = np.array([1, 1, 1, 1, 1, 6, 6, 8, 1])
        atoms_list = npz_to_ase_atoms(coords, charges)
        assert len(atoms_list) == 3
        assert len(atoms_list[0]) == 9

    def test_with_energies_and_forces(self) -> None:
        """Energies and forces are stored in info/arrays."""
        coords = np.random.default_rng(1).standard_normal((2, 9, 3))
        charges = np.array([1, 1, 1, 1, 1, 6, 6, 8, 1])
        energies = np.array([100.0, 200.0])  # kcal/mol
        forces = np.random.default_rng(2).standard_normal((2, 9, 3))
        atoms_list = npz_to_ase_atoms(coords, charges, energies, forces)
        assert "energy" in atoms_list[0].info
        assert abs(atoms_list[0].info["energy"] - 100.0 * KCAL_TO_EV) < 1e-8
        assert "forces" in atoms_list[0].arrays
        assert atoms_list[0].arrays["forces"].shape == (9, 3)

    def test_with_indices(self) -> None:
        """Select a subset of structures."""
        coords = np.random.default_rng(3).standard_normal((10, 9, 3))
        charges = np.array([1, 1, 1, 1, 1, 6, 6, 8, 1])
        atoms_list = npz_to_ase_atoms(coords, charges, indices=[0, 5, 9])
        assert len(atoms_list) == 3


class TestGetHoccDihedral:
    """Test ethanol H-O-C-C dihedral extraction."""

    def test_shape(self) -> None:
        """Output shape matches number of indices."""
        rng = np.random.default_rng(7)
        # 5 conformations of 9-atom ethanol
        coords = rng.standard_normal((5, 9, 3))
        indices = np.array([0, 2, 4])
        # Ethanol atom ordering: H(0-4), C(5), C(6), O(7), H(8)
        # This is synthetic — just checking shape
        dihedrals = get_hocc_dihedral(
            coords, indices, c_indices=[5, 6], o_index=7, h_indices=[0, 1, 2, 3, 4, 8]
        )
        assert dihedrals.shape == (3,)
        assert np.all(np.abs(dihedrals) <= np.pi + 1e-6)


class TestImports:
    """Verify that public API is importable."""

    def test_model_backend_abc(self) -> None:
        from fyp_al.model_backend import ModelBackend

        assert hasattr(ModelBackend, "train_single")
        assert hasattr(ModelBackend, "train_committee")
        assert hasattr(ModelBackend, "compute_committee_disagreement")
        assert hasattr(ModelBackend, "evaluate")

    def test_training_result(self) -> None:
        from fyp_al.model_backend import TrainingResult

        from pathlib import Path

        tr = TrainingResult(
            checkpoint_path=Path("/tmp/test.ckpt"),
            training_time=1.0,
            best_val_metric=0.5,
        )
        assert tr.training_time == 1.0

    def test_geometry_exports(self) -> None:
        """All geometry.py exports are available from the package."""
        import fyp_al

        assert hasattr(fyp_al, "compute_dihedral")
        assert hasattr(fyp_al, "get_hocc_dihedral")
        assert hasattr(fyp_al, "load_al_selected_indices")
        assert hasattr(fyp_al, "npz_to_ase_atoms")
        assert hasattr(fyp_al, "KCAL_TO_EV")
