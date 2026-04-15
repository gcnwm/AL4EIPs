"""ASE Calculator wrapper for NequIP models.

Loads a NequIP checkpoint via ``ModelFromCheckpoint`` and exposes it as a
standard ASE ``Calculator`` so it can be used with ASE's MD drivers
(Langevin, VelocityVerlet, etc.).

Usage::

    from fyp_al.nequip_calc import NequIPCalculator

    calc = NequIPCalculator("results/al/qbc_seed123/iter09_committee/committee_0/best.ckpt")
    atoms.calc = calc
    print(atoms.get_potential_energy())
    print(atoms.get_forces())
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from nequip.data import AtomicDataDict, from_ase
from nequip.data.transforms import (
    ChemicalSpeciesToAtomTypeMapper,
    NeighborListTransform,
    NonPeriodicCellTransform,
)
from nequip.model import ModelFromCheckpoint
from nequip.utils.global_state import set_global_state

set_global_state()

from fyp_al.transforms import ZeroEdgeCellShiftTransform  # noqa: E402


class NequIPCalculator(Calculator):
    """ASE Calculator backed by a NequIP checkpoint.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to ``best.ckpt`` produced by ``nequip-train``.
    model_type_names : list[str]
        Atom type ordering used during training (default: ethanol ``["H", "C", "O"]``).
    r_max : float
        Cutoff radius in Å (must match training config).
    device : str
        ``"cuda"`` or ``"cpu"``.  Auto-detected if omitted.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_type_names: list[str] | None = None,
        r_max: float = 4.0,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if model_type_names is None:
            model_type_names = ["H", "C", "O"]

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = ModelFromCheckpoint(str(checkpoint_path))["sole_model"]
        self.model.eval().to(self.device)

        self.transforms = [
            NonPeriodicCellTransform(),
            ChemicalSpeciesToAtomTypeMapper(model_type_names=model_type_names),
            NeighborListTransform(r_max=r_max),
            ZeroEdgeCellShiftTransform(),
        ]

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """Compute energy and forces for *atoms*."""
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            msg = "No Atoms object provided"
            raise ValueError(msg)

        adata = from_ase(atoms)
        for t in self.transforms:
            adata = t(adata)

        batch = AtomicDataDict.batched_from_list([adata])
        AtomicDataDict.to_(batch, self.device)
        batch[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)

        with torch.no_grad():
            outputs = self.model(batch.copy())

        energy = outputs[AtomicDataDict.TOTAL_ENERGY_KEY].detach().cpu().item()
        forces = outputs[AtomicDataDict.FORCE_KEY].detach().cpu().numpy()

        self.results["energy"] = float(energy)
        self.results["forces"] = np.array(forces, dtype=np.float64)
