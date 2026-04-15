"""FYP Active Learning — reusable modules for NequIP/MACE active learning experiments."""

from fyp_al.geometry import (
    KCAL_TO_EV,
    compute_dihedral,
    get_hocc_dihedral,
    load_al_selected_indices,
    npz_to_ase_atoms,
)
from fyp_al.mace_backend import MACEMHCBackend, MACEQBCBackend
from fyp_al.model_backend import (
    CommitteeTrainingResult,
    ModelBackend,
    TrainingResult,
)
from fyp_al.nequip_backend import NequIPBackend

__all__ = [
    "CommitteeTrainingResult",
    "compute_dihedral",
    "get_hocc_dihedral",
    "KCAL_TO_EV",
    "load_al_selected_indices",
    "MACEMHCBackend",
    "MACEQBCBackend",
    "ModelBackend",
    "NequIPBackend",
    "npz_to_ase_atoms",
    "TrainingResult",
]
