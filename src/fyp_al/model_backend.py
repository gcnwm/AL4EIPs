"""Abstract backend interface for ML potential active learning.

Defines the `ModelBackend` ABC that both NequIP and MACE backends implement,
plus shared dataclasses for training results.  The AL loop in
``scripts/experiments/run_al_experiment.py`` programs against this interface
so that switching architectures requires zero loop changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from ase import Atoms

EvaluationPolicyName = Literal["ensemble_mean", "best_validation", "fixed_member"]


@dataclass
class TrainingResult:
    """Outcome of training a single model."""

    checkpoint_path: Path
    training_time: float  # seconds
    best_val_metric: float


@dataclass
class CommitteeTrainingResult:
    """Outcome of training a committee (QBC or MHC)."""

    checkpoints: dict[int, Path]  # member_id → checkpoint
    training_times: dict[int, float]  # member_id → seconds
    best_checkpoint: Path  # single best member
    extra: dict[str, object] = field(default_factory=dict)
    """Backend-specific metadata (e.g. MHC model path with all heads)."""


class ModelBackend(ABC):
    """Architecture-agnostic interface consumed by the AL loop.

    Concrete subclasses: ``NequIPBackend``, ``MACEQBCBackend``,
    ``MACEMHCBackend``.
    """

    @abstractmethod
    def train_single(
        self,
        train_xyz: Path,
        val_xyz: Path,
        output_dir: Path,
        seed: int,
        max_epochs: int = 200,
        pretrained_ckpt: Path | None = None,
    ) -> TrainingResult:
        """Train one model and return its checkpoint."""
        ...

    @abstractmethod
    def train_committee(
        self,
        train_xyz: Path,
        val_xyz: Path,
        output_dir: Path,
        seeds: list[int],
        max_epochs: int = 200,
        warm_start_ckpts: dict[int, Path] | None = None,
    ) -> CommitteeTrainingResult:
        """Train a committee (independent ensemble or multi-head)."""
        ...

    @abstractmethod
    def compute_committee_disagreement(
        self,
        committee_result: CommitteeTrainingResult,
        atoms_list: list[Atoms],
        batch_size: int = 64,
        max_eval: int = 2000,
    ) -> np.ndarray:
        """Per-structure force disagreement.  Returns shape ``(N,)``."""
        ...

    @abstractmethod
    def evaluate(
        self,
        checkpoint: Path,
        test_atoms: list[Atoms],
        test_energies: np.ndarray,
        test_forces: list[np.ndarray],
    ) -> dict[str, float]:
        """Evaluate one checkpoint on a held-out test set.

        Returns
        -------
        dict
            Keys: ``energy_mae``, ``forces_mae``, ``energy_rmse``,
            ``forces_rmse`` — all in eV / eV/Å.
        """
        ...

    def evaluate_committee(
        self,
        committee_result: CommitteeTrainingResult,
        test_atoms: list[Atoms],
        test_energies: np.ndarray,
        test_forces: list[np.ndarray],
        policy: EvaluationPolicyName = "best_validation",
    ) -> dict[str, float]:
        """Evaluate a committee according to the preregistered policy.

        Backends override this for true ensemble/head-mean predictions.  The
        fallback supports fixed-member and best-validation policies only.
        """
        if policy == "fixed_member":
            first_checkpoint = next(iter(committee_result.checkpoints.values()))
            return self.evaluate(
                first_checkpoint, test_atoms, test_energies, test_forces
            )
        if policy in {"best_validation", "ensemble_mean"}:
            return self.evaluate(
                committee_result.best_checkpoint,
                test_atoms,
                test_energies,
                test_forces,
            )
        msg = f"Unknown evaluation policy: {policy}"
        raise ValueError(msg)
