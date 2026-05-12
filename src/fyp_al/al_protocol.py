"""Protocol utilities for reproducible active-learning experiments.

The helpers in this module are intentionally backend-independent so tests can
verify acquisition semantics without launching MACE, NequIP, or GPU work.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from typing import Literal

import numpy as np
from ase import Atoms

CandidatePolicy = Literal["proposal_stream_filtered", "full_unlabeled_pool"]
AcquisitionRule = Literal["uncertainty", "random"]
TrainingPolicy = Literal["scratch", "warm_start"]
EvaluationPolicy = Literal["ensemble_mean", "best_validation", "fixed_member"]


@dataclass(frozen=True)
class SplitIntegrityReport:
    """Summary of split integrity checks."""

    n_total: int
    n_test: int
    n_val: int
    n_pool: int
    is_disjoint: bool
    covers_dataset: bool

    @property
    def ok(self) -> bool:
        """Return True when all integrity checks passed."""
        return self.is_disjoint and self.covers_dataset


def stable_seed(*parts: object) -> int:
    """Return a stable unsigned 64-bit seed from arbitrary text parts.

    Python's built-in ``hash`` is salted per process, so it must not be used for
    reproducibility-critical seeds.
    """
    text = ":".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def array_sha256(array: np.ndarray) -> str:
    """Compute a stable SHA256 digest for a NumPy array's bytes and metadata."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def remove_indices_preserve_order(values: np.ndarray, remove: np.ndarray) -> np.ndarray:
    """Return ``values`` with ``remove`` excluded while preserving order."""
    remove_set = set(np.asarray(remove, dtype=np.intp).tolist())
    return np.asarray(
        [int(value) for value in values if int(value) not in remove_set], dtype=np.intp
    )


def initialize_labeled_pool(
    pool_indices: np.ndarray,
    *,
    seed: int,
    n_initial: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the initial labeled set and order-preserving unlabeled remainder."""
    if n_initial < 0 or n_initial > len(pool_indices):
        msg = f"n_initial={n_initial} outside pool length {len(pool_indices)}"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    chosen_positions = rng.choice(len(pool_indices), size=n_initial, replace=False)
    labeled = np.asarray(pool_indices[chosen_positions], dtype=np.intp)
    unlabeled = remove_indices_preserve_order(pool_indices, labeled)
    return labeled, unlabeled


def query_indices(
    labeled: np.ndarray,
    unlabeled: np.ndarray,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Add selected indices to labeled and remove them from unlabeled."""
    selected = np.asarray(selected, dtype=np.intp)
    if not np.all(np.isin(selected, unlabeled)):
        msg = "selected indices must be a subset of the current unlabeled pool"
        raise ValueError(msg)
    return (
        np.concatenate([np.asarray(labeled, dtype=np.intp), selected]),
        remove_indices_preserve_order(unlabeled, selected),
    )


def candidate_seed_for(
    *,
    run_id: str,
    arm_name: str,
    al_seed: int,
    iteration: int,
    purpose: str = "candidate",
) -> int:
    """Stable candidate/random seed for a run, arm, AL seed, and iteration."""
    return stable_seed(run_id, arm_name, al_seed, iteration, purpose)


def select_candidate_indices(
    unlabeled: np.ndarray,
    *,
    run_id: str,
    arm_name: str,
    al_seed: int,
    iteration: int,
    policy: CandidatePolicy,
    candidate_size: int | None,
) -> tuple[np.ndarray, int | None]:
    """Select acquisition candidates from the current unlabeled pool.

    ``proposal_stream_filtered`` uses a deterministic random permutation of the
    *current eligible unlabeled pool*. This avoids sorted/index-window bias while
    respecting the fact that AL arms diverge after different acquisitions.
    """
    unlabeled = np.asarray(unlabeled, dtype=np.intp)
    if policy == "full_unlabeled_pool":
        return unlabeled.copy(), None
    if policy != "proposal_stream_filtered":
        msg = f"Unknown candidate policy: {policy}"
        raise ValueError(msg)
    if candidate_size is None or candidate_size <= 0:
        msg = "candidate_size must be positive for proposal_stream_filtered"
        raise ValueError(msg)
    seed = candidate_seed_for(
        run_id=run_id,
        arm_name=arm_name,
        al_seed=al_seed,
        iteration=iteration,
    )
    rng = np.random.default_rng(seed)
    n_take = min(candidate_size, len(unlabeled))
    positions = rng.permutation(len(unlabeled))[:n_take]
    return unlabeled[positions].astype(np.intp, copy=True), seed


def select_top_k_by_score(
    candidate_indices: np.ndarray,
    scores: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """Select the ``k`` highest-scoring candidate indices."""
    candidate_indices = np.asarray(candidate_indices, dtype=np.intp)
    scores = np.asarray(scores, dtype=float)
    if len(candidate_indices) != len(scores):
        msg = "candidate_indices and scores must have identical length"
        raise ValueError(msg)
    if k < 0 or k > len(candidate_indices):
        msg = f"k={k} outside candidate length {len(candidate_indices)}"
        raise ValueError(msg)
    if k == 0:
        return np.array([], dtype=np.intp)
    order = np.argsort(scores, kind="stable")
    return candidate_indices[order[-k:]].astype(np.intp, copy=True)


def select_random_k(
    candidate_indices: np.ndarray,
    *,
    k: int,
    run_id: str,
    arm_name: str,
    al_seed: int,
    iteration: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Select ``k`` candidates uniformly at random and return scores for logging."""
    candidate_indices = np.asarray(candidate_indices, dtype=np.intp)
    if k < 0 or k > len(candidate_indices):
        msg = f"k={k} outside candidate length {len(candidate_indices)}"
        raise ValueError(msg)
    seed = candidate_seed_for(
        run_id=run_id,
        arm_name=arm_name,
        al_seed=al_seed,
        iteration=iteration,
        purpose="random_acquisition",
    )
    rng = np.random.default_rng(seed)
    scores = rng.random(len(candidate_indices))
    selected = select_top_k_by_score(candidate_indices, scores, k=k)
    return selected, scores, seed


def force_disagreement_score(forces_comm: np.ndarray) -> float:
    """Maximum atom-wise norm of committee force standard deviation.

    Parameters
    ----------
    forces_comm
        Array with shape ``(n_members, n_atoms, 3)``.
    """
    forces = np.asarray(forces_comm, dtype=float)
    if forces.ndim != 3 or forces.shape[-1] != 3:
        msg = "forces_comm must have shape (n_members, n_atoms, 3)"
        raise ValueError(msg)
    force_std = np.std(forces, axis=0, ddof=0)
    atom_scores = np.linalg.norm(force_std, axis=1)
    return float(np.max(atom_scores))


def has_oracle_labels(atoms: Atoms) -> bool:
    """Return True when an ASE structure carries oracle energy/force labels."""
    return "energy" in atoms.info or "forces" in atoms.arrays


def ensure_label_free_candidates(atoms_list: Sequence[Atoms]) -> None:
    """Raise if any acquisition candidate contains hidden oracle labels."""
    for idx, atoms in enumerate(atoms_list):
        if has_oracle_labels(atoms):
            msg = f"candidate atom at position {idx} carries oracle labels"
            raise ValueError(msg)


def split_integrity_report(
    *,
    n_total: int,
    test_indices: np.ndarray,
    val_indices: np.ndarray,
    pool_indices: np.ndarray,
) -> SplitIntegrityReport:
    """Check whether dataset splits are disjoint and cover the dataset."""
    test = set(np.asarray(test_indices, dtype=np.intp).tolist())
    val = set(np.asarray(val_indices, dtype=np.intp).tolist())
    pool = set(np.asarray(pool_indices, dtype=np.intp).tolist())
    is_disjoint = not (test & val or test & pool or val & pool)
    union = test | val | pool
    covers_dataset = len(union) == n_total and union == set(range(n_total))
    return SplitIntegrityReport(
        n_total=n_total,
        n_test=len(test),
        n_val=len(val),
        n_pool=len(pool),
        is_disjoint=is_disjoint,
        covers_dataset=covers_dataset,
    )


def json_ready_array(values: np.ndarray) -> list[int | float]:
    """Convert a one-dimensional numeric array to JSON-serializable scalars."""
    array = np.asarray(values)
    if array.ndim != 1:
        msg = "json_ready_array expects a one-dimensional array"
        raise ValueError(msg)
    if np.issubdtype(array.dtype, np.integer):
        return [int(value) for value in array]
    return [float(value) for value in array]
