"""MACE backends for the active learning pipeline.

Provides two ``ModelBackend`` implementations:

* **MACEQBCBackend** — trains *N* independent MACE models (distinct seeds)
  and computes force disagreement via ``MACECalculator(model_paths=[…])``.
* **MACEMHCBackend** — trains **one** MACE model with *N* disjoint heads
  (``--multiheads_finetuning``) and computes per-head force disagreement.

Both delegate training to the ``mace_run_train`` CLI and inference to
``mace.calculators.MACECalculator``.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

from fyp_al.al_protocol import force_disagreement_score
from fyp_al.model_backend import (
    CommitteeTrainingResult,
    ModelBackend,
    TrainingResult,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _mace_base_args(
    *,
    name: str,
    seed: int,
    train_file: Path,
    valid_file: Path,
    model_dir: Path,
    checkpoints_dir: Path,
    results_dir: Path,
    max_epochs: int,
    restart_latest: bool = False,
) -> list[str]:
    """Build the CLI args common to both QBC and MHC backends.

    Hyperparameters match plan §3.5:
      model=MACE, max_ell=3, correlation=3, num_interactions=2,
      hidden_irreps="32x0e + 32x1o", r_max=6.0, batch_size=5,
      lr=0.01, forces_weight=100, energy_weight=1, SWA last 25%.
    """
    swa_start = int(max_epochs * 0.75)
    cmd = [
        "mace_run_train",
        f"--name={name}",
        f"--seed={seed}",
        f"--train_file={train_file}",
        f"--valid_file={valid_file}",
        # Architecture
        "--model=MACE",
        "--hidden_irreps=32x0e + 32x1o",
        "--r_max=6.0",
        "--max_ell=3",
        "--correlation=3",
        "--num_interactions=2",
        # Training
        f"--max_num_epochs={max_epochs}",
        "--batch_size=5",
        "--valid_batch_size=10",
        "--lr=0.01",
        "--amsgrad",
        "--scheduler=ReduceLROnPlateau",
        "--lr_scheduler_gamma=0.5",
        "--patience=10",
        # Loss
        "--energy_weight=1.0",
        "--forces_weight=100.0",
        "--E0s=average",
        # SWA + EMA
        "--swa",
        f"--start_swa={swa_start}",
        "--ema",
        "--ema_decay=0.99",
        # I/O
        f"--model_dir={model_dir}",
        f"--checkpoints_dir={checkpoints_dir}",
        f"--results_dir={results_dir}",
        # Data keys (ASE standard: "energy" in info, "forces" in arrays)
        "--energy_key=energy",
        "--forces_key=forces",
        # Precision & device
        "--device=cuda",
        "--default_dtype=float64",
    ]
    if restart_latest:
        cmd.append("--restart_latest")
    return cmd


def _run_mace_train(cmd: list[str], cwd: Path) -> None:
    """Execute ``mace_run_train`` and raise on failure."""
    log.info("Running: %s", " ".join(cmd[:6]) + " …")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        msg = (
            f"mace_run_train failed (exit {result.returncode})\n"
            f"STDOUT: {result.stdout[-2000:]}\n"
            f"STDERR: {result.stderr[-2000:]}"
        )
        raise RuntimeError(msg)


def _find_mace_model(model_dir: Path, name: str) -> Path:
    """Return the best MACE model file (prefer SWA stage-two if it exists)."""
    stage_two = model_dir / f"{name}_stagetwo.model"
    if stage_two.exists():
        return stage_two
    base = model_dir / f"{name}.model"
    if base.exists():
        return base
    msg = f"No .model found in {model_dir} for name={name}"
    raise FileNotFoundError(msg)


def _metrics_from_predictions(
    pred_energies: np.ndarray,
    pred_forces: list[np.ndarray],
    test_energies: np.ndarray,
    test_forces: list[np.ndarray],
) -> dict[str, float]:
    """Compute standard energy/force error metrics from predictions."""
    pred_e_arr = np.asarray(pred_energies, dtype=float)
    true_e_arr = np.asarray(test_energies, dtype=float)
    pred_f_flat = np.concatenate([f.reshape(-1) for f in pred_forces])
    true_f_flat = np.concatenate([f.reshape(-1) for f in test_forces])

    return {
        "energy_mae": float(np.mean(np.abs(pred_e_arr - true_e_arr))),
        "forces_mae": float(np.mean(np.abs(pred_f_flat - true_f_flat))),
        "energy_rmse": float(np.sqrt(np.mean((pred_e_arr - true_e_arr) ** 2))),
        "forces_rmse": float(np.sqrt(np.mean((pred_f_flat - true_f_flat) ** 2))),
    }


def _mace_predict_single(
    model_path: Path,
    test_atoms: list[Atoms],
    *,
    device: str = "cuda",
    head: str | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Predict energies and forces for one MACE model/head."""
    from mace.calculators import MACECalculator

    if head is not None:
        calc = MACECalculator(
            model_paths=str(model_path),
            device=device,
            default_dtype="float64",
            head=head,
        )
    else:
        calc = MACECalculator(
            model_paths=str(model_path),
            device=device,
            default_dtype="float64",
        )

    pred_e: list[float] = []
    pred_f: list[np.ndarray] = []
    for atoms in test_atoms:
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        pred_e.append(float(atoms_copy.get_potential_energy()))
        pred_f.append(atoms_copy.get_forces())
    return np.asarray(pred_e, dtype=float), pred_f


def _mean_predictions(
    predictions: list[tuple[np.ndarray, list[np.ndarray]]],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Average a list of structure-aligned energy/force predictions."""
    if not predictions:
        msg = "at least one prediction set is required"
        raise ValueError(msg)
    mean_energies = np.mean(np.stack([energies for energies, _ in predictions]), axis=0)
    n_structures = len(predictions[0][1])
    mean_forces = [
        np.mean(np.stack([forces[idx] for _, forces in predictions]), axis=0)
        for idx in range(n_structures)
    ]
    return mean_energies, mean_forces


def _mace_evaluate(
    model_path: Path,
    test_atoms: list[Atoms],
    test_energies: np.ndarray,
    test_forces: list[np.ndarray],
    device: str = "cuda",
    head: str | None = None,
) -> dict[str, float]:
    """Evaluate a single MACE model on a test set."""
    pred_e, pred_f = _mace_predict_single(
        model_path,
        test_atoms,
        device=device,
        head=head,
    )
    return _metrics_from_predictions(pred_e, pred_f, test_energies, test_forces)


# ---------------------------------------------------------------------------
# MACE QBC — independent ensemble
# ---------------------------------------------------------------------------


class MACEQBCBackend(ModelBackend):
    """MACE query-by-committee: *N* independent models with different seeds."""

    def __init__(
        self,
        project_root: Path,
        device: str = "cuda",
    ) -> None:
        self.project_root = project_root
        self.device = device if torch.cuda.is_available() else "cpu"

    # -- train_single -------------------------------------------------------

    def train_single(
        self,
        train_xyz: Path,
        val_xyz: Path,
        output_dir: Path,
        seed: int,
        max_epochs: int = 200,
        pretrained_ckpt: Path | None = None,
    ) -> TrainingResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        name = f"mace_single_{seed}"
        model_dir = output_dir / "models"
        ckpt_dir = output_dir / "checkpoints"
        results_dir = output_dir / "results"
        for d in (model_dir, ckpt_dir, results_dir):
            d.mkdir(parents=True, exist_ok=True)

        cmd = _mace_base_args(
            name=name,
            seed=seed,
            train_file=train_xyz,
            valid_file=val_xyz,
            model_dir=model_dir,
            checkpoints_dir=ckpt_dir,
            results_dir=results_dir,
            max_epochs=max_epochs,
            restart_latest=pretrained_ckpt is not None,
        )
        if pretrained_ckpt is not None and pretrained_ckpt.exists():
            dest = ckpt_dir / pretrained_ckpt.name
            if not dest.exists():
                shutil.copy2(pretrained_ckpt, dest)

        t0 = time.time()
        _run_mace_train(cmd, cwd=self.project_root)
        elapsed = time.time() - t0

        model_path = _find_mace_model(model_dir, name)
        return TrainingResult(
            checkpoint_path=model_path,
            training_time=elapsed,
            best_val_metric=float("inf"),
        )

    # -- train_committee ----------------------------------------------------

    def train_committee(
        self,
        train_xyz: Path,
        val_xyz: Path,
        output_dir: Path,
        seeds: list[int],
        max_epochs: int = 200,
        warm_start_ckpts: dict[int, Path] | None = None,
    ) -> CommitteeTrainingResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints: dict[int, Path] = {}
        training_times: dict[int, float] = {}

        for i, seed in enumerate(seeds):
            log.info(
                "Training MACE-QBC member %d/%d (seed=%d)", i + 1, len(seeds), seed
            )
            member_dir = output_dir / f"member_{seed}"
            model_dir = member_dir / "models"
            ckpt_dir = member_dir / "checkpoints"
            results_dir = member_dir / "results"
            for d in (model_dir, ckpt_dir, results_dir):
                d.mkdir(parents=True, exist_ok=True)

            name = f"mace_qbc_{seed}"
            has_warm_start = (
                warm_start_ckpts is not None
                and seed in warm_start_ckpts
                and warm_start_ckpts[seed].exists()
            )
            if has_warm_start:
                assert warm_start_ckpts is not None
                src = warm_start_ckpts[seed]
                dest = ckpt_dir / src.name
                if not dest.exists():
                    shutil.copy2(src, dest)
                log.info("  Warm-starting from %s", src)

            cmd = _mace_base_args(
                name=name,
                seed=seed,
                train_file=train_xyz,
                valid_file=val_xyz,
                model_dir=model_dir,
                checkpoints_dir=ckpt_dir,
                results_dir=results_dir,
                max_epochs=max_epochs,
                restart_latest=has_warm_start,
            )

            t0 = time.time()
            _run_mace_train(cmd, cwd=self.project_root)
            elapsed = time.time() - t0

            model_path = _find_mace_model(model_dir, name)
            checkpoints[seed] = model_path
            training_times[seed] = elapsed
            log.info("  Done in %.1fs — %s", elapsed, model_path)

        # Best = arbitrary first; evaluation picks the best later
        best_ckpt = checkpoints[seeds[0]]
        return CommitteeTrainingResult(
            checkpoints=checkpoints,
            training_times=training_times,
            best_checkpoint=best_ckpt,
        )

    # -- compute_committee_disagreement -------------------------------------

    def compute_committee_disagreement(
        self,
        committee_result: CommitteeTrainingResult,
        atoms_list: list[Atoms],
        batch_size: int = 64,
        max_eval: int = 2000,
    ) -> np.ndarray:
        """Force disagreement via MACECalculator with multiple model_paths.

        ``MACECalculator(model_paths=[p1, p2, …])`` internally runs all
        models and stores per-model forces in ``results["forces_comm"]``
        with shape ``(n_models, n_atoms, 3)``.
        """
        from mace.calculators import MACECalculator

        model_paths = [str(p) for p in committee_result.checkpoints.values()]
        calc = MACECalculator(
            model_paths=model_paths,
            device=self.device,
            default_dtype="float64",
        )

        eval_atoms = atoms_list[:max_eval]
        log.info(
            "Computing MACE-QBC force disagreement on %d structures (%d models)…",
            len(eval_atoms),
            len(model_paths),
        )

        disagreements = np.zeros(len(eval_atoms))
        for idx, atoms in enumerate(eval_atoms):
            atoms_copy = atoms.copy()
            atoms_copy.calc = calc
            atoms_copy.get_potential_energy()
            forces_comm = atoms_copy.calc.results["forces_comm"]
            # forces_comm: (n_models, n_atoms, 3)
            disagreements[idx] = force_disagreement_score(forces_comm)

        log.info(
            "  Disagreement — mean: %.6f, max: %.6f eV/Å",
            disagreements.mean(),
            disagreements.max(),
        )
        return disagreements

    # -- evaluate -----------------------------------------------------------

    def evaluate(
        self,
        checkpoint: Path,
        test_atoms: list[Atoms],
        test_energies: np.ndarray,
        test_forces: list[np.ndarray],
    ) -> dict[str, float]:
        return _mace_evaluate(
            checkpoint,
            test_atoms,
            test_energies,
            test_forces,
            device=self.device,
        )

    def evaluate_committee(
        self,
        committee_result: CommitteeTrainingResult,
        test_atoms: list[Atoms],
        test_energies: np.ndarray,
        test_forces: list[np.ndarray],
        policy: str = "best_validation",
    ) -> dict[str, float]:
        """Evaluate QBC by fixed member, best checkpoint, or ensemble mean."""
        if policy == "ensemble_mean":
            predictions = [
                _mace_predict_single(path, test_atoms, device=self.device)
                for path in committee_result.checkpoints.values()
            ]
            pred_e, pred_f = _mean_predictions(predictions)
            return _metrics_from_predictions(pred_e, pred_f, test_energies, test_forces)
        if policy == "fixed_member":
            checkpoint = next(iter(committee_result.checkpoints.values()))
        elif policy == "best_validation":
            checkpoint = committee_result.best_checkpoint
        else:
            msg = f"Unknown evaluation policy: {policy}"
            raise ValueError(msg)
        return self.evaluate(checkpoint, test_atoms, test_energies, test_forces)


# ---------------------------------------------------------------------------
# MACE MHC — multi-head committee (single model, disjoint heads)
# ---------------------------------------------------------------------------


class MACEMHCBackend(ModelBackend):
    """MACE multi-head committee: **one** model with N disjoint data heads.

    Training data is split into N disjoint subsets (one per head).  A single
    ``mace_run_train`` invocation with ``--heads='{…}'`` and
    ``--multiheads_finetuning=True`` produces a model whose heads can be
    queried independently for per-head force predictions.
    """

    def __init__(
        self,
        project_root: Path,
        device: str = "cuda",
    ) -> None:
        self.project_root = project_root
        self.device = device if torch.cuda.is_available() else "cpu"

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _split_train_file(
        train_xyz: Path,
        n_heads: int,
        output_dir: Path,
        seed: int,
    ) -> list[Path]:
        """Split training XYZ into N disjoint subsets for MHC heads."""
        from ase.io import read as ase_read
        from ase.io import write as ase_write

        all_atoms = ase_read(str(train_xyz), index=":")
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(all_atoms))
        splits = np.array_split(indices, n_heads)

        split_paths: list[Path] = []
        for i, split_idx in enumerate(splits):
            split_path = output_dir / f"head{i}_train.xyz"
            ase_write(str(split_path), [all_atoms[j] for j in split_idx])
            split_paths.append(split_path)
            log.info("  Head %d: %d structures → %s", i, len(split_idx), split_path)

        return split_paths

    # -- train_single -------------------------------------------------------

    def train_single(
        self,
        train_xyz: Path,
        val_xyz: Path,
        output_dir: Path,
        seed: int,
        max_epochs: int = 200,
        pretrained_ckpt: Path | None = None,
    ) -> TrainingResult:
        """MHC single = standard single-head MACE (same as QBC single)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        name = f"mace_mhc_single_{seed}"
        model_dir = output_dir / "models"
        ckpt_dir = output_dir / "checkpoints"
        results_dir = output_dir / "results"
        for d in (model_dir, ckpt_dir, results_dir):
            d.mkdir(parents=True, exist_ok=True)

        cmd = _mace_base_args(
            name=name,
            seed=seed,
            train_file=train_xyz,
            valid_file=val_xyz,
            model_dir=model_dir,
            checkpoints_dir=ckpt_dir,
            results_dir=results_dir,
            max_epochs=max_epochs,
            restart_latest=pretrained_ckpt is not None,
        )
        if pretrained_ckpt is not None and pretrained_ckpt.exists():
            dest = ckpt_dir / pretrained_ckpt.name
            if not dest.exists():
                shutil.copy2(pretrained_ckpt, dest)

        t0 = time.time()
        _run_mace_train(cmd, cwd=self.project_root)
        elapsed = time.time() - t0

        model_path = _find_mace_model(model_dir, name)
        return TrainingResult(
            checkpoint_path=model_path,
            training_time=elapsed,
            best_val_metric=float("inf"),
        )

    # -- train_committee (multi-head) ---------------------------------------

    def train_committee(
        self,
        train_xyz: Path,
        val_xyz: Path,
        output_dir: Path,
        seeds: list[int],
        max_epochs: int = 200,
        warm_start_ckpts: dict[int, Path] | None = None,
    ) -> CommitteeTrainingResult:
        """Train a single model with ``len(seeds)`` disjoint heads.

        The ``seeds`` parameter determines the number of heads.  Only ``seeds[0]``
        is used as the model seed; the rest only define the head count.  Data
        splitting uses a fixed RNG seeded by ``seeds[0]``.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        n_heads = len(seeds)
        name = f"mace_mhc_{n_heads}h"

        model_dir = output_dir / "models"
        ckpt_dir = output_dir / "checkpoints"
        results_dir = output_dir / "results"
        splits_dir = output_dir / "splits"
        for d in (model_dir, ckpt_dir, results_dir, splits_dir):
            d.mkdir(parents=True, exist_ok=True)

        split_paths = self._split_train_file(
            train_xyz, n_heads, splits_dir, seed=seeds[0]
        )

        heads_dict: dict[str, dict[str, str]] = {}
        for i, sp in enumerate(split_paths):
            heads_dict[f"head{i}"] = {"train_file": str(sp)}
        heads_json = json.dumps(heads_dict)

        # MACE CLI requires --train_file even with --heads; pass first split
        cmd = _mace_base_args(
            name=name,
            seed=seeds[0],
            train_file=split_paths[0],
            valid_file=val_xyz,
            model_dir=model_dir,
            checkpoints_dir=ckpt_dir,
            results_dir=results_dir,
            max_epochs=max_epochs,
            restart_latest=(
                warm_start_ckpts is not None
                and any(p.exists() for p in warm_start_ckpts.values())
            ),
        )
        cmd.extend(
            [
                f"--heads={heads_json}",
                "--multiheads_finetuning=True",
            ]
        )

        if warm_start_ckpts is not None:
            for ckpt_path in warm_start_ckpts.values():
                if ckpt_path.exists():
                    dest = ckpt_dir / ckpt_path.name
                    if not dest.exists():
                        shutil.copy2(ckpt_path, dest)
                    log.info("  Warm-starting MHC from %s", ckpt_path)
                    break

        t0 = time.time()
        _run_mace_train(cmd, cwd=self.project_root)
        elapsed = time.time() - t0

        model_path = _find_mace_model(model_dir, name)

        checkpoints = {seed: model_path for seed in seeds}
        training_times = {seed: elapsed / n_heads for seed in seeds}

        return CommitteeTrainingResult(
            checkpoints=checkpoints,
            training_times=training_times,
            best_checkpoint=model_path,
            extra={
                "n_heads": n_heads,
                "head_names": [f"head{i}" for i in range(n_heads)],
                "model_path": str(model_path),
            },
        )

    # -- compute_committee_disagreement (per-head) --------------------------

    def compute_committee_disagreement(
        self,
        committee_result: CommitteeTrainingResult,
        atoms_list: list[Atoms],
        batch_size: int = 64,
        max_eval: int = 2000,
    ) -> np.ndarray:
        """Per-head force disagreement for a multi-head MACE model.

        Loads the single model N times with different ``head=`` parameters,
        collects per-head forces, and computes max atom-wise std.
        """
        from mace.calculators import MACECalculator

        head_names_raw = committee_result.extra.get(
            "head_names", [f"head{i}" for i in range(len(committee_result.checkpoints))]
        )
        head_names = (
            list(head_names_raw)
            if isinstance(head_names_raw, list)
            else [str(head_names_raw)]
        )
        model_path = str(committee_result.best_checkpoint)

        eval_atoms = atoms_list[:max_eval]
        n_heads = len(head_names)
        log.info(
            "Computing MACE-MHC force disagreement on %d structures (%d heads)…",
            len(eval_atoms),
            n_heads,
        )

        calcs = [
            MACECalculator(
                model_paths=model_path,
                device=self.device,
                default_dtype="float64",
                head=head_name,
            )
            for head_name in head_names
        ]

        disagreements = np.zeros(len(eval_atoms))
        for idx, atoms in enumerate(eval_atoms):
            forces_per_head: list[np.ndarray] = []
            for calc in calcs:
                atoms_copy = atoms.copy()
                atoms_copy.calc = calc
                atoms_copy.get_potential_energy()
                forces_per_head.append(atoms_copy.get_forces())
            stacked = np.stack(forces_per_head, axis=0)  # (n_heads, n_atoms, 3)
            disagreements[idx] = force_disagreement_score(stacked)

        log.info(
            "  Disagreement — mean: %.6f, max: %.6f eV/Å",
            disagreements.mean(),
            disagreements.max(),
        )
        return disagreements

    # -- evaluate -----------------------------------------------------------

    def evaluate(
        self,
        checkpoint: Path,
        test_atoms: list[Atoms],
        test_energies: np.ndarray,
        test_forces: list[np.ndarray],
    ) -> dict[str, float]:
        return _mace_evaluate(
            checkpoint,
            test_atoms,
            test_energies,
            test_forces,
            device=self.device,
            head="head0",
        )

    def evaluate_committee(
        self,
        committee_result: CommitteeTrainingResult,
        test_atoms: list[Atoms],
        test_energies: np.ndarray,
        test_forces: list[np.ndarray],
        policy: str = "best_validation",
    ) -> dict[str, float]:
        """Evaluate MHC by fixed head or preregistered head-mean prediction."""
        head_names_raw = committee_result.extra.get(
            "head_names", [f"head{i}" for i in range(len(committee_result.checkpoints))]
        )
        head_names = (
            list(head_names_raw)
            if isinstance(head_names_raw, list)
            else [str(head_names_raw)]
        )
        if policy == "ensemble_mean":
            predictions = [
                _mace_predict_single(
                    committee_result.best_checkpoint,
                    test_atoms,
                    device=self.device,
                    head=str(head_name),
                )
                for head_name in head_names
            ]
            pred_e, pred_f = _mean_predictions(predictions)
            return _metrics_from_predictions(pred_e, pred_f, test_energies, test_forces)
        if policy in {"fixed_member", "best_validation"}:
            return _mace_evaluate(
                committee_result.best_checkpoint,
                test_atoms,
                test_energies,
                test_forces,
                device=self.device,
                head=str(head_names[0]),
            )
        msg = f"Unknown evaluation policy: {policy}"
        raise ValueError(msg)
