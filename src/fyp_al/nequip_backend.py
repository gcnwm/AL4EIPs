"""NequIP backend for the active learning pipeline.

Wraps the existing ``nequip-train`` subprocess workflow and NequIP inference
into the ``ModelBackend`` interface.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ruamel.yaml import YAML

from fyp_al.al_protocol import force_disagreement_score
from fyp_al.model_backend import (
    CommitteeTrainingResult,
    ModelBackend,
    TrainingResult,
)

log = logging.getLogger(__name__)

KCAL_TO_EV = 0.0433641153


def _ensure_src_on_path() -> str:
    src_dir = str(Path(__file__).resolve().parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{src_dir}:{existing}" if existing else src_dir
    return src_dir


def _build_nequip_model_config(
    seed: int,
    pretrained_ckpt: Path | None = None,
) -> dict:
    """Build the NequIP model sub-config.

    Uses Hydra interpolation (``${...}``) for shared parameters so the
    generated YAML mirrors hand-written configs.
    """
    if pretrained_ckpt is not None:
        return {
            "_target_": "nequip.model.ModelFromCheckpoint",
            "checkpoint_path": str(pretrained_ckpt),
        }
    return {
        "_target_": "nequip.model.NequIPGNNModel",
        "seed": seed,
        "model_dtype": "float32",
        "type_names": "${model_type_names}",
        "r_max": "${cutoff_radius}",
        # Bessel radial basis
        "num_bessels": 8,
        "bessel_trainable": False,
        "polynomial_cutoff_p": 6,
        # Interaction layers
        "num_layers": "${num_layers}",
        "l_max": "${l_max}",
        "parity": True,
        "num_features": "${num_features}",
        # Radial MLP
        "radial_mlp_depth": 2,
        "radial_mlp_width": 64,
        # Energy shifts & scales from dataset statistics
        "avg_num_neighbors": "${training_data_stats:num_neighbors_mean}",
        "per_type_energy_scales": "${training_data_stats:per_type_forces_rms}",
        "per_type_energy_shifts": "${training_data_stats:per_atom_energy_mean}",
        "per_type_energy_scales_trainable": False,
        "per_type_energy_shifts_trainable": False,
        # Force output via autograd
        "do_derivatives": True,
    }


def _build_nequip_config(
    seed: int,
    train_xyz_path: Path,
    val_xyz_path: Path,
    output_name: str,
    max_epochs: int = 200,
    pretrained_ckpt: Path | None = None,
) -> dict:
    """Build a valid NequIP v0.7+ Hydra config dict.

    The config must have exactly four required top-level sections:
    ``run``, ``data``, ``trainer``, ``training_module``.

    The ``data`` section uses ``ASEDataModule`` with separate
    ``train_file_path`` / ``val_file_path`` since the AL loop provides
    pre-split XYZ files each iteration.
    """
    return {
        # ── top-level shared parameters (for ${...} interpolation) ──
        "run": ["train"],
        "cutoff_radius": 4.0,
        "num_layers": 4,
        "l_max": 2,
        "num_features": 64,
        "model_type_names": ["H", "C", "O"],
        "monitored_metric": "val0_epoch/weighted_sum",
        # ── data ──
        "data": {
            "_target_": "nequip.data.datamodule.ASEDataModule",
            "seed": seed,
            "train_file_path": str(train_xyz_path.resolve()),
            "val_file_path": str(val_xyz_path.resolve()),
            "key_mapping": {"energy": "total_energy"},
            "transforms": [
                {"_target_": "nequip.data.transforms.NonPeriodicCellTransform"},
                {
                    "_target_": "nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper",
                    "model_type_names": "${model_type_names}",
                },
                {
                    "_target_": "nequip.data.transforms.NeighborListTransform",
                    "r_max": "${cutoff_radius}",
                },
                {"_target_": "fyp_al.transforms.ZeroEdgeCellShiftTransform"},
            ],
            "train_dataloader": {
                "_target_": "torch.utils.data.DataLoader",
                "batch_size": 5,
                "num_workers": 0,
                "shuffle": True,
            },
            "val_dataloader": {
                "_target_": "torch.utils.data.DataLoader",
                "batch_size": 5,
                "num_workers": 0,
            },
            "stats_manager": {
                "_target_": "nequip.data.CommonDataStatisticsManager",
                "type_names": "${model_type_names}",
                "dataloader_kwargs": {"batch_size": 5},
            },
        },
        # ── trainer ──
        "trainer": {
            "_target_": "lightning.Trainer",
            "accelerator": "gpu",
            "precision": "bf16-mixed",
            "enable_checkpointing": True,
            "max_epochs": max_epochs,
            "log_every_n_steps": 100,
            "logger": {
                "_target_": "lightning.pytorch.loggers.TensorBoardLogger",
                "save_dir": "${hydra:runtime.output_dir}",
                "version": "${hydra:job.name}",
            },
            "callbacks": [
                {
                    "_target_": "lightning.pytorch.callbacks.EarlyStopping",
                    "monitor": "${monitored_metric}",
                    "min_delta": 1e-4,
                    "patience": 25,
                },
                {
                    "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "monitor": "${monitored_metric}",
                    "dirpath": "${hydra:runtime.output_dir}",
                    "filename": "best",
                    "save_last": True,
                },
                {
                    "_target_": "lightning.pytorch.callbacks.LearningRateMonitor",
                    "logging_interval": "epoch",
                },
                {
                    "_target_": "nequip.train.callbacks.TF32Scheduler",
                    "schedule": {0: True, 100: False},
                },
            ],
        },
        # ── training_module ──
        "training_module": {
            "_target_": "nequip.train.EMALightningModule",
            "ema_decay": 0.999,
            "loss": {
                "_target_": "nequip.train.EnergyForceLoss",
                "per_atom_energy": True,
                "coeffs": {"total_energy": 1.0, "forces": 100.0},
            },
            "val_metrics": {
                "_target_": "nequip.train.EnergyForceMetrics",
                "coeffs": {
                    "total_energy_rmse": 1.0,
                    "forces_rmse": 1.0,
                    "total_energy_mae": None,
                    "per_atom_energy_mae": None,
                    "forces_mae": None,
                },
            },
            "train_metrics": "${training_module.val_metrics}",
            "test_metrics": "${training_module.val_metrics}",
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.01},
            "lr_scheduler": {
                "scheduler": {
                    "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                    "factor": 0.5,
                    "patience": 10,
                    "threshold": 0.1,
                    "min_lr": 1e-6,
                },
                "monitor": "${monitored_metric}",
                "interval": "epoch",
                "frequency": 1,
            },
            "model": _build_nequip_model_config(seed, pretrained_ckpt),
        },
    }


def _write_yaml_config(config: dict, config_path: Path) -> Path:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml = YAML(typ="unsafe", pure=False)
        yaml.default_flow_style = False
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.dump(config, f)
    return config_path


def _normalize_hydra_config(run_dir: Path) -> None:
    hydra_config = run_dir / ".hydra" / "config.yaml"
    if not hydra_config.exists():
        return
    yaml_in = YAML(typ="safe", pure=True)
    yaml_out = YAML(typ="unsafe", pure=True)
    yaml_out.default_flow_style = False
    yaml_out.indent(mapping=2, sequence=4, offset=2)
    yaml_out.width = 10_000
    try:
        with open(hydra_config) as f:
            data = yaml_in.load(f)
        with open(hydra_config, "w") as f:
            yaml_out.dump(data, f)
    except Exception as exc:
        log.warning("Could not normalize Hydra config %s: %s", hydra_config, exc)


def _run_nequip_train(
    config_path: Path,
    run_dir: Path,
    project_root: Path,
) -> None:
    cmd = [
        "nequip-train",
        "-cn",
        config_path.stem,
        "-cp",
        str(config_path.parent.resolve()),
        f"hydra.run.dir={run_dir}",
    ]
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    _normalize_hydra_config(run_dir)
    if result.returncode != 0:
        msg = (
            f"nequip-train failed (exit {result.returncode})\n"
            f"STDOUT: {result.stdout[-2000:]}\n"
            f"STDERR: {result.stderr[-2000:]}"
        )
        raise RuntimeError(msg)


def _get_best_val_score(ckpt_path: Path) -> float:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key, value in ckpt.get("callbacks", {}).items():
        if "ModelCheckpoint" not in key:
            continue
        score = value.get("best_model_score")
        if score is not None:
            return score.item() if hasattr(score, "item") else float(score)
    return float("inf")


class NequIPBackend(ModelBackend):
    """NequIP backend using ``nequip-train`` subprocess + NequIP inference."""

    def __init__(
        self,
        project_root: Path,
        config_dir: Path | None = None,
        model_type_names: list[str] | None = None,
        r_max: float = 4.0,
    ) -> None:
        self.project_root = project_root
        self.config_dir = config_dir or (project_root / "data" / "al")
        self.type_names = model_type_names or ["H", "C", "O"]
        self.r_max = r_max
        _ensure_src_on_path()

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
        name = f"nequip_single_{seed}"
        config = _build_nequip_config(
            seed=seed,
            train_xyz_path=train_xyz,
            val_xyz_path=val_xyz,
            output_name=name,
            max_epochs=max_epochs,
            pretrained_ckpt=pretrained_ckpt,
        )
        config_path = _write_yaml_config(config, self.config_dir / f"{name}.yaml")
        run_dir = output_dir / name

        t0 = time.time()
        _run_nequip_train(config_path, run_dir, self.project_root)
        elapsed = time.time() - t0

        best_ckpt = run_dir / "best.ckpt"
        if not best_ckpt.exists():
            msg = f"Checkpoint not found: {best_ckpt}"
            raise FileNotFoundError(msg)

        return TrainingResult(
            checkpoint_path=best_ckpt,
            training_time=elapsed,
            best_val_metric=_get_best_val_score(best_ckpt),
        )

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
                "Training NequIP committee member %d/%d (seed=%d)",
                i + 1,
                len(seeds),
                seed,
            )
            pretrained = None
            if warm_start_ckpts and seed in warm_start_ckpts:
                ckpt = warm_start_ckpts[seed]
                if ckpt.exists():
                    pretrained = ckpt
                    log.info("  Warm-starting from %s", ckpt)

            name = f"committee_{seed}"
            config = _build_nequip_config(
                seed=seed,
                train_xyz_path=train_xyz,
                val_xyz_path=val_xyz,
                output_name=name,
                max_epochs=max_epochs,
                pretrained_ckpt=pretrained,
            )
            config_path = _write_yaml_config(config, self.config_dir / f"{name}.yaml")
            run_dir = output_dir / name

            t0 = time.time()
            _run_nequip_train(config_path, run_dir, self.project_root)
            elapsed = time.time() - t0

            best_ckpt = run_dir / "best.ckpt"
            if not best_ckpt.exists():
                msg = f"Checkpoint not found: {best_ckpt}"
                raise FileNotFoundError(msg)

            checkpoints[seed] = best_ckpt
            training_times[seed] = elapsed
            log.info("  Done in %.1fs — %s", elapsed, best_ckpt)

        best_score = float("inf")
        best_checkpoint = checkpoints[seeds[0]]
        for seed, ckpt_path in checkpoints.items():
            score = _get_best_val_score(ckpt_path)
            if score < best_score:
                best_score = score
                best_checkpoint = ckpt_path

        return CommitteeTrainingResult(
            checkpoints=checkpoints,
            training_times=training_times,
            best_checkpoint=best_checkpoint,
        )

    def compute_committee_disagreement(
        self,
        committee_result: CommitteeTrainingResult,
        atoms_list: list[Atoms],
        batch_size: int = 64,
        max_eval: int = 2000,
    ) -> np.ndarray:
        from nequip.data import AtomicDataDict, from_ase
        from nequip.data.transforms import (
            ChemicalSpeciesToAtomTypeMapper,
            NeighborListTransform,
            NonPeriodicCellTransform,
        )
        from nequip.model import ModelFromCheckpoint
        from nequip.utils.global_state import set_global_state

        from fyp_al.transforms import ZeroEdgeCellShiftTransform

        set_global_state()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        eval_atoms = atoms_list[:max_eval]
        log.info(
            "Computing NequIP force disagreement on %d structures (%d committee members)...",
            len(eval_atoms),
            len(committee_result.checkpoints),
        )

        transforms = [
            NonPeriodicCellTransform(),
            ChemicalSpeciesToAtomTypeMapper(model_type_names=self.type_names),
            NeighborListTransform(r_max=self.r_max),
            ZeroEdgeCellShiftTransform(),
        ]

        models = []
        for seed, ckpt_path in committee_result.checkpoints.items():
            model = ModelFromCheckpoint(str(ckpt_path))["sole_model"]
            model.eval().to(device)
            models.append(model)

        disagreements = np.zeros(len(eval_atoms))

        for start in range(0, len(eval_atoms), batch_size):
            batch_atoms = eval_atoms[start : start + batch_size]
            data_list = []
            for atoms in batch_atoms:
                data = from_ase(atoms)
                for t in transforms:
                    data = t(data)
                data_list.append(data)

            batch = AtomicDataDict.batched_from_list(data_list)
            AtomicDataDict.to_(batch, device)

            committee_forces: list[list[np.ndarray]] = []
            for model in models:
                batch_copy = batch.copy()
                batch_copy[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)
                outputs = model(batch_copy)
                force_tensor = outputs[AtomicDataDict.FORCE_KEY].detach().cpu()
                batch_idx = outputs[AtomicDataDict.BATCH_KEY].detach().cpu().numpy()
                per_struct = [
                    force_tensor[batch_idx == i].numpy()
                    for i in range(len(batch_atoms))
                ]
                committee_forces.append(per_struct)
                del outputs

            for local_idx in range(len(batch_atoms)):
                stacked = np.stack(
                    [member[local_idx] for member in committee_forces], axis=0
                )
                disagreements[start + local_idx] = force_disagreement_score(stacked)

        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log.info(
            "  Disagreement — mean: %.6f, max: %.6f eV/Å",
            disagreements.mean(),
            disagreements.max(),
        )
        return disagreements

    @staticmethod
    def _metrics_from_predictions(
        pred_energies: np.ndarray,
        pred_forces: list[np.ndarray],
        test_energies: np.ndarray,
        test_forces: list[np.ndarray],
    ) -> dict[str, float]:
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

    @staticmethod
    def _mean_predictions(
        predictions: list[tuple[np.ndarray, list[np.ndarray]]],
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        if not predictions:
            msg = "at least one prediction set is required"
            raise ValueError(msg)
        mean_energies = np.mean(
            np.stack([energies for energies, _ in predictions]),
            axis=0,
        )
        n_structures = len(predictions[0][1])
        mean_forces = [
            np.mean(np.stack([forces[idx] for _, forces in predictions]), axis=0)
            for idx in range(n_structures)
        ]
        return mean_energies, mean_forces

    def _predict_checkpoint(
        self,
        checkpoint: Path,
        test_atoms: list[Atoms],
        *,
        batch_size: int = 64,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        from nequip.data import AtomicDataDict, from_ase
        from nequip.data.transforms import (
            ChemicalSpeciesToAtomTypeMapper,
            NeighborListTransform,
            NonPeriodicCellTransform,
        )
        from nequip.model import ModelFromCheckpoint
        from nequip.utils.global_state import set_global_state

        from fyp_al.transforms import ZeroEdgeCellShiftTransform

        set_global_state()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ModelFromCheckpoint(str(checkpoint))["sole_model"]
        model.eval().to(device)

        transforms = [
            NonPeriodicCellTransform(),
            ChemicalSpeciesToAtomTypeMapper(model_type_names=self.type_names),
            NeighborListTransform(r_max=self.r_max),
            ZeroEdgeCellShiftTransform(),
        ]

        pred_e: list[float] = []
        pred_f: list[np.ndarray] = []
        for start in range(0, len(test_atoms), batch_size):
            batch_atoms = test_atoms[start : start + batch_size]
            data_list = []
            for atoms in batch_atoms:
                data = from_ase(atoms)
                for transform in transforms:
                    data = transform(data)
                data_list.append(data)

            batch = AtomicDataDict.batched_from_list(data_list)
            AtomicDataDict.to_(batch, device)
            # NequIP computes forces via autograd.grad(energy, positions),
            # so positions must have requires_grad=True and no_grad is forbidden.
            batch[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)
            outputs = model(batch.copy())

            pred_e.extend(
                outputs[AtomicDataDict.TOTAL_ENERGY_KEY]
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .tolist()
            )
            force_tensor = outputs[AtomicDataDict.FORCE_KEY].detach().cpu()
            batch_idx = outputs[AtomicDataDict.BATCH_KEY].detach().cpu().numpy()
            pred_f.extend(
                [force_tensor[batch_idx == i].numpy() for i in range(len(batch_atoms))]
            )
            del outputs

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.asarray(pred_e, dtype=float), pred_f

    def evaluate(
        self,
        checkpoint: Path,
        test_atoms: list[Atoms],
        test_energies: np.ndarray,
        test_forces: list[np.ndarray],
    ) -> dict[str, float]:
        pred_e, pred_f = self._predict_checkpoint(checkpoint, test_atoms)
        return self._metrics_from_predictions(
            pred_e, pred_f, test_energies, test_forces
        )

    def evaluate_committee(
        self,
        committee_result: CommitteeTrainingResult,
        test_atoms: list[Atoms],
        test_energies: np.ndarray,
        test_forces: list[np.ndarray],
        policy: str = "best_validation",
    ) -> dict[str, float]:
        if policy == "ensemble_mean":
            predictions = [
                self._predict_checkpoint(checkpoint, test_atoms)
                for checkpoint in committee_result.checkpoints.values()
            ]
            pred_e, pred_f = self._mean_predictions(predictions)
            return self._metrics_from_predictions(
                pred_e, pred_f, test_energies, test_forces
            )
        if policy == "fixed_member":
            checkpoint = next(iter(committee_result.checkpoints.values()))
        elif policy == "best_validation":
            checkpoint = committee_result.best_checkpoint
        else:
            msg = f"Unknown evaluation policy: {policy}"
            raise ValueError(msg)
        return self.evaluate(checkpoint, test_atoms, test_energies, test_forces)
