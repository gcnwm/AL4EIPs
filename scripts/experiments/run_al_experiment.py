#!/usr/bin/env python
# pyright: reportMissingImports=false
"""Backend-agnostic active learning experiment runner.

The default CLI remains compatible with the original thesis runner.  New v3
options add preregistration-oriented controls for candidate generation,
training policy, run namespaces, and acquisition metadata logging.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import sys
import time
from typing import Literal, cast

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

_src_dir = str(PROJECT_ROOT / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
existing = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{_src_dir}:{existing}" if existing else _src_dir

from fyp_al.al_protocol import (  # noqa: E402
    AcquisitionRule,
    CandidatePolicy,
    EvaluationPolicy,
    TrainingPolicy,
    array_sha256,
    ensure_label_free_candidates,
    initialize_labeled_pool,
    json_ready_array,
    query_indices,
    select_candidate_indices,
    select_random_k,
    select_top_k_by_score,
    split_integrity_report,
)
from fyp_al.geometry import KCAL_TO_EV, npz_to_ase_atoms  # noqa: E402
from fyp_al.model_backend import ModelBackend  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AL protocol constants
# ---------------------------------------------------------------------------

SHUFFLE_SEED = 42
N_TEST = 1000
N_VAL = 500
N_INITIAL = 100
N_QUERY = 50
# Historical name: 10 metric points = 9 acquisition rounds + final evaluation.
N_ITERATIONS = 10
AL_SEEDS = [1, 2, 3]

DATA_DIR = PROJECT_ROOT / "data"
NPZ_PATH = DATA_DIR / "rmd17_ethanol.npz"
RESULTS_DIR = PROJECT_ROOT / "results" / "al"
RESULTS_V3_DIR = PROJECT_ROOT / "results" / "al_v3"
RESULTS_V4_DIR = PROJECT_ROOT / "results" / "al_v4"
AL_DATA_DIR = DATA_DIR / "al"
FROZEN_V4_CONFIG_DIR = PROJECT_ROOT / "configs" / "frozen_v4"

SplitMode = Literal["legacy", "v4_audit"]
ProtocolVersion = Literal["legacy", "v3", "v4"]
EndpointOption = Literal["fixed_member", "random_committee"]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return np.ravel(value).tolist()
    return str(value)


def write_json(path: Path, data: object) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str | None:
    """Return a file SHA256 digest, or None when the file is absent."""
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(args: list[str]) -> str | None:
    """Run a small git metadata command without failing the experiment."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def infer_protocol_version(run_id: str) -> ProtocolVersion:
    """Infer protocol version from run namespace for backward compatibility."""
    if run_id.startswith("v4"):
        return "v4"
    if run_id.startswith("v3"):
        return "v3"
    if run_id == "legacy":
        return "legacy"
    return "v4"


def default_results_root(run_id: str, protocol_version: ProtocolVersion) -> Path:
    """Return the default result namespace for a run id and protocol version."""
    if protocol_version == "v4":
        return RESULTS_V4_DIR / run_id
    if protocol_version == "v3":
        return RESULTS_V3_DIR / run_id
    return RESULTS_DIR


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------


def load_npz() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load rMD17 ethanol.  Returns (nuclear_charges, coords, energies, forces)."""
    data = np.load(NPZ_PATH)
    return (
        data["nuclear_charges"],
        data["coords"],
        data["energies"],
        data["forces"],
    )


def build_splits(split_mode: SplitMode = "legacy") -> dict[str, np.ndarray]:
    """Build deterministic splits.

    ``legacy`` keeps the original 1000 test + 500 validation + 98500 pool split.
    ``v4_audit`` reserves an additional 1000-structure final audit set and uses
    it as the returned primary test set; this supports the v3 plan's stronger
    leakage-control option.
    """
    n_total = int(np.load(NPZ_PATH)["energies"].shape[0])
    rng = np.random.default_rng(seed=SHUFFLE_SEED)
    shuffled = rng.permutation(n_total)

    if split_mode == "legacy":
        test_idx = shuffled[:N_TEST]
        val_idx = shuffled[N_TEST : N_TEST + N_VAL]
        pool_idx = shuffled[N_TEST + N_VAL :]
        return {
            "test_indices": test_idx.astype(np.intp),
            "val_indices": val_idx.astype(np.intp),
            "pool_indices": pool_idx.astype(np.intp),
        }

    if split_mode == "v4_audit":
        val_idx = shuffled[:N_VAL]
        dev_test_idx = shuffled[N_VAL : N_VAL + N_TEST]
        audit_test_idx = shuffled[N_VAL + N_TEST : N_VAL + 2 * N_TEST]
        pool_idx = shuffled[N_VAL + 2 * N_TEST :]
        return {
            "test_indices": audit_test_idx.astype(np.intp),
            "dev_test_indices": dev_test_idx.astype(np.intp),
            "val_indices": val_idx.astype(np.intp),
            "pool_indices": pool_idx.astype(np.intp),
        }

    msg = f"Unknown split mode: {split_mode}"
    raise ValueError(msg)


def write_split_metadata(
    *,
    output_dir: Path,
    split_mode: SplitMode,
    nuclear_charges: np.ndarray,
    coords: np.ndarray,
    energies: np.ndarray,
    forces: np.ndarray,
    split_data: dict[str, np.ndarray],
) -> None:
    """Persist split metadata and integrity checks."""
    n_total = len(energies)
    report = split_integrity_report(
        n_total=n_total,
        test_indices=split_data["test_indices"],
        val_indices=split_data["val_indices"],
        pool_indices=split_data["pool_indices"],
    )
    # v4_audit has an extra dev test set, so the 3-way report intentionally
    # does not cover the full dataset.  Record full disjointness separately.
    split_sets = [set(values.tolist()) for values in split_data.values()]
    pairwise_disjoint = all(
        not (left & right)
        for idx, left in enumerate(split_sets)
        for right in split_sets[idx + 1 :]
    )
    covered = set().union(*split_sets) if split_sets else set()
    metadata: dict[str, object] = {
        "split_mode": split_mode,
        "shuffle_seed": SHUFFLE_SEED,
        "dataset_path": str(NPZ_PATH),
        "n_total": n_total,
        "nuclear_charges_shape": list(nuclear_charges.shape),
        "coords_shape": list(coords.shape),
        "energies_shape": list(energies.shape),
        "forces_shape": list(forces.shape),
        "nuclear_charges_sha256": array_sha256(nuclear_charges),
        "coords_sha256": array_sha256(coords),
        "energies_sha256": array_sha256(energies),
        "forces_sha256": array_sha256(forces),
        "pairwise_disjoint": pairwise_disjoint,
        "covered_count": len(covered),
        "three_way_report": asdict(report),
        "all_split_report": {
            "is_disjoint": pairwise_disjoint,
            "covers_dataset": len(covered) == n_total,
            "covered_count": len(covered),
            "split_names": sorted(split_data),
        },
    }
    for key, values in split_data.items():
        metadata[f"n_{key.removesuffix('_indices')}"] = len(values)
        metadata[f"{key}_sha256"] = array_sha256(values)
    write_json(output_dir / "split_metadata.json", metadata)
    if "dev_test_indices" in split_data:
        np.savez_compressed(
            output_dir / "split_indices.npz",
            test_indices=split_data["test_indices"],
            dev_test_indices=split_data["dev_test_indices"],
            val_indices=split_data["val_indices"],
            pool_indices=split_data["pool_indices"],
        )
    else:
        np.savez_compressed(
            output_dir / "split_indices.npz",
            test_indices=split_data["test_indices"],
            val_indices=split_data["val_indices"],
            pool_indices=split_data["pool_indices"],
        )


def save_shuffle_indices() -> Path:
    """Persist the historical shuffled index order for reproducibility."""
    out = AL_DATA_DIR / "shuffle_indices.npy"
    if out.exists():
        return out
    n_total = int(np.load(NPZ_PATH)["energies"].shape[0])
    rng = np.random.default_rng(seed=SHUFFLE_SEED)
    shuffled = rng.permutation(n_total)
    AL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(out, shuffled)
    log.info("Saved shuffle indices to %s", out)
    return out


def write_xyz_subset(
    indices: np.ndarray,
    nuclear_charges: np.ndarray,
    coords: np.ndarray,
    energies: np.ndarray,
    forces: np.ndarray,
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    structures = npz_to_ase_atoms(coords, nuclear_charges, energies, forces, indices)
    ase_write(out_path, structures)
    return out_path


# ---------------------------------------------------------------------------
# Pool manager
# ---------------------------------------------------------------------------


class ALPoolManager:
    """Tracks labeled / unlabeled splits and simulates oracle queries."""

    def __init__(self, pool_indices: np.ndarray, seed: int) -> None:
        self.seed = seed
        self.all_pool = np.asarray(pool_indices, dtype=np.intp).copy()
        self.labeled: np.ndarray = np.array([], dtype=np.intp)
        self.unlabeled: np.ndarray = self.all_pool.copy()

    def initialize(self, n_initial: int) -> None:
        self.labeled, self.unlabeled = initialize_labeled_pool(
            self.all_pool,
            seed=self.seed,
            n_initial=n_initial,
        )

    def query(self, indices: np.ndarray) -> None:
        self.labeled, self.unlabeled = query_indices(
            self.labeled,
            self.unlabeled,
            np.asarray(indices, dtype=np.intp),
        )

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"labeled": self.labeled, "unlabeled": self.unlabeled}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        self.labeled = np.asarray(state["labeled"], dtype=np.intp)
        self.unlabeled = np.asarray(state["unlabeled"], dtype=np.intp)


# ---------------------------------------------------------------------------
# Iteration metrics
# ---------------------------------------------------------------------------


@dataclass
class ALIterationMetrics:
    iteration: int
    n_labeled: int
    energy_mae: float
    forces_mae: float
    energy_rmse: float
    forces_rmse: float
    mean_disagreement: float
    max_disagreement: float
    training_time: float
    train_n_before_acquisition: int = 0
    labeled_n_after_acquisition: int = 0
    candidate_n: int = 0
    selected_n: int = 0
    is_acquisition_iteration: bool = False
    candidate_policy: str = ""
    candidate_seed: int | None = None
    acquisition_seed: int | None = None
    acquisition_rule: str = ""
    training_policy: str = "warm_start"
    evaluation_policy: str = "fixed_member"


@dataclass(frozen=True)
class ExperimentProtocol:
    run_id: str
    results_root: Path
    candidate_policy: CandidatePolicy
    candidate_size: int | None
    training_policy: TrainingPolicy
    evaluation_policy: EvaluationPolicy
    n_acquisition_rounds: int
    n_metric_points: int
    split_mode: SplitMode
    allow_resume: bool
    dry_run_protocol: bool = False
    protocol_version: ProtocolVersion = "legacy"
    endpoint_option: EndpointOption = "fixed_member"


# ---------------------------------------------------------------------------
# Metadata and acquisition logging
# ---------------------------------------------------------------------------


def write_environment_metadata(output_dir: Path) -> None:
    """Write environment metadata for reproducibility."""
    metadata: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(PROJECT_ROOT),
        "pixi_toml_sha256": file_sha256(PROJECT_ROOT / "pixi.toml"),
        "pixi_lock_sha256": file_sha256(PROJECT_ROOT / "pixi.lock"),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_branch": _git_value(["branch", "--show-current"]),
        "git_status_short": _git_value(["status", "--short"]),
    }
    write_json(output_dir / "environment.json", metadata)


def v4_expected_runs(endpoint_option: EndpointOption) -> list[dict[str, object]]:
    """Return the preregistered v4 primary run matrix."""
    runs: list[dict[str, object]] = []
    for seed in AL_SEEDS:
        runs.extend(
            [
                {
                    "arm": "mace_qbc",
                    "backend": "mace_qbc",
                    "seed": seed,
                    "acquisition_rule": "uncertainty",
                    "committee_size": 4,
                    "primary_role": "qbc",
                },
                {
                    "arm": "mace_random",
                    "backend": "mace_qbc",
                    "seed": seed,
                    "acquisition_rule": "random",
                    "committee_size": 1 if endpoint_option == "fixed_member" else 4,
                    "primary_role": "random",
                },
                {
                    "arm": "nequip_qbc",
                    "backend": "nequip_qbc",
                    "seed": seed,
                    "acquisition_rule": "uncertainty",
                    "committee_size": 4,
                    "primary_role": "qbc",
                },
                {
                    "arm": "nequip_random",
                    "backend": "nequip_qbc",
                    "seed": seed,
                    "acquisition_rule": "random",
                    "committee_size": 1 if endpoint_option == "fixed_member" else 4,
                    "primary_role": "random",
                },
            ]
        )
    return runs


def write_v4_manifest_and_flow(protocol: ExperimentProtocol) -> None:
    """Persist the v4 primary manifest and initial run-flow table."""
    if protocol.protocol_version != "v4":
        return
    expected_runs = v4_expected_runs(protocol.endpoint_option)
    manifest = {
        "protocol_version": protocol.protocol_version,
        "run_id": protocol.run_id,
        "endpoint_option": protocol.endpoint_option,
        "split_mode": protocol.split_mode,
        "candidate_policy": protocol.candidate_policy,
        "candidate_size": protocol.candidate_size,
        "candidate_stream_policy": "same_mechanism_current_pool",
        "training_policy": protocol.training_policy,
        "evaluation_policy": protocol.evaluation_policy,
        "n_initial": N_INITIAL,
        "n_query": N_QUERY,
        "n_acquisition_rounds": protocol.n_acquisition_rounds,
        "n_metric_points": protocol.n_metric_points,
        "final_training_size": N_INITIAL + N_QUERY * protocol.n_acquisition_rounds,
        "allow_resume_required": False,
        "expected_runs": expected_runs,
        "primary_claim_boundary": (
            "fixed-member single-inference acquisition comparison"
            if protocol.endpoint_option == "fixed_member"
            else "matched random-committee ensemble comparison"
        ),
    }
    write_json(protocol.results_root / "run_manifest.json", manifest)

    flow_path = protocol.results_root / "run_flow.json"
    if not flow_path.exists():
        flow = {
            "run_id": protocol.run_id,
            "protocol_version": "v4",
            "runs": {
                f"{item['arm']}_seed{item['seed']}": {**item, "status": "planned"}
                for item in expected_runs
            },
        }
        write_json(flow_path, flow)

    if FROZEN_V4_CONFIG_DIR.exists():
        dest = protocol.results_root / "frozen_configs"
        dest.mkdir(parents=True, exist_ok=True)
        for src in FROZEN_V4_CONFIG_DIR.iterdir():
            if src.is_file():
                shutil.copy2(src, dest / src.name)
        write_json(
            protocol.results_root / "frozen_config_hashes.json",
            {
                path.name: file_sha256(path)
                for path in sorted(dest.iterdir())
                if path.is_file()
            },
        )


def update_run_flow(
    *,
    results_root: Path,
    run_name: str,
    status: str,
    detail: str | None = None,
) -> None:
    """Update v4 run-flow status when the artifact exists."""
    flow_path = results_root / "run_flow.json"
    if not flow_path.exists():
        return
    flow = json.loads(flow_path.read_text(encoding="utf-8"))
    runs = flow.setdefault("runs", {})
    entry = runs.setdefault(run_name, {})
    entry["status"] = status
    entry["updated_at_unix"] = time.time()
    if detail is not None:
        entry["detail"] = detail
    write_json(flow_path, flow)


def write_run_metadata(
    *,
    run_dir: Path,
    command: list[str],
    backend_name: str,
    arm_name: str,
    seed: int,
    protocol: ExperimentProtocol,
    committee_seeds: list[int] | None,
    acquisition_rule: AcquisitionRule,
) -> None:
    """Persist run-level protocol metadata."""
    metadata: dict[str, object] = {
        "run_id": protocol.run_id,
        "command": command,
        "backend": backend_name,
        "arm": arm_name,
        "al_seed": seed,
        "committee_size": len(committee_seeds) if committee_seeds is not None else 1,
        "member_seeds": committee_seeds or [],
        "training_policy": protocol.training_policy,
        "candidate_policy": protocol.candidate_policy,
        "candidate_size": protocol.candidate_size,
        "evaluation_policy": protocol.evaluation_policy,
        "effective_evaluation_policy": (
            "single_model" if committee_seeds is None else protocol.evaluation_policy
        ),
        "acquisition_rule": acquisition_rule,
        "n_acquisition_rounds": protocol.n_acquisition_rounds,
        "n_metric_points": protocol.n_metric_points,
        "split_mode": protocol.split_mode,
        "allow_resume": protocol.allow_resume,
        "dry_run_protocol": protocol.dry_run_protocol,
        "protocol_version": protocol.protocol_version,
        "endpoint_option": protocol.endpoint_option,
    }
    write_json(run_dir / "run_metadata.json", metadata)


def write_acquisition_artifact(
    *,
    run_dir: Path,
    iteration: int,
    candidate_indices: np.ndarray,
    selected_indices: np.ndarray,
    scores: np.ndarray,
    candidate_seed: int | None,
    acquisition_seed: int | None,
    acquisition_rule: AcquisitionRule,
    candidate_policy: CandidatePolicy,
) -> None:
    """Persist iteration-level acquisition metadata and arrays."""
    acquisition_dir = run_dir / "acquisition"
    acquisition_dir.mkdir(parents=True, exist_ok=True)
    stem = f"iteration_{iteration:02d}"
    np.savez_compressed(
        acquisition_dir / f"{stem}.npz",
        candidate_indices=np.asarray(candidate_indices, dtype=np.intp),
        selected_indices=np.asarray(selected_indices, dtype=np.intp),
        acquisition_scores=np.asarray(scores, dtype=float),
    )
    summary = {
        "iteration": iteration,
        "candidate_policy": candidate_policy,
        "candidate_seed": candidate_seed,
        "acquisition_seed": acquisition_seed,
        "candidate_n": int(len(candidate_indices)),
        "selected_n": int(len(selected_indices)),
        "acquisition_rule": acquisition_rule,
        "selected_indices_preview": json_ready_array(selected_indices[:10]),
        "score_min": float(np.min(scores)) if len(scores) else None,
        "score_mean": float(np.mean(scores)) if len(scores) else None,
        "score_max": float(np.max(scores)) if len(scores) else None,
    }
    write_json(acquisition_dir / f"{stem}.json", summary)


# ---------------------------------------------------------------------------
# Core AL loops
# ---------------------------------------------------------------------------


def _prepare_eval_sets(
    *,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    nuclear_charges: np.ndarray,
    coords: np.ndarray,
    energies: np.ndarray,
    forces: np.ndarray,
) -> tuple[Path, list[Atoms], np.ndarray, list[np.ndarray]]:
    val_xyz = AL_DATA_DIR / "val_set_v3.xyz"
    test_xyz = AL_DATA_DIR / "test_set_v3.xyz"
    # Always rewrite these small held-out XYZ files because the same filenames
    # may be used with different split modes across dry runs and corrected runs.
    write_xyz_subset(val_indices, nuclear_charges, coords, energies, forces, val_xyz)
    write_xyz_subset(test_indices, nuclear_charges, coords, energies, forces, test_xyz)

    test_atoms = npz_to_ase_atoms(
        coords, nuclear_charges, energies, forces, test_indices
    )
    test_e = np.array([energies[i] * KCAL_TO_EV for i in test_indices])
    test_f = [forces[i] * KCAL_TO_EV for i in test_indices]
    return val_xyz, test_atoms, test_e, test_f


def run_al_experiment(
    backend: ModelBackend,
    arm_name: str,
    seed: int,
    nuclear_charges: np.ndarray,
    coords: np.ndarray,
    energies: np.ndarray,
    forces: np.ndarray,
    test_indices: np.ndarray,
    val_indices: np.ndarray,
    pool_indices: np.ndarray,
    n_initial: int = N_INITIAL,
    n_query: int = N_QUERY,
    n_iterations: int | None = None,
    max_epochs: int = 200,
    committee_seeds: list[int] | None = None,
    protocol: ExperimentProtocol | None = None,
    backend_name: str | None = None,
    command: list[str] | None = None,
) -> list[ALIterationMetrics]:
    """Run one complete QBC/MHC AL experiment for a backend/arm/seed."""
    if committee_seeds is None:
        committee_seeds = [0, 1, 2, 3]
    if protocol is None:
        metric_points = n_iterations or N_ITERATIONS
        protocol = ExperimentProtocol(
            run_id="legacy",
            results_root=RESULTS_DIR,
            candidate_policy="proposal_stream_filtered",
            candidate_size=2000,
            training_policy="warm_start",
            evaluation_policy="fixed_member",
            n_acquisition_rounds=metric_points - 1,
            n_metric_points=metric_points,
            split_mode="legacy",
            allow_resume=True,
        )
    n_metric_points = n_iterations or protocol.n_metric_points
    run_dir = protocol.results_root / f"{arm_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "al_checkpoint.pkl"
    write_run_metadata(
        run_dir=run_dir,
        command=command or sys.argv,
        backend_name=backend_name or arm_name,
        arm_name=arm_name,
        seed=seed,
        protocol=protocol,
        committee_seeds=committee_seeds,
        acquisition_rule="uncertainty",
    )
    write_environment_metadata(run_dir)
    update_run_flow(
        results_root=protocol.results_root,
        run_name=run_dir.name,
        status="started" if not protocol.dry_run_protocol else "metadata_written",
    )

    val_xyz, test_atoms, test_e, test_f = _prepare_eval_sets(
        val_indices=val_indices,
        test_indices=test_indices,
        nuclear_charges=nuclear_charges,
        coords=coords,
        energies=energies,
        forces=forces,
    )

    pool = ALPoolManager(pool_indices, seed=seed)
    metrics_history: list[ALIterationMetrics] = []
    warm_ckpts: dict[int, Path] | None = None
    start_iter = 0

    if protocol.allow_resume and checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        pool.load_state_dict(ckpt["pool_state"])
        metrics_history = [ALIterationMetrics(**m) for m in ckpt["metrics"]]
        start_iter = int(ckpt["iteration"]) + 1
        warm_ckpts = ckpt.get("warm_ckpts")
        log.info("Resuming %s seed=%d from iteration %d", arm_name, seed, start_iter)
    else:
        pool.initialize(n_initial)

    if protocol.dry_run_protocol:
        update_run_flow(
            results_root=protocol.results_root,
            run_name=run_dir.name,
            status="metadata_written",
        )
        log.info("Dry-run protocol requested; metadata written to %s", run_dir)
        return metrics_history

    for iteration in range(start_iter, n_metric_points):
        train_indices_before = pool.labeled.copy()
        train_n_before = len(train_indices_before)
        is_acquisition_iteration = iteration < protocol.n_acquisition_rounds
        log.info(
            "=== %s seed=%d iter %d/%d — train_n=%d ===",
            arm_name,
            seed,
            iteration + 1,
            n_metric_points,
            train_n_before,
        )
        t0 = time.time()

        train_xyz = run_dir / f"iter{iteration:02d}_train.xyz"
        write_xyz_subset(
            train_indices_before, nuclear_charges, coords, energies, forces, train_xyz
        )

        committee_dir = run_dir / f"iter{iteration:02d}_committee"
        committee_result = backend.train_committee(
            train_xyz=train_xyz,
            val_xyz=val_xyz,
            output_dir=committee_dir,
            seeds=committee_seeds,
            max_epochs=max_epochs,
            warm_start_ckpts=(
                warm_ckpts if protocol.training_policy == "warm_start" else None
            ),
        )

        test_metrics = backend.evaluate_committee(
            committee_result,
            test_atoms,
            test_e,
            test_f,
            policy=protocol.evaluation_policy,
        )

        mean_dis = 0.0
        max_dis = 0.0
        candidate_n = 0
        selected_n = 0
        candidate_seed: int | None = None
        if is_acquisition_iteration:
            candidate_indices, candidate_seed = select_candidate_indices(
                pool.unlabeled,
                run_id=protocol.run_id,
                arm_name=arm_name,
                al_seed=seed,
                iteration=iteration,
                policy=protocol.candidate_policy,
                candidate_size=protocol.candidate_size,
            )
            candidate_atoms = npz_to_ase_atoms(
                coords,
                nuclear_charges,
                indices=candidate_indices,
            )
            ensure_label_free_candidates(candidate_atoms)
            disagreements = backend.compute_committee_disagreement(
                committee_result,
                candidate_atoms,
                max_eval=len(candidate_atoms),
            )
            if len(disagreements) != len(candidate_indices):
                msg = (
                    "Backend disagreement length does not match candidate length: "
                    f"{len(disagreements)} != {len(candidate_indices)}"
                )
                raise RuntimeError(msg)
            selected_indices = select_top_k_by_score(
                candidate_indices,
                disagreements,
                k=n_query,
            )
            pool.query(selected_indices)
            candidate_n = len(candidate_indices)
            selected_n = len(selected_indices)
            mean_dis = float(disagreements.mean())
            max_dis = float(disagreements.max())
            write_acquisition_artifact(
                run_dir=run_dir,
                iteration=iteration,
                candidate_indices=candidate_indices,
                selected_indices=selected_indices,
                scores=disagreements,
                candidate_seed=candidate_seed,
                acquisition_seed=None,
                acquisition_rule="uncertainty",
                candidate_policy=protocol.candidate_policy,
            )

        elapsed = time.time() - t0
        metrics = ALIterationMetrics(
            iteration=iteration,
            n_labeled=len(pool.labeled),
            energy_mae=test_metrics["energy_mae"],
            forces_mae=test_metrics["forces_mae"],
            energy_rmse=test_metrics["energy_rmse"],
            forces_rmse=test_metrics["forces_rmse"],
            mean_disagreement=mean_dis,
            max_disagreement=max_dis,
            training_time=elapsed,
            train_n_before_acquisition=train_n_before,
            labeled_n_after_acquisition=len(pool.labeled),
            candidate_n=candidate_n,
            selected_n=selected_n,
            is_acquisition_iteration=is_acquisition_iteration,
            candidate_policy=protocol.candidate_policy,
            candidate_seed=candidate_seed,
            acquisition_seed=None,
            acquisition_rule="uncertainty",
            training_policy=protocol.training_policy,
            evaluation_policy=protocol.evaluation_policy,
        )
        metrics_history.append(metrics)
        log.info(
            "  forces_mae=%.4f eV/Å  energy_mae=%.6f eV  time=%.0fs",
            metrics.forces_mae,
            metrics.energy_mae,
            metrics.training_time,
        )

        warm_ckpts = dict(committee_result.checkpoints)
        with open(checkpoint_path, "wb") as f:
            pickle.dump(
                {
                    "iteration": iteration,
                    "pool_state": pool.state_dict(),
                    "metrics": [asdict(m) for m in metrics_history],
                    "warm_ckpts": warm_ckpts,
                    "protocol": asdict(protocol),
                },
                f,
            )

    metrics_file = run_dir / "metrics.json"
    write_json(metrics_file, [asdict(m) for m in metrics_history])
    update_run_flow(
        results_root=protocol.results_root,
        run_name=run_dir.name,
        status="completed",
    )
    log.info("Saved metrics to %s", metrics_file)
    return metrics_history


def run_random_baseline(
    backend: ModelBackend,
    seed: int,
    nuclear_charges: np.ndarray,
    coords: np.ndarray,
    energies: np.ndarray,
    forces: np.ndarray,
    test_indices: np.ndarray,
    val_indices: np.ndarray,
    pool_indices: np.ndarray,
    n_initial: int = N_INITIAL,
    n_query: int = N_QUERY,
    n_iterations: int | None = None,
    max_epochs: int = 200,
    model_seed: int = 0,
    arm_name: str = "random",
    protocol: ExperimentProtocol | None = None,
    backend_name: str | None = None,
    command: list[str] | None = None,
) -> list[ALIterationMetrics]:
    """Random acquisition baseline with v3 candidate-policy logging."""
    if protocol is None:
        metric_points = n_iterations or N_ITERATIONS
        protocol = ExperimentProtocol(
            run_id="legacy",
            results_root=RESULTS_DIR,
            candidate_policy="proposal_stream_filtered",
            candidate_size=2000,
            training_policy="warm_start",
            evaluation_policy="fixed_member",
            n_acquisition_rounds=metric_points - 1,
            n_metric_points=metric_points,
            split_mode="legacy",
            allow_resume=True,
        )
    n_metric_points = n_iterations or protocol.n_metric_points
    run_dir = protocol.results_root / f"{arm_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "al_checkpoint.pkl"
    write_run_metadata(
        run_dir=run_dir,
        command=command or sys.argv,
        backend_name=backend_name or arm_name,
        arm_name=arm_name,
        seed=seed,
        protocol=protocol,
        committee_seeds=None,
        acquisition_rule="random",
    )
    write_environment_metadata(run_dir)
    update_run_flow(
        results_root=protocol.results_root,
        run_name=run_dir.name,
        status="started" if not protocol.dry_run_protocol else "metadata_written",
    )

    val_xyz, test_atoms, test_e, test_f = _prepare_eval_sets(
        val_indices=val_indices,
        test_indices=test_indices,
        nuclear_charges=nuclear_charges,
        coords=coords,
        energies=energies,
        forces=forces,
    )

    pool = ALPoolManager(pool_indices, seed=seed)
    metrics_history: list[ALIterationMetrics] = []
    warm_ckpt: Path | None = None
    start_iter = 0

    if protocol.allow_resume and checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        pool.load_state_dict(ckpt["pool_state"])
        metrics_history = [ALIterationMetrics(**m) for m in ckpt["metrics"]]
        start_iter = int(ckpt["iteration"]) + 1
        warm_ckpt = ckpt.get("warm_ckpt")
        log.info("Resuming %s seed=%d from iteration %d", arm_name, seed, start_iter)
    else:
        pool.initialize(n_initial)

    if protocol.dry_run_protocol:
        update_run_flow(
            results_root=protocol.results_root,
            run_name=run_dir.name,
            status="metadata_written",
        )
        log.info("Dry-run protocol requested; metadata written to %s", run_dir)
        return metrics_history

    for iteration in range(start_iter, n_metric_points):
        train_indices_before = pool.labeled.copy()
        train_n_before = len(train_indices_before)
        is_acquisition_iteration = iteration < protocol.n_acquisition_rounds
        log.info(
            "=== %s seed=%d iter %d/%d — train_n=%d ===",
            arm_name,
            seed,
            iteration + 1,
            n_metric_points,
            train_n_before,
        )
        t0 = time.time()

        train_xyz = run_dir / f"iter{iteration:02d}_train.xyz"
        write_xyz_subset(
            train_indices_before, nuclear_charges, coords, energies, forces, train_xyz
        )

        model_dir = run_dir / f"iter{iteration:02d}_model"
        result = backend.train_single(
            train_xyz=train_xyz,
            val_xyz=val_xyz,
            output_dir=model_dir,
            seed=model_seed,
            max_epochs=max_epochs,
            pretrained_ckpt=(
                warm_ckpt if protocol.training_policy == "warm_start" else None
            ),
        )

        test_metrics = backend.evaluate(
            result.checkpoint_path, test_atoms, test_e, test_f
        )

        candidate_n = 0
        selected_n = 0
        candidate_seed: int | None = None
        acquisition_seed: int | None = None
        if is_acquisition_iteration:
            candidate_indices, candidate_seed = select_candidate_indices(
                pool.unlabeled,
                run_id=protocol.run_id,
                arm_name=arm_name,
                al_seed=seed,
                iteration=iteration,
                policy=protocol.candidate_policy,
                candidate_size=protocol.candidate_size,
            )
            candidate_atoms = npz_to_ase_atoms(
                coords,
                nuclear_charges,
                indices=candidate_indices,
            )
            ensure_label_free_candidates(candidate_atoms)
            selected_indices, random_scores, acquisition_seed = select_random_k(
                candidate_indices,
                k=n_query,
                run_id=protocol.run_id,
                arm_name=arm_name,
                al_seed=seed,
                iteration=iteration,
            )
            pool.query(selected_indices)
            candidate_n = len(candidate_indices)
            selected_n = len(selected_indices)
            write_acquisition_artifact(
                run_dir=run_dir,
                iteration=iteration,
                candidate_indices=candidate_indices,
                selected_indices=selected_indices,
                scores=random_scores,
                candidate_seed=candidate_seed,
                acquisition_seed=acquisition_seed,
                acquisition_rule="random",
                candidate_policy=protocol.candidate_policy,
            )

        elapsed = time.time() - t0
        metrics = ALIterationMetrics(
            iteration=iteration,
            n_labeled=len(pool.labeled),
            energy_mae=test_metrics["energy_mae"],
            forces_mae=test_metrics["forces_mae"],
            energy_rmse=test_metrics["energy_rmse"],
            forces_rmse=test_metrics["forces_rmse"],
            mean_disagreement=0.0,
            max_disagreement=0.0,
            training_time=elapsed,
            train_n_before_acquisition=train_n_before,
            labeled_n_after_acquisition=len(pool.labeled),
            candidate_n=candidate_n,
            selected_n=selected_n,
            is_acquisition_iteration=is_acquisition_iteration,
            candidate_policy=protocol.candidate_policy,
            candidate_seed=candidate_seed,
            acquisition_seed=acquisition_seed,
            acquisition_rule="random",
            training_policy=protocol.training_policy,
            evaluation_policy=protocol.evaluation_policy,
        )
        metrics_history.append(metrics)
        log.info("  forces_mae=%.4f eV/Å  time=%.0fs", metrics.forces_mae, elapsed)

        warm_ckpt = result.checkpoint_path
        with open(checkpoint_path, "wb") as f:
            pickle.dump(
                {
                    "iteration": iteration,
                    "pool_state": pool.state_dict(),
                    "metrics": [asdict(m) for m in metrics_history],
                    "warm_ckpt": warm_ckpt,
                    "protocol": asdict(protocol),
                },
                f,
            )

    metrics_file = run_dir / "metrics.json"
    write_json(metrics_file, [asdict(m) for m in metrics_history])
    update_run_flow(
        results_root=protocol.results_root,
        run_name=run_dir.name,
        status="completed",
    )
    log.info("Saved metrics to %s", metrics_file)
    return metrics_history


# ---------------------------------------------------------------------------
# Backend factory and CLI
# ---------------------------------------------------------------------------


def make_backend(name: str) -> ModelBackend:
    """Instantiate a backend by short name."""
    if name == "nequip_qbc":
        from fyp_al.nequip_backend import NequIPBackend

        return NequIPBackend(project_root=PROJECT_ROOT)

    if name == "mace_qbc":
        from fyp_al.mace_backend import MACEQBCBackend

        return MACEQBCBackend(project_root=PROJECT_ROOT)

    if name == "mace_mhc":
        from fyp_al.mace_backend import MACEMHCBackend

        return MACEMHCBackend(project_root=PROJECT_ROOT)

    msg = f"Unknown backend: {name}"
    raise ValueError(msg)


COMMITTEE_SEEDS_MAP: dict[str, list[int]] = {
    "nequip_qbc": [0, 1, 2, 3],
    "mace_qbc": [0, 1, 2, 3],
    "mace_mhc": [0, 1, 2, 3],
}


def _protocol_from_args(args: argparse.Namespace) -> ExperimentProtocol:
    run_id = cast(str | None, args.run_id) or "legacy"
    protocol_version = cast(ProtocolVersion | None, args.protocol_version)
    if protocol_version is None:
        protocol_version = infer_protocol_version(run_id)
    if args.results_root is not None:
        results_root = Path(cast(str, args.results_root))
    elif args.run_id is not None:
        results_root = default_results_root(run_id, protocol_version)
    else:
        results_root = RESULTS_DIR
    n_acquisition_rounds = int(args.n_acquisition_rounds)
    n_metric_points = n_acquisition_rounds + 1
    candidate_size = cast(int | None, args.candidate_size)
    candidate_policy = cast(CandidatePolicy, args.candidate_policy)
    if candidate_policy == "full_unlabeled_pool":
        candidate_size = None
    return ExperimentProtocol(
        run_id=run_id,
        results_root=results_root,
        candidate_policy=candidate_policy,
        candidate_size=candidate_size,
        training_policy=cast(TrainingPolicy, args.training_policy),
        evaluation_policy=cast(EvaluationPolicy, args.evaluation_policy),
        n_acquisition_rounds=n_acquisition_rounds,
        n_metric_points=n_metric_points,
        split_mode=cast(SplitMode, args.split_mode),
        allow_resume=not bool(args.no_resume),
        dry_run_protocol=bool(args.dry_run_protocol),
        protocol_version=protocol_version,
        endpoint_option=cast(EndpointOption, args.endpoint_option),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backend-agnostic AL experiment runner"
    )
    parser.add_argument(
        "--backend",
        choices=["nequip_qbc", "mace_qbc", "mace_mhc"],
        help="Backend / arm to run",
    )
    parser.add_argument("--seed", type=int, choices=AL_SEEDS, help="AL seed")
    parser.add_argument("--random", action="store_true", help="Run random baseline")
    parser.add_argument("--all", action="store_true", help="Run all arms × all seeds")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument(
        "--random-arm-name",
        default="random",
        help="Directory prefix for random baseline (e.g. nequip_random)",
    )
    parser.add_argument(
        "--run-id", help="run namespace; v4* defaults to results/al_v4/RUN_ID"
    )
    parser.add_argument(
        "--protocol-version", choices=["v3", "v4"], help="defaults from run id prefix"
    )
    parser.add_argument(
        "--endpoint-option",
        choices=["fixed_member", "random_committee"],
        default="fixed_member",
    )
    parser.add_argument("--results-root", help="Explicit results root override")
    parser.add_argument(
        "--candidate-policy",
        choices=["proposal_stream_filtered", "full_unlabeled_pool"],
        default="proposal_stream_filtered",
    )
    parser.add_argument("--candidate-size", type=int, default=10_000)
    parser.add_argument(
        "--training-policy",
        choices=["scratch", "warm_start"],
        default="warm_start",
    )
    parser.add_argument(
        "--evaluation-policy",
        choices=["ensemble_mean", "best_validation", "fixed_member"],
        default="ensemble_mean",
        help="Committee evaluation rule: QBC ensemble mean, MHC head mean, best validation, or fixed member/head.",
    )
    parser.add_argument("--committee-size", type=int, default=4)
    parser.add_argument("--n-acquisition-rounds", type=int, default=9)
    parser.add_argument(
        "--split-mode",
        choices=["legacy", "v4_audit"],
        default="legacy",
    )
    parser.add_argument("--dry-run-protocol", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    protocol = _protocol_from_args(args)
    if protocol.protocol_version == "v4" and args.all:
        parser.error(
            "v4 primary runs must be launched explicitly; --all has an unsafe run matrix"
        )
    nuclear_charges, coords, energies, forces_arr = load_npz()
    split_data = build_splits(protocol.split_mode)
    protocol.results_root.mkdir(parents=True, exist_ok=True)
    write_split_metadata(
        output_dir=protocol.results_root,
        split_mode=protocol.split_mode,
        nuclear_charges=nuclear_charges,
        coords=coords,
        energies=energies,
        forces=forces_arr,
        split_data=split_data,
    )
    write_v4_manifest_and_flow(protocol)
    save_shuffle_indices()

    committee_seeds = list(range(int(args.committee_size)))
    test_idx = split_data["test_indices"]
    val_idx = split_data["val_indices"]
    pool_idx = split_data["pool_indices"]

    if args.dry_run_protocol and not args.backend and not args.random and not args.all:
        log.info("Dry-run protocol metadata written to %s", protocol.results_root)
        return

    if args.all:
        for backend_name in ["mace_mhc", "mace_qbc", "nequip_qbc"]:
            backend = make_backend(backend_name)
            for s in AL_SEEDS:
                run_al_experiment(
                    backend=backend,
                    arm_name=backend_name,
                    seed=s,
                    nuclear_charges=nuclear_charges,
                    coords=coords,
                    energies=energies,
                    forces=forces_arr,
                    test_indices=test_idx,
                    val_indices=val_idx,
                    pool_indices=pool_idx,
                    committee_seeds=committee_seeds,
                    max_epochs=args.max_epochs,
                    protocol=protocol,
                    backend_name=backend_name,
                    command=sys.argv,
                )
        mace_backend = make_backend("mace_qbc")
        for s in AL_SEEDS:
            run_random_baseline(
                backend=mace_backend,
                seed=s,
                nuclear_charges=nuclear_charges,
                coords=coords,
                energies=energies,
                forces=forces_arr,
                test_indices=test_idx,
                val_indices=val_idx,
                pool_indices=pool_idx,
                max_epochs=args.max_epochs,
                arm_name="random",
                protocol=protocol,
                backend_name="mace_qbc",
                command=sys.argv,
            )
        return

    if args.random:
        backend_name = args.backend or "mace_qbc"
        backend = make_backend(backend_name)
        seeds = [args.seed] if args.seed else AL_SEEDS
        for s in seeds:
            run_random_baseline(
                backend=backend,
                seed=s,
                nuclear_charges=nuclear_charges,
                coords=coords,
                energies=energies,
                forces=forces_arr,
                test_indices=test_idx,
                val_indices=val_idx,
                pool_indices=pool_idx,
                max_epochs=args.max_epochs,
                arm_name=args.random_arm_name,
                protocol=protocol,
                backend_name=backend_name,
                command=sys.argv,
            )
        return

    if not args.backend:
        parser.error("--backend is required (or use --all/--random/--dry-run-protocol)")

    backend = make_backend(args.backend)
    seeds = [args.seed] if args.seed else AL_SEEDS
    for s in seeds:
        run_al_experiment(
            backend=backend,
            arm_name=args.backend,
            seed=s,
            nuclear_charges=nuclear_charges,
            coords=coords,
            energies=energies,
            forces=forces_arr,
            test_indices=test_idx,
            val_indices=val_idx,
            pool_indices=pool_idx,
            committee_seeds=committee_seeds,
            max_epochs=args.max_epochs,
            protocol=protocol,
            backend_name=args.backend,
            command=sys.argv,
        )


if __name__ == "__main__":
    main()
