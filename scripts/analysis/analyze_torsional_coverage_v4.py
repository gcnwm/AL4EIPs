#!/usr/bin/env python
"""Analyse ethanol torsional coverage for the final v4 selected structures.

The analysis is intentionally lightweight: it does not retrain or evaluate any
model. It maps the final labelled sets onto two chemically interpretable ethanol
coordinates and reports whether selected labels are enriched in high-reference-
force structures from the fixed rMD17 pool.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
_src_dir = str(PROJECT_ROOT / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from fyp_al.geometry import KCAL_TO_EV, compute_dihedral  # noqa: E402

DEFAULT_RUN_ID = "v4_primary_fixed_member"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "al_v4" / DEFAULT_RUN_ID
DEFAULT_V4_FIGURES = PROJECT_ROOT / "results" / "figures_v4" / DEFAULT_RUN_ID
DEFAULT_FINAL_FIGURES = PROJECT_ROOT / "results" / "figures_final" / "fixed_member"
DEFAULT_NPZ = PROJECT_ROOT / "data" / "rmd17_ethanol.npz"


@dataclass(frozen=True)
class CoverageSummary:
    method: str
    source: str
    n_selected: int
    n_unique: int
    occupied_30deg_bins: int
    occupied_30deg_bins_percent: float
    high_force_decile_percent: float
    mean_reference_force_rms_ev_per_a: float
    median_cc_torsion_deg: float
    median_co_torsion_deg: float


def _array_sha256(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode())
    digest.update(str(arr.dtype).encode())
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_final_labeled(results_root: Path, run_name: str) -> np.ndarray:
    checkpoint_path = results_root / run_name / "al_checkpoint.pkl"
    if not checkpoint_path.exists():
        msg = f"Missing AL checkpoint: {checkpoint_path}"
        raise FileNotFoundError(msg)
    with checkpoint_path.open("rb") as handle:
        checkpoint: dict[str, Any] = pickle.load(handle)
    pool_state = checkpoint["pool_state"]
    # Current v4 checkpoints use ``labeled``. Older helper code used
    # ``labeled_indices``, so support both to keep the script reusable.
    if "labeled" in pool_state:
        return np.asarray(pool_state["labeled"], dtype=np.intp)
    return np.asarray(pool_state["labeled_indices"], dtype=np.intp)


def _passive_indices(
    pool_indices: np.ndarray, seeds: list[int], n_train: int
) -> np.ndarray:
    selections: list[np.ndarray] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        positions = rng.choice(len(pool_indices), size=n_train, replace=False)
        selections.append(np.asarray(pool_indices[positions], dtype=np.intp))
    return np.concatenate(selections)


def _identify_ethanol_atoms(
    nuclear_charges: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray]:
    carbon_indices = np.where(nuclear_charges == 6)[0]
    oxygen_indices = np.where(nuclear_charges == 8)[0]
    hydrogen_indices = np.where(nuclear_charges == 1)[0]
    if (
        len(carbon_indices) != 2
        or len(oxygen_indices) != 1
        or len(hydrogen_indices) != 6
    ):
        msg = "Expected ethanol atom order with 2 C, 1 O, and 6 H atoms"
        raise ValueError(msg)
    return carbon_indices, int(oxygen_indices[0]), hydrogen_indices


def _ethanol_torsions_deg(
    coords: np.ndarray,
    indices: np.ndarray,
    carbon_indices: np.ndarray,
    oxygen_index: int,
    hydrogen_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return C-C and C-O torsions in degrees for selected rMD17 structures.

    The C-O torsion is H(O)-O-C-C, using the H nearest oxygen. The C-C torsion
    is O-C-C-H, using a methyl H nearest the carbon not bonded to oxygen.
    """
    cc_torsion = np.empty(len(indices), dtype=float)
    co_torsion = np.empty(len(indices), dtype=float)
    for row, index in enumerate(indices):
        pos = coords[int(index)]
        c_o_position = int(
            np.argmin(np.linalg.norm(pos[carbon_indices] - pos[oxygen_index], axis=1))
        )
        carbon_bonded_to_oxygen = int(carbon_indices[c_o_position])
        methyl_carbon = int(carbon_indices[1 - c_o_position])

        hydroxyl_h_position = int(
            np.argmin(np.linalg.norm(pos[hydrogen_indices] - pos[oxygen_index], axis=1))
        )
        hydroxyl_hydrogen = int(hydrogen_indices[hydroxyl_h_position])
        non_hydroxyl_hydrogens = np.asarray(
            [int(atom) for atom in hydrogen_indices if int(atom) != hydroxyl_hydrogen],
            dtype=np.intp,
        )
        methyl_h_position = int(
            np.argmin(
                np.linalg.norm(pos[non_hydroxyl_hydrogens] - pos[methyl_carbon], axis=1)
            )
        )
        methyl_hydrogen = int(non_hydroxyl_hydrogens[methyl_h_position])

        co_torsion[row] = compute_dihedral(
            pos[hydroxyl_hydrogen : hydroxyl_hydrogen + 1],
            pos[oxygen_index : oxygen_index + 1],
            pos[carbon_bonded_to_oxygen : carbon_bonded_to_oxygen + 1],
            pos[methyl_carbon : methyl_carbon + 1],
        )[0]
        cc_torsion[row] = compute_dihedral(
            pos[oxygen_index : oxygen_index + 1],
            pos[carbon_bonded_to_oxygen : carbon_bonded_to_oxygen + 1],
            pos[methyl_carbon : methyl_carbon + 1],
            pos[methyl_hydrogen : methyl_hydrogen + 1],
        )[0]
    return np.degrees(cc_torsion), np.degrees(co_torsion)


def _coverage_summary(
    *,
    method: str,
    source: str,
    indices: np.ndarray,
    cc_torsion: np.ndarray,
    co_torsion: np.ndarray,
    reference_force_rms: np.ndarray,
    high_force_threshold: float,
) -> CoverageSummary:
    bins = np.linspace(-180.0, 180.0, 13)
    histogram, _, _ = np.histogram2d(cc_torsion, co_torsion, bins=(bins, bins))
    occupied = int(np.count_nonzero(histogram))
    selected_force = reference_force_rms[indices]
    return CoverageSummary(
        method=method,
        source=source,
        n_selected=int(len(indices)),
        n_unique=int(len(np.unique(indices))),
        occupied_30deg_bins=occupied,
        occupied_30deg_bins_percent=100.0 * occupied / 144.0,
        high_force_decile_percent=100.0
        * float(np.mean(selected_force >= high_force_threshold)),
        mean_reference_force_rms_ev_per_a=float(np.mean(selected_force)),
        median_cc_torsion_deg=float(np.median(cc_torsion)),
        median_co_torsion_deg=float(np.median(co_torsion)),
    )


def _write_csv(path: Path, rows: list[CoverageSummary]) -> None:
    header = list(asdict(rows[0]).keys())
    lines = [",".join(header)]
    for row in rows:
        values = asdict(row)
        lines.append(
            ",".join(
                str(values[column])
                if not isinstance(values[column], str)
                else values[column].replace(",", ";")
                for column in header
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_torsional_coverage(
    *,
    output_path: Path,
    plot_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    pool_background: tuple[np.ndarray, np.ndarray],
    high_force_threshold: float,
) -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(12.0, 3.8), sharex=True, sharey=True, constrained_layout=True
    )
    methods = ["QBC", "Random AL", "Passive"]
    bg_cc, bg_co = pool_background
    all_forces = np.concatenate([plot_data[name][2] for name in methods])
    vmin = float(np.quantile(all_forces, 0.02))
    vmax = float(np.quantile(all_forces, 0.98))
    scatter = None
    for axis, method in zip(axes, methods, strict=True):
        cc_torsion, co_torsion, force_rms = plot_data[method]
        axis.scatter(
            bg_cc, bg_co, s=2, c="0.88", alpha=0.18, linewidths=0, rasterized=True
        )
        scatter = axis.scatter(
            cc_torsion,
            co_torsion,
            s=8,
            c=force_rms,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            alpha=0.72,
            linewidths=0,
            rasterized=True,
        )
        high_force_fraction = 100.0 * float(np.mean(force_rms >= high_force_threshold))
        axis.set_title(f"{method}\n{high_force_fraction:.1f}% top-force decile")
        axis.set_xlim(-180, 180)
        axis.set_ylim(-180, 180)
        axis.set_xticks([-180, -90, 0, 90, 180])
        axis.set_yticks([-180, -90, 0, 90, 180])
        axis.grid(alpha=0.18, linewidth=0.5)
        axis.set_xlabel("C-C torsion O-C-C-H / degrees")
    axes[0].set_ylabel("C-O torsion H-O-C-C / degrees")
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("Reference force RMS / eV Å$^{-1}$")
    fig.suptitle("Final labelled-set coverage in ethanol torsional space", y=1.03)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_V4_FIGURES)
    parser.add_argument("--final-figures-dir", type=Path, default=DEFAULT_FINAL_FIGURES)
    parser.add_argument("--background-sample", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.npz)
    nuclear_charges = data["nuclear_charges"]
    coords = data["coords"]
    forces = data["forces"] * KCAL_TO_EV
    split = np.load(args.results_root / "split_indices.npz")
    pool_indices = np.asarray(split["pool_indices"], dtype=np.intp)

    carbon_indices, oxygen_index, hydrogen_indices = _identify_ethanol_atoms(
        nuclear_charges
    )
    reference_force_rms = np.sqrt(np.mean(np.sum(forces * forces, axis=2), axis=1))
    high_force_threshold = float(np.quantile(reference_force_rms[pool_indices], 0.90))

    qbc_indices = np.concatenate(
        [
            _load_final_labeled(args.results_root, f"{architecture}_qbc_seed{seed}")
            for architecture in ("mace", "nequip")
            for seed in (1, 2, 3)
        ]
    )
    random_indices = np.concatenate(
        [
            _load_final_labeled(args.results_root, f"{architecture}_random_seed{seed}")
            for architecture in ("mace", "nequip")
            for seed in (1, 2, 3)
        ]
    )
    passive_indices = _passive_indices(pool_indices, seeds=[1, 2, 3], n_train=550)

    method_indices = {
        "QBC": (
            qbc_indices,
            "MACE-QBC and NequIP-QBC final labelled sets; 3 seeds each",
        ),
        "Random AL": (
            random_indices,
            "MACE-Random and NequIP-Random final labelled sets; 3 seeds each",
        ),
        "Passive": (
            passive_indices,
            "Passive 550-label train sets reconstructed from v4 pool seeds 1-3",
        ),
    }

    summaries: list[CoverageSummary] = []
    plot_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for method, (indices, source) in method_indices.items():
        cc_torsion, co_torsion = _ethanol_torsions_deg(
            coords,
            indices,
            carbon_indices,
            oxygen_index,
            hydrogen_indices,
        )
        summaries.append(
            _coverage_summary(
                method=method,
                source=source,
                indices=indices,
                cc_torsion=cc_torsion,
                co_torsion=co_torsion,
                reference_force_rms=reference_force_rms,
                high_force_threshold=high_force_threshold,
            )
        )
        plot_data[method] = (cc_torsion, co_torsion, reference_force_rms[indices])

    rng = np.random.default_rng(20260516)
    background_indices = rng.choice(
        pool_indices,
        size=min(args.background_sample, len(pool_indices)),
        replace=False,
    )
    background = _ethanol_torsions_deg(
        coords,
        background_indices,
        carbon_indices,
        oxygen_index,
        hydrogen_indices,
    )

    metadata = {
        "description": "Torsional coverage and high-reference-force enrichment for final v4 selected labels.",
        "run_id": args.results_root.name,
        "npz_path": str(args.npz),
        "split_root": str(args.results_root),
        "pool_indices_sha256": _array_sha256(pool_indices),
        "high_force_threshold_ev_per_a": high_force_threshold,
        "torsions": {
            "cc_torsion_deg": "O-C-C-H, where H is a methyl hydrogen nearest the methyl carbon",
            "co_torsion_deg": "H-O-C-C, where H is the hydroxyl hydrogen nearest oxygen",
        },
        "summaries": [asdict(row) for row in summaries],
    }
    _write_json(args.figures_dir / "torsional_coverage_summary.json", metadata)
    _write_json(args.final_figures_dir / "torsional_coverage_summary.json", metadata)
    _write_csv(args.figures_dir / "torsional_coverage_summary.csv", summaries)
    _write_csv(args.final_figures_dir / "torsional_coverage_summary.csv", summaries)
    _plot_torsional_coverage(
        output_path=args.figures_dir / "torsional_coverage_selected_sets.png",
        plot_data=plot_data,
        pool_background=background,
        high_force_threshold=high_force_threshold,
    )
    _plot_torsional_coverage(
        output_path=args.final_figures_dir / "torsional_coverage_selected_sets.png",
        plot_data=plot_data,
        pool_background=background,
        high_force_threshold=high_force_threshold,
    )
    print(json.dumps(metadata, indent=2)[:2000])


if __name__ == "__main__":
    main()
