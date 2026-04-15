#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
rMD17 Ethanol Data Pipeline — Phase 0
Downloads the revised MD17 ethanol dataset, converts to EXTXYZ format
with unit conversion (kcal/mol → eV), and creates train/val/test splits.

Source: Christensen & von Lilienfeld (2020)
        https://archive.materialscloud.org/records/pfffs-fff86

Usage:
    pixi run python scripts/01_fetch_md17.py
    pixi run python scripts/01_fetch_md17.py --n-train 200 --n-val 50 --n-test 500
    pixi run python scripts/01_fetch_md17.py --force  # re-download even if files exist
"""

import argparse
import hashlib
import logging
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fyp_al.geometry import KCAL_TO_EV  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
RMD17_URL = "https://archive.materialscloud.org/records/pfffs-fff86/files/rmd17.tar.bz2"
RMD17_MD5 = "cb1a927628d96f2e966025da4fb63d18"
RMD17_SIZE_MB = 1017  # approximate size in MiB

NPZ_MEMBER = "rmd17/npz_data/rmd17_ethanol.npz"  # path inside tarball

# Atomic number → element symbol
Z_TO_SYMBOL = {1: "H", 6: "C", 8: "O"}


# ── Download utilities ─────────────────────────────────────────────────────────
class DownloadProgressReporter:
    """Report download progress to stderr."""

    def __init__(self, total_size_mb: int):
        self.total_size_mb = total_size_mb
        self.downloaded = 0
        self.last_pct = -1

    def __call__(self, block_num: int, block_size: int, total_size: int):
        self.downloaded += block_size
        pct = min(100, int(self.downloaded / (self.total_size_mb * 1024 * 1024) * 100))
        if pct != self.last_pct:
            self.last_pct = pct
            bar = "=" * (pct // 2) + ">" + " " * (50 - pct // 2)
            mb_done = self.downloaded / (1024 * 1024)
            sys.stderr.write(
                f"\r  [{bar}] {pct:3d}%  ({mb_done:.0f}/{self.total_size_mb} MiB)"
            )
            sys.stderr.flush()
            if pct >= 100:
                sys.stderr.write("\n")


def verify_md5(filepath: Path, expected_md5: str) -> bool:
    """Verify MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest() == expected_md5


def download_rmd17(cache_dir: Path, force: bool = False) -> Path:
    """Download the rMD17 tarball.

    Returns path to the downloaded tar.bz2 file.
    """
    tarball_path = cache_dir / "rmd17.tar.bz2"

    if tarball_path.exists() and not force:
        logger.info("Tarball already exists: %s", tarball_path)
        logger.info("Verifying MD5 checksum...")
        if verify_md5(tarball_path, RMD17_MD5):
            logger.info("Checksum OK — skipping download.")
            return tarball_path
        else:
            logger.warning("Checksum mismatch — re-downloading.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading rMD17 tarball (~%d MiB)...", RMD17_SIZE_MB)
    logger.info("URL: %s", RMD17_URL)

    reporter = DownloadProgressReporter(RMD17_SIZE_MB)
    urllib.request.urlretrieve(RMD17_URL, tarball_path, reporthook=reporter)

    logger.info("Verifying MD5 checksum...")
    if not verify_md5(tarball_path, RMD17_MD5):
        tarball_path.unlink()
        raise RuntimeError(
            f"MD5 checksum mismatch after download. "
            f"Expected {RMD17_MD5}. File deleted — please retry."
        )
    logger.info("Checksum OK.")
    return tarball_path


def extract_ethanol_npz(tarball_path: Path, cache_dir: Path) -> Path:
    """Extract only the ethanol NPZ from the rMD17 tarball.

    Returns path to the extracted NPZ file.
    """
    npz_path = cache_dir / "rmd17_ethanol.npz"

    if npz_path.exists():
        logger.info("NPZ already extracted: %s", npz_path)
        return npz_path

    logger.info("Extracting %s from tarball...", NPZ_MEMBER)
    with tarfile.open(tarball_path, "r:bz2") as tar:
        member = tar.getmember(NPZ_MEMBER)
        f = tar.extractfile(member)
        if f is None:
            raise RuntimeError(f"Could not extract {NPZ_MEMBER} from tarball.")
        npz_path.write_bytes(f.read())

    logger.info(
        "Extracted: %s (%.1f MiB)", npz_path, npz_path.stat().st_size / 1024 / 1024
    )
    return npz_path


# ── Data processing ────────────────────────────────────────────────────────────
def load_and_convert(npz_path: Path) -> dict:
    """Load rMD17 ethanol NPZ and convert units to eV/Å.

    Returns dict with keys: symbols, coords, energies, forces, n_frames.
    """
    logger.info("Loading NPZ: %s", npz_path)
    data = np.load(npz_path)

    nuclear_charges = data["nuclear_charges"]  # (9,)
    coords = data["coords"]  # (N, 9, 3) in Å
    energies_kcal = data["energies"]  # (N,) in kcal/mol
    forces_kcal = data["forces"]  # (N, 9, 3) in kcal/mol/Å

    n_frames = coords.shape[0]
    n_atoms = coords.shape[1]

    # Convert atomic numbers to symbols
    symbols = [Z_TO_SYMBOL[int(z)] for z in nuclear_charges]

    # Convert units
    energies_ev = energies_kcal * KCAL_TO_EV
    forces_ev = forces_kcal * KCAL_TO_EV  # kcal/mol/Å → eV/Å

    logger.info("Loaded %d frames, %d atoms per frame", n_frames, n_atoms)
    logger.info("Elements: %s", symbols)
    logger.info(
        "Energy range: [%.4f, %.4f] eV  (mean=%.4f, std=%.4f)",
        energies_ev.min(),
        energies_ev.max(),
        energies_ev.mean(),
        energies_ev.std(),
    )
    force_magnitudes = np.linalg.norm(forces_ev, axis=-1)
    logger.info(
        "Force magnitude range: [%.4f, %.4f] eV/Å  (mean=%.4f)",
        force_magnitudes.min(),
        force_magnitudes.max(),
        force_magnitudes.mean(),
    )

    return {
        "symbols": symbols,
        "coords": coords,
        "energies": energies_ev,
        "forces": forces_ev,
        "n_frames": n_frames,
    }


def create_splits(
    n_frames: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    stride: int,
) -> dict:
    """Create train/val/test index splits with strided sampling.

    Strided sampling reduces autocorrelation between consecutive MD frames
    (recommended by rMD17 authors).

    Returns dict with keys: train, val, test (each an ndarray of indices).
    """
    total_needed = n_train + n_val + n_test
    if total_needed > n_frames:
        raise ValueError(
            f"Requested {total_needed} samples but only {n_frames} available."
        )

    rng = np.random.default_rng(seed)

    # Strided candidate pool: take every `stride`-th frame to decorrelate
    candidates = np.arange(0, n_frames, stride)
    logger.info(
        "Strided sampling: stride=%d → %d candidate frames from %d total",
        stride,
        len(candidates),
        n_frames,
    )

    if len(candidates) < total_needed:
        logger.warning(
            "Stride %d yields only %d candidates (need %d). Falling back to full pool.",
            stride,
            len(candidates),
            total_needed,
        )
        candidates = np.arange(n_frames)

    # Shuffle candidates and pick
    rng.shuffle(candidates)
    selected = candidates[:total_needed]

    train_idx = np.sort(selected[:n_train])
    val_idx = np.sort(selected[n_train : n_train + n_val])
    test_idx = np.sort(selected[n_train + n_val : n_train + n_val + n_test])

    logger.info(
        "Split sizes: train=%d, val=%d, test=%d",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def indices_to_atoms_list(data: dict, indices: np.ndarray) -> list:
    """Convert frame indices to a list of ASE Atoms objects."""
    atoms_list = []
    for i in indices:
        atoms = Atoms(
            symbols=data["symbols"],
            positions=data["coords"][i],
            pbc=False,
        )
        atoms.info["energy"] = float(data["energies"][i])
        atoms.arrays["forces"] = data["forces"][i].copy()
        atoms_list.append(atoms)
    return atoms_list


def write_split(atoms_list: list, filepath: Path) -> None:
    """Write a list of Atoms to an EXTXYZ file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    write(str(filepath), atoms_list, format="extxyz")
    logger.info(
        "Wrote %d frames → %s (%.1f MiB)",
        len(atoms_list),
        filepath,
        filepath.stat().st_size / 1024 / 1024,
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download rMD17 ethanol, convert to EXTXYZ, create splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/rmd17"),
        help="Output directory for EXTXYZ files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/rmd17/.cache"),
        help="Cache directory for downloaded tarball and extracted NPZ.",
    )
    parser.add_argument(
        "--n-train", type=int, default=950, help="Number of training samples."
    )
    parser.add_argument(
        "--n-val", type=int, default=50, help="Number of validation samples."
    )
    parser.add_argument(
        "--n-test", type=int, default=1000, help="Number of test samples."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting."
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Stride for sampling frames (reduces autocorrelation). "
        "stride=10 gives 10K candidates from 100K frames.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if tarball exists."
    )
    parser.add_argument(
        "--keep-tarball",
        action="store_true",
        help="Keep the tarball after extraction (default: delete to save ~1 GiB).",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("rMD17 Ethanol Data Pipeline")
    logger.info("=" * 60)

    # Check if output files already exist
    train_path = args.output_dir / "ethanol_train.xyz"
    val_path = args.output_dir / "ethanol_val.xyz"
    test_path = args.output_dir / "ethanol_test.xyz"

    if all(p.exists() for p in [train_path, val_path, test_path]) and not args.force:
        logger.info("Output files already exist:")
        for p in [train_path, val_path, test_path]:
            logger.info("  %s (%.1f MiB)", p, p.stat().st_size / 1024 / 1024)
        logger.info("Use --force to regenerate. Exiting.")
        return

    # Step 1: Download
    tarball_path = download_rmd17(args.cache_dir, force=args.force)
    # tarball_path = args.cache_dir / "rmd17.tar.bz2"

    # Step 2: Extract ethanol NPZ only
    npz_path = extract_ethanol_npz(tarball_path, args.cache_dir)

    # Step 3: Clean up tarball to save disk space
    if not args.keep_tarball and tarball_path.exists():
        logger.info("Removing tarball to save disk space (~1 GiB)...")
        tarball_path.unlink()
        logger.info("Tarball removed.")

    # Step 4: Load and convert units
    data = load_and_convert(npz_path)

    # Step 5: Create splits
    splits = create_splits(
        n_frames=data["n_frames"],
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
        stride=args.stride,
    )

    # Step 6: Write EXTXYZ files
    logger.info("Writing EXTXYZ files...")
    for split_name, indices in splits.items():
        atoms_list = indices_to_atoms_list(data, indices)
        filepath = args.output_dir / f"ethanol_{split_name}.xyz"
        write_split(atoms_list, filepath)

    # Step 7: Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("DONE — rMD17 Ethanol Dataset Ready")
    logger.info("=" * 60)
    logger.info("Output directory: %s", args.output_dir.resolve())
    logger.info("")
    logger.info("  %-25s %5d frames", "ethanol_train.xyz", args.n_train)
    logger.info("  %-25s %5d frames", "ethanol_val.xyz", args.n_val)
    logger.info("  %-25s %5d frames", "ethanol_test.xyz", args.n_test)
    logger.info("")
    logger.info(
        "Energy range: [%.4f, %.4f] eV", data["energies"].min(), data["energies"].max()
    )
    logger.info("Seed: %d  |  Stride: %d", args.seed, args.stride)
    logger.info("")
    logger.info("Next step: run debug training with NequIP")
    logger.info("  nequip-train -cn ethanol_debug -cp configs/nequip/")


if __name__ == "__main__":
    main()
