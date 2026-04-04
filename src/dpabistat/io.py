"""NIfTI image loading and saving utilities."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def load_images(
    paths: str | Path | list[str | Path],
) -> tuple[np.ndarray, nib.Nifti1Header, np.ndarray]:
    """Load NIfTI images and stack into a 4D array.

    Parameters
    ----------
    paths : path-like or list of path-likes
        A single 4D NIfTI file, a directory of 3D NIfTI files,
        or an explicit list of 3D NIfTI file paths.

    Returns
    -------
    data : np.ndarray, shape (X, Y, Z, N)
    header : nib.Nifti1Header from the first image
    affine : np.ndarray, shape (4, 4)
    """
    paths = _resolve_paths(paths)

    first_img = nib.load(paths[0])
    header = first_img.header.copy()
    affine = first_img.affine.copy()
    first_data = first_img.get_fdata(dtype=np.float32)

    # Single 4D file
    if len(paths) == 1 and first_data.ndim == 4:
        return first_data, header, affine

    # Multiple 3D files
    ref_shape = first_data.shape[:3]
    volumes = [first_data[..., np.newaxis] if first_data.ndim == 3 else first_data]

    for p in paths[1:]:
        img = nib.load(p)
        d = img.get_fdata(dtype=np.float32)
        if d.shape[:3] != ref_shape:
            raise ValueError(
                f"Image shape mismatch: {p} has shape {d.shape[:3]}, "
                f"expected {ref_shape}"
            )
        volumes.append(d[..., np.newaxis] if d.ndim == 3 else d)

    data = np.concatenate(volumes, axis=3)
    return data, header, affine


def load_mask(path: str | Path | None) -> np.ndarray | None:
    """Load a brain mask as a boolean array.

    Returns None if path is None.
    """
    if path is None:
        return None
    img = nib.load(path)
    return img.get_fdata(dtype=np.float32) > 0


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header,
    path: str | Path,
    description: str | None = None,
) -> None:
    """Save data as a NIfTI image (float32, gzipped)."""
    header = header.copy()
    header.set_data_dtype(np.float32)
    if description is not None:
        header["descrip"] = description[:80]
    img = nib.Nifti1Image(data.astype(np.float32), affine, header)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, path)


def _resolve_paths(paths: str | Path | list[str | Path]) -> list[Path]:
    """Normalize input to a sorted list of NIfTI file paths."""
    paths = Path(paths) if isinstance(paths, str) else paths

    if isinstance(paths, Path):
        if paths.is_dir():
            files = sorted(
                p for p in paths.iterdir()
                if p.suffix in (".gz", ".nii") and ".nii" in p.name
            )
            if not files:
                raise FileNotFoundError(f"No NIfTI files found in {paths}")
            return files
        return [paths]

    return [Path(p) for p in paths]
