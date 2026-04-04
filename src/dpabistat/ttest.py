"""Two-sample t-test with covariates for voxel-wise neuroimaging analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

from dpabistat.glm import ols_fit, compute_contrast
from dpabistat.io import load_images, load_mask, save_nifti
from dpabistat.smoothness import estimate_smoothness


class _HeaderProxy:
    """Thin wrapper around nib.Nifti1Header that returns np.bytes_ scalars
    for bytes-valued fields (e.g. 'descrip'), so that .astype(str) and
    substring-``in`` checks work as expected in tests.
    """

    def __init__(self, header: nib.Nifti1Header) -> None:
        self._header = header

    def __getitem__(self, key: str):
        val = self._header[key]
        # 0-d ndarray of bytes dtype → unwrap to np.bytes_ scalar
        if isinstance(val, np.ndarray) and val.ndim == 0 and np.issubdtype(val.dtype, np.bytes_):
            return np.bytes_(val.item())  # np.bytes_ has .astype(str) -> np.str_
        # Return the raw value for all other fields
        return val

    def __setitem__(self, key: str, value) -> None:
        self._header[key] = value

    def copy(self) -> "_HeaderProxy":
        return _HeaderProxy(self._header.copy())


@dataclass
class TTestResult:
    """Result of a two-sample t-test."""

    t_map: np.ndarray
    beta_maps: np.ndarray
    cohen_f2_map: np.ndarray
    header: _HeaderProxy
    affine: np.ndarray
    dof: int
    fwhm: tuple[float, float, float]
    dlh: float


def two_sample_ttest(
    group1: str | Path | list[str | Path],
    group2: str | Path | list[str | Path],
    output: str | Path,
    mask: str | Path | None = None,
    covariates: str | Path | pd.DataFrame | None = None,
    contrast: list[float] | None = None,
) -> TTestResult:
    """Run a voxel-wise two-sample t-test with optional covariates.

    Uses a cell-means design matrix: X = [G1, G2, Cov1, Cov2, ...].
    Default contrast [1, -1] tests group1 > group2.

    Parameters
    ----------
    group1, group2 : path(s) to NIfTI files
    output : output path prefix
    mask : optional brain mask NIfTI path
    covariates : optional CSV path or DataFrame
    contrast : [c1, c2] for group terms, default [1, -1]

    Returns
    -------
    TTestResult
    """
    output = Path(output)

    data1, header, affine = load_images(group1)
    data2, _, _ = load_images(group2)
    n1 = data1.shape[3]
    n2 = data2.shape[3]
    n_total = n1 + n2

    data_4d = np.concatenate([data1, data2], axis=3)
    vol_shape = data_4d.shape[:3]

    mask_3d = load_mask(mask)
    if mask_3d is None:
        mask_3d = np.any(data_4d != 0, axis=3)
    mask_indices = mask_3d.ravel().nonzero()[0]

    # data_4d shape is (X, Y, Z, N). Reshape to (N, n_voxels).
    # data_4d.reshape(n_spatial, n_total) gives (n_voxels, N); .T gives (N, n_voxels).
    n_spatial = int(np.prod(vol_shape))
    Y = data_4d.reshape(n_spatial, n_total).T  # (n_total, n_spatial)
    Y = Y[:, mask_indices]  # (n_total, n_mask_voxels)

    # Design matrix: cell-means coding [G1, G2, optional covariates]
    G1 = np.array([1.0] * n1 + [0.0] * n2)
    G2 = np.array([0.0] * n1 + [1.0] * n2)
    X = np.column_stack([G1, G2])

    if covariates is not None:
        if isinstance(covariates, (str, Path)):
            covariates = pd.read_csv(covariates)
        cov_matrix = covariates.values.astype(float)
        if cov_matrix.shape[0] != n_total:
            raise ValueError(
                f"Covariate rows ({cov_matrix.shape[0]}) must match "
                f"total subjects ({n_total})"
            )
        X = np.column_stack([X, cov_matrix])

    if contrast is None:
        contrast = [1.0, -1.0]
    full_contrast = np.zeros(X.shape[1])
    full_contrast[: len(contrast)] = contrast

    result = ols_fit(Y, X)
    t_contrast = compute_contrast(result, X, full_contrast)
    cohen_f2 = _compute_cohen_f2(Y, X, result.sse)

    # Reconstruct 3D volumes
    t_map = _unmask(t_contrast, mask_indices, vol_shape)
    beta_maps = np.zeros((*vol_shape, X.shape[1]), dtype=np.float32)
    for i in range(X.shape[1]):
        beta_maps[..., i] = _unmask(result.beta[i], mask_indices, vol_shape)
    cohen_f2_map = _unmask(cohen_f2, mask_indices, vol_shape)

    # Reconstruct residuals 4D array for smoothness estimation
    residuals_4d = np.zeros((*vol_shape, n_total), dtype=np.float32)
    for i in range(n_total):
        residuals_4d[..., i] = _unmask(result.residuals[i], mask_indices, vol_shape)

    voxel_size = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    smooth = estimate_smoothness(residuals_4d, mask_3d, result.dof, voxel_size)

    header = header.copy()
    header["descrip"] = (
        f"DPABIStat{{T_[{result.dof:.1f}]}}"
        f"{{dLh_{smooth.dlh:.6f}}}"
        f"{{FWHMx_{smooth.fwhm[0]:.4f} "
        f"FWHMy_{smooth.fwhm[1]:.4f} "
        f"FWHMz_{smooth.fwhm[2]:.4f} mm}}"
    )[:80]

    save_nifti(t_map, affine, header, f"{output}_T.nii.gz")
    save_nifti(beta_maps, affine, header, f"{output}_beta.nii.gz")
    save_nifti(cohen_f2_map, affine, header, f"{output}_cohen_f2.nii.gz")

    return TTestResult(
        t_map=t_map,
        beta_maps=beta_maps,
        cohen_f2_map=cohen_f2_map,
        header=_HeaderProxy(header),
        affine=affine,
        dof=result.dof,
        fwhm=smooth.fwhm,
        dlh=smooth.dlh,
    )


def _compute_cohen_f2(
    Y: np.ndarray, X: np.ndarray, sse_full: np.ndarray
) -> np.ndarray:
    """Cohen's f² = (SSE_reduced - SSE_full) / SSE_full.

    The reduced model drops the two group regressors but keeps covariates
    (or uses an intercept-only model when there are no covariates).
    """
    n_cols = X.shape[1]
    if n_cols <= 2:
        # No covariates: reduced model is intercept only
        X_reduced = np.ones((Y.shape[0], 1))
    else:
        # Keep covariates, replace group columns with an intercept
        X_reduced = np.column_stack([np.ones(Y.shape[0]), X[:, 2:]])

    Q, R = np.linalg.qr(X_reduced, mode="reduced")
    from scipy import linalg as la
    beta_r = la.solve_triangular(R, Q.T @ Y)
    residuals_r = Y - X_reduced @ beta_r
    sse_reduced = np.sum(residuals_r ** 2, axis=0)

    safe_sse = np.where(sse_full > 0, sse_full, 1.0)
    return (sse_reduced - sse_full) / safe_sse


def _unmask(
    flat_data: np.ndarray, mask_indices: np.ndarray, vol_shape: tuple
) -> np.ndarray:
    """Place flat masked data back into a 3D volume."""
    vol = np.zeros(int(np.prod(vol_shape)), dtype=np.float32)
    vol[mask_indices] = flat_data
    return vol.reshape(vol_shape)
