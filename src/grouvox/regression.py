"""Voxel-wise regression on a continuous predictor with optional covariates.

Tests: image ~ intercept + predictor + covariates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from grouvox.glm import ols_fit, compute_contrast
from grouvox.io import load_images, load_mask, save_nifti
from grouvox.smoothness import estimate_smoothness
from grouvox.ttest import _HeaderProxy, _unmask


@dataclass
class RegressionResult:
    """Result of a voxel-wise regression."""

    t_map: np.ndarray
    beta_maps: np.ndarray
    cohen_f2_map: np.ndarray
    header: _HeaderProxy
    affine: np.ndarray
    dof: int
    fwhm: tuple[float, float, float]
    dlh: float


def regression(
    images: str | Path | list[str | Path],
    predictor: str | Path | np.ndarray | pd.Series,
    output: str | Path,
    mask: str | Path | None = None,
    covariates: str | Path | pd.DataFrame | None = None,
) -> RegressionResult:
    """Run a voxel-wise regression of imaging data on a continuous predictor.

    Design matrix: X = [intercept, predictor, cov1, cov2, ...].
    The T-statistic is computed for the predictor slope (contrast [0, 1, 0, ...]).

    Parameters
    ----------
    images : path(s) to NIfTI files (directory, 4D, or list of 3D files)
    predictor : 1D array, pandas Series, or CSV path (single column) of values,
        one per subject, in the same order as the images.
    output : output path prefix
    mask : optional brain mask NIfTI path
    covariates : optional CSV path or DataFrame of nuisance regressors

    Returns
    -------
    RegressionResult
    """
    output = Path(output)

    data_4d, header, affine = load_images(images)
    n_total = data_4d.shape[3]
    vol_shape = data_4d.shape[:3]

    mask_3d = load_mask(mask)
    if mask_3d is None:
        mask_3d = np.any(data_4d != 0, axis=3)
    mask_indices = mask_3d.ravel().nonzero()[0]

    n_spatial = int(np.prod(vol_shape))
    Y = data_4d.reshape(n_spatial, n_total).T
    Y = Y[:, mask_indices]

    pred_vec = _load_predictor(predictor, n_total)

    X = np.column_stack([np.ones(n_total), pred_vec])

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

    contrast = np.zeros(X.shape[1])
    contrast[1] = 1.0

    result = ols_fit(Y, X)
    t_contrast = compute_contrast(result, X, contrast)
    cohen_f2 = _compute_cohen_f2(Y, X, result.sse)

    t_map = _unmask(t_contrast, mask_indices, vol_shape)
    beta_maps = np.zeros((*vol_shape, X.shape[1]), dtype=np.float32)
    for i in range(X.shape[1]):
        beta_maps[..., i] = _unmask(result.beta[i], mask_indices, vol_shape)
    cohen_f2_map = _unmask(cohen_f2, mask_indices, vol_shape)

    residuals_4d = np.zeros((*vol_shape, n_total), dtype=np.float32)
    for i in range(n_total):
        residuals_4d[..., i] = _unmask(result.residuals[i], mask_indices, vol_shape)

    voxel_size = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    smooth = estimate_smoothness(residuals_4d, mask_3d, result.dof, voxel_size)

    header = header.copy()
    header["descrip"] = (
        f"GrouVox{{T_[{result.dof:.1f}]}}"
        f"{{dLh_{smooth.dlh:.6f}}}"
        f"{{FWHMx_{smooth.fwhm[0]:.4f} "
        f"FWHMy_{smooth.fwhm[1]:.4f} "
        f"FWHMz_{smooth.fwhm[2]:.4f} mm}}"
    )[:80]

    save_nifti(t_map, affine, header, f"{output}_T.nii.gz")
    save_nifti(beta_maps, affine, header, f"{output}_beta.nii.gz")
    save_nifti(cohen_f2_map, affine, header, f"{output}_cohen_f2.nii.gz")

    return RegressionResult(
        t_map=t_map,
        beta_maps=beta_maps,
        cohen_f2_map=cohen_f2_map,
        header=_HeaderProxy(header),
        affine=affine,
        dof=result.dof,
        fwhm=smooth.fwhm,
        dlh=smooth.dlh,
    )


def _load_predictor(
    predictor: str | Path | np.ndarray | pd.Series, n_total: int
) -> np.ndarray:
    """Coerce a predictor input into a 1D float array of length n_total."""
    if isinstance(predictor, (str, Path)):
        df = pd.read_csv(predictor)
        if df.shape[1] != 1:
            raise ValueError(
                f"Predictor CSV must have exactly 1 column, got {df.shape[1]}"
            )
        pred_vec = df.iloc[:, 0].values.astype(float)
    elif isinstance(predictor, pd.Series):
        pred_vec = predictor.values.astype(float)
    else:
        pred_vec = np.asarray(predictor, dtype=float).ravel()

    if pred_vec.shape[0] != n_total:
        raise ValueError(
            f"Predictor length ({pred_vec.shape[0]}) must match "
            f"number of subjects ({n_total})"
        )
    return pred_vec


def _compute_cohen_f2(
    Y: np.ndarray, X: np.ndarray, sse_full: np.ndarray
) -> np.ndarray:
    """Cohen's f² = (SSE_reduced - SSE_full) / SSE_full.

    Reduced model drops the predictor column (index 1) but keeps the
    intercept (index 0) and any covariates (indices 2+).
    """
    keep_cols = [0] + list(range(2, X.shape[1]))
    X_reduced = X[:, keep_cols]

    Q, R = np.linalg.qr(X_reduced, mode="reduced")
    from scipy import linalg as la
    beta_r = la.solve_triangular(R, Q.T @ Y)
    residuals_r = Y - X_reduced @ beta_r
    sse_reduced = np.sum(residuals_r ** 2, axis=0)

    safe_sse = np.where(sse_full > 0, sse_full, 1.0)
    return (sse_reduced - sse_full) / safe_sse
