"""Vectorized OLS regression engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg


@dataclass
class OLSResult:
    """Result of vectorized OLS fit across voxels."""

    beta: np.ndarray       # (n_regressors, n_voxels)
    residuals: np.ndarray  # (n_subjects, n_voxels)
    t_values: np.ndarray   # (n_regressors, n_voxels)
    sse: np.ndarray        # (n_voxels,)
    dof: int               # n_subjects - rank(X)


def ols_fit(Y: np.ndarray, X: np.ndarray) -> OLSResult:
    """Vectorized OLS regression: fits Y = X @ beta + residuals for all voxels.

    Parameters
    ----------
    Y : array, shape (n_subjects, n_voxels)
    X : array, shape (n_subjects, n_regressors)

    Returns
    -------
    OLSResult
    """
    n, p = X.shape
    Q, R = np.linalg.qr(X, mode="reduced")
    rank = np.linalg.matrix_rank(R)
    dof = n - rank

    QtY = Q.T @ Y
    beta = linalg.solve_triangular(R, QtY)

    residuals = Y - X @ beta
    sse = np.sum(residuals ** 2, axis=0)

    mse = sse / dof
    R_inv = linalg.solve_triangular(R, np.eye(p))
    var_beta_factor = np.sum(R_inv ** 2, axis=1)
    t_values = beta / np.sqrt(var_beta_factor[:, np.newaxis] * mse[np.newaxis, :])

    return OLSResult(
        beta=beta,
        residuals=residuals,
        t_values=t_values,
        sse=sse,
        dof=dof,
    )


def compute_contrast(
    result: OLSResult,
    X: np.ndarray,
    contrast: np.ndarray,
) -> np.ndarray:
    """Compute T-statistic for a linear contrast.

    T = (c @ beta) / (std_e * sqrt(c @ (X'X)^-1 @ c'))

    Parameters
    ----------
    result : OLSResult from ols_fit
    X : design matrix, shape (n_subjects, n_regressors)
    contrast : array, shape (n_regressors,)

    Returns
    -------
    t_contrast : array, shape (n_voxels,)
    """
    c = np.asarray(contrast, dtype=float)
    c_beta = c @ result.beta

    XtX_inv = np.linalg.inv(X.T @ X)
    denom_factor = np.sqrt(c @ XtX_inv @ c)

    std_e = np.sqrt(result.sse / result.dof)
    t_contrast = c_beta / (std_e * denom_factor)
    return t_contrast
