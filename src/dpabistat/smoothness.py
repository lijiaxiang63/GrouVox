"""Residual smoothness estimation (FWHM, dLh).

Implements the autocorrelation-based method used in FSL and DPABI.
Reference: Flitney & Jenkinson (2000), Worsley et al. (1999).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SmoothnessResult:
    """Smoothness estimation output."""

    fwhm: tuple[float, float, float]
    dlh: float
    resels: float
    n_voxels: int


def estimate_smoothness(
    residuals: np.ndarray,
    mask: np.ndarray,
    dof: int,
    voxel_size: np.ndarray,
) -> SmoothnessResult:
    """Estimate spatial smoothness of residuals within a mask.

    Uses lag-1 autocorrelation in each axis direction, matching the
    FSL/DPABI algorithm.

    Parameters
    ----------
    residuals : array, shape (X, Y, Z, N)
    mask : boolean array, shape (X, Y, Z)
    dof : int
    voxel_size : array, shape (3,)

    Returns
    -------
    SmoothnessResult
    """
    n_voxels = int(mask.sum())

    std = np.std(residuals, axis=-1, ddof=1, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    normed = residuals / std

    ss_minus = np.zeros(3)
    ss_total = np.zeros(3)

    for axis, slices in enumerate(_axis_neighbor_slices()):
        fwd, bwd = slices
        both_in_mask = mask[fwd] & mask[bwd]
        for t in range(residuals.shape[3]):
            a = normed[fwd][..., t][both_in_mask]
            b = normed[bwd][..., t][both_in_mask]
            ss_minus[axis] += np.dot(a, b)
            ss_total[axis] += 0.5 * (np.dot(a, a) + np.dot(b, b))

    sigma_sq = np.zeros(3)
    fwhm = np.zeros(3)

    for i in range(3):
        rho = ss_minus[i] / ss_total[i] if ss_total[i] > 0 else 0.0
        rho = np.clip(rho, 1e-15, 1.0 - 1e-15)
        sigma_sq[i] = -1.0 / (4.0 * np.log(abs(rho)))
        fwhm[i] = np.sqrt(8.0 * np.log(2.0) * sigma_sq[i]) * voxel_size[i]

    dlh = (sigma_sq[0] * sigma_sq[1] * sigma_sq[2]) ** (-0.5) / np.sqrt(8.0)
    dlh = _dof_scale(dlh, dof)
    resels = n_voxels * dlh

    return SmoothnessResult(
        fwhm=tuple(float(f) for f in fwhm),
        dlh=float(dlh),
        resels=float(resels),
        n_voxels=n_voxels,
    )


def _axis_neighbor_slices():
    """Return forward/backward slice pairs for x, y, z neighbor differences."""
    x_fwd = (slice(1, None), slice(None), slice(None))
    x_bwd = (slice(None, -1), slice(None), slice(None))
    y_fwd = (slice(None), slice(1, None), slice(None))
    y_bwd = (slice(None), slice(None, -1), slice(None))
    z_fwd = (slice(None), slice(None), slice(1, None))
    z_bwd = (slice(None), slice(None), slice(None, -1))
    return [(x_fwd, x_bwd), (y_fwd, y_bwd), (z_fwd, z_bwd)]


def _dof_scale(dlh: float, dof: int) -> float:
    """Apply DOF-dependent scaling correction (matching DPABI)."""
    if dof < 6:
        return dlh * 1.1
    if dof > 500:
        return dlh * np.sqrt(1.0321 / dof + 1)

    dof_table = np.array([6, 7, 8, 9, 10, 12, 15, 20, 30, 50, 100, 200, 500])
    scale_table = np.array([
        1.08, 1.07, 1.06, 1.05, 1.04, 1.03, 1.025, 1.02, 1.015, 1.01, 1.005, 1.002, 1.001
    ])
    scale = float(np.interp(dof, dof_table, scale_table))
    return dlh * scale
