# DPABIStat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python package for voxel-wise two-sample t-test with covariates, T-map output, and GRF/FDR multiple comparison correction.

**Architecture:** Pure numpy/scipy vectorized GLM with nibabel for NIfTI I/O. Cell-means parameterization for intuitive `[1, -1]` contrasts. Design matrix `X` is shared across voxels so QR decomposition happens once; all voxels are solved via a single matrix multiply.

**Tech Stack:** Python 3.10+, numpy, scipy, nibabel, pandas, click, pytest. Package managed with uv.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/dpabistat/__init__.py`
- Create: `src/dpabistat/io.py` (empty placeholder)
- Create: `src/dpabistat/glm.py` (empty placeholder)
- Create: `src/dpabistat/ttest.py` (empty placeholder)
- Create: `src/dpabistat/smoothness.py` (empty placeholder)
- Create: `src/dpabistat/correction.py` (empty placeholder)
- Create: `src/dpabistat/cli.py` (empty placeholder)
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "dpabistat"
version = "0.1.0"
description = "Voxel-wise group-level statistical analysis for neuroimaging"
requires-python = ">=3.10"
dependencies = [
    "nibabel>=5.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "pandas>=2.0",
    "click>=8.0",
]

[project.scripts]
dpabistat = "dpabistat.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/dpabistat"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[dependency-groups]
dev = ["pytest>=8.0"]
```

- [ ] **Step 2: Create package init with public API**

`src/dpabistat/__init__.py`:
```python
"""DPABIStat: Voxel-wise group-level statistical analysis for neuroimaging."""

from dpabistat.ttest import two_sample_ttest, TTestResult
from dpabistat.glm import ols_fit, compute_contrast, OLSResult
from dpabistat.correction import grf_correction, fdr_correction, GRFResult, FDRResult
from dpabistat.smoothness import estimate_smoothness, SmoothnessResult
from dpabistat.io import load_images, load_mask, save_nifti

__all__ = [
    "two_sample_ttest",
    "TTestResult",
    "ols_fit",
    "compute_contrast",
    "OLSResult",
    "grf_correction",
    "fdr_correction",
    "GRFResult",
    "FDRResult",
    "estimate_smoothness",
    "SmoothnessResult",
    "load_images",
    "load_mask",
    "save_nifti",
]
```

- [ ] **Step 3: Create empty module placeholders**

Each of these files gets a single docstring so imports don't crash yet:

`src/dpabistat/io.py`:
```python
"""NIfTI image loading and saving utilities."""
```

`src/dpabistat/glm.py`:
```python
"""Vectorized OLS regression engine."""
```

`src/dpabistat/ttest.py`:
```python
"""Two-sample t-test with covariates."""
```

`src/dpabistat/smoothness.py`:
```python
"""Residual smoothness estimation (FWHM, dLh)."""
```

`src/dpabistat/correction.py`:
```python
"""Multiple comparison correction: GRF and FDR."""
```

`src/dpabistat/cli.py`:
```python
"""Command-line interface."""
```

- [ ] **Step 4: Create test conftest with synthetic NIfTI fixtures**

`tests/conftest.py`:
```python
import numpy as np
import nibabel as nib
import pytest
from pathlib import Path


@pytest.fixture
def tmp_nifti_factory(tmp_path):
    """Factory to create synthetic NIfTI files for testing."""

    def _create(data: np.ndarray, filename: str = "test.nii.gz") -> Path:
        affine = np.diag([2.0, 2.0, 2.0, 1.0])  # 2mm isotropic
        img = nib.Nifti1Image(data.astype(np.float32), affine)
        path = tmp_path / filename
        nib.save(img, path)
        return path

    return _create


@pytest.fixture
def synthetic_two_groups(tmp_nifti_factory, tmp_path):
    """Create two groups of 3D NIfTI images with a planted difference.

    Group 1: voxel values drawn from N(10, 1) with a 'hot spot' at [3:6,3:6,3:6] of +5
    Group 2: voxel values drawn from N(10, 1) with no hot spot
    Shape: (10, 10, 10), 8 subjects per group.
    """
    rng = np.random.default_rng(42)
    shape = (10, 10, 10)
    n_per_group = 8

    g1_dir = tmp_path / "group1"
    g2_dir = tmp_path / "group2"
    g1_dir.mkdir()
    g2_dir.mkdir()

    for i in range(n_per_group):
        data1 = rng.normal(10, 1, shape)
        data1[3:6, 3:6, 3:6] += 5  # planted difference
        tmp_nifti_factory(data1, f"group1/sub{i:02d}.nii.gz")

        data2 = rng.normal(10, 1, shape)
        tmp_nifti_factory(data2, f"group2/sub{i:02d}.nii.gz")

    mask = np.ones(shape, dtype=np.float32)
    mask_path = tmp_nifti_factory(mask, "mask.nii.gz")

    return g1_dir, g2_dir, mask_path
```

- [ ] **Step 5: Install package in dev mode and verify**

Run: `cd /Users/jiaxiangli/neuroimaging/mriscript/DPABIStat && uv sync`
Expected: Dependencies install successfully.

Run: `uv run python -c "import dpabistat; print('OK')"`
Expected: Prints `OK` (imports will fail until modules are populated — that's fine for now, we'll fill them in subsequent tasks).

- [ ] **Step 6: Commit scaffolding**

```bash
git init
git add pyproject.toml src/ tests/conftest.py
git commit -m "chore: scaffold dpabistat package with uv"
```

---

### Task 2: NIfTI I/O Module

**Files:**
- Create: `src/dpabistat/io.py`
- Create: `tests/test_io.py`

- [ ] **Step 1: Write failing tests for io.py**

`tests/test_io.py`:
```python
import numpy as np
import nibabel as nib
import pytest
from pathlib import Path

from dpabistat.io import load_images, load_mask, save_nifti


class TestLoadImages:
    def test_load_from_directory(self, tmp_nifti_factory, tmp_path):
        shape = (5, 5, 5)
        d = tmp_path / "imgs"
        d.mkdir()
        for i in range(3):
            tmp_nifti_factory(np.ones(shape) * i, f"imgs/s{i}.nii.gz")

        data, header, affine = load_images(d)
        assert data.shape == (5, 5, 5, 3)
        # files sorted by name, so values should be 0, 1, 2
        assert np.isclose(data[0, 0, 0, 0], 0.0)
        assert np.isclose(data[0, 0, 0, 2], 2.0)

    def test_load_from_list(self, tmp_nifti_factory):
        shape = (5, 5, 5)
        paths = [
            tmp_nifti_factory(np.ones(shape) * i, f"s{i}.nii.gz")
            for i in range(3)
        ]
        data, header, affine = load_images(paths)
        assert data.shape == (5, 5, 5, 3)

    def test_load_4d_file(self, tmp_nifti_factory):
        data_4d = np.random.default_rng(0).normal(size=(5, 5, 5, 4))
        path = tmp_nifti_factory(data_4d, "4d.nii.gz")
        data, header, affine = load_images(path)
        assert data.shape == (5, 5, 5, 4)
        np.testing.assert_allclose(data, data_4d, atol=1e-5)

    def test_shape_mismatch_raises(self, tmp_nifti_factory):
        p1 = tmp_nifti_factory(np.ones((5, 5, 5)), "a.nii.gz")
        p2 = tmp_nifti_factory(np.ones((6, 6, 6)), "b.nii.gz")
        with pytest.raises(ValueError, match="shape"):
            load_images([p1, p2])


class TestLoadMask:
    def test_load_binary_mask(self, tmp_nifti_factory):
        mask_data = np.zeros((5, 5, 5))
        mask_data[1:4, 1:4, 1:4] = 1
        path = tmp_nifti_factory(mask_data, "mask.nii.gz")
        mask = load_mask(path)
        assert mask.dtype == bool
        assert mask.sum() == 27

    def test_none_returns_none(self):
        assert load_mask(None) is None


class TestSaveNifti:
    def test_save_and_reload(self, tmp_path):
        data = np.random.default_rng(0).normal(size=(5, 5, 5)).astype(np.float32)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        header = nib.Nifti1Header()
        out = tmp_path / "out.nii.gz"
        save_nifti(data, affine, header, out, description="test desc")

        img = nib.load(out)
        np.testing.assert_allclose(img.get_fdata(), data, atol=1e-6)
        assert "test desc" in img.header["descrip"].astype(str)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jiaxiangli/neuroimaging/mriscript/DPABIStat && uv run pytest tests/test_io.py -v`
Expected: FAIL — `load_images`, `load_mask`, `save_nifti` not defined.

- [ ] **Step 3: Implement io.py**

`src/dpabistat/io.py`:
```python
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
        header["descrip"] = description[:80]  # NIfTI descrip field is 80 chars max
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_io.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dpabistat/io.py tests/test_io.py
git commit -m "feat: add NIfTI I/O module (load_images, load_mask, save_nifti)"
```

---

### Task 3: Vectorized OLS Engine

**Files:**
- Create: `src/dpabistat/glm.py`
- Create: `tests/test_glm.py`

- [ ] **Step 1: Write failing tests for glm.py**

`tests/test_glm.py`:
```python
import numpy as np
from scipy import stats
import pytest

from dpabistat.glm import ols_fit, compute_contrast, OLSResult


class TestOLSFit:
    def test_simple_regression(self):
        """OLS beta should match known values for y = 2*x + 3 + noise."""
        rng = np.random.default_rng(42)
        n = 50
        x = rng.normal(0, 1, n)
        noise = rng.normal(0, 0.1, n)
        y = 2 * x + 3 + noise

        X = np.column_stack([x, np.ones(n)])
        Y = y.reshape(-1, 1)  # (n_subjects, 1 voxel)

        result = ols_fit(Y, X)
        assert isinstance(result, OLSResult)
        assert result.beta.shape == (2, 1)
        assert result.dof == n - 2
        np.testing.assert_allclose(result.beta[0, 0], 2.0, atol=0.1)
        np.testing.assert_allclose(result.beta[1, 0], 3.0, atol=0.1)

    def test_matches_scipy_ttest(self):
        """For two-group cell-means model without covariates, T should match scipy."""
        rng = np.random.default_rng(99)
        n1, n2 = 15, 15
        n_voxels = 100
        g1 = rng.normal(5, 1, (n1, n_voxels))
        g2 = rng.normal(3, 1, (n2, n_voxels))

        Y = np.vstack([g1, g2])
        G1 = np.array([1]*n1 + [0]*n2, dtype=float)
        G2 = np.array([0]*n1 + [1]*n2, dtype=float)
        X = np.column_stack([G1, G2])

        result = ols_fit(Y, X)
        contrast = np.array([1.0, -1.0])
        t_vals = compute_contrast(result, X, contrast)

        # Compare with scipy independent t-test (equal_var=True)
        scipy_t, _ = stats.ttest_ind(g1, g2, axis=0, equal_var=True)
        np.testing.assert_allclose(t_vals, scipy_t, atol=1e-10)

    def test_residuals_orthogonal_to_design(self):
        """Residuals should be orthogonal to design matrix columns."""
        rng = np.random.default_rng(7)
        n, p, v = 20, 3, 10
        X = rng.normal(size=(n, p))
        Y = rng.normal(size=(n, v))

        result = ols_fit(Y, X)
        # X.T @ residuals should be ~0
        product = X.T @ result.residuals
        np.testing.assert_allclose(product, 0, atol=1e-10)

    def test_multiple_voxels(self):
        """Verify vectorization works across many voxels."""
        rng = np.random.default_rng(0)
        n, v = 30, 500
        X = np.column_stack([rng.normal(size=n), np.ones(n)])
        Y = rng.normal(size=(n, v))

        result = ols_fit(Y, X)
        assert result.beta.shape == (2, v)
        assert result.residuals.shape == (n, v)
        assert result.t_values.shape == (2, v)
        assert result.sse.shape == (v,)


class TestComputeContrast:
    def test_contrast_direction(self):
        """[1, -1] should give positive T when group1 > group2."""
        rng = np.random.default_rng(42)
        n1 = n2 = 20
        g1 = rng.normal(10, 1, (n1, 1))
        g2 = rng.normal(5, 1, (n2, 1))
        Y = np.vstack([g1, g2])
        G1 = np.array([1]*n1 + [0]*n2, dtype=float)
        G2 = np.array([0]*n1 + [1]*n2, dtype=float)
        X = np.column_stack([G1, G2])

        result = ols_fit(Y, X)
        t_pos = compute_contrast(result, X, np.array([1.0, -1.0]))
        t_neg = compute_contrast(result, X, np.array([-1.0, 1.0]))

        assert t_pos[0] > 0
        assert t_neg[0] < 0
        np.testing.assert_allclose(t_pos, -t_neg, atol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_glm.py -v`
Expected: FAIL — `ols_fit`, `compute_contrast`, `OLSResult` not defined.

- [ ] **Step 3: Implement glm.py**

`src/dpabistat/glm.py`:
```python
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
    # QR decomposition (computed once — X is the same for all voxels)
    Q, R = np.linalg.qr(X, mode="reduced")
    rank = np.linalg.matrix_rank(R)
    dof = n - rank

    # Beta: (p, n_voxels) = solve R @ beta = Q.T @ Y
    QtY = Q.T @ Y  # (p, n_voxels)
    beta = linalg.solve_triangular(R, QtY)  # (p, n_voxels)

    # Residuals and SSE
    residuals = Y - X @ beta  # (n, n_voxels)
    sse = np.sum(residuals ** 2, axis=0)  # (n_voxels,)

    # T-values for each regressor
    mse = sse / dof  # (n_voxels,)
    # Variance of beta: diag(R_inv @ R_inv.T) * mse
    R_inv = linalg.solve_triangular(R, np.eye(p))  # (p, p)
    var_beta_factor = np.sum(R_inv ** 2, axis=1)  # (p,) — diagonal of (X'X)^-1
    # se_beta[i, v] = sqrt(var_beta_factor[i] * mse[v])
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
    c_beta = c @ result.beta  # (n_voxels,)

    XtX_inv = np.linalg.inv(X.T @ X)
    denom_factor = np.sqrt(c @ XtX_inv @ c)

    std_e = np.sqrt(result.sse / result.dof)  # (n_voxels,)
    t_contrast = c_beta / (std_e * denom_factor)
    return t_contrast
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_glm.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dpabistat/glm.py tests/test_glm.py
git commit -m "feat: add vectorized OLS engine with contrast T-statistic"
```

---

### Task 4: Smoothness Estimation

**Files:**
- Create: `src/dpabistat/smoothness.py`
- Create: `tests/test_smoothness.py`

- [ ] **Step 1: Write failing tests for smoothness.py**

`tests/test_smoothness.py`:
```python
import numpy as np
import pytest

from dpabistat.smoothness import estimate_smoothness, SmoothnessResult


class TestEstimateSmoothness:
    def test_returns_dataclass(self):
        rng = np.random.default_rng(42)
        shape = (10, 10, 10)
        n_subjects = 20
        residuals = rng.normal(size=(*shape, n_subjects))
        mask = np.ones(shape, dtype=bool)
        voxel_size = np.array([2.0, 2.0, 2.0])

        result = estimate_smoothness(residuals, mask, dof=18, voxel_size=voxel_size)
        assert isinstance(result, SmoothnessResult)
        assert len(result.fwhm) == 3
        assert result.dlh > 0
        assert result.n_voxels == 1000

    def test_smooth_data_has_larger_fwhm(self):
        """Smoothed data should have larger FWHM than white noise."""
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(42)
        shape = (20, 20, 20)
        n_subjects = 15
        voxel_size = np.array([2.0, 2.0, 2.0])
        mask = np.ones(shape, dtype=bool)

        # White noise residuals
        noise = rng.normal(size=(*shape, n_subjects))
        res_noise = estimate_smoothness(noise, mask, dof=13, voxel_size=voxel_size)

        # Smoothed residuals
        smoothed = np.stack(
            [gaussian_filter(noise[..., i], sigma=2.0) for i in range(n_subjects)],
            axis=-1,
        )
        res_smooth = estimate_smoothness(smoothed, mask, dof=13, voxel_size=voxel_size)

        assert all(s > n for s, n in zip(res_smooth.fwhm, res_noise.fwhm))

    def test_mask_restricts_voxels(self):
        rng = np.random.default_rng(42)
        shape = (10, 10, 10)
        residuals = rng.normal(size=(*shape, 10))
        mask = np.zeros(shape, dtype=bool)
        mask[2:8, 2:8, 2:8] = True
        voxel_size = np.array([2.0, 2.0, 2.0])

        result = estimate_smoothness(residuals, mask, dof=8, voxel_size=voxel_size)
        assert result.n_voxels == mask.sum()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_smoothness.py -v`
Expected: FAIL — `estimate_smoothness`, `SmoothnessResult` not defined.

- [ ] **Step 3: Implement smoothness.py**

`src/dpabistat/smoothness.py`:
```python
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

    fwhm: tuple[float, float, float]  # Full Width at Half Maximum in mm
    dlh: float                         # resel density (sqrt(det(Lambda))^-1 factor)
    resels: float                      # number of resolution elements
    n_voxels: int                      # number of in-mask voxels


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
        Residual images from OLS fit.
    mask : boolean array, shape (X, Y, Z)
        Brain mask.
    dof : int
        Degrees of freedom of the residuals.
    voxel_size : array, shape (3,)
        Voxel dimensions in mm.

    Returns
    -------
    SmoothnessResult
    """
    n_voxels = int(mask.sum())

    # Normalize residuals to unit variance per voxel (across subjects)
    std = np.std(residuals, axis=-1, ddof=1, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    normed = residuals / std

    # Lag-1 autocorrelation products and sum-of-squares for each axis
    ss_minus = np.zeros(3)
    ss_total = np.zeros(3)

    for axis, slices in enumerate(_axis_neighbor_slices()):
        fwd, bwd = slices
        # Both voxels must be in mask
        both_in_mask = mask[fwd] & mask[bwd]
        for t in range(residuals.shape[3]):
            a = normed[fwd][..., t][both_in_mask]
            b = normed[bwd][..., t][both_in_mask]
            ss_minus[axis] += np.dot(a, b)
            ss_total[axis] += 0.5 * (np.dot(a, a) + np.dot(b, b))

    # Autocorrelation and smoothness per axis
    sigma_sq = np.zeros(3)
    fwhm = np.zeros(3)

    for i in range(3):
        rho = ss_minus[i] / ss_total[i] if ss_total[i] > 0 else 0.0
        rho = np.clip(rho, 1e-15, 1.0 - 1e-15)
        sigma_sq[i] = -1.0 / (4.0 * np.log(abs(rho)))
        fwhm[i] = np.sqrt(8.0 * np.log(2.0) * sigma_sq[i]) * voxel_size[i]

    # Resel density: dLh
    dlh = (sigma_sq[0] * sigma_sq[1] * sigma_sq[2]) ** (-0.5) / np.sqrt(8.0)

    # Apply DOF-dependent scaling (matching DPABI lookup table)
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
    # x-axis: compare voxel (x, y, z) with (x-1, y, z)
    x_fwd = (slice(1, None), slice(None), slice(None))
    x_bwd = (slice(None, -1), slice(None), slice(None))
    # y-axis
    y_fwd = (slice(None), slice(1, None), slice(None))
    y_bwd = (slice(None), slice(None, -1), slice(None))
    # z-axis
    z_fwd = (slice(None), slice(None), slice(1, None))
    z_bwd = (slice(None), slice(None), slice(None, -1))
    return [(x_fwd, x_bwd), (y_fwd, y_bwd), (z_fwd, z_bwd)]


def _dof_scale(dlh: float, dof: int) -> float:
    """Apply DOF-dependent scaling correction (matching DPABI).

    For low DOF, smoothness tends to be overestimated; this scaling
    corrects for that bias.
    """
    if dof < 6:
        return dlh * 1.1
    if dof > 500:
        return dlh * np.sqrt(1.0321 / dof + 1)

    # Lookup table from DPABI (interpolated from FSL)
    dof_table = np.array([6, 7, 8, 9, 10, 12, 15, 20, 30, 50, 100, 200, 500])
    scale_table = np.array([
        1.08, 1.07, 1.06, 1.05, 1.04, 1.03, 1.025, 1.02, 1.015, 1.01, 1.005, 1.002, 1.001
    ])
    scale = float(np.interp(dof, dof_table, scale_table))
    return dlh * scale
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_smoothness.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dpabistat/smoothness.py tests/test_smoothness.py
git commit -m "feat: add residual smoothness estimation (FWHM, dLh)"
```

---

### Task 5: Two-Sample T-Test

**Files:**
- Create: `src/dpabistat/ttest.py`
- Create: `tests/test_ttest.py`

- [ ] **Step 1: Write failing tests for ttest.py**

`tests/test_ttest.py`:
```python
import numpy as np
import pandas as pd
import nibabel as nib
import pytest
from pathlib import Path
from scipy import stats

from dpabistat.ttest import two_sample_ttest, TTestResult


class TestTwoSampleTTest:
    def test_basic_two_groups(self, synthetic_two_groups, tmp_path):
        g1_dir, g2_dir, mask_path = synthetic_two_groups
        output = tmp_path / "results" / "ttest"

        result = two_sample_ttest(
            group1=g1_dir,
            group2=g2_dir,
            output=output,
            mask=mask_path,
        )

        assert isinstance(result, TTestResult)
        assert result.t_map.shape == (10, 10, 10)
        assert result.dof == 14  # 8 + 8 - 2

        # Hot spot region should have high positive T
        hotspot_t = result.t_map[3:6, 3:6, 3:6].mean()
        background_t = np.abs(result.t_map[0, 0, 0])
        assert hotspot_t > 3.0
        assert hotspot_t > background_t

    def test_output_files_created(self, synthetic_two_groups, tmp_path):
        g1_dir, g2_dir, mask_path = synthetic_two_groups
        output = tmp_path / "results" / "ttest"

        two_sample_ttest(g1_dir, g2_dir, output, mask=mask_path)

        assert (tmp_path / "results" / "ttest_T.nii.gz").exists()
        assert (tmp_path / "results" / "ttest_beta.nii.gz").exists()
        assert (tmp_path / "results" / "ttest_cohen_f2.nii.gz").exists()

    def test_header_has_metadata(self, synthetic_two_groups, tmp_path):
        g1_dir, g2_dir, mask_path = synthetic_two_groups
        output = tmp_path / "results" / "ttest"

        result = two_sample_ttest(g1_dir, g2_dir, output, mask=mask_path)

        descrip = result.header["descrip"].astype(str)
        assert "DPABIStat{T_" in descrip
        assert "dLh_" in descrip
        assert "FWHM" in descrip

    def test_with_covariates(self, synthetic_two_groups, tmp_path):
        g1_dir, g2_dir, mask_path = synthetic_two_groups
        output = tmp_path / "results" / "ttest_cov"

        covars = pd.DataFrame({
            "age": np.random.default_rng(0).normal(30, 5, 16),
            "sex": [0, 1] * 8,
        })

        result = two_sample_ttest(
            g1_dir, g2_dir, output, mask=mask_path, covariates=covars,
        )

        assert result.dof == 12  # 16 - 2 (groups) - 2 (covariates)
        assert result.beta_maps.shape[-1] == 4  # G1, G2, age, sex

    def test_covariates_from_csv(self, synthetic_two_groups, tmp_path):
        g1_dir, g2_dir, mask_path = synthetic_two_groups
        output = tmp_path / "results" / "ttest_csv"

        csv_path = tmp_path / "covars.csv"
        covars = pd.DataFrame({
            "age": np.random.default_rng(0).normal(30, 5, 16),
            "sex": [0, 1] * 8,
        })
        covars.to_csv(csv_path, index=False)

        result = two_sample_ttest(
            g1_dir, g2_dir, output, mask=mask_path, covariates=csv_path,
        )
        assert result.dof == 12

    def test_contrast_direction(self, synthetic_two_groups, tmp_path):
        g1_dir, g2_dir, mask_path = synthetic_two_groups

        r1 = two_sample_ttest(
            g1_dir, g2_dir, tmp_path / "r1", mask=mask_path,
            contrast=[1, -1],
        )
        r2 = two_sample_ttest(
            g1_dir, g2_dir, tmp_path / "r2", mask=mask_path,
            contrast=[-1, 1],
        )

        np.testing.assert_allclose(r1.t_map, -r2.t_map, atol=1e-5)

    def test_no_mask_uses_all_voxels(self, synthetic_two_groups, tmp_path):
        g1_dir, g2_dir, _ = synthetic_two_groups
        output = tmp_path / "results" / "ttest_nomask"

        result = two_sample_ttest(g1_dir, g2_dir, output)
        assert result.t_map.shape == (10, 10, 10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ttest.py -v`
Expected: FAIL — `two_sample_ttest`, `TTestResult` not defined.

- [ ] **Step 3: Implement ttest.py**

`src/dpabistat/ttest.py`:
```python
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


@dataclass
class TTestResult:
    """Result of a two-sample t-test."""

    t_map: np.ndarray
    beta_maps: np.ndarray
    cohen_f2_map: np.ndarray
    header: nib.Nifti1Header
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
    group1, group2 : path(s) to NIfTI files (3D per subject, or 4D).
    output : output path prefix (e.g., "results/ttest").
    mask : optional brain mask NIfTI path.
    covariates : optional CSV path or DataFrame. Rows = all subjects
        (group1 first, then group2). Columns = covariate names.
    contrast : [c1, c2] for group terms. Default [1, -1].

    Returns
    -------
    TTestResult
    """
    output = Path(output)

    # Load data
    data1, header, affine = load_images(group1)
    data2, _, _ = load_images(group2)
    n1 = data1.shape[3]
    n2 = data2.shape[3]
    n_total = n1 + n2

    # Stack: (X, Y, Z, N)
    data_4d = np.concatenate([data1, data2], axis=3)
    vol_shape = data_4d.shape[:3]

    # Mask
    mask_3d = load_mask(mask)
    if mask_3d is None:
        mask_3d = np.any(data_4d != 0, axis=3)
    mask_indices = mask_3d.ravel().nonzero()[0]

    # Flatten to (N, n_voxels)
    Y = data_4d.reshape(-1, n_total).T  # (n_total, n_all_voxels)
    Y = Y[:, mask_indices]  # (n_total, n_mask_voxels)

    # Design matrix: cell-means model
    G1 = np.array([1.0] * n1 + [0.0] * n2)
    G2 = np.array([0.0] * n1 + [1.0] * n2)
    X = np.column_stack([G1, G2])

    # Covariates
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

    # Contrast
    if contrast is None:
        contrast = [1.0, -1.0]
    full_contrast = np.zeros(X.shape[1])
    full_contrast[: len(contrast)] = contrast

    # Fit OLS
    result = ols_fit(Y, X)

    # Contrast T-statistic
    t_contrast = compute_contrast(result, X, full_contrast)

    # Cohen's f² effect size: (SSE_reduced - SSE_full) / SSE_full
    # Reduced model: remove group columns, keep only covariates + intercept
    cohen_f2 = _compute_cohen_f2(Y, X, result.sse)

    # Reconstruct 3D volumes
    t_map = _unmask(t_contrast, mask_indices, vol_shape)
    beta_maps = np.zeros((*vol_shape, X.shape[1]), dtype=np.float32)
    for i in range(X.shape[1]):
        beta_maps[..., i] = _unmask(result.beta[i], mask_indices, vol_shape)
    cohen_f2_map = _unmask(cohen_f2, mask_indices, vol_shape)

    # Reconstruct residuals for smoothness estimation
    residuals_4d = np.zeros((*vol_shape, n_total), dtype=np.float32)
    for i in range(n_total):
        residuals_4d[..., i] = _unmask(result.residuals[i], mask_indices, vol_shape)

    # Smoothness estimation
    voxel_size = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    smooth = estimate_smoothness(residuals_4d, mask_3d, result.dof, voxel_size)

    # Header metadata
    header = header.copy()
    header["descrip"] = (
        f"DPABIStat{{T_[{result.dof:.1f}]}}"
        f"{{dLh_{smooth.dlh:.6f}}}"
        f"{{FWHMx_{smooth.fwhm[0]:.4f} "
        f"FWHMy_{smooth.fwhm[1]:.4f} "
        f"FWHMz_{smooth.fwhm[2]:.4f} mm}}"
    )[:80]

    # Save outputs
    save_nifti(t_map, affine, header, f"{output}_T.nii.gz")
    save_nifti(beta_maps, affine, header, f"{output}_beta.nii.gz")
    save_nifti(cohen_f2_map, affine, header, f"{output}_cohen_f2.nii.gz")

    return TTestResult(
        t_map=t_map,
        beta_maps=beta_maps,
        cohen_f2_map=cohen_f2_map,
        header=header,
        affine=affine,
        dof=result.dof,
        fwhm=smooth.fwhm,
        dlh=smooth.dlh,
    )


def _compute_cohen_f2(
    Y: np.ndarray, X: np.ndarray, sse_full: np.ndarray
) -> np.ndarray:
    """Cohen's f² = (SSE_reduced - SSE_full) / SSE_full.

    Reduced model: intercept + covariates only (no group effect).
    Uses a single intercept column for the reduced model.
    """
    n = X.shape[1]
    if n <= 2:
        # No covariates — reduced model is intercept-only
        X_reduced = np.ones((Y.shape[0], 1))
    else:
        # Reduced: intercept + covariate columns (drop G1, G2, add intercept)
        X_reduced = np.column_stack([np.ones(Y.shape[0]), X[:, 2:]])

    Q, R = np.linalg.qr(X_reduced, mode="reduced")
    from scipy import linalg as la

    beta_r = la.solve_triangular(R, Q.T @ Y)
    residuals_r = Y - X_reduced @ beta_r
    sse_reduced = np.sum(residuals_r ** 2, axis=0)

    # Avoid division by zero
    safe_sse = np.where(sse_full > 0, sse_full, 1.0)
    return (sse_reduced - sse_full) / safe_sse


def _unmask(
    flat_data: np.ndarray, mask_indices: np.ndarray, vol_shape: tuple
) -> np.ndarray:
    """Place flat masked data back into a 3D volume."""
    vol = np.zeros(np.prod(vol_shape), dtype=np.float32)
    vol[mask_indices] = flat_data
    return vol.reshape(vol_shape)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ttest.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dpabistat/ttest.py tests/test_ttest.py
git commit -m "feat: add two-sample t-test with covariates and Cohen's f²"
```

---

### Task 6: FDR Correction

**Files:**
- Create: `src/dpabistat/correction.py`
- Create: `tests/test_correction.py`

- [ ] **Step 1: Write failing tests for FDR correction**

`tests/test_correction.py`:
```python
import numpy as np
import nibabel as nib
import pytest
from pathlib import Path

from dpabistat.correction import fdr_correction, grf_correction, FDRResult, GRFResult


def _make_stat_nifti(data, tmp_path, filename="stat_T.nii.gz", dof=14, dlh=0.01,
                     fwhm=(8.0, 8.0, 8.0)):
    """Helper: create a T-stat NIfTI with DPABI-style header metadata."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    header = nib.Nifti1Header()
    header.set_data_dtype(np.float32)
    header["descrip"] = (
        f"DPABIStat{{T_[{dof:.1f}]}}{{dLh_{dlh:.6f}}}"
        f"{{FWHMx_{fwhm[0]:.4f} FWHMy_{fwhm[1]:.4f} FWHMz_{fwhm[2]:.4f} mm}}"
    )[:80]
    img = nib.Nifti1Image(data.astype(np.float32), affine, header)
    path = tmp_path / filename
    nib.save(img, path)
    return path


def _make_mask(shape, tmp_path, filename="mask.nii.gz"):
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    mask = np.ones(shape, dtype=np.float32)
    img = nib.Nifti1Image(mask, affine)
    path = tmp_path / filename
    nib.save(img, path)
    return path


class TestFDRCorrection:
    def test_basic_fdr(self, tmp_path):
        """FDR should retain voxels with large T and remove small T."""
        rng = np.random.default_rng(42)
        shape = (20, 20, 20)
        data = rng.normal(0, 1, shape).astype(np.float32)
        # Plant strong signal
        data[8:12, 8:12, 8:12] = 6.0

        stat_path = _make_stat_nifti(data, tmp_path, dof=28)
        mask_path = _make_mask(shape, tmp_path)

        result = fdr_correction(stat_path, mask_path, q=0.05)

        assert isinstance(result, FDRResult)
        assert result.n_significant > 0
        # The planted signal voxels should survive
        assert result.thresholded_map[9, 9, 9] != 0.0
        # Most background voxels should be zeroed
        assert result.thresholded_map[0, 0, 0] == 0.0

    def test_fdr_output_file(self, tmp_path):
        shape = (10, 10, 10)
        data = np.zeros(shape, dtype=np.float32)
        data[4:6, 4:6, 4:6] = 5.0
        stat_path = _make_stat_nifti(data, tmp_path, filename="mystat_T.nii.gz", dof=20)
        mask_path = _make_mask(shape, tmp_path)

        fdr_correction(stat_path, mask_path, q=0.05)

        expected_out = tmp_path / "FDR_Thresholded_mystat_T.nii.gz"
        assert expected_out.exists()

    def test_fdr_no_signal(self, tmp_path):
        """With pure noise, FDR should find few or no significant voxels."""
        rng = np.random.default_rng(42)
        shape = (15, 15, 15)
        data = rng.normal(0, 1, shape).astype(np.float32)
        stat_path = _make_stat_nifti(data, tmp_path, dof=28)
        mask_path = _make_mask(shape, tmp_path)

        result = fdr_correction(stat_path, mask_path, q=0.05)
        # Very few or no voxels should survive in pure noise
        assert result.n_significant < shape[0] * shape[1] * shape[2] * 0.1


class TestGRFCorrection:
    def test_basic_grf(self, tmp_path):
        """GRF should retain large clusters and remove isolated voxels."""
        shape = (30, 30, 30)
        data = np.zeros(shape, dtype=np.float32)
        # Large cluster of high T-values
        data[10:20, 10:20, 10:20] = 5.0
        # Single isolated high voxel (should be removed by cluster threshold)
        data[2, 2, 2] = 5.0

        stat_path = _make_stat_nifti(data, tmp_path, dof=28, dlh=0.005,
                                     fwhm=(6.0, 6.0, 6.0))
        mask_path = _make_mask(shape, tmp_path)

        result = grf_correction(
            stat_path, mask_path, voxel_p=0.001, cluster_p=0.05,
            two_tailed=False,
        )

        assert isinstance(result, GRFResult)
        assert result.z_threshold > 0
        assert result.cluster_size_threshold > 1
        # Large cluster should survive
        assert result.thresholded_t_map[15, 15, 15] != 0
        # Isolated voxel should be removed
        assert result.thresholded_t_map[2, 2, 2] == 0

    def test_grf_output_files(self, tmp_path):
        shape = (20, 20, 20)
        data = np.zeros(shape, dtype=np.float32)
        data[8:12, 8:12, 8:12] = 5.0
        stat_path = _make_stat_nifti(data, tmp_path, filename="res_T.nii.gz",
                                     dof=20, dlh=0.005)
        mask_path = _make_mask(shape, tmp_path)

        grf_correction(stat_path, mask_path, voxel_p=0.001, cluster_p=0.05)

        parent = tmp_path
        assert (parent / "Z_ClusterThresholded_res_T.nii.gz").exists()
        assert (parent / "ClusterThresholded_res_T.nii.gz").exists()

    def test_grf_two_tailed(self, tmp_path):
        """Two-tailed GRF should retain both positive and negative clusters."""
        shape = (30, 30, 30)
        data = np.zeros(shape, dtype=np.float32)
        data[5:12, 5:12, 5:12] = 5.0    # positive cluster
        data[18:25, 18:25, 18:25] = -5.0  # negative cluster

        stat_path = _make_stat_nifti(data, tmp_path, dof=28, dlh=0.005)
        mask_path = _make_mask(shape, tmp_path)

        result = grf_correction(
            stat_path, mask_path, voxel_p=0.001, cluster_p=0.05,
            two_tailed=True,
        )

        assert result.thresholded_t_map[8, 8, 8] > 0
        assert result.thresholded_t_map[21, 21, 21] < 0

    def test_grf_cluster_table(self, tmp_path):
        shape = (30, 30, 30)
        data = np.zeros(shape, dtype=np.float32)
        data[10:20, 10:20, 10:20] = 5.0

        stat_path = _make_stat_nifti(data, tmp_path, dof=28, dlh=0.005)
        mask_path = _make_mask(shape, tmp_path)

        result = grf_correction(
            stat_path, mask_path, voxel_p=0.001, cluster_p=0.05,
            two_tailed=False,
        )

        assert len(result.cluster_table) > 0
        cluster = result.cluster_table[0]
        assert "label" in cluster
        assert "size" in cluster
        assert "peak_value" in cluster
        assert "peak_coords" in cluster
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_correction.py -v`
Expected: FAIL — `fdr_correction`, `grf_correction`, `FDRResult`, `GRFResult` not defined.

- [ ] **Step 3: Implement correction.py**

`src/dpabistat/correction.py`:
```python
"""Multiple comparison correction: GRF and FDR."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage, stats
from scipy.special import gamma

from dpabistat.io import load_mask, save_nifti


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FDRResult:
    """Result of FDR correction."""

    p_threshold: float
    n_significant: int
    thresholded_map: np.ndarray


@dataclass
class GRFResult:
    """Result of GRF cluster-level correction."""

    z_threshold: float
    cluster_size_threshold: int
    thresholded_z_map: np.ndarray
    thresholded_t_map: np.ndarray
    n_clusters: int
    cluster_table: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Header metadata parsing
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(
    r"DPABIStat\{T_\[([0-9.]+)\]\}"
    r"\{dLh_([0-9.eE+-]+)\}"
    r"\{FWHMx_([0-9.]+)\s+FWHMy_([0-9.]+)\s+FWHMz_([0-9.]+)\s+mm\}"
)


def _parse_header_meta(header: nib.Nifti1Header) -> dict:
    """Extract DOF, dLh, FWHM from NIfTI header description."""
    descrip = header["descrip"].astype(str)
    m = _HEADER_RE.search(descrip)
    if not m:
        raise ValueError(
            f"Cannot parse DPABIStat metadata from header description: {descrip!r}. "
            "Run two_sample_ttest first to generate a compatible T-map."
        )
    return {
        "dof": float(m.group(1)),
        "dlh": float(m.group(2)),
        "fwhm": (float(m.group(3)), float(m.group(4)), float(m.group(5))),
    }


# ---------------------------------------------------------------------------
# FDR Correction (Benjamini-Hochberg)
# ---------------------------------------------------------------------------


def fdr_correction(
    stat_path: str | Path,
    mask_path: str | Path | None = None,
    q: float = 0.05,
    two_tailed: bool = True,
) -> FDRResult:
    """Apply FDR (Benjamini-Hochberg) correction to a T-statistic map.

    Parameters
    ----------
    stat_path : path to the T-statistic NIfTI file.
    mask_path : optional brain mask.
    q : FDR threshold (default 0.05).
    two_tailed : if True, use two-tailed p-values.

    Returns
    -------
    FDRResult
    """
    stat_path = Path(stat_path)
    img = nib.load(stat_path)
    data = img.get_fdata(dtype=np.float32)
    header = img.header
    affine = img.affine

    meta = _parse_header_meta(header)
    dof = meta["dof"]

    mask = load_mask(mask_path)
    if mask is None:
        mask = np.ones(data.shape, dtype=bool)

    # Extract in-mask values and compute p-values
    t_vals = data[mask]
    if two_tailed:
        p_vals = 2.0 * stats.t.sf(np.abs(t_vals), dof)
    else:
        p_vals = stats.t.sf(t_vals, dof)

    # Benjamini-Hochberg procedure
    m = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    sorted_p = p_vals[sorted_idx]

    thresholds = np.arange(1, m + 1) / m * q
    below = sorted_p <= thresholds

    if not np.any(below):
        # No significant voxels
        thresholded = np.zeros_like(data)
        _save_fdr_output(thresholded, affine, header, stat_path)
        return FDRResult(p_threshold=0.0, n_significant=0, thresholded_map=thresholded)

    # Find the largest k where P_k <= k/m * q
    k_max = np.max(np.where(below)[0])
    p_threshold = sorted_p[k_max]

    # Threshold the map
    significant = p_vals <= p_threshold
    mask_flat = np.zeros(np.prod(data.shape), dtype=bool)
    mask_flat[mask.ravel().nonzero()[0]] = significant
    sig_mask = mask_flat.reshape(data.shape)

    thresholded = data * sig_mask

    _save_fdr_output(thresholded, affine, header, stat_path)

    return FDRResult(
        p_threshold=float(p_threshold),
        n_significant=int(significant.sum()),
        thresholded_map=thresholded,
    )


def _save_fdr_output(data, affine, header, stat_path):
    out_path = stat_path.parent / f"FDR_Thresholded_{stat_path.name}"
    save_nifti(data, affine, header, out_path)


# ---------------------------------------------------------------------------
# GRF Correction (Friston et al. 1994)
# ---------------------------------------------------------------------------

# 26-connectivity structuring element for 3D
_STRUCT_26 = ndimage.generate_binary_structure(3, 3)


def grf_correction(
    stat_path: str | Path,
    mask_path: str | Path | None = None,
    voxel_p: float = 0.001,
    cluster_p: float = 0.05,
    two_tailed: bool = True,
) -> GRFResult:
    """Apply GRF cluster-level correction to a T-statistic map.

    Parameters
    ----------
    stat_path : path to the T-statistic NIfTI file.
    mask_path : optional brain mask.
    voxel_p : voxel-level p-value threshold (default 0.001).
    cluster_p : cluster-level p-value threshold (default 0.05).
    two_tailed : if True, threshold both tails.

    Returns
    -------
    GRFResult
    """
    stat_path = Path(stat_path)
    img = nib.load(stat_path)
    t_data = img.get_fdata(dtype=np.float32)
    header = img.header
    affine = img.affine

    meta = _parse_header_meta(header)
    dof = meta["dof"]
    dlh = meta["dlh"]

    mask = load_mask(mask_path)
    if mask is None:
        mask = np.ones(t_data.shape, dtype=bool)
    n_voxels = int(mask.sum())

    # Convert T -> Z (preserving sign)
    z_data = _t_to_z(t_data, dof)
    z_data = z_data * mask

    # Voxel-level Z threshold
    if two_tailed:
        z_thr = stats.norm.ppf(1.0 - voxel_p / 2.0)
    else:
        z_thr = stats.norm.ppf(1.0 - voxel_p)

    # Cluster size threshold (GRF theory, Friston et al. 1994)
    cluster_size_thr = _grf_cluster_threshold(
        n_voxels, dlh, z_thr, cluster_p, D=3,
    )

    # Apply thresholds
    thresholded_z = np.zeros_like(z_data)
    thresholded_t = np.zeros_like(t_data)
    cluster_table = []

    # Positive tail
    pos_clusters, pos_table = _threshold_tail(
        z_data, t_data, z_thr, cluster_size_thr, positive=True,
    )
    thresholded_z += pos_clusters[0]
    thresholded_t += pos_clusters[1]
    cluster_table.extend(pos_table)

    # Negative tail (two-tailed only)
    if two_tailed:
        neg_clusters, neg_table = _threshold_tail(
            z_data, t_data, z_thr, cluster_size_thr, positive=False,
        )
        thresholded_z += neg_clusters[0]
        thresholded_t += neg_clusters[1]
        cluster_table.extend(neg_table)

    # Sort cluster table by size descending
    cluster_table.sort(key=lambda c: c["size"], reverse=True)

    # Save outputs
    parent = stat_path.parent
    name = stat_path.name
    save_nifti(thresholded_z, affine, header, parent / f"Z_ClusterThresholded_{name}")
    save_nifti(thresholded_t, affine, header, parent / f"ClusterThresholded_{name}")

    return GRFResult(
        z_threshold=float(z_thr),
        cluster_size_threshold=int(cluster_size_thr),
        thresholded_z_map=thresholded_z,
        thresholded_t_map=thresholded_t,
        n_clusters=len(cluster_table),
        cluster_table=cluster_table,
    )


def _t_to_z(t_data: np.ndarray, dof: float) -> np.ndarray:
    """Convert T-statistic to Z-score, preserving sign."""
    # p = P(T > |t|) for each voxel
    p = stats.t.sf(np.abs(t_data), dof)
    # Clamp to avoid inf
    p = np.clip(p, 1e-300, 1.0 - 1e-15)
    z = stats.norm.ppf(1.0 - p)
    return z * np.sign(t_data)


def _grf_cluster_threshold(
    n_voxels: int,
    dlh: float,
    z_thr: float,
    cluster_p: float,
    D: int = 3,
) -> int:
    """Calculate the minimum cluster size using GRF theory.

    Friston et al. (1994) formula for expected number of clusters
    and extreme-value cluster size distribution.
    """
    # Expected number of clusters above threshold
    Em = (
        n_voxels
        * (2 * np.pi) ** (-(D + 1) / 2.0)
        * dlh
        * (z_thr**2 - 1) ** ((D - 1) / 2.0)
        * np.exp(-(z_thr**2) / 2.0)
    )

    # Expected number of suprathreshold voxels
    EN = n_voxels * stats.norm.sf(z_thr)

    if Em <= 0 or EN <= 0:
        return 1

    # Shape parameter beta
    beta_param = (gamma(D / 2.0 + 1) * Em / EN) ** (2.0 / D)

    # Find minimum cluster size K such that P(cluster >= K) <= cluster_p
    cluster_size = 0
    p_temp = 1.0
    while p_temp >= cluster_p and cluster_size < n_voxels:
        cluster_size += 1
        p_temp = 1.0 - np.exp(-Em * np.exp(-beta_param * cluster_size ** (2.0 / D)))

    return max(cluster_size, 1)


def _threshold_tail(
    z_data: np.ndarray,
    t_data: np.ndarray,
    z_thr: float,
    min_cluster_size: int,
    positive: bool,
) -> tuple[tuple[np.ndarray, np.ndarray], list[dict]]:
    """Threshold one tail, remove small clusters, return surviving data + table."""
    if positive:
        suprathreshold = z_data >= z_thr
    else:
        suprathreshold = z_data <= -z_thr

    labeled, n_labels = ndimage.label(suprathreshold, structure=_STRUCT_26)

    z_out = np.zeros_like(z_data)
    t_out = np.zeros_like(t_data)
    table = []

    for label_id in range(1, n_labels + 1):
        cluster_mask = labeled == label_id
        size = int(cluster_mask.sum())
        if size >= min_cluster_size:
            z_out[cluster_mask] = z_data[cluster_mask]
            t_out[cluster_mask] = t_data[cluster_mask]

            # Peak info
            cluster_z = np.abs(z_data[cluster_mask])
            peak_idx = np.argmax(cluster_z)
            peak_coords = np.array(np.where(cluster_mask)).T[peak_idx]
            peak_val = z_data[cluster_mask][peak_idx]

            table.append({
                "label": label_id,
                "size": size,
                "peak_value": float(peak_val),
                "peak_coords": tuple(int(c) for c in peak_coords),
            })

    return (z_out, t_out), table
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_correction.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dpabistat/correction.py tests/test_correction.py
git commit -m "feat: add GRF and FDR multiple comparison correction"
```

---

### Task 7: CLI

**Files:**
- Create: `src/dpabistat/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for CLI**

`tests/test_cli.py`:
```python
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from pathlib import Path

from dpabistat.cli import main


@pytest.fixture
def cli_runner():
    return CliRunner()


class TestTTest2Command:
    def test_basic_invocation(self, cli_runner, synthetic_two_groups, tmp_path):
        g1_dir, g2_dir, mask_path = synthetic_two_groups
        output = str(tmp_path / "cli_results" / "ttest")

        result = cli_runner.invoke(main, [
            "ttest2",
            "--group1", str(g1_dir),
            "--group2", str(g2_dir),
            "--output", output,
            "--mask", str(mask_path),
        ])

        assert result.exit_code == 0, result.output
        assert (tmp_path / "cli_results" / "ttest_T.nii.gz").exists()

    def test_with_covariates(self, cli_runner, synthetic_two_groups, tmp_path):
        g1_dir, g2_dir, mask_path = synthetic_two_groups
        output = str(tmp_path / "cli_results" / "ttest_cov")

        csv_path = tmp_path / "covars.csv"
        pd.DataFrame({
            "age": np.random.default_rng(0).normal(30, 5, 16),
            "sex": [0, 1] * 8,
        }).to_csv(csv_path, index=False)

        result = cli_runner.invoke(main, [
            "ttest2",
            "--group1", str(g1_dir),
            "--group2", str(g2_dir),
            "--output", output,
            "--mask", str(mask_path),
            "--covariates", str(csv_path),
        ])

        assert result.exit_code == 0, result.output


class TestCorrectCommand:
    def test_fdr(self, cli_runner, synthetic_two_groups, tmp_path):
        g1_dir, g2_dir, mask_path = synthetic_two_groups
        output = str(tmp_path / "res" / "ttest")

        # First run t-test to get a stat map
        cli_runner.invoke(main, [
            "ttest2",
            "--group1", str(g1_dir),
            "--group2", str(g2_dir),
            "--output", output,
            "--mask", str(mask_path),
        ])

        stat_file = tmp_path / "res" / "ttest_T.nii.gz"
        result = cli_runner.invoke(main, [
            "correct",
            "--input", str(stat_file),
            "--method", "fdr",
            "--q", "0.05",
            "--mask", str(mask_path),
        ])

        assert result.exit_code == 0, result.output

    def test_grf(self, cli_runner, synthetic_two_groups, tmp_path):
        g1_dir, g2_dir, mask_path = synthetic_two_groups
        output = str(tmp_path / "res" / "ttest")

        cli_runner.invoke(main, [
            "ttest2",
            "--group1", str(g1_dir),
            "--group2", str(g2_dir),
            "--output", output,
            "--mask", str(mask_path),
        ])

        stat_file = tmp_path / "res" / "ttest_T.nii.gz"
        result = cli_runner.invoke(main, [
            "correct",
            "--input", str(stat_file),
            "--method", "grf",
            "--voxel-p", "0.001",
            "--cluster-p", "0.05",
            "--mask", str(mask_path),
        ])

        assert result.exit_code == 0, result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `main` not defined.

- [ ] **Step 3: Implement cli.py**

`src/dpabistat/cli.py`:
```python
"""Command-line interface for DPABIStat."""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
def main():
    """DPABIStat: Voxel-wise group-level statistical analysis for neuroimaging."""


@main.command()
@click.option("--group1", required=True, type=click.Path(exists=True),
              help="Directory or 4D NIfTI for group 1.")
@click.option("--group2", required=True, type=click.Path(exists=True),
              help="Directory or 4D NIfTI for group 2.")
@click.option("--output", required=True, type=click.Path(),
              help="Output path prefix (e.g., results/ttest).")
@click.option("--mask", default=None, type=click.Path(exists=True),
              help="Brain mask NIfTI file.")
@click.option("--covariates", default=None, type=click.Path(exists=True),
              help="CSV file with covariates (rows=subjects, group1 first).")
@click.option("--contrast", default=None, type=float, nargs=2,
              help="Contrast for group terms, e.g., --contrast 1 -1.")
def ttest2(group1, group2, output, mask, covariates, contrast):
    """Run a two-sample t-test with optional covariates."""
    from dpabistat.ttest import two_sample_ttest

    contrast_list = list(contrast) if contrast else None

    result = two_sample_ttest(
        group1=group1,
        group2=group2,
        output=output,
        mask=mask,
        covariates=covariates,
        contrast=contrast_list,
    )

    click.echo(f"T-map saved: {output}_T.nii.gz")
    click.echo(f"DOF: {result.dof}")
    click.echo(f"FWHM: {result.fwhm[0]:.2f} x {result.fwhm[1]:.2f} x {result.fwhm[2]:.2f} mm")


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="T-statistic NIfTI file from ttest2.")
@click.option("--method", required=True, type=click.Choice(["grf", "fdr"]),
              help="Correction method: grf or fdr.")
@click.option("--mask", default=None, type=click.Path(exists=True),
              help="Brain mask NIfTI file.")
@click.option("--voxel-p", default=0.001, type=float,
              help="Voxel-level p threshold (GRF only, default 0.001).")
@click.option("--cluster-p", default=0.05, type=float,
              help="Cluster-level p threshold (GRF only, default 0.05).")
@click.option("--q", "q_value", default=0.05, type=float,
              help="FDR q threshold (FDR only, default 0.05).")
@click.option("--two-tailed/--one-tailed", default=True,
              help="Two-tailed test (default) or one-tailed.")
def correct(input_path, method, mask, voxel_p, cluster_p, q_value, two_tailed):
    """Apply multiple comparison correction to a T-statistic map."""
    from dpabistat.correction import grf_correction, fdr_correction

    if method == "grf":
        result = grf_correction(
            stat_path=input_path,
            mask_path=mask,
            voxel_p=voxel_p,
            cluster_p=cluster_p,
            two_tailed=two_tailed,
        )
        click.echo(f"GRF correction applied:")
        click.echo(f"  Z threshold: {result.z_threshold:.4f}")
        click.echo(f"  Cluster size threshold: {result.cluster_size_threshold} voxels")
        click.echo(f"  Surviving clusters: {result.n_clusters}")
        for c in result.cluster_table:
            click.echo(
                f"    Cluster {c['label']}: {c['size']} voxels, "
                f"peak Z={c['peak_value']:.3f} at {c['peak_coords']}"
            )
    else:
        result = fdr_correction(
            stat_path=input_path,
            mask_path=mask,
            q=q_value,
            two_tailed=two_tailed,
        )
        click.echo(f"FDR correction applied (q={q_value}):")
        click.echo(f"  P threshold: {result.p_threshold:.6f}")
        click.echo(f"  Significant voxels: {result.n_significant}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dpabistat/cli.py tests/test_cli.py
git commit -m "feat: add CLI for ttest2 and multiple comparison correction"
```

---

### Task 8: Package Init and Full Test Suite

**Files:**
- Modify: `src/dpabistat/__init__.py` (ensure imports work)

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 2: Verify CLI entry point works**

Run: `uv run dpabistat --help`
Expected: Shows help text with `ttest2` and `correct` subcommands.

Run: `uv run dpabistat ttest2 --help`
Expected: Shows ttest2 options.

Run: `uv run dpabistat correct --help`
Expected: Shows correct options.

- [ ] **Step 3: Commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: ensure package imports and CLI entry point work"
```

---

### Task 9: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

```markdown
# DPABIStat

Voxel-wise group-level statistical analysis for neuroimaging data.

DPABIStat performs two-sample t-tests on NIfTI brain images with optional covariates (age, sex, etc.), outputs T-statistic maps, and applies multiple comparison correction using Gaussian Random Field (GRF) theory or False Discovery Rate (FDR).

## Installation

```bash
# With uv
uv add dpabistat

# From source
git clone <repo-url>
cd DPABIStat
uv sync
```

## Quick Start

### Python API

```python
import dpabistat

# Two-sample t-test
result = dpabistat.two_sample_ttest(
    group1="data/patients/",
    group2="data/controls/",
    output="results/group_comparison",
    mask="brain_mask.nii.gz",
    covariates="covariates.csv",  # CSV with age, sex, etc.
    contrast=[1, -1],             # patients > controls
)

print(f"DOF: {result.dof}, FWHM: {result.fwhm}")

# GRF cluster-level correction
grf_result = dpabistat.grf_correction(
    "results/group_comparison_T.nii.gz",
    mask_path="brain_mask.nii.gz",
    voxel_p=0.001,
    cluster_p=0.05,
)

print(f"Cluster size threshold: {grf_result.cluster_size_threshold} voxels")
for c in grf_result.cluster_table:
    print(f"  Cluster {c['label']}: {c['size']} voxels, peak Z={c['peak_value']:.2f}")

# FDR correction
fdr_result = dpabistat.fdr_correction(
    "results/group_comparison_T.nii.gz",
    mask_path="brain_mask.nii.gz",
    q=0.05,
)

print(f"FDR p-threshold: {fdr_result.p_threshold:.6f}")
print(f"Significant voxels: {fdr_result.n_significant}")
```

### CLI

```bash
# Two-sample t-test
dpabistat ttest2 \
    --group1 data/patients/ \
    --group2 data/controls/ \
    --output results/group_comparison \
    --mask brain_mask.nii.gz \
    --covariates covariates.csv \
    --contrast 1 -1

# GRF correction
dpabistat correct \
    --input results/group_comparison_T.nii.gz \
    --method grf \
    --voxel-p 0.001 \
    --cluster-p 0.05 \
    --mask brain_mask.nii.gz

# FDR correction
dpabistat correct \
    --input results/group_comparison_T.nii.gz \
    --method fdr \
    --q 0.05 \
    --mask brain_mask.nii.gz
```

## Covariate File Format

A CSV file where each row is a subject (group 1 first, then group 2) and each column is a covariate:

```csv
age,sex
25,0
30,1
28,0
...
```

## Output Files

### T-test outputs

| File | Description |
|------|-------------|
| `{output}_T.nii.gz` | T-statistic map (from contrast) |
| `{output}_beta.nii.gz` | Beta coefficients (4D: one volume per regressor) |
| `{output}_cohen_f2.nii.gz` | Cohen's f² effect size map |

### GRF correction outputs

| File | Description |
|------|-------------|
| `Z_ClusterThresholded_{name}.nii.gz` | Thresholded Z-map |
| `ClusterThresholded_{name}.nii.gz` | Thresholded original T-map |

### FDR correction outputs

| File | Description |
|------|-------------|
| `FDR_Thresholded_{name}.nii.gz` | FDR-thresholded T-map |

## Statistical Methods

### Design Matrix

Uses a cell-means parameterization:

```
X = [G1, G2, Cov1, Cov2, ...]
```

- `G1`: indicator for group 1 (1 for group 1, 0 for group 2)
- `G2`: indicator for group 2
- Contrast `[1, -1]` tests group 1 > group 2

### GRF Correction

Implements cluster-level inference based on Gaussian Random Field theory (Friston et al., 1994):

1. Converts T-map to Z-map
2. Applies voxel-level threshold
3. Calculates cluster size threshold from expected Euler characteristic
4. Removes clusters below threshold (26-connectivity)

### FDR Correction

Implements the Benjamini-Hochberg procedure to control the false discovery rate.

## Development

```bash
uv sync
uv run pytest tests/ -v
```

## References

- Friston, K. J., et al. (1994). Assessing the significance of focal activations using their spatial extent. *Human Brain Mapping*, 1(3), 210-220.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B*, 57(1), 289-300.
- Worsley, K. J., et al. (1999). Detecting changes in nonisotropic images. *Human Brain Mapping*, 8(2-3), 98-101.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with API and CLI usage examples"
```

---

### Task 10: Git Setup and Final Commit

- [ ] **Step 1: Create .gitignore**

```gitignore
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.venv/
*.nii
*.nii.gz
.pytest_cache/
uv.lock
```

- [ ] **Step 2: Final commit with all files**

```bash
git add .gitignore
git commit -m "chore: add .gitignore"
```

- [ ] **Step 3: Verify final state**

Run: `git log --oneline`
Expected: Shows all commits in order.

Run: `uv run pytest tests/ -v`
Expected: All tests PASS.

Run: `uv run dpabistat --help`
Expected: CLI help output.
