# DPABIStat: Voxel-wise Group Analysis in Python

**Date:** 2026-04-04
**Status:** Design

## Overview

A Python package for voxel-wise group-level statistical analysis of neuroimaging (NIfTI) data. Initial scope: two-sample t-test with covariates, outputting T-maps, with GRF and FDR multiple comparison correction. Modeled after DPABI V9.0's StatisticalAnalysis but with Pythonic design, vectorized computation, and CLI + Python API interfaces.

## Architecture

```
DPABIStat/
├── pyproject.toml
├── README.md
├── src/
│   └── dpabistat/
│       ├── __init__.py        # Public API exports
│       ├── io.py              # NIfTI loading/saving
│       ├── glm.py             # Vectorized OLS regression
│       ├── ttest.py           # Two-sample t-test with covariates
│       ├── correction.py      # GRF and FDR correction
│       ├── smoothness.py      # Residual smoothness estimation
│       └── cli.py             # Click CLI
└── tests/
    ├── conftest.py            # Shared fixtures (synthetic NIfTI data)
    ├── test_glm.py
    ├── test_ttest.py
    └── test_correction.py
```

### Dependencies

- **nibabel** — NIfTI I/O
- **numpy** — vectorized computation
- **scipy** — stats distributions, ndimage for connected components
- **pandas** — covariate CSV loading
- **click** — CLI framework

Dev dependencies: **pytest**

## Components

### 1. `io.py` — NIfTI I/O

**`load_images(paths) -> tuple[np.ndarray, nib.Nifti1Header, np.ndarray]`**

- Accepts: single 4D NIfTI path, list of 3D NIfTI paths, or a directory of NIfTI files
- Returns: `(data_4d, header, affine)` where `data_4d` is `(X, Y, Z, N)` shaped
- Validates all images share the same affine/shape

**`load_mask(path) -> np.ndarray`**

- Returns boolean 3D mask
- If None, generates mask from non-zero voxels across all subjects

**`save_nifti(data, affine, header, path, description=None)`**

- Writes 3D or 4D float32 NIfTI
- Embeds metadata (DOF, FWHM, dLh) in header description field, matching DPABI convention: `DPABIStat{T_[df]}{dLh_val}{FWHMx_val FWHMy_val FWHMz_val mm}`

### 2. `glm.py` — Vectorized OLS Engine

**`ols_fit(Y, X) -> OLSResult`**

- `Y`: `(n_subjects, n_voxels)` — flattened masked dependent data
- `X`: `(n_subjects, n_regressors)` — design matrix (same for all voxels)
- QR decomposition computed once on X
- All voxels solved via matrix multiply: `beta = R_inv @ Q.T @ Y`
- Returns dataclass:

```python
@dataclass
class OLSResult:
    beta: np.ndarray       # (n_regressors, n_voxels)
    residuals: np.ndarray  # (n_subjects, n_voxels)
    t_values: np.ndarray   # (n_regressors, n_voxels)
    sse: np.ndarray        # (n_voxels,)
    dof: int               # n_subjects - rank(X)
```

**`compute_contrast(result, X, contrast, stat_type="T") -> np.ndarray`**

- Computes contrast T-statistic: `T = (c @ beta) / (std_e * sqrt(c @ (X'X)^-1 @ c'))`
- Returns `(n_voxels,)` T-values

### 3. `ttest.py` — Two-Sample T-Test

**`two_sample_ttest(group1, group2, output, mask=None, covariates=None, contrast=None) -> TTestResult`**

Parameters:
- `group1`, `group2`: paths (str/Path/list) to NIfTI files
- `output`: output path prefix (e.g., `results/ttest`)
- `mask`: optional brain mask NIfTI path
- `covariates`: optional CSV path or DataFrame. Columns are covariate names, rows are subjects (group1 first, then group2). Example:

```csv
age,sex
25,0
30,1
...
```

- `contrast`: optional `[c1, c2]` for group terms, default `[1, -1]` (group1 > group2). Covariate contrasts are always zero.

**Design matrix construction (cell-means model):**

```
X = [G1, G2, Cov1, Cov2, ...]
     ↑   ↑   ↑covariate columns
     group indicators (0/1)
```

- G1: 1 for group1 subjects, 0 otherwise
- G2: 1 for group2 subjects, 0 otherwise
- No separate intercept (G1 + G2 span the intercept)
- Full contrast vector: `[c1, c2, 0, 0, ...]`

**Outputs:**
- `{output}_T.nii.gz` — T-statistic map (from contrast)
- `{output}_beta.nii.gz` — beta coefficients (4D, one volume per regressor)
- `{output}_cohen_f2.nii.gz` — Cohen's f² effect size map

**Returns:**

```python
@dataclass
class TTestResult:
    t_map: np.ndarray          # 3D T-statistic volume
    beta_maps: np.ndarray      # 4D beta coefficients
    cohen_f2_map: np.ndarray   # 3D effect size
    header: nib.Nifti1Header   # with DOF, FWHM, dLh in description
    affine: np.ndarray
    dof: int
    fwhm: tuple[float, float, float]
    dlh: float
```

### 4. `smoothness.py` — Smoothness Estimation

**`estimate_smoothness(residuals_4d, mask, dof, voxel_size) -> SmoothnessResult`**

Algorithm (matching DPABI/FSL):
1. Compute lag-1 autocorrelation of residuals in x, y, z within mask
2. Derive per-axis variance: `sigma_sq[i] = -1 / (4 * ln(|rho[i]|))`
3. FWHM per axis: `FWHM[i] = sqrt(8 * ln(2) * sigma_sq[i]) * voxel_size[i]`
4. Resel density: `dLh = (sigma_sq[0] * sigma_sq[1] * sigma_sq[2])^(-0.5) / sqrt(8)`
5. Number of resels: `resels = n_voxels * dLh` (but computed properly via FWHM)
6. DOF-dependent scaling correction (lookup table matching DPABI)

Returns:

```python
@dataclass
class SmoothnessResult:
    fwhm: tuple[float, float, float]  # in mm
    dlh: float
    resels: float
    n_voxels: int
```

### 5. `correction.py` — Multiple Comparison Correction

#### GRF Correction

**`grf_correction(stat_path, mask_path=None, voxel_p=0.001, cluster_p=0.05, two_tailed=True) -> GRFResult`**

Algorithm (Friston et al. 1994):
1. Load T-map, parse DOF and smoothness from header description
2. Convert T → Z: `Z = norm.ppf(t.cdf(|T|, dof))` preserving sign
3. Voxel threshold: `z_thr = norm.ppf(1 - voxel_p/2)` (two-tailed) or `norm.ppf(1 - voxel_p)` (one-tailed)
4. Expected clusters: `Em = n_voxels * (2*pi)^(-2) * dLh * (z_thr^2 - 1) * exp(-z_thr^2 / 2)`
5. Expected suprathreshold voxels: `EN = n_voxels * (1 - norm.cdf(z_thr))`
6. Shape parameter: `beta = (gamma(D/2 + 1) * Em / EN)^(2/D)` where D=3
7. Find minimum cluster size K: iterate until `P(K) = 1 - exp(-Em * exp(-beta * K^(2/D))) <= cluster_p`
8. Connected component labeling (26-connectivity via `scipy.ndimage.label`)
9. Remove clusters smaller than K voxels
10. Handle positive and negative tails separately for two-tailed tests

Outputs:
- `Z_ClusterThresholded_{basename}.nii.gz` — thresholded Z-map
- `ClusterThresholded_{basename}.nii.gz` — thresholded original T-map

Returns:

```python
@dataclass
class GRFResult:
    z_threshold: float
    cluster_size_threshold: int
    thresholded_z_map: np.ndarray
    thresholded_t_map: np.ndarray
    n_clusters: int
    cluster_table: list[dict]  # label, size, peak_z, peak_coords
```

#### FDR Correction

**`fdr_correction(stat_path, mask_path=None, q=0.05, two_tailed=True) -> FDRResult`**

Algorithm (Benjamini-Hochberg):
1. Load statistic map, extract within mask
2. Convert to p-values (two-tailed for T/Z, one-tailed for F)
3. Sort p-values ascending
4. Find largest k where `P_k <= k/m * q`
5. Threshold: keep voxels with `p <= P_k`

Outputs:
- `FDR_Thresholded_{basename}.nii.gz`

Returns:

```python
@dataclass
class FDRResult:
    p_threshold: float
    n_significant: int
    thresholded_map: np.ndarray
```

### 6. `cli.py` — Command Line Interface

```bash
# Two-sample t-test
dpabistat ttest2 \
    --group1 /data/group1/ \
    --group2 /data/group2/ \
    --output results/my_ttest \
    --mask brain_mask.nii.gz \
    --covariates covars.csv \
    --contrast 1 -1

# GRF correction
dpabistat correct \
    --input results/my_ttest_T.nii.gz \
    --method grf \
    --voxel-p 0.001 \
    --cluster-p 0.05 \
    --two-tailed \
    --mask brain_mask.nii.gz

# FDR correction
dpabistat correct \
    --input results/my_ttest_T.nii.gz \
    --method fdr \
    --q 0.05 \
    --mask brain_mask.nii.gz
```

## Testing Strategy

- **GLM**: Compare vectorized OLS against `scipy.stats.ttest_ind` for the no-covariate case; verify beta, t, dof match
- **FDR**: Compare against `statsmodels.stats.multitest.multipletests` with method='fdr_bh'
- **GRF**: Synthetic data with known smoothness; verify cluster thresholds are reasonable
- **Integration**: Small synthetic 3D phantom (e.g., 10x10x10) with planted group differences; verify the full pipeline detects them

## Design Decisions

1. **Cell-means parameterization** over effect coding — `[1, -1, 0, ...]` contrast is more intuitive for neuroimaging users
2. **Vectorized across voxels** — QR once, matrix multiply for all voxels, instead of MATLAB's triple-nested loop
3. **Header metadata convention** — store DOF/FWHM/dLh in NIfTI description field so correction tools can read it without needing residual images
4. **26-connectivity** for GRF cluster detection — matches FSL/DPABI default
5. **`.nii.gz` output** by default — compressed to save disk space
6. **Pandas for covariates** — CSV is the simplest interchange format; DataFrame for API use
