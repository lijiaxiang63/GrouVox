# DPABIStat

Voxel-wise group-level statistical analysis for neuroimaging data.

DPABIStat performs two-sample t-tests on NIfTI brain images with optional covariates (age, sex, etc.), outputs T-statistic maps, and applies multiple comparison correction using Gaussian Random Field (GRF) theory or False Discovery Rate (FDR).

## Installation

```bash
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
