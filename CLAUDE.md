# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GrouVox performs voxel-wise group-level statistical analysis on NIfTI neuroimaging data. It runs two-sample t-tests and continuous-predictor regression with optional covariates, and applies multiple comparison correction (GRF cluster-level or FDR). Python package name: `grouvox`.

## Commands

```bash
uv sync                          # Install dependencies
uv run pytest tests/ -v          # Run all tests
uv run pytest tests/test_glm.py  # Run a single test file
uv run pytest tests/ -v -k "test_name"  # Run a specific test
```

CLI entry point: `grouvox` (defined in `pyproject.toml` → `grouvox.cli:main`).

## Architecture

Source layout: `src/grouvox/` with `hatchling` build backend.

### Data Flow

```
io.load_images → ttest.two_sample_ttest     → correction.grf_correction / fdr_correction
              └─ regression.regression      ↗
                        ↓
                  glm.ols_fit + glm.compute_contrast
                        ↓
                  smoothness.estimate_smoothness
```

### Module Responsibilities

- **`io.py`** — Load NIfTI images (3D files, 4D files, or directories) into 4D arrays; load masks; save NIfTI outputs. `_resolve_paths()` handles path normalization.
- **`glm.py`** — Vectorized OLS via QR decomposition. `ols_fit(Y, X)` operates on `(n_subjects, n_voxels)` matrices. `compute_contrast()` computes T-statistics for arbitrary linear contrasts using `(X'X)^-1`.
- **`ttest.py`** — Orchestrator for two-sample t-tests. Builds cell-means design matrix `[G1, G2, covariates]`, calls GLM, computes Cohen's f², estimates smoothness, writes output NIfTIs. Encodes DOF/FWHM/dLh into the NIfTI `descrip` header field for downstream correction. Exposes `_HeaderProxy` and `_unmask` helpers that `regression.py` reuses.
- **`regression.py`** — Orchestrator for continuous-predictor regression. Builds design matrix `[intercept, predictor, covariates]`, computes T-statistic for the predictor slope (contrast `[0, 1, 0, ...]`), and writes outputs with the same `descrip` header format as `ttest.py` so downstream correction works unchanged. Accepts `predictor` as a CSV path (single column), numpy array, or pandas Series.
- **`smoothness.py`** — Estimates residual smoothness (FWHM per axis, dLh) using lag-1 autocorrelation with DOF-dependent scaling correction. `estimate_smoothness()` works on 4D residuals (used by `ttest.py`); `estimate_smoothness_from_map()` works on a 3D Z-map (used by `correction.py` when re-estimating).
- **`correction.py`** — GRF cluster-level correction (T→Z conversion, Euler characteristic-based cluster threshold, 26-connectivity labeling) and FDR Benjamini-Hochberg. Parses metadata from the `descrip` header field via `_HEADER_RE` regex. GRF supports `reestimate=True` to re-estimate smoothness from the Z-map instead of using header values. In two-tailed mode, cluster_p is halved per tail to maintain the correct family-wise error rate. Both methods annotate clusters with atlas-based region labels and write a ClusterReport CSV.
- **`atlas.py`** — Loads bundled atlases (AAL, HarvardOxford-cortical, HarvardOxford-subcortical), resamples them to match the statistical map, and annotates clusters with peak MNI coordinates, peak atlas labels, and region overlap percentages. Atlas NIfTI and label JSON files are stored in `src/grouvox/atlases/`.
- **`cli.py`** — Click CLI with `ttest2`, `regress`, `correct`, `plot`, and `plot-subcortical` subcommands. Lazy-imports analysis modules. Prints atlas-annotated cluster tables for both GRF and FDR results.

### Key Design Decisions

- Statistical metadata (DOF, dLh, FWHM) is stored in the NIfTI header `descrip` field with a specific format parsed by `_HEADER_RE` in `correction.py`. This couples t-test/regression output to correction input. When header metadata is present, smoothness values are used by default; `reestimate=True` overrides this to re-estimate from the Z-map.
- Cell-means parameterization for t-tests (not dummy coding): design matrix uses separate indicator columns for each group rather than intercept + group difference. Regression, by contrast, uses intercept + predictor.
- Cohen's f² reduced model differs by orchestrator: `ttest.py` drops both group columns (keeps covariates, or intercept-only when no covariates); `regression.py` drops only the predictor column (keeps intercept + covariates).
- All voxel-wise operations work on masked flat arrays `(n_subjects, n_mask_voxels)` for memory efficiency; results are unmasked back to 3D via `_unmask()`.

### Test Fixtures

`conftest.py` provides `tmp_nifti_factory` (creates NIfTI files from arrays) and `synthetic_two_groups` (two groups of 8 subjects each on a 10x10x10 grid with a planted signal at `[3:6, 3:6, 3:6]`).
