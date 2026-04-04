# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DPABIStat performs voxel-wise group-level statistical analysis on NIfTI neuroimaging data. It runs two-sample t-tests with optional covariates and applies multiple comparison correction (GRF cluster-level or FDR). Python package name: `grouvox`.

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
io.load_images → ttest.two_sample_ttest → correction.grf_correction / fdr_correction
                        ↓
                  glm.ols_fit + glm.compute_contrast
                        ↓
                  smoothness.estimate_smoothness
```

### Module Responsibilities

- **`io.py`** — Load NIfTI images (3D files, 4D files, or directories) into 4D arrays; load masks; save NIfTI outputs. `_resolve_paths()` handles path normalization.
- **`glm.py`** — Vectorized OLS via QR decomposition. `ols_fit(Y, X)` operates on `(n_subjects, n_voxels)` matrices. `compute_contrast()` computes T-statistics for arbitrary linear contrasts using `(X'X)^-1`.
- **`ttest.py`** — Orchestrator. Builds cell-means design matrix `[G1, G2, covariates]`, calls GLM, computes Cohen's f², estimates smoothness, writes output NIfTIs. Encodes DOF/FWHM/dLh into the NIfTI `descrip` header field for downstream correction.
- **`smoothness.py`** — Estimates residual smoothness (FWHM per axis, dLh) using lag-1 autocorrelation with DOF-dependent scaling correction.
- **`correction.py`** — GRF cluster-level correction (T→Z conversion, Euler characteristic-based cluster threshold, 26-connectivity labeling) and FDR Benjamini-Hochberg. Both parse metadata from the `descrip` header field via `_HEADER_RE` regex.
- **`cli.py`** — Click CLI with `ttest2` and `correct` subcommands. Lazy-imports analysis modules.

### Key Design Decisions

- Statistical metadata (DOF, dLh, FWHM) is stored in the NIfTI header `descrip` field with a specific format parsed by `_HEADER_RE` in `correction.py`. This couples t-test output to correction input.
- Cell-means parameterization (not dummy coding): design matrix uses separate indicator columns for each group rather than intercept + group difference.
- All voxel-wise operations work on masked flat arrays `(n_subjects, n_mask_voxels)` for memory efficiency; results are unmasked back to 3D via `_unmask()`.

### Test Fixtures

`conftest.py` provides `tmp_nifti_factory` (creates NIfTI files from arrays) and `synthetic_two_groups` (two groups of 8 subjects each on a 10x10x10 grid with a planted signal at `[3:6, 3:6, 3:6]`).
