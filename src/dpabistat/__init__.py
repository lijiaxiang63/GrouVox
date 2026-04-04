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
