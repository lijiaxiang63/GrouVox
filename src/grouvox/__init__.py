"""GrouVox: Voxel-wise group-level statistical analysis for neuroimaging."""

from grouvox.ttest import two_sample_ttest, TTestResult
from grouvox.glm import ols_fit, compute_contrast, OLSResult
from grouvox.correction import grf_correction, fdr_correction, GRFResult, FDRResult
from grouvox.smoothness import estimate_smoothness, SmoothnessResult
from grouvox.io import load_images, load_mask, save_nifti

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
