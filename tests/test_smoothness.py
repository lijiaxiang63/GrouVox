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

        noise = rng.normal(size=(*shape, n_subjects))
        res_noise = estimate_smoothness(noise, mask, dof=13, voxel_size=voxel_size)

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
