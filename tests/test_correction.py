import numpy as np
import nibabel as nib
import pytest
from pathlib import Path

from grouvox.correction import fdr_correction, grf_correction, FDRResult, GRFResult


def _make_stat_nifti(data, tmp_path, filename="stat_T.nii.gz", dof=14, dlh=0.01,
                     fwhm=(8.0, 8.0, 8.0)):
    """Helper: create a T-stat NIfTI with GrouVox-style header metadata."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    header = nib.Nifti1Header()
    header.set_data_dtype(np.float32)
    header["descrip"] = (
        f"GrouVox{{T_[{dof:.1f}]}}{{dLh_{dlh:.6f}}}"
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
        rng = np.random.default_rng(42)
        shape = (20, 20, 20)
        data = rng.normal(0, 1, shape).astype(np.float32)
        data[8:12, 8:12, 8:12] = 6.0

        stat_path = _make_stat_nifti(data, tmp_path, dof=28)
        mask_path = _make_mask(shape, tmp_path)

        result = fdr_correction(stat_path, mask_path, q=0.05)

        assert isinstance(result, FDRResult)
        assert result.n_significant > 0
        assert result.thresholded_map[9, 9, 9] != 0.0
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
        rng = np.random.default_rng(42)
        shape = (15, 15, 15)
        data = rng.normal(0, 1, shape).astype(np.float32)
        stat_path = _make_stat_nifti(data, tmp_path, dof=28)
        mask_path = _make_mask(shape, tmp_path)

        result = fdr_correction(stat_path, mask_path, q=0.05)
        assert result.n_significant < shape[0] * shape[1] * shape[2] * 0.1


class TestGRFCorrection:
    def test_basic_grf(self, tmp_path):
        shape = (30, 30, 30)
        data = np.zeros(shape, dtype=np.float32)
        data[10:20, 10:20, 10:20] = 5.0
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
        assert result.thresholded_t_map[15, 15, 15] != 0
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
        shape = (30, 30, 30)
        data = np.zeros(shape, dtype=np.float32)
        data[5:12, 5:12, 5:12] = 5.0
        data[18:25, 18:25, 18:25] = -5.0

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
