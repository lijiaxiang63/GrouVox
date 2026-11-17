import numpy as np
import nibabel as nib
import pytest
from pathlib import Path

from scipy import ndimage, stats

from grouvox.correction import (
    FDRResult,
    GRFResult,
    _t_to_z,
    fdr_correction,
    grf_correction,
)


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

    def test_fdr_cluster_table(self, tmp_path):
        shape = (20, 20, 20)
        data = np.zeros(shape, dtype=np.float32)
        data[8:12, 8:12, 8:12] = 6.0

        stat_path = _make_stat_nifti(data, tmp_path, dof=28)
        mask_path = _make_mask(shape, tmp_path)

        result = fdr_correction(stat_path, mask_path, q=0.05)

        assert hasattr(result, "n_clusters")
        assert hasattr(result, "cluster_table")
        assert result.n_clusters >= 1
        cluster = result.cluster_table[0]
        assert "label" in cluster
        assert "size" in cluster
        assert "peak_value" in cluster
        assert "peak_coords" in cluster
        assert "peak_coords_mni" in cluster
        assert "peak_atlas" in cluster
        assert "atlas_regions" in cluster

    def test_fdr_csv_report(self, tmp_path):
        import csv
        shape = (20, 20, 20)
        data = np.zeros(shape, dtype=np.float32)
        data[8:12, 8:12, 8:12] = 6.0
        stat_path = _make_stat_nifti(data, tmp_path, filename="mystat_T.nii.gz", dof=28)
        mask_path = _make_mask(shape, tmp_path)
        fdr_correction(stat_path, mask_path, q=0.05)
        csv_path = tmp_path / "ClusterReport_FDR_mystat_T.csv"
        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 1
        assert "Cluster" in rows[0]
        assert "AAL_Peak" in rows[0]

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

    def test_grf_two_tailed_touching_opposite_sign_clusters_keep_distinct_labels(self, tmp_path):
        shape = (5, 5, 5)
        data = np.zeros(shape, dtype=np.float32)
        data[2, 2, 2] = 8.0
        data[2, 2, 3] = -8.0

        stat_path = _make_stat_nifti(data, tmp_path, dof=28, dlh=100.0)
        mask_path = _make_mask(shape, tmp_path)

        result = grf_correction(
            stat_path, mask_path, voxel_p=0.05, cluster_p=0.99,
            two_tailed=True,
        )

        assert result.n_clusters == 2
        assert len(result.cluster_table) == 2
        assert len({c["label"] for c in result.cluster_table}) == 2
        peak_values = sorted(c["peak_value"] for c in result.cluster_table)
        assert peak_values[0] < 0 < peak_values[1]

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

    def test_grf_atlas_annotation(self, tmp_path):
        shape = (30, 30, 30)
        data = np.zeros(shape, dtype=np.float32)
        data[10:20, 10:20, 10:20] = 5.0
        stat_path = _make_stat_nifti(data, tmp_path, dof=28, dlh=0.005)
        mask_path = _make_mask(shape, tmp_path)
        result = grf_correction(stat_path, mask_path, voxel_p=0.001, cluster_p=0.05, two_tailed=False)
        assert len(result.cluster_table) > 0
        cluster = result.cluster_table[0]
        assert "peak_coords_mni" in cluster
        assert "peak_atlas" in cluster
        assert "atlas_regions" in cluster
        assert "AAL" in cluster["peak_atlas"]

    def test_grf_csv_report(self, tmp_path):
        import csv
        shape = (30, 30, 30)
        data = np.zeros(shape, dtype=np.float32)
        data[10:20, 10:20, 10:20] = 5.0
        stat_path = _make_stat_nifti(data, tmp_path, filename="mystat_T.nii.gz", dof=28, dlh=0.005)
        mask_path = _make_mask(shape, tmp_path)
        grf_correction(stat_path, mask_path, voxel_p=0.001, cluster_p=0.05, two_tailed=False)
        csv_path = tmp_path / "ClusterReport_mystat_T.csv"
        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 1
        assert "Cluster" in rows[0]
        assert "AAL_Peak" in rows[0]
        assert "HO_Cort_Peak" in rows[0]

    def test_grf_peak_is_largest_abs_t_for_strong_effects(self, tmp_path):
        """Strong effects keep a finite Z and the true peak location.

        With T this large the one-sided p-value is below 1e-16; a Z computed
        through ``1 - p`` is infinite for the whole cluster, so the reported
        peak degenerates to the first voxel of the cluster.
        """
        import csv

        shape = (30, 30, 30)
        data = np.zeros(shape, dtype=np.float32)
        data[5:12, 5:12, 5:12] = 10.0
        data[9, 7, 10] = 19.0
        data[18:25, 18:25, 18:25] = -15.0
        data[22, 23, 19] = -18.0

        stat_path = _make_stat_nifti(data, tmp_path, filename="strong_T.nii.gz",
                                     dof=379, dlh=0.005)
        mask_path = _make_mask(shape, tmp_path)

        result = grf_correction(
            stat_path, mask_path, voxel_p=0.001, cluster_p=0.05,
            two_tailed=True,
        )

        assert np.all(np.isfinite(result.thresholded_z_map))
        assert result.n_clusters == 2
        peaks = {c["peak_coords"]: c["peak_value"] for c in result.cluster_table}
        assert set(peaks) == {(9, 7, 10), (22, 23, 19)}
        assert peaks[(9, 7, 10)] == pytest.approx(15.9147242039993, rel=1e-9)
        assert peaks[(22, 23, 19)] == pytest.approx(_t_to_z(np.array([-18.0]), 379)[0])
        assert peaks[(22, 23, 19)] < 0

        z_img = nib.load(tmp_path / "Z_ClusterThresholded_strong_T.nii.gz")
        assert np.all(np.isfinite(z_img.get_fdata()))

        with open(tmp_path / "ClusterReport_strong_T.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert all(np.isfinite(float(r["PeakZ"])) for r in rows)

    def test_grf_reestimate_with_strong_effects(self, tmp_path):
        """Smoothness re-estimated from the Z-map stays finite for strong effects.

        An infinite Z makes the estimated dLh NaN, which silently collapses
        the cluster-size threshold to a single voxel.
        """
        shape = (30, 30, 30)
        rng = np.random.default_rng(0)
        noise = ndimage.gaussian_filter(rng.standard_normal(shape), sigma=2.0)
        data = (noise / noise.std()).astype(np.float32)
        data[10:20, 10:20, 10:20] += 12.0

        stat_path = _make_stat_nifti(data, tmp_path, dof=379, dlh=0.005)
        mask_path = _make_mask(shape, tmp_path)

        result = grf_correction(
            stat_path, mask_path, voxel_p=0.001, cluster_p=0.05,
            two_tailed=False, reestimate=True,
        )

        assert result.cluster_size_threshold > 1
        assert np.all(np.isfinite(result.thresholded_z_map))
        assert result.thresholded_t_map[15, 15, 15] != 0


def _t_to_z_via_one_minus_p(t, dof):
    """The pre-fix conversion: accurate only while 1 - p is representable."""
    p = stats.t.sf(np.abs(t), dof)
    return stats.norm.ppf(1.0 - p) * np.sign(t)


class TestTToZ:
    @pytest.mark.parametrize("dof", [20, 379])
    def test_finite_and_strictly_increasing(self, dof):
        # Step 0.05 up to |T| = 200; at dof 379 this crosses both the point
        # where 1 - p rounds to 1 (|T| ~ 8.6) and the point where p itself
        # underflows double precision (|T| ~ 125).
        t = np.linspace(0.0, 200.0, 4001)
        z = _t_to_z(t, dof)

        assert z[0] == 0.0
        assert np.all(np.isfinite(z))
        assert np.all(np.diff(z) > 0)

    @pytest.mark.parametrize("dof", [20, 379])
    def test_finite_and_increasing_for_extreme_t(self, dof):
        t = np.array([1e3, 1e6, 1e12, 1e20, np.finfo(np.float32).max], dtype=np.float64)
        z = _t_to_z(t, dof)

        assert np.all(np.isfinite(z))
        assert np.all(np.diff(z) > 0)

    @pytest.mark.parametrize("dof", [20, 379])
    def test_matches_one_minus_p_formula_where_it_is_accurate(self, dof):
        t = np.linspace(-4.99, 4.99, 999)

        np.testing.assert_allclose(
            _t_to_z(t, dof), _t_to_z_via_one_minus_p(t, dof), rtol=1e-8, atol=1e-12,
        )

    @pytest.mark.parametrize(
        "dof, t, expected",
        [
            # Reference values: 50-digit mpmath, solving sf_norm(z) = sf_t(t).
            (20, 10.0, 5.92286624031131),
            (20, 40.0, 9.29605973703433),
            (379, 9.0, 8.56232440611972),
            (379, 19.0, 15.9147242039993),
            (379, 40.0, 25.01401054777),
            (379, 150.0, 39.4035461107778),
            (379, 1000.0, 54.624470143243),
        ],
    )
    def test_reference_values_in_the_tail(self, dof, t, expected):
        assert _t_to_z(np.array([t]), dof)[0] == pytest.approx(expected, rel=1e-10)

    def test_sign_is_preserved(self):
        t = np.array([-150.0, -19.0, -2.0, 0.0, 2.0, 19.0, 150.0])
        z = _t_to_z(t, 379)

        np.testing.assert_array_equal(np.sign(z), np.sign(t))
        np.testing.assert_array_equal(z, -z[::-1])

    def test_float32_input_and_shape(self):
        t = np.full((3, 4, 5), 19.0, dtype=np.float32)
        z = _t_to_z(t, 379)

        assert z.shape == t.shape
        assert z.dtype == np.float64
        np.testing.assert_allclose(z, 15.9147242039993, rtol=1e-10)
