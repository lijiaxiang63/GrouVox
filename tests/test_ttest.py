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
