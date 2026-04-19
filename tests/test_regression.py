import numpy as np
import pandas as pd
import pytest

from grouvox.regression import regression, RegressionResult


@pytest.fixture
def synthetic_regression(tmp_nifti_factory, tmp_path):
    """16 subjects with a continuous predictor that modulates a hotspot.

    Voxels at [3:6, 3:6, 3:6] have signal = 2 * predictor + noise;
    the rest is pure noise. Predictor is saved alongside images.
    """
    rng = np.random.default_rng(42)
    shape = (10, 10, 10)
    n = 16

    predictor = rng.normal(0, 1, n)

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(n):
        data = rng.normal(10, 1, shape)
        data[3:6, 3:6, 3:6] += 2.0 * predictor[i]
        tmp_nifti_factory(data, f"images/sub{i:02d}.nii.gz")

    mask = np.ones(shape, dtype=np.float32)
    mask_path = tmp_nifti_factory(mask, "mask.nii.gz")

    pred_path = tmp_path / "predictor.csv"
    pd.DataFrame({"x": predictor}).to_csv(pred_path, index=False)

    return img_dir, pred_path, mask_path, predictor


class TestRegression:
    def test_basic(self, synthetic_regression, tmp_path):
        img_dir, pred_path, mask_path, _ = synthetic_regression
        output = tmp_path / "results" / "reg"

        result = regression(
            images=img_dir,
            predictor=pred_path,
            output=output,
            mask=mask_path,
        )

        assert isinstance(result, RegressionResult)
        assert result.t_map.shape == (10, 10, 10)
        assert result.dof == 14  # 16 - 2 (intercept + predictor)

        hotspot_t = result.t_map[3:6, 3:6, 3:6].mean()
        background_t = np.abs(result.t_map[0, 0, 0])
        assert hotspot_t > 3.0
        assert hotspot_t > background_t

    def test_output_files_created(self, synthetic_regression, tmp_path):
        img_dir, pred_path, mask_path, _ = synthetic_regression
        output = tmp_path / "results" / "reg"

        regression(img_dir, pred_path, output, mask=mask_path)

        assert (tmp_path / "results" / "reg_T.nii.gz").exists()
        assert (tmp_path / "results" / "reg_beta.nii.gz").exists()
        assert (tmp_path / "results" / "reg_cohen_f2.nii.gz").exists()

    def test_header_metadata(self, synthetic_regression, tmp_path):
        img_dir, pred_path, mask_path, _ = synthetic_regression
        output = tmp_path / "results" / "reg"

        result = regression(img_dir, pred_path, output, mask=mask_path)
        descrip = result.header["descrip"].astype(str)
        assert "GrouVox{T_" in descrip
        assert "dLh_" in descrip
        assert "FWHM" in descrip

    def test_with_covariates(self, synthetic_regression, tmp_path):
        img_dir, pred_path, mask_path, _ = synthetic_regression
        output = tmp_path / "results" / "reg_cov"

        covars = pd.DataFrame({
            "age": np.random.default_rng(0).normal(30, 5, 16),
            "sex": [0, 1] * 8,
        })

        result = regression(
            img_dir, pred_path, output, mask=mask_path, covariates=covars,
        )

        assert result.dof == 12  # 16 - 2 (intercept + predictor) - 2 covariates
        assert result.beta_maps.shape[-1] == 4  # intercept, predictor, age, sex

    def test_predictor_array(self, synthetic_regression, tmp_path):
        img_dir, _, mask_path, predictor = synthetic_regression
        output = tmp_path / "results" / "reg_arr"

        result = regression(
            images=img_dir,
            predictor=predictor,
            output=output,
            mask=mask_path,
        )
        assert result.dof == 14

    def test_predictor_length_mismatch(self, synthetic_regression, tmp_path):
        img_dir, _, mask_path, _ = synthetic_regression
        output = tmp_path / "results" / "reg_bad"

        with pytest.raises(ValueError, match="Predictor length"):
            regression(
                images=img_dir,
                predictor=np.arange(5),
                output=output,
                mask=mask_path,
            )

    def test_sign_matches_predictor_direction(self, synthetic_regression, tmp_path):
        img_dir, _, mask_path, predictor = synthetic_regression

        r_pos = regression(
            img_dir, predictor, tmp_path / "r_pos", mask=mask_path,
        )
        r_neg = regression(
            img_dir, -predictor, tmp_path / "r_neg", mask=mask_path,
        )
        np.testing.assert_allclose(r_pos.t_map, -r_neg.t_map, atol=1e-5)
