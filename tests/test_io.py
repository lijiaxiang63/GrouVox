import numpy as np
import nibabel as nib
import pytest
from pathlib import Path

from dpabistat.io import load_images, load_mask, save_nifti


class TestLoadImages:
    def test_load_from_directory(self, tmp_nifti_factory, tmp_path):
        shape = (5, 5, 5)
        d = tmp_path / "imgs"
        d.mkdir()
        for i in range(3):
            tmp_nifti_factory(np.ones(shape) * i, f"imgs/s{i}.nii.gz")

        data, header, affine = load_images(d)
        assert data.shape == (5, 5, 5, 3)
        assert np.isclose(data[0, 0, 0, 0], 0.0)
        assert np.isclose(data[0, 0, 0, 2], 2.0)

    def test_load_from_list(self, tmp_nifti_factory):
        shape = (5, 5, 5)
        paths = [
            tmp_nifti_factory(np.ones(shape) * i, f"s{i}.nii.gz")
            for i in range(3)
        ]
        data, header, affine = load_images(paths)
        assert data.shape == (5, 5, 5, 3)

    def test_load_4d_file(self, tmp_nifti_factory):
        data_4d = np.random.default_rng(0).normal(size=(5, 5, 5, 4))
        path = tmp_nifti_factory(data_4d, "4d.nii.gz")
        data, header, affine = load_images(path)
        assert data.shape == (5, 5, 5, 4)
        np.testing.assert_allclose(data, data_4d, atol=1e-5)

    def test_shape_mismatch_raises(self, tmp_nifti_factory):
        p1 = tmp_nifti_factory(np.ones((5, 5, 5)), "a.nii.gz")
        p2 = tmp_nifti_factory(np.ones((6, 6, 6)), "b.nii.gz")
        with pytest.raises(ValueError, match="shape"):
            load_images([p1, p2])


class TestLoadMask:
    def test_load_binary_mask(self, tmp_nifti_factory):
        mask_data = np.zeros((5, 5, 5))
        mask_data[1:4, 1:4, 1:4] = 1
        path = tmp_nifti_factory(mask_data, "mask.nii.gz")
        mask = load_mask(path)
        assert mask.dtype == bool
        assert mask.sum() == 27

    def test_none_returns_none(self):
        assert load_mask(None) is None


class TestSaveNifti:
    def test_save_and_reload(self, tmp_path):
        data = np.random.default_rng(0).normal(size=(5, 5, 5)).astype(np.float32)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        header = nib.Nifti1Header()
        out = tmp_path / "out.nii.gz"
        save_nifti(data, affine, header, out, description="test desc")

        img = nib.load(out)
        np.testing.assert_allclose(img.get_fdata(), data, atol=1e-6)
        assert "test desc" in img.header["descrip"].astype(str)
