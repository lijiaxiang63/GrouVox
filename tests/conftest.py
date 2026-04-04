import numpy as np
import nibabel as nib
import pytest
from pathlib import Path


@pytest.fixture
def tmp_nifti_factory(tmp_path):
    """Factory to create synthetic NIfTI files for testing."""

    def _create(data: np.ndarray, filename: str = "test.nii.gz") -> Path:
        affine = np.diag([2.0, 2.0, 2.0, 1.0])  # 2mm isotropic
        img = nib.Nifti1Image(data.astype(np.float32), affine)
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(img, path)
        return path

    return _create


@pytest.fixture
def synthetic_two_groups(tmp_nifti_factory, tmp_path):
    """Create two groups of 3D NIfTI images with a planted difference.

    Group 1: voxel values drawn from N(10, 1) with a 'hot spot' at [3:6,3:6,3:6] of +5
    Group 2: voxel values drawn from N(10, 1) with no hot spot
    Shape: (10, 10, 10), 8 subjects per group.
    """
    rng = np.random.default_rng(42)
    shape = (10, 10, 10)
    n_per_group = 8

    g1_dir = tmp_path / "group1"
    g2_dir = tmp_path / "group2"
    g1_dir.mkdir()
    g2_dir.mkdir()

    for i in range(n_per_group):
        data1 = rng.normal(10, 1, shape)
        data1[3:6, 3:6, 3:6] += 5  # planted difference
        tmp_nifti_factory(data1, f"group1/sub{i:02d}.nii.gz")

        data2 = rng.normal(10, 1, shape)
        tmp_nifti_factory(data2, f"group2/sub{i:02d}.nii.gz")

    mask = np.ones(shape, dtype=np.float32)
    mask_path = tmp_nifti_factory(mask, "mask.nii.gz")

    return g1_dir, g2_dir, mask_path
