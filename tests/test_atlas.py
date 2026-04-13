"""Tests for grouvox.atlas module."""

import numpy as np
import pytest

from grouvox import atlas


class TestLoadAtlas:
    def test_load_aal(self):
        data, labels, affine = atlas.load_atlas("AAL")
        assert data.ndim == 3
        assert len(labels) == 116
        assert labels["1"] == "Precentral_L"
        assert affine.shape == (4, 4)

    def test_load_ho_cort(self):
        data, labels, affine = atlas.load_atlas("HarvardOxford-cortical")
        assert data.ndim == 3
        assert len(labels) == 96

    def test_load_ho_sub(self):
        data, labels, affine = atlas.load_atlas("HarvardOxford-subcortical")
        assert data.ndim == 3
        assert len(labels) == 16

    def test_unknown_atlas_raises(self):
        with pytest.raises(KeyError, match="Nonexistent"):
            atlas.load_atlas("Nonexistent")

    def test_caching(self):
        a = atlas.load_atlas("AAL")
        b = atlas.load_atlas("AAL")
        assert a is b


class TestResampleAtlas:
    def test_identity_resample(self):
        data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
        affine = np.eye(4)
        result = atlas.resample_atlas_to_stat(data, affine, data.shape, affine)
        np.testing.assert_array_equal(result, data)

    def test_downsampled_resample(self):
        # 4×4×4 atlas at 1mm
        atlas_data = np.zeros((4, 4, 4), dtype=np.int32)
        atlas_data[0:2, 0:2, 0:2] = 1
        atlas_data[2:4, 2:4, 2:4] = 2
        atlas_affine = np.eye(4)

        # 2×2×2 stat map at 2mm
        stat_shape = (2, 2, 2)
        stat_affine = np.diag([2.0, 2.0, 2.0, 1.0])

        result = atlas.resample_atlas_to_stat(atlas_data, atlas_affine, stat_shape, stat_affine)
        assert result.shape == stat_shape
        # Voxel (0,0,0) in stat → world (0,0,0) → atlas voxel (0,0,0) → label 1
        assert result[0, 0, 0] == 1
        # Voxel (1,1,1) in stat → world (2,2,2) → atlas voxel (2,2,2) → label 2
        assert result[1, 1, 1] == 2


class TestLabelPeak:
    def test_peak_in_region(self):
        atlas_data = np.zeros((5, 5, 5), dtype=np.int32)
        atlas_data[2, 2, 2] = 7
        affine = np.eye(4)
        labels = {"7": "Hippocampus_L"}
        result = atlas.label_peak((2, 2, 2), affine, atlas_data, affine, labels)
        assert result == "Hippocampus_L"

    def test_peak_outside_regions(self):
        atlas_data = np.zeros((5, 5, 5), dtype=np.int32)
        affine = np.eye(4)
        labels = {"7": "Hippocampus_L"}
        result = atlas.label_peak((0, 0, 0), affine, atlas_data, affine, labels)
        assert result == "\u2014"


class TestLabelCluster:
    def test_single_region_cluster(self):
        atlas_data = np.zeros((5, 5, 5), dtype=np.int32)
        atlas_data[1:4, 1:4, 1:4] = 3
        affine = np.eye(4)
        labels = {"3": "Putamen_L"}

        cluster_mask = np.zeros((5, 5, 5), dtype=bool)
        cluster_mask[2, 2, 2] = True
        cluster_mask[2, 2, 3] = True
        cluster_mask[2, 3, 2] = True

        result = atlas.label_cluster(cluster_mask, affine, atlas_data, affine, labels)
        assert len(result) == 1
        assert result[0]["name"] == "Putamen_L"
        assert result[0]["pct"] == 100.0

    def test_multi_region_cluster(self):
        atlas_data = np.zeros((10, 10, 10), dtype=np.int32)
        atlas_data[0:5, :, :] = 1
        atlas_data[5:10, :, :] = 2
        affine = np.eye(4)
        labels = {"1": "RegionA", "2": "RegionB"}

        cluster_mask = np.zeros((10, 10, 10), dtype=bool)
        cluster_mask[3:7, 5, 5] = True  # 4 voxels: 2 in region 1, 2 in region 2

        result = atlas.label_cluster(cluster_mask, affine, atlas_data, affine, labels)
        assert len(result) == 2
        names = [r["name"] for r in result]
        assert "RegionA" in names
        assert "RegionB" in names
        assert result[0]["voxels"] >= result[1]["voxels"]

    def test_cluster_outside_atlas(self):
        atlas_data = np.zeros((5, 5, 5), dtype=np.int32)
        affine = np.eye(4)
        labels = {"1": "SomeRegion"}

        cluster_mask = np.zeros((5, 5, 5), dtype=bool)
        cluster_mask[0, 0, 0] = True

        result = atlas.label_cluster(cluster_mask, affine, atlas_data, affine, labels)
        assert result == []


class TestAnnotateClusters:
    def test_adds_mni_and_atlas_fields(self):
        shape = (5, 5, 5)
        stat_affine = np.diag([2.0, 2.0, 2.0, 1.0])

        labeled_array = np.zeros(shape, dtype=np.int32)
        labeled_array[2, 2, 2] = 1
        labeled_array[2, 2, 3] = 1

        cluster_table = [{
            "label": 1,
            "size": 2,
            "peak_value": 3.5,
            "peak_coords": (2, 2, 2),
        }]

        atlas.annotate_clusters(cluster_table, stat_affine, shape, labeled_array)

        c = cluster_table[0]
        assert "peak_coords_mni" in c
        assert len(c["peak_coords_mni"]) == 3
        # World coords: affine @ [2,2,2,1] → (4.0, 4.0, 4.0)
        assert c["peak_coords_mni"] == (4.0, 4.0, 4.0)

        assert "peak_atlas" in c
        assert isinstance(c["peak_atlas"], dict)
        for atlas_name in atlas._ATLAS_REGISTRY:
            assert atlas_name in c["peak_atlas"]

        assert "atlas_regions" in c
        assert isinstance(c["atlas_regions"], dict)
        for atlas_name in atlas._ATLAS_REGISTRY:
            assert atlas_name in c["atlas_regions"]
            assert isinstance(c["atlas_regions"][atlas_name], list)

    def test_empty_cluster_table(self):
        atlas.annotate_clusters([], np.eye(4), (5, 5, 5), None)
