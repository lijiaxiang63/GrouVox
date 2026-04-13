# Atlas-Based Cluster Reporting

**Date:** 2026-04-13

## Overview

Enhance GRF cluster reporting to identify which brain atlas regions each cluster overlaps, and which region the peak voxel falls in. Supports three bundled atlases: AAL, HarvardOxford-cortical, and HarvardOxford-subcortical.

## Bundled Atlases

Three atlas NIfTI files and their label mappings are bundled into the package:

| Atlas | Source File | Resolution | Labels |
|-------|-----------|------------|--------|
| AAL | `aal.nii` | 1mm, 181×217×181 | 116 regions |
| HarvardOxford-cortical | `HarvardOxford-cort-maxprob-thr25-2mm_YCG.nii` | 2mm, 91×109×91 | 96 regions |
| HarvardOxford-subcortical | `HarvardOxford-sub-maxprob-thr25-2mm_YCG.nii` | 2mm, 91×109×91 | 16 regions |

All atlases are in MNI space. Label-to-name mappings are extracted from the corresponding `*_Labels.mat` files and stored as JSON for pure-Python loading (no `scipy.io` dependency at runtime).

### Package layout

```
src/grouvox/atlases/
    aal.nii.gz
    aal_labels.json
    HarvardOxford-cort-maxprob-thr25-2mm.nii.gz
    HarvardOxford-cort_labels.json
    HarvardOxford-sub-maxprob-thr25-2mm.nii.gz
    HarvardOxford-sub_labels.json
```

### Label JSON format

```json
{
  "1": "Precentral_L",
  "2": "Precentral_R",
  ...
}
```

Keys are string-encoded integer label values from the NIfTI volume. Value `0` ("None") is excluded.

## Enhanced Cluster Table Entry

Each cluster dict in `GRFResult.cluster_table` gains three new fields:

```python
{
    # Existing fields
    "label": 1,
    "size": 523,
    "peak_value": 4.32,
    "peak_coords": (45, 62, 38),

    # New fields
    "peak_coords_mni": (-12.0, 34.0, 56.0),
    "peak_atlas": {
        "AAL": "Frontal_Mid_L",
        "HarvardOxford-cortical": "Middle Frontal Gyrus (L)",
        "HarvardOxford-subcortical": "—",
    },
    "atlas_regions": {
        "AAL": [
            {"name": "Frontal_Mid_L",   "voxels": 312, "pct": 59.7},
            {"name": "Frontal_Sup_L",   "voxels": 189, "pct": 36.1},
            {"name": "Precentral_L",    "voxels":  22, "pct":  4.2},
        ],
        "HarvardOxford-cortical": [
            {"name": "Middle Frontal Gyrus (L)", "voxels": 401, "pct": 76.7},
            {"name": "Superior Frontal Gyrus (L)", "voxels": 122, "pct": 23.3},
        ],
        "HarvardOxford-subcortical": []
    },
}
```

- `peak_coords_mni`: voxel coords converted to MNI mm via the stat-map affine.
- `peak_atlas`: atlas region name at the peak voxel for each atlas. `"—"` if the peak falls outside all labeled regions (label 0).
- `atlas_regions`: per-atlas list of overlapping regions sorted by voxel count descending. Each entry has `name`, `voxels` (count of cluster voxels in that region), and `pct` (percentage of the cluster).

## CLI Output

### Terminal (GRF)

```
GRF correction applied:
  Z threshold: 3.2905
  Cluster size threshold: 82 voxels
  Surviving clusters: 3

  Cluster 1: 523 voxels, peak Z=4.320 at MNI (-12, 34, 56)
    Peak location:  AAL: Frontal_Mid_L | HO-cort: Middle Frontal Gyrus (L)
    AAL:
       59.7%  Frontal_Mid_L (312 voxels)
       36.1%  Frontal_Sup_L (189 voxels)
        4.2%  Precentral_L (22 voxels)
    HarvardOxford-cortical:
       76.7%  Middle Frontal Gyrus (L) (401 voxels)
       23.3%  Superior Frontal Gyrus (L) (122 voxels)

  Cluster 2: 201 voxels, peak Z=-3.891 at MNI (24, -8, 12)
    Peak location:  AAL: Putamen_R | HO-sub: Putamen
    AAL:
      100.0%  Putamen_R (201 voxels)
    HarvardOxford-subcortical:
      100.0%  Putamen (201 voxels)
```

Atlases with no overlap for a cluster are omitted from terminal output.

### CSV File — One Row Per Cluster

Saved as `ClusterReport_{stat_name}.csv` alongside the thresholded maps.

| Cluster | Size | PeakZ | MNI_X | MNI_Y | MNI_Z | AAL_Peak | AAL_Regions | HO_Cort_Peak | HO_Cort_Regions | HO_Sub_Peak | HO_Sub_Regions |
|---------|------|-------|-------|-------|-------|----------|-------------|--------------|-----------------|-------------|----------------|
| 1 | 523 | 4.320 | -12 | 34 | 56 | Frontal_Mid_L | Frontal_Mid_L(59.7%); Frontal_Sup_L(36.1%); Precentral_L(4.2%) | Middle Frontal Gyrus (L) | Middle Frontal Gyrus (L)(76.7%); Superior Frontal Gyrus (L)(23.3%) | — | — |
| 2 | 201 | -3.891 | 24 | -8 | 12 | Putamen_R | Putamen_R(100.0%) | — | — | Putamen | Putamen(100.0%) |

- `*_Peak` columns: atlas region name at the peak voxel.
- `*_Regions` columns: semicolon-delimited `RegionName(pct%)` sorted by overlap descending. `"—"` if no overlap.

## New Module: `src/grouvox/atlas.py`

Responsibilities:

1. **Load atlas**: load NIfTI + JSON labels for a named atlas. Cache loaded atlases in a module-level dict.
2. **Resample to stat-map space**: map atlas voxels to the stat-map grid via affine composition (nearest-neighbor). Both are MNI-space so this is a pure affine transform, no external resampling library needed.
3. **`label_peak(peak_voxel, affine, atlas_data, atlas_labels)`**: return the atlas region name at a given voxel coordinate.
4. **`label_cluster(cluster_mask, affine, atlas_data, atlas_labels)`**: return a list of `{"name", "voxels", "pct"}` dicts for all atlas regions overlapping the cluster.
5. **`annotate_clusters(cluster_table, stat_affine, stat_shape)`**: top-level function called from `grf_correction`. Adds `peak_coords_mni`, `peak_atlas`, and `atlas_regions` to each cluster dict in-place.

## Integration Points

### `correction.py`

After `cluster_table` is built (line ~205), call `annotate_clusters(cluster_table, affine, t_data.shape)`. The atlas annotation is always performed (atlases are bundled, lightweight).

### `cli.py`

Update the `correct` command's GRF output section to:
- Print MNI coordinates instead of voxel coordinates.
- Print peak atlas labels.
- Print per-atlas region breakdown.
- Write `ClusterReport_*.csv` via `csv.writer`.

### `pyproject.toml`

Add `[tool.hatch.build.targets.wheel] packages = ["src/grouvox"]` and ensure `src/grouvox/atlases/` is included as package data.

## Coordinate Transform

Voxel-to-MNI: `mni = affine @ [i, j, k, 1]`.

Atlas-to-stat-map resampling: compute `T = inv(atlas_affine) @ stat_affine`, then for each stat-map voxel, find the corresponding atlas voxel via nearest-neighbor rounding. This handles resolution mismatches (1mm AAL vs 2mm stat map, or vice versa).
