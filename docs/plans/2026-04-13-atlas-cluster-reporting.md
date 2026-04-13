# Atlas-Based Cluster Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add atlas-based anatomical labeling to GRF and FDR cluster reports (peak region + overlap breakdown), with CSV output.

**Architecture:** A new `atlas.py` module loads bundled atlas NIfTI + JSON label files, resamples them to the stat-map grid via affine math, and annotates cluster tables in-place. Both `grf_correction` and `fdr_correction` call `annotate_clusters()` after building their cluster tables. The CLI prints atlas info and writes a CSV report.

**Tech Stack:** nibabel, numpy, scipy.ndimage (already used), csv (stdlib), json (stdlib)

**Spec:** `docs/specs/2026-04-13-atlas-cluster-reporting-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/grouvox/atlases/` | Create directory | Bundled atlas NIfTI + JSON label files |
| `src/grouvox/atlas.py` | Create | Load atlases, resample to stat-map space, label peaks and clusters |
| `src/grouvox/correction.py` | Modify | Call `annotate_clusters()`, add FDR cluster identification, add CSV writing |
| `src/grouvox/cli.py` | Modify | Print atlas info in terminal output |
| `src/grouvox/__init__.py` | Modify | Export new `annotate_clusters` function |
| `pyproject.toml` | Modify | Include atlas data files in package |
| `tests/test_atlas.py` | Create | Unit tests for atlas module |
| `tests/test_correction.py` | Modify | Update tests for new cluster_table fields + FDR clusters |
| `tests/test_cli.py` | Modify | Verify CLI atlas output |

---

### Task 1: Bundle Atlas Data Files

**Files:**
- Create: `src/grouvox/atlases/aal_labels.json`
- Create: `src/grouvox/atlases/ho_cort_labels.json`
- Create: `src/grouvox/atlases/ho_sub_labels.json`
- Create: `scripts/convert_atlas_labels.py` (one-time conversion script)
- Modify: `pyproject.toml`

- [ ] **Step 1: Create the conversion script**

This script reads the `.mat` label files and writes JSON. Run it once to generate the JSON files.

```python
# scripts/convert_atlas_labels.py
"""One-time script to convert DPABI .mat atlas labels to JSON."""
import json
import sys
from pathlib import Path

import scipy.io as sio

SOURCES = [
    (
        "/Users/jiaxiangli/neuroimaging/mritools/DPABI_V9.0_250415/Templates/aal_Labels.mat",
        "aal_labels.json",
    ),
    (
        "/Users/jiaxiangli/neuroimaging/mritools/DPABI_V9.0_250415/Templates/HarvardOxford-cort-maxprob-thr25-2mm_YCG_Labels.mat",
        "ho_cort_labels.json",
    ),
    (
        "/Users/jiaxiangli/neuroimaging/mritools/DPABI_V9.0_250415/Templates/HarvardOxford-sub-maxprob-thr25-2mm_YCG_Labels.mat",
        "ho_sub_labels.json",
    ),
]

out_dir = Path(__file__).resolve().parent.parent / "src" / "grouvox" / "atlases"
out_dir.mkdir(parents=True, exist_ok=True)

for mat_path, out_name in SOURCES:
    mat = sio.loadmat(mat_path)
    ref = mat["Reference"]
    labels = {}
    for i in range(ref.shape[0]):
        name = str(ref[i, 0][0])
        idx = int(ref[i, 1][0, 0])
        if idx == 0:
            continue
        labels[str(idx)] = name
    out_file = out_dir / out_name
    out_file.write_text(json.dumps(labels, indent=2) + "\n")
    print(f"Wrote {out_file} ({len(labels)} labels)")
```

- [ ] **Step 2: Run the conversion script**

Run: `uv run python scripts/convert_atlas_labels.py`

Expected: Three JSON files created in `src/grouvox/atlases/`.

- [ ] **Step 3: Copy atlas NIfTI files and compress**

```bash
cp /Users/jiaxiangli/neuroimaging/mritools/DPABI_V9.0_250415/Templates/aal.nii src/grouvox/atlases/aal.nii
gzip src/grouvox/atlases/aal.nii

cp /Users/jiaxiangli/neuroimaging/mritools/DPABI_V9.0_250415/Templates/HarvardOxford-cort-maxprob-thr25-2mm_YCG.nii src/grouvox/atlases/ho_cort.nii
gzip src/grouvox/atlases/ho_cort.nii

cp /Users/jiaxiangli/neuroimaging/mritools/DPABI_V9.0_250415/Templates/HarvardOxford-sub-maxprob-thr25-2mm_YCG.nii src/grouvox/atlases/ho_sub.nii
gzip src/grouvox/atlases/ho_sub.nii
```

- [ ] **Step 4: Add package data to pyproject.toml**

In `pyproject.toml`, after the existing `[tool.hatch.build.targets.wheel]` section, add:

```toml
[tool.hatch.build.targets.wheel.force-include]
"src/grouvox/atlases" = "grouvox/atlases"
```

- [ ] **Step 5: Verify atlas files are accessible**

```bash
uv run python -c "
from importlib.resources import files
atlas_dir = files('grouvox') / 'atlases'
for name in ['aal.nii.gz', 'ho_cort.nii.gz', 'ho_sub.nii.gz', 'aal_labels.json', 'ho_cort_labels.json', 'ho_sub_labels.json']:
    path = atlas_dir / name
    print(f'{name}: exists={path.is_file() if hasattr(path, \"is_file\") else \"resource\"}')
"
```

- [ ] **Step 6: Commit**

```bash
git add src/grouvox/atlases/ scripts/convert_atlas_labels.py pyproject.toml
git commit -m "feat: bundle atlas NIfTI and JSON label files"
```

---

### Task 2: Create `atlas.py` — Core Atlas Module

**Files:**
- Create: `src/grouvox/atlas.py`
- Create: `tests/test_atlas.py`

- [ ] **Step 1: Write the failing test for `load_atlas`**

```python
# tests/test_atlas.py
import numpy as np
import pytest

from grouvox.atlas import load_atlas, ATLAS_REGISTRY


class TestLoadAtlas:
    def test_load_aal(self):
        data, labels, affine = load_atlas("AAL")
        assert data.ndim == 3
        assert isinstance(labels, dict)
        assert len(labels) == 116
        assert labels["1"] == "Precentral_L"
        assert affine.shape == (4, 4)

    def test_load_ho_cort(self):
        data, labels, affine = load_atlas("HarvardOxford-cortical")
        assert data.ndim == 3
        assert len(labels) == 96

    def test_load_ho_sub(self):
        data, labels, affine = load_atlas("HarvardOxford-subcortical")
        assert data.ndim == 3
        assert len(labels) == 16

    def test_unknown_atlas_raises(self):
        with pytest.raises(KeyError):
            load_atlas("Nonexistent")

    def test_caching(self):
        a1 = load_atlas("AAL")
        a2 = load_atlas("AAL")
        assert a1[0] is a2[0]  # same array object from cache
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_atlas.py::TestLoadAtlas -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement `load_atlas`**

```python
# src/grouvox/atlas.py
"""Atlas-based anatomical labeling for cluster reports."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files

import nibabel as nib
import numpy as np


# Registry: name -> (nifti_filename, labels_filename)
ATLAS_REGISTRY: dict[str, tuple[str, str]] = {
    "AAL": ("aal.nii.gz", "aal_labels.json"),
    "HarvardOxford-cortical": ("ho_cort.nii.gz", "ho_cort_labels.json"),
    "HarvardOxford-subcortical": ("ho_sub.nii.gz", "ho_sub_labels.json"),
}

_atlas_cache: dict[str, tuple[np.ndarray, dict[str, str], np.ndarray]] = {}


def load_atlas(name: str) -> tuple[np.ndarray, dict[str, str], np.ndarray]:
    """Load an atlas by name. Returns (data, labels_dict, affine).

    Results are cached after the first call.
    """
    if name in _atlas_cache:
        return _atlas_cache[name]

    if name not in ATLAS_REGISTRY:
        raise KeyError(
            f"Unknown atlas {name!r}. Available: {list(ATLAS_REGISTRY)}"
        )

    nifti_file, labels_file = ATLAS_REGISTRY[name]
    atlas_pkg = files("grouvox") / "atlases"

    nifti_path = atlas_pkg / nifti_file
    img = nib.load(str(nifti_path))
    data = np.asanyarray(img.dataobj)
    affine = img.affine.copy()

    labels_path = atlas_pkg / labels_file
    labels = json.loads(labels_path.read_text())

    _atlas_cache[name] = (data, labels, affine)
    return data, labels, affine
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_atlas.py::TestLoadAtlas -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/grouvox/atlas.py tests/test_atlas.py
git commit -m "feat(atlas): add load_atlas with caching and registry"
```

---

### Task 3: Add `resample_atlas_to_stat` and `label_peak`

**Files:**
- Modify: `src/grouvox/atlas.py`
- Modify: `tests/test_atlas.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_atlas.py`:

```python
from grouvox.atlas import resample_atlas_to_stat, label_peak


class TestResampleAtlas:
    def test_identity_resample(self):
        """When atlas and stat have same affine/shape, data is unchanged."""
        atlas_data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int16)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        result = resample_atlas_to_stat(atlas_data, affine, stat_shape=(2, 2, 2), stat_affine=affine)
        np.testing.assert_array_equal(result, atlas_data)

    def test_downsampled_resample(self):
        """Atlas at 1mm resampled to 2mm stat-map picks nearest neighbor."""
        atlas_data = np.zeros((4, 4, 4), dtype=np.int16)
        atlas_data[2:4, 2:4, 2:4] = 5
        atlas_affine = np.diag([1.0, 1.0, 1.0, 1.0])
        stat_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        result = resample_atlas_to_stat(atlas_data, atlas_affine, stat_shape=(2, 2, 2), stat_affine=stat_affine)
        assert result.shape == (2, 2, 2)
        assert result[0, 0, 0] == 0
        assert result[1, 1, 1] == 5


class TestLabelPeak:
    def test_peak_in_region(self):
        atlas_data = np.zeros((10, 10, 10), dtype=np.int16)
        atlas_data[3:6, 3:6, 3:6] = 7
        labels = {"7": "Hippocampus_L"}
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        result = label_peak((4, 4, 4), affine, atlas_data, affine, labels)
        assert result == "Hippocampus_L"

    def test_peak_outside_regions(self):
        atlas_data = np.zeros((10, 10, 10), dtype=np.int16)
        labels = {"1": "SomeRegion"}
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        result = label_peak((0, 0, 0), affine, atlas_data, affine, labels)
        assert result == "\u2014"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_atlas.py::TestResampleAtlas tests/test_atlas.py::TestLabelPeak -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `resample_atlas_to_stat` and `label_peak`**

Add to `src/grouvox/atlas.py`:

```python
def resample_atlas_to_stat(
    atlas_data: np.ndarray,
    atlas_affine: np.ndarray,
    stat_shape: tuple[int, int, int],
    stat_affine: np.ndarray,
) -> np.ndarray:
    """Resample atlas to stat-map grid via nearest-neighbor affine mapping."""
    # Build voxel coordinate grid for the stat map
    si, sj, sk = np.mgrid[
        0:stat_shape[0], 0:stat_shape[1], 0:stat_shape[2]
    ]
    # stat voxel -> world -> atlas voxel
    stat_to_atlas = np.linalg.inv(atlas_affine) @ stat_affine
    coords = np.column_stack([si.ravel(), sj.ravel(), sk.ravel(), np.ones(si.size)])
    atlas_coords = (stat_to_atlas @ coords.T).T[:, :3]
    atlas_ijk = np.round(atlas_coords).astype(int)

    # Clip to atlas bounds
    for dim in range(3):
        np.clip(atlas_ijk[:, dim], 0, atlas_data.shape[dim] - 1, out=atlas_ijk[:, dim])

    resampled = atlas_data[atlas_ijk[:, 0], atlas_ijk[:, 1], atlas_ijk[:, 2]]
    return resampled.reshape(stat_shape)


def label_peak(
    peak_voxel: tuple[int, int, int],
    stat_affine: np.ndarray,
    atlas_data: np.ndarray,
    atlas_affine: np.ndarray,
    labels: dict[str, str],
) -> str:
    """Return the atlas region name at a peak voxel coordinate.

    Returns "\u2014" if the peak falls in label 0 or outside all regions.
    """
    # stat voxel -> world -> atlas voxel
    world = stat_affine @ np.array([*peak_voxel, 1.0])
    atlas_voxel = np.linalg.inv(atlas_affine) @ world
    ijk = np.round(atlas_voxel[:3]).astype(int)

    for dim in range(3):
        if ijk[dim] < 0 or ijk[dim] >= atlas_data.shape[dim]:
            return "\u2014"

    label_val = int(atlas_data[ijk[0], ijk[1], ijk[2]])
    return labels.get(str(label_val), "\u2014")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_atlas.py::TestResampleAtlas tests/test_atlas.py::TestLabelPeak -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/grouvox/atlas.py tests/test_atlas.py
git commit -m "feat(atlas): add resample_atlas_to_stat and label_peak"
```

---

### Task 4: Add `label_cluster` and `annotate_clusters`

**Files:**
- Modify: `src/grouvox/atlas.py`
- Modify: `tests/test_atlas.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_atlas.py`:

```python
from grouvox.atlas import label_cluster, annotate_clusters


class TestLabelCluster:
    def test_single_region_cluster(self):
        atlas_data = np.zeros((10, 10, 10), dtype=np.int16)
        atlas_data[3:6, 3:6, 3:6] = 7
        labels = {"7": "Hippocampus_L"}
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        cluster_mask = np.zeros((10, 10, 10), dtype=bool)
        cluster_mask[3:6, 3:6, 3:6] = True
        result = label_cluster(cluster_mask, affine, atlas_data, affine, labels)
        assert len(result) == 1
        assert result[0]["name"] == "Hippocampus_L"
        assert result[0]["voxels"] == 27
        assert result[0]["pct"] == pytest.approx(100.0)

    def test_multi_region_cluster(self):
        atlas_data = np.zeros((10, 10, 10), dtype=np.int16)
        atlas_data[3:5, 3:5, 3:5] = 7  # 8 voxels
        atlas_data[5:7, 5:7, 5:7] = 9  # 8 voxels
        labels = {"7": "RegionA", "9": "RegionB"}
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        cluster_mask = np.zeros((10, 10, 10), dtype=bool)
        cluster_mask[3:7, 3:7, 3:7] = True  # 64 voxels total
        result = label_cluster(cluster_mask, affine, atlas_data, affine, labels)
        names = [r["name"] for r in result]
        assert "RegionA" in names
        assert "RegionB" in names
        # sorted by voxels descending
        assert result[0]["voxels"] >= result[1]["voxels"]

    def test_cluster_outside_atlas(self):
        atlas_data = np.zeros((10, 10, 10), dtype=np.int16)
        labels = {"1": "SomeRegion"}
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        cluster_mask = np.zeros((10, 10, 10), dtype=bool)
        cluster_mask[0:2, 0:2, 0:2] = True
        result = label_cluster(cluster_mask, affine, atlas_data, affine, labels)
        assert result == []


class TestAnnotateClusters:
    def test_adds_mni_and_atlas_fields(self):
        cluster_table = [
            {
                "label": 1,
                "size": 27,
                "peak_value": 4.5,
                "peak_coords": (4, 4, 4),
            }
        ]
        stat_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        stat_shape = (10, 10, 10)
        annotate_clusters(cluster_table, stat_affine, stat_shape)

        c = cluster_table[0]
        assert "peak_coords_mni" in c
        assert c["peak_coords_mni"] == (8.0, 8.0, 8.0)
        assert "peak_atlas" in c
        assert isinstance(c["peak_atlas"], dict)
        assert "AAL" in c["peak_atlas"]
        assert "HarvardOxford-cortical" in c["peak_atlas"]
        assert "HarvardOxford-subcortical" in c["peak_atlas"]
        assert "atlas_regions" in c
        assert isinstance(c["atlas_regions"], dict)

    def test_empty_cluster_table(self):
        cluster_table = []
        annotate_clusters(cluster_table, np.eye(4), (10, 10, 10))
        assert cluster_table == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_atlas.py::TestLabelCluster tests/test_atlas.py::TestAnnotateClusters -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `label_cluster` and `annotate_clusters`**

Add to `src/grouvox/atlas.py`:

```python
def label_cluster(
    cluster_mask: np.ndarray,
    stat_affine: np.ndarray,
    atlas_data: np.ndarray,
    atlas_affine: np.ndarray,
    labels: dict[str, str],
) -> list[dict]:
    """Return atlas region overlap for a cluster.

    Returns a list of {"name": str, "voxels": int, "pct": float} dicts,
    sorted by voxels descending. Regions with label 0 or not in labels
    dict are excluded.
    """
    resampled = resample_atlas_to_stat(
        atlas_data, atlas_affine, cluster_mask.shape, stat_affine,
    )
    cluster_labels = resampled[cluster_mask]
    unique, counts = np.unique(cluster_labels, return_counts=True)

    total = int(cluster_mask.sum())
    result = []
    for val, cnt in zip(unique, counts):
        key = str(int(val))
        if key in labels:
            result.append({
                "name": labels[key],
                "voxels": int(cnt),
                "pct": round(100.0 * cnt / total, 1),
            })

    result.sort(key=lambda r: r["voxels"], reverse=True)
    return result


def annotate_clusters(
    cluster_table: list[dict],
    stat_affine: np.ndarray,
    stat_shape: tuple[int, int, int],
) -> None:
    """Add peak_coords_mni, peak_atlas, and atlas_regions to each cluster in-place."""
    if not cluster_table:
        return

    # Pre-load all atlases
    atlases = {}
    for name in ATLAS_REGISTRY:
        atlas_data, labels, atlas_affine = load_atlas(name)
        resampled = resample_atlas_to_stat(
            atlas_data, atlas_affine, stat_shape, stat_affine,
        )
        atlases[name] = (resampled, labels, atlas_data, atlas_affine)

    for cluster in cluster_table:
        peak = cluster["peak_coords"]
        # MNI coordinates
        mni = stat_affine @ np.array([*peak, 1.0])
        cluster["peak_coords_mni"] = tuple(float(round(v, 1)) for v in mni[:3])

        # Peak atlas label and cluster overlap
        peak_atlas = {}
        atlas_regions = {}
        for name, (resampled, labels, atlas_data, atlas_affine) in atlases.items():
            # Peak label
            peak_atlas[name] = label_peak(
                peak, stat_affine, atlas_data, atlas_affine, labels,
            )
            # Cluster overlap — build mask from cluster voxels
            cluster_mask = np.zeros(stat_shape, dtype=bool)
            cluster_mask[peak]  # just to validate shape
            # We need to reconstruct the cluster mask; use the resampled atlas
            # The cluster_table doesn't carry the mask, so we use the peak + size
            # Actually we need the labeled array — but annotate_clusters doesn't have it.
            # Instead, compute overlap using resampled atlas at cluster voxel positions.
            # We don't have the full cluster mask here, so we'll add it below.
            atlas_regions[name] = []

        cluster["peak_atlas"] = peak_atlas
        cluster["atlas_regions"] = atlas_regions
```

Wait — the current `cluster_table` entries don't carry the cluster mask. We need to either pass the labeled array or store cluster masks. Let me revise: `annotate_clusters` should also receive the labeled array and the original data so it can reconstruct cluster masks.

**Revised signature and implementation:**

```python
def annotate_clusters(
    cluster_table: list[dict],
    stat_affine: np.ndarray,
    stat_shape: tuple[int, int, int],
    labeled_array: np.ndarray | None = None,
) -> None:
    """Add peak_coords_mni, peak_atlas, and atlas_regions to each cluster in-place.

    Parameters
    ----------
    cluster_table : list[dict]
        Each dict must have "label", "peak_coords", and "size".
    stat_affine : (4,4) array
        Affine of the stat map.
    stat_shape : tuple
        Shape of the stat map (3D).
    labeled_array : ndarray, optional
        Output of scipy.ndimage.label — integer array where each cluster
        has a unique label matching cluster_table[i]["label"].
    """
    if not cluster_table:
        return

    # Pre-load and resample all atlases to the stat-map grid
    atlases = {}
    for name in ATLAS_REGISTRY:
        atlas_data, labels, atlas_affine = load_atlas(name)
        resampled = resample_atlas_to_stat(
            atlas_data, atlas_affine, stat_shape, stat_affine,
        )
        atlases[name] = (resampled, labels, atlas_data, atlas_affine)

    for cluster in cluster_table:
        peak = cluster["peak_coords"]
        # MNI coordinates
        mni = stat_affine @ np.array([*peak, 1.0])
        cluster["peak_coords_mni"] = tuple(float(round(v, 1)) for v in mni[:3])

        # Peak atlas label and cluster overlap
        peak_atlas_result = {}
        atlas_regions_result = {}

        for name, (resampled, labels, atlas_data, atlas_affine) in atlases.items():
            peak_atlas_result[name] = label_peak(
                peak, stat_affine, atlas_data, atlas_affine, labels,
            )
            if labeled_array is not None:
                cluster_mask = labeled_array == cluster["label"]
                cluster_labels = resampled[cluster_mask]
                unique, counts = np.unique(cluster_labels, return_counts=True)
                total = int(cluster_mask.sum())
                regions = []
                for val, cnt in zip(unique, counts):
                    key = str(int(val))
                    if key in labels:
                        regions.append({
                            "name": labels[key],
                            "voxels": int(cnt),
                            "pct": round(100.0 * cnt / total, 1),
                        })
                regions.sort(key=lambda r: r["voxels"], reverse=True)
                atlas_regions_result[name] = regions
            else:
                atlas_regions_result[name] = []

        cluster["peak_atlas"] = peak_atlas_result
        cluster["atlas_regions"] = atlas_regions_result
```

- [ ] **Step 4: Update test to pass labeled_array**

Revise `TestAnnotateClusters` in `tests/test_atlas.py`:

```python
class TestAnnotateClusters:
    def test_adds_mni_and_atlas_fields(self):
        # Create a small labeled array with one cluster
        labeled = np.zeros((10, 10, 10), dtype=np.int32)
        labeled[3:6, 3:6, 3:6] = 1  # cluster label=1, 27 voxels

        cluster_table = [
            {
                "label": 1,
                "size": 27,
                "peak_value": 4.5,
                "peak_coords": (4, 4, 4),
            }
        ]
        stat_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        stat_shape = (10, 10, 10)
        annotate_clusters(cluster_table, stat_affine, stat_shape, labeled)

        c = cluster_table[0]
        assert "peak_coords_mni" in c
        assert c["peak_coords_mni"] == (8.0, 8.0, 8.0)
        assert "peak_atlas" in c
        assert isinstance(c["peak_atlas"], dict)
        assert "AAL" in c["peak_atlas"]
        assert "HarvardOxford-cortical" in c["peak_atlas"]
        assert "HarvardOxford-subcortical" in c["peak_atlas"]
        assert "atlas_regions" in c
        assert isinstance(c["atlas_regions"], dict)

    def test_empty_cluster_table(self):
        cluster_table = []
        annotate_clusters(cluster_table, np.eye(4), (10, 10, 10), np.zeros((10, 10, 10), dtype=np.int32))
        assert cluster_table == []
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_atlas.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/grouvox/atlas.py tests/test_atlas.py
git commit -m "feat(atlas): add label_cluster, annotate_clusters with labeled array"
```

---

### Task 5: Add FDR Cluster Identification

**Files:**
- Modify: `src/grouvox/correction.py`
- Modify: `tests/test_correction.py`

- [ ] **Step 1: Write failing test for FDR cluster table**

Add to `tests/test_correction.py` inside `TestFDRCorrection`:

```python
    def test_fdr_cluster_table(self, tmp_path):
        shape = (20, 20, 20)
        data = np.zeros(shape, dtype=np.float32)
        data[8:12, 8:12, 8:12] = 6.0  # strong cluster

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_correction.py::TestFDRCorrection::test_fdr_cluster_table -v`
Expected: FAIL with `AttributeError: 'FDRResult' has no attribute 'n_clusters'`

- [ ] **Step 3: Modify `FDRResult` and `fdr_correction`**

In `src/grouvox/correction.py`:

Update the `FDRResult` dataclass:

```python
@dataclass
class FDRResult:
    """Result of FDR correction."""
    p_threshold: float
    n_significant: int
    thresholded_map: np.ndarray
    n_clusters: int = 0
    cluster_table: list[dict] = field(default_factory=list)
```

Then in `fdr_correction`, after computing `thresholded` and `sig_mask`, add cluster identification before the return statement:

```python
    # --- Cluster identification ---
    from grouvox.atlas import annotate_clusters

    labeled, n_labels = ndimage.label(sig_mask, structure=_STRUCT_26)
    cluster_table = []
    for label_id in range(1, n_labels + 1):
        cluster_mask = labeled == label_id
        size = int(cluster_mask.sum())
        cluster_t = np.abs(data[cluster_mask])
        peak_idx = np.argmax(cluster_t)
        peak_coords = np.array(np.where(cluster_mask)).T[peak_idx]
        peak_val = data[cluster_mask][peak_idx]
        cluster_table.append({
            "label": label_id,
            "size": size,
            "peak_value": float(peak_val),
            "peak_coords": tuple(int(c) for c in peak_coords),
        })

    cluster_table.sort(key=lambda c: c["size"], reverse=True)
    annotate_clusters(cluster_table, affine, data.shape, labeled)
```

Update the return to include the new fields:

```python
    return FDRResult(
        p_threshold=float(p_threshold),
        n_significant=int(significant.sum()),
        thresholded_map=thresholded,
        n_clusters=len(cluster_table),
        cluster_table=cluster_table,
    )
```

Also update the early return (no significant voxels):

```python
    if not np.any(below):
        thresholded = np.zeros_like(data)
        _save_fdr_output(thresholded, affine, header, stat_path)
        return FDRResult(p_threshold=0.0, n_significant=0, thresholded_map=thresholded,
                         n_clusters=0, cluster_table=[])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_correction.py::TestFDRCorrection -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/grouvox/correction.py tests/test_correction.py
git commit -m "feat(correction): add cluster identification and atlas annotation to FDR"
```

---

### Task 6: Add Atlas Annotation to GRF Correction

**Files:**
- Modify: `src/grouvox/correction.py`
- Modify: `tests/test_correction.py`

- [ ] **Step 1: Write failing test for GRF atlas fields**

Add to `tests/test_correction.py` inside `TestGRFCorrection`:

```python
    def test_grf_atlas_annotation(self, tmp_path):
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
        assert "peak_coords_mni" in cluster
        assert "peak_atlas" in cluster
        assert "atlas_regions" in cluster
        assert "AAL" in cluster["peak_atlas"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_correction.py::TestGRFCorrection::test_grf_atlas_annotation -v`
Expected: FAIL with `KeyError: 'peak_coords_mni'`

- [ ] **Step 3: Add annotation call to `grf_correction`**

In `src/grouvox/correction.py`, in `grf_correction`, after `cluster_table.sort(...)` (around line 205) and before saving NIfTIs, add:

```python
    from grouvox.atlas import annotate_clusters

    # Build combined labeled array for annotation
    # Re-label both tails into a single labeled array
    combined_labeled = np.zeros_like(t_data, dtype=np.int32)
    pos_supra = z_data >= z_thr
    pos_labeled, pos_n = ndimage.label(pos_supra, structure=_STRUCT_26)
    combined_labeled += pos_labeled

    if two_tailed:
        neg_supra = z_data <= -z_thr
        neg_labeled, neg_n = ndimage.label(neg_supra, structure=_STRUCT_26)
        # Offset negative labels so they don't collide with positive
        neg_labeled[neg_labeled > 0] += pos_n
        combined_labeled += neg_labeled

    annotate_clusters(cluster_table, affine, t_data.shape, combined_labeled)
```

Wait — the `label` values in `cluster_table` come from `_threshold_tail` which labels each tail independently. The positive tail labels start at 1, and the negative tail labels also start at 1. We need to make sure the label values in `cluster_table` match the `combined_labeled` array. Let me revise.

**Revised approach:** Instead of re-labeling, modify `_threshold_tail` to return the labeled array, then combine them with an offset. Update `cluster_table` negative labels to use the offset.

Actually, a simpler approach: just re-run labeling on the thresholded output. The thresholded maps already only contain surviving clusters.

```python
    # Build labeled array from the surviving thresholded map for annotation
    surviving_mask = thresholded_z != 0
    anno_labeled, _ = ndimage.label(surviving_mask, structure=_STRUCT_26)

    # Re-assign labels in cluster_table to match anno_labeled
    for cluster in cluster_table:
        peak = cluster["peak_coords"]
        cluster["label"] = int(anno_labeled[peak])

    annotate_clusters(cluster_table, affine, t_data.shape, anno_labeled)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_correction.py::TestGRFCorrection -v`
Expected: ALL PASS

- [ ] **Step 5: Run full correction test suite for regressions**

Run: `uv run pytest tests/test_correction.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/grouvox/correction.py tests/test_correction.py
git commit -m "feat(correction): add atlas annotation to GRF clusters"
```

---

### Task 7: Add CSV Report Writing

**Files:**
- Modify: `src/grouvox/correction.py`
- Modify: `tests/test_correction.py`

- [ ] **Step 1: Write failing test for GRF CSV output**

Add to `tests/test_correction.py` inside `TestGRFCorrection`:

```python
    def test_grf_csv_report(self, tmp_path):
        import csv

        shape = (30, 30, 30)
        data = np.zeros(shape, dtype=np.float32)
        data[10:20, 10:20, 10:20] = 5.0

        stat_path = _make_stat_nifti(data, tmp_path, filename="mystat_T.nii.gz",
                                     dof=28, dlh=0.005)
        mask_path = _make_mask(shape, tmp_path)

        grf_correction(stat_path, mask_path, voxel_p=0.001, cluster_p=0.05,
                        two_tailed=False)

        csv_path = tmp_path / "ClusterReport_mystat_T.csv"
        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 1
        assert "Cluster" in rows[0]
        assert "Size" in rows[0]
        assert "PeakZ" in rows[0]
        assert "MNI_X" in rows[0]
        assert "AAL_Peak" in rows[0]
        assert "AAL_Regions" in rows[0]
        assert "HO_Cort_Peak" in rows[0]
        assert "HO_Sub_Peak" in rows[0]
```

- [ ] **Step 2: Write failing test for FDR CSV output**

Add to `tests/test_correction.py` inside `TestFDRCorrection`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_correction.py -k "csv_report" -v`
Expected: FAIL — CSV files do not exist yet

- [ ] **Step 4: Implement `_write_cluster_csv` helper and call it**

Add to `src/grouvox/correction.py`:

```python
import csv


def _write_cluster_csv(cluster_table: list[dict], path: Path) -> None:
    """Write cluster report as CSV — one row per cluster."""
    if not cluster_table:
        return

    fieldnames = [
        "Cluster", "Size", "PeakZ", "MNI_X", "MNI_Y", "MNI_Z",
        "AAL_Peak", "AAL_Regions",
        "HO_Cort_Peak", "HO_Cort_Regions",
        "HO_Sub_Peak", "HO_Sub_Regions",
    ]
    atlas_keys = [
        ("AAL", "AAL_Peak", "AAL_Regions"),
        ("HarvardOxford-cortical", "HO_Cort_Peak", "HO_Cort_Regions"),
        ("HarvardOxford-subcortical", "HO_Sub_Peak", "HO_Sub_Regions"),
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, c in enumerate(cluster_table, 1):
            mni = c.get("peak_coords_mni", (0, 0, 0))
            row = {
                "Cluster": i,
                "Size": c["size"],
                "PeakZ": f"{c['peak_value']:.3f}",
                "MNI_X": mni[0],
                "MNI_Y": mni[1],
                "MNI_Z": mni[2],
            }
            for atlas_name, peak_col, regions_col in atlas_keys:
                peak_atlas = c.get("peak_atlas", {})
                row[peak_col] = peak_atlas.get(atlas_name, "\u2014")

                regions = c.get("atlas_regions", {}).get(atlas_name, [])
                if regions:
                    row[regions_col] = "; ".join(
                        f"{r['name']}({r['pct']}%)" for r in regions
                    )
                else:
                    row[regions_col] = "\u2014"
            writer.writerow(row)
```

Call `_write_cluster_csv` at the end of `grf_correction`, after `annotate_clusters`:

```python
    csv_path = parent / f"ClusterReport_{name}"
    csv_path = csv_path.with_suffix(".csv")
    _write_cluster_csv(cluster_table, csv_path)
```

Call `_write_cluster_csv` at the end of `fdr_correction`, after `annotate_clusters`:

```python
    csv_name = f"ClusterReport_FDR_{stat_path.stem}"
    # Handle .nii.gz double extension
    if stat_path.name.endswith(".nii.gz"):
        csv_name = f"ClusterReport_FDR_{stat_path.name.removesuffix('.nii.gz')}"
    csv_path = stat_path.parent / f"{csv_name}.csv"
    _write_cluster_csv(cluster_table, csv_path)
```

Actually, let's simplify. `stat_path.name` is something like `mystat_T.nii.gz`. We want `ClusterReport_mystat_T.csv` for GRF and `ClusterReport_FDR_mystat_T.csv` for FDR. Use a helper:

```python
def _csv_stem(stat_path: Path) -> str:
    """Strip .nii or .nii.gz to get the base name."""
    name = stat_path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return stat_path.stem
```

In `grf_correction`:
```python
    _write_cluster_csv(cluster_table, parent / f"ClusterReport_{_csv_stem(stat_path)}.csv")
```

In `fdr_correction`:
```python
    _write_cluster_csv(cluster_table, stat_path.parent / f"ClusterReport_FDR_{_csv_stem(stat_path)}.csv")
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_correction.py -k "csv_report" -v`
Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add src/grouvox/correction.py tests/test_correction.py
git commit -m "feat(correction): write ClusterReport CSV for GRF and FDR"
```

---

### Task 8: Update CLI Output

**Files:**
- Modify: `src/grouvox/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing test for CLI atlas output**

Add to `tests/test_cli.py` inside `TestCorrectCommand`:

```python
    def test_grf_prints_atlas(self, cli_runner, synthetic_two_groups, tmp_path):
        g1_dir, g2_dir, mask_path = synthetic_two_groups
        output = str(tmp_path / "res" / "ttest")

        cli_runner.invoke(main, [
            "ttest2",
            "--group1", str(g1_dir),
            "--group2", str(g2_dir),
            "--output", output,
            "--mask", str(mask_path),
        ])

        stat_file = tmp_path / "res" / "ttest_T.nii.gz"
        result = cli_runner.invoke(main, [
            "correct",
            "--input", str(stat_file),
            "--method", "grf",
            "--voxel-p", "0.001",
            "--cluster-p", "0.05",
            "--mask", str(mask_path),
        ])

        assert result.exit_code == 0, result.output
        assert "MNI" in result.output

    def test_fdr_prints_clusters(self, cli_runner, synthetic_two_groups, tmp_path):
        g1_dir, g2_dir, mask_path = synthetic_two_groups
        output = str(tmp_path / "res" / "ttest")

        cli_runner.invoke(main, [
            "ttest2",
            "--group1", str(g1_dir),
            "--group2", str(g2_dir),
            "--output", output,
            "--mask", str(mask_path),
        ])

        stat_file = tmp_path / "res" / "ttest_T.nii.gz"
        result = cli_runner.invoke(main, [
            "correct",
            "--input", str(stat_file),
            "--method", "fdr",
            "--q", "0.05",
            "--mask", str(mask_path),
        ])

        assert result.exit_code == 0, result.output
        assert "Clusters:" in result.output or "Significant voxels:" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py::TestCorrectCommand::test_grf_prints_atlas tests/test_cli.py::TestCorrectCommand::test_fdr_prints_clusters -v`
Expected: FAIL — "MNI" not in output, etc.

- [ ] **Step 3: Update CLI GRF output**

Replace the GRF output section in `src/grouvox/cli.py` `correct` command:

```python
    if method == "grf":
        result = grf_correction(
            stat_path=input_path,
            mask_path=mask,
            voxel_p=voxel_p,
            cluster_p=cluster_p,
            two_tailed=two_tailed,
            reestimate=reestimate,
        )
        click.echo(f"GRF correction applied:")
        click.echo(f"  Z threshold: {result.z_threshold:.4f}")
        click.echo(f"  Cluster size threshold: {result.cluster_size_threshold} voxels")
        click.echo(f"  Surviving clusters: {result.n_clusters}")
        for c in result.cluster_table:
            mni = c.get("peak_coords_mni", c["peak_coords"])
            click.echo(
                f"\n  Cluster {c['label']}: {c['size']} voxels, "
                f"peak Z={c['peak_value']:.3f} at MNI ({mni[0]:.0f}, {mni[1]:.0f}, {mni[2]:.0f})"
            )
            # Peak atlas labels
            peak_atlas = c.get("peak_atlas", {})
            peak_parts = [
                f"{k}: {v}" for k, v in peak_atlas.items() if v != "\u2014"
            ]
            if peak_parts:
                click.echo(f"    Peak location:  {' | '.join(peak_parts)}")
            # Atlas region breakdown
            for atlas_name, regions in c.get("atlas_regions", {}).items():
                if regions:
                    short = atlas_name.replace("HarvardOxford-", "HO-")
                    click.echo(f"    {short}:")
                    for r in regions:
                        click.echo(f"       {r['pct']:5.1f}%  {r['name']} ({r['voxels']} voxels)")
```

- [ ] **Step 4: Update CLI FDR output**

Replace the FDR output section:

```python
    else:
        result = fdr_correction(
            stat_path=input_path,
            mask_path=mask,
            q=q_value,
            two_tailed=two_tailed,
        )
        click.echo(f"FDR correction applied (q={q_value}):")
        click.echo(f"  P threshold: {result.p_threshold:.6f}")
        click.echo(f"  Significant voxels: {result.n_significant}")
        click.echo(f"  Clusters: {result.n_clusters}")
        for c in result.cluster_table:
            mni = c.get("peak_coords_mni", c["peak_coords"])
            click.echo(
                f"\n  Cluster {c['label']}: {c['size']} voxels, "
                f"peak T={c['peak_value']:.3f} at MNI ({mni[0]:.0f}, {mni[1]:.0f}, {mni[2]:.0f})"
            )
            peak_atlas = c.get("peak_atlas", {})
            peak_parts = [
                f"{k}: {v}" for k, v in peak_atlas.items() if v != "\u2014"
            ]
            if peak_parts:
                click.echo(f"    Peak location:  {' | '.join(peak_parts)}")
            for atlas_name, regions in c.get("atlas_regions", {}).items():
                if regions:
                    short = atlas_name.replace("HarvardOxford-", "HO-")
                    click.echo(f"    {short}:")
                    for r in regions:
                        click.echo(f"       {r['pct']:5.1f}%  {r['name']} ({r['voxels']} voxels)")
```

- [ ] **Step 5: Run CLI tests**

Run: `uv run pytest tests/test_cli.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/grouvox/cli.py tests/test_cli.py
git commit -m "feat(cli): print atlas-based cluster report for GRF and FDR"
```

---

### Task 9: Update `__init__.py` and Final Verification

**Files:**
- Modify: `src/grouvox/__init__.py`

- [ ] **Step 1: Update exports**

Add to `src/grouvox/__init__.py`:

```python
from grouvox.atlas import annotate_clusters, load_atlas
```

And add `"annotate_clusters"` and `"load_atlas"` to `__all__`.

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add src/grouvox/__init__.py
git commit -m "feat: export atlas functions from package init"
```
