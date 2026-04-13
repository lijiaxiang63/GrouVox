"""Atlas loading, resampling, and cluster annotation."""

from __future__ import annotations

import importlib.resources
import json

import nibabel as nib
import numpy as np

_ATLAS_REGISTRY: dict[str, tuple[str, str]] = {
    "AAL": ("aal.nii.gz", "aal_labels.json"),
    "HarvardOxford-cortical": ("ho_cort.nii.gz", "ho_cort_labels.json"),
    "HarvardOxford-subcortical": ("ho_sub.nii.gz", "ho_sub_labels.json"),
}

_atlas_cache: dict[str, tuple[np.ndarray, dict[str, str], np.ndarray]] = {}


def load_atlas(name: str) -> tuple[np.ndarray, dict[str, str], np.ndarray]:
    """Load a bundled atlas by name.

    Parameters
    ----------
    name : str
        Atlas name (e.g. ``"AAL"``, ``"HarvardOxford-cortical"``).

    Returns
    -------
    data : np.ndarray
        3-D integer label volume.
    labels : dict[str, str]
        Mapping from string label ID to region name.
    affine : np.ndarray, shape (4, 4)
        Voxel-to-world affine.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    if name in _atlas_cache:
        return _atlas_cache[name]

    if name not in _ATLAS_REGISTRY:
        raise KeyError(
            f"Unknown atlas {name!r}. Available: {sorted(_ATLAS_REGISTRY)}"
        )

    nii_fname, json_fname = _ATLAS_REGISTRY[name]
    pkg = importlib.resources.files("grouvox.atlases")

    nii_path = pkg / nii_fname
    with importlib.resources.as_file(nii_path) as p:
        img = nib.load(p)
        data = np.asarray(img.dataobj, dtype=np.int32)
        affine = img.affine.copy()

    json_path = pkg / json_fname
    with importlib.resources.as_file(json_path) as p:
        with open(p) as f:
            labels: dict[str, str] = json.load(f)

    result = (data, labels, affine)
    _atlas_cache[name] = result
    return result


def resample_atlas_to_stat(
    atlas_data: np.ndarray,
    atlas_affine: np.ndarray,
    stat_shape: tuple[int, ...],
    stat_affine: np.ndarray,
) -> np.ndarray:
    """Resample an atlas volume onto a stat-map grid (nearest-neighbor).

    Parameters
    ----------
    atlas_data : np.ndarray
        3-D integer atlas volume.
    atlas_affine : np.ndarray
        Atlas voxel-to-world affine (4×4).
    stat_shape : tuple
        Spatial shape of the stat map (I, J, K).
    stat_affine : np.ndarray
        Stat-map voxel-to-world affine (4×4).

    Returns
    -------
    np.ndarray
        Integer label array with shape *stat_shape*.
    """
    stat_to_atlas = np.linalg.inv(atlas_affine) @ stat_affine

    gi, gj, gk = np.meshgrid(
        np.arange(stat_shape[0]),
        np.arange(stat_shape[1]),
        np.arange(stat_shape[2]),
        indexing="ij",
    )
    coords = np.stack([gi.ravel(), gj.ravel(), gk.ravel(), np.ones(gi.size)], axis=0)
    atlas_coords = stat_to_atlas @ coords
    ai = np.rint(atlas_coords[0]).astype(int)
    aj = np.rint(atlas_coords[1]).astype(int)
    ak = np.rint(atlas_coords[2]).astype(int)

    np.clip(ai, 0, atlas_data.shape[0] - 1, out=ai)
    np.clip(aj, 0, atlas_data.shape[1] - 1, out=aj)
    np.clip(ak, 0, atlas_data.shape[2] - 1, out=ak)

    resampled = atlas_data[ai, aj, ak].reshape(stat_shape)
    return resampled


def label_peak(
    peak_voxel: tuple[int, ...],
    stat_affine: np.ndarray,
    atlas_data: np.ndarray,
    atlas_affine: np.ndarray,
    labels: dict[str, str],
) -> str:
    """Return the atlas region name at a peak voxel coordinate.

    Parameters
    ----------
    peak_voxel : tuple of int
        (i, j, k) voxel indices in the stat map.
    stat_affine : np.ndarray
        Stat-map voxel-to-world affine (4×4).
    atlas_data : np.ndarray
        3-D integer atlas volume.
    atlas_affine : np.ndarray
        Atlas voxel-to-world affine (4×4).
    labels : dict[str, str]
        Label-ID-to-region-name mapping.

    Returns
    -------
    str
        Region name, or ``"\\u2014"`` (em dash) if no label is found.
    """
    world = stat_affine @ np.array([*peak_voxel, 1.0])
    atlas_vox = np.linalg.inv(atlas_affine) @ world
    ai, aj, ak = np.rint(atlas_vox[:3]).astype(int)

    if (
        ai < 0 or ai >= atlas_data.shape[0]
        or aj < 0 or aj >= atlas_data.shape[1]
        or ak < 0 or ak >= atlas_data.shape[2]
    ):
        return "\u2014"

    label_id = int(atlas_data[ai, aj, ak])
    return labels.get(str(label_id), "\u2014") if label_id != 0 else "\u2014"


def label_cluster(
    cluster_mask: np.ndarray,
    stat_affine: np.ndarray,
    atlas_data: np.ndarray,
    atlas_affine: np.ndarray,
    labels: dict[str, str],
) -> list[dict]:
    """Return region overlap for a cluster mask.

    Parameters
    ----------
    cluster_mask : np.ndarray
        Boolean 3-D mask for one cluster.
    stat_affine : np.ndarray
        Stat-map voxel-to-world affine (4×4).
    atlas_data : np.ndarray
        3-D integer atlas volume.
    atlas_affine : np.ndarray
        Atlas voxel-to-world affine (4×4).
    labels : dict[str, str]
        Label-ID-to-region-name mapping.

    Returns
    -------
    list[dict]
        Each dict has ``name``, ``voxels``, ``pct`` keys, sorted by
        voxels descending.
    """
    resampled = resample_atlas_to_stat(
        atlas_data, atlas_affine, cluster_mask.shape, stat_affine,
    )
    cluster_labels = resampled[cluster_mask]
    total = int(cluster_mask.sum())
    if total == 0:
        return []

    unique, counts = np.unique(cluster_labels, return_counts=True)
    result = []
    for uid, cnt in zip(unique, counts):
        uid_int = int(uid)
        if uid_int == 0:
            continue
        name = labels.get(str(uid_int))
        if name is None:
            continue
        result.append({
            "name": name,
            "voxels": int(cnt),
            "pct": round(100.0 * cnt / total, 1),
        })

    result.sort(key=lambda d: d["voxels"], reverse=True)
    return result


def annotate_clusters(
    cluster_table: list[dict],
    stat_affine: np.ndarray,
    stat_shape: tuple[int, ...],
    labeled_array: np.ndarray | None,
) -> None:
    """Add MNI coordinates and atlas annotations to a cluster table in-place.

    Parameters
    ----------
    cluster_table : list[dict]
        Cluster dicts as produced by ``grf_correction``.  Modified in-place.
    stat_affine : np.ndarray
        Stat-map voxel-to-world affine (4×4).
    stat_shape : tuple
        Spatial shape of the stat map.
    labeled_array : np.ndarray or None
        Integer label volume from ``scipy.ndimage.label``.
    """
    if not cluster_table:
        return

    # Pre-load and resample all atlases once
    atlas_info: list[tuple[str, np.ndarray, dict[str, str]]] = []
    for atlas_name in _ATLAS_REGISTRY:
        data, labels, affine = load_atlas(atlas_name)
        resampled = resample_atlas_to_stat(data, affine, stat_shape, stat_affine)
        atlas_info.append((atlas_name, resampled, labels))

    for cluster in cluster_table:
        peak = cluster["peak_coords"]
        world = stat_affine @ np.array([*peak, 1.0])
        cluster["peak_coords_mni"] = tuple(round(float(c), 1) for c in world[:3])

        peak_atlas: dict[str, str] = {}
        atlas_regions: dict[str, list[dict]] = {}

        for atlas_name, resampled, labels in atlas_info:
            # Peak label
            label_id = int(resampled[peak[0], peak[1], peak[2]])
            if label_id != 0 and str(label_id) in labels:
                peak_atlas[atlas_name] = labels[str(label_id)]
            else:
                peak_atlas[atlas_name] = "\u2014"

            # Cluster region overlap
            if labeled_array is not None:
                cmask = labeled_array == cluster["label"]
                cluster_labels = resampled[cmask]
                total = int(cmask.sum())
                unique, counts = np.unique(cluster_labels, return_counts=True)
                regions = []
                for uid, cnt in zip(unique, counts):
                    uid_int = int(uid)
                    if uid_int == 0:
                        continue
                    name = labels.get(str(uid_int))
                    if name is None:
                        continue
                    regions.append({
                        "name": name,
                        "voxels": int(cnt),
                        "pct": round(100.0 * cnt / total, 1),
                    })
                regions.sort(key=lambda d: d["voxels"], reverse=True)
                atlas_regions[atlas_name] = regions
            else:
                atlas_regions[atlas_name] = []

        cluster["peak_atlas"] = peak_atlas
        cluster["atlas_regions"] = atlas_regions
