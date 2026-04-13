"""Multiple comparison correction: GRF and FDR."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage, stats
from scipy.special import gamma

from grouvox.io import load_mask, save_nifti
from grouvox.smoothness import estimate_smoothness_from_map


@dataclass
class FDRResult:
    """Result of FDR correction."""
    p_threshold: float
    n_significant: int
    thresholded_map: np.ndarray
    n_clusters: int = 0
    cluster_table: list[dict] = field(default_factory=list)


@dataclass
class GRFResult:
    """Result of GRF cluster-level correction."""
    z_threshold: float
    cluster_size_threshold: int
    thresholded_z_map: np.ndarray
    thresholded_t_map: np.ndarray
    n_clusters: int
    cluster_table: list[dict] = field(default_factory=list)


_HEADER_RE = re.compile(
    r"GrouVox\{T_\[([0-9.]+)\]\}"
    r"\{dLh_([0-9.eE+-]+)\}"
    r"\{FWHMx_([0-9.]+)\s+FWHMy_([0-9.]+)\s+FWHMz_([0-9.]+)\s+mm\}"
)


def _parse_header_meta(header: nib.Nifti1Header, raise_on_missing: bool = True) -> dict:
    """Extract DOF, dLh, FWHM from NIfTI header description."""
    descrip = header["descrip"].astype(str) if hasattr(header["descrip"], 'astype') else str(header["descrip"])
    # Handle bytes
    if isinstance(descrip, bytes):
        descrip = descrip.decode("utf-8", errors="replace")
    m = _HEADER_RE.search(str(descrip))
    if not m:
        if raise_on_missing:
            raise ValueError(
                f"Cannot parse GrouVox metadata from header description: {descrip!r}. "
                "Run two_sample_ttest first to generate a compatible T-map."
            )
        return {}
    return {
        "dof": float(m.group(1)),
        "dlh": float(m.group(2)),
        "fwhm": (float(m.group(3)), float(m.group(4)), float(m.group(5))),
    }


def fdr_correction(
    stat_path: str | Path,
    mask_path: str | Path | None = None,
    q: float = 0.05,
    two_tailed: bool = True,
) -> FDRResult:
    """Apply FDR (Benjamini-Hochberg) correction to a T-statistic map."""
    stat_path = Path(stat_path)
    img = nib.load(stat_path)
    data = img.get_fdata(dtype=np.float32)
    header = img.header
    affine = img.affine

    meta = _parse_header_meta(header)
    dof = meta["dof"]

    mask = load_mask(mask_path)
    if mask is None:
        mask = np.ones(data.shape, dtype=bool)

    t_vals = data[mask]
    if two_tailed:
        p_vals = 2.0 * stats.t.sf(np.abs(t_vals), dof)
    else:
        p_vals = stats.t.sf(t_vals, dof)

    m_total = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    sorted_p = p_vals[sorted_idx]

    thresholds = np.arange(1, m_total + 1) / m_total * q
    below = sorted_p <= thresholds

    if not np.any(below):
        thresholded = np.zeros_like(data)
        _save_fdr_output(thresholded, affine, header, stat_path)
        return FDRResult(
            p_threshold=0.0, n_significant=0, thresholded_map=thresholded,
            n_clusters=0, cluster_table=[],
        )

    k_max = np.max(np.where(below)[0])
    p_threshold = sorted_p[k_max]

    significant = p_vals <= p_threshold
    mask_flat = np.zeros(np.prod(data.shape), dtype=bool)
    mask_flat[mask.ravel().nonzero()[0]] = significant
    sig_mask = mask_flat.reshape(data.shape)

    thresholded = data * sig_mask

    _save_fdr_output(thresholded, affine, header, stat_path)

    # Cluster identification
    labeled, n_labels = ndimage.label(sig_mask, structure=_STRUCT_26)
    cluster_table: list[dict] = []
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

    from grouvox.atlas import annotate_clusters

    annotate_clusters(cluster_table, affine, data.shape, labeled)
    _write_cluster_csv(
        cluster_table,
        stat_path.parent / f"ClusterReport_FDR_{_csv_stem(stat_path)}.csv",
    )

    return FDRResult(
        p_threshold=float(p_threshold),
        n_significant=int(significant.sum()),
        thresholded_map=thresholded,
        n_clusters=n_labels,
        cluster_table=cluster_table,
    )


def _save_fdr_output(data, affine, header, stat_path):
    out_path = stat_path.parent / f"FDR_Thresholded_{stat_path.name}"
    save_nifti(data, affine, header, out_path)


_STRUCT_26 = ndimage.generate_binary_structure(3, 3)


def grf_correction(
    stat_path: str | Path,
    mask_path: str | Path | None = None,
    voxel_p: float = 0.001,
    cluster_p: float = 0.05,
    two_tailed: bool = True,
    reestimate: bool = False,
) -> GRFResult:
    """Apply GRF cluster-level correction to a T-statistic map.

    Parameters
    ----------
    reestimate : bool
        If True, re-estimate smoothness (dLh) from the Z-map even when
        GrouVox header metadata is present.  When the header has no
        metadata, smoothness is always estimated automatically.
    """
    stat_path = Path(stat_path)
    img = nib.load(stat_path)
    t_data = img.get_fdata(dtype=np.float32)
    header = img.header
    affine = img.affine

    meta = _parse_header_meta(header, raise_on_missing=False)
    if not meta:
        raise ValueError(
            "Cannot parse GrouVox metadata (DOF) from header description. "
            "Run two_sample_ttest first to generate a compatible T-map."
        )
    dof = meta["dof"]

    mask = load_mask(mask_path)
    if mask is None:
        mask = np.ones(t_data.shape, dtype=bool)
    n_voxels = int(mask.sum())

    z_data = _t_to_z(t_data, dof)
    z_data = z_data * mask

    if reestimate or "dlh" not in meta:
        voxel_size = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
        smooth = estimate_smoothness_from_map(z_data, mask, voxel_size)
        dlh = smooth.dlh
    else:
        dlh = meta["dlh"]

    if two_tailed:
        z_thr = stats.norm.ppf(1.0 - voxel_p / 2.0)
    else:
        z_thr = stats.norm.ppf(1.0 - voxel_p)

    # When two-tailed, each tail is corrected at cluster_p/2 so the
    # combined false-positive rate across both tails equals cluster_p.
    effective_cluster_p = cluster_p / 2.0 if two_tailed else cluster_p
    cluster_size_thr = _grf_cluster_threshold(n_voxels, dlh, z_thr, effective_cluster_p, D=3)

    thresholded_z = np.zeros_like(z_data)
    thresholded_t = np.zeros_like(t_data)
    cluster_table = []

    pos_clusters, pos_table = _threshold_tail(
        z_data, t_data, z_thr, cluster_size_thr, positive=True,
    )
    thresholded_z += pos_clusters[0]
    thresholded_t += pos_clusters[1]
    cluster_table.extend(pos_table)

    if two_tailed:
        neg_clusters, neg_table = _threshold_tail(
            z_data, t_data, z_thr, cluster_size_thr, positive=False,
        )
        thresholded_z += neg_clusters[0]
        thresholded_t += neg_clusters[1]
        cluster_table.extend(neg_table)

    cluster_table.sort(key=lambda c: c["size"], reverse=True)

    # Atlas annotation
    surviving_mask = thresholded_z != 0
    anno_labeled, _ = ndimage.label(surviving_mask, structure=_STRUCT_26)
    for cluster in cluster_table:
        peak = cluster["peak_coords"]
        cluster["label"] = int(anno_labeled[peak])

    from grouvox.atlas import annotate_clusters

    annotate_clusters(cluster_table, affine, t_data.shape, anno_labeled)

    parent = stat_path.parent
    name = stat_path.name
    save_nifti(thresholded_z, affine, header, parent / f"Z_ClusterThresholded_{name}")
    save_nifti(thresholded_t, affine, header, parent / f"ClusterThresholded_{name}")
    _write_cluster_csv(cluster_table, parent / f"ClusterReport_{_csv_stem(stat_path)}.csv")

    return GRFResult(
        z_threshold=float(z_thr),
        cluster_size_threshold=int(cluster_size_thr),
        thresholded_z_map=thresholded_z,
        thresholded_t_map=thresholded_t,
        n_clusters=len(cluster_table),
        cluster_table=cluster_table,
    )


def _t_to_z(t_data: np.ndarray, dof: float) -> np.ndarray:
    """Convert T-statistic to Z-score, preserving sign."""
    p = stats.t.sf(np.abs(t_data), dof)
    p = np.clip(p, 1e-300, 1.0 - 1e-15)
    z = stats.norm.ppf(1.0 - p)
    return z * np.sign(t_data)


def _grf_cluster_threshold(
    n_voxels: int, dlh: float, z_thr: float, cluster_p: float, D: int = 3,
) -> int:
    """Calculate minimum cluster size using GRF theory (Friston et al. 1994)."""
    Em = (
        n_voxels
        * (2 * np.pi) ** (-(D + 1) / 2.0)
        * dlh
        * (z_thr**2 - 1) ** ((D - 1) / 2.0)
        * np.exp(-(z_thr**2) / 2.0)
    )

    EN = n_voxels * stats.norm.sf(z_thr)

    if Em <= 0 or EN <= 0:
        return 1

    beta_param = (gamma(D / 2.0 + 1) * Em / EN) ** (2.0 / D)

    cluster_size = 0
    p_temp = 1.0
    while p_temp >= cluster_p and cluster_size < n_voxels:
        cluster_size += 1
        p_temp = 1.0 - np.exp(-Em * np.exp(-beta_param * cluster_size ** (2.0 / D)))

    return max(cluster_size, 1)


def _threshold_tail(
    z_data: np.ndarray, t_data: np.ndarray,
    z_thr: float, min_cluster_size: int, positive: bool,
) -> tuple[tuple[np.ndarray, np.ndarray], list[dict]]:
    """Threshold one tail, remove small clusters, return surviving data + table."""
    if positive:
        suprathreshold = z_data >= z_thr
    else:
        suprathreshold = z_data <= -z_thr

    labeled, n_labels = ndimage.label(suprathreshold, structure=_STRUCT_26)

    z_out = np.zeros_like(z_data)
    t_out = np.zeros_like(t_data)
    table = []

    for label_id in range(1, n_labels + 1):
        cluster_mask = labeled == label_id
        size = int(cluster_mask.sum())
        if size >= min_cluster_size:
            z_out[cluster_mask] = z_data[cluster_mask]
            t_out[cluster_mask] = t_data[cluster_mask]

            cluster_z = np.abs(z_data[cluster_mask])
            peak_idx = np.argmax(cluster_z)
            peak_coords = np.array(np.where(cluster_mask)).T[peak_idx]
            peak_val = z_data[cluster_mask][peak_idx]

            table.append({
                "label": label_id,
                "size": size,
                "peak_value": float(peak_val),
                "peak_coords": tuple(int(c) for c in peak_coords),
            })

    return (z_out, t_out), table


def _csv_stem(stat_path: Path) -> str:
    """Strip .nii or .nii.gz to get the base name."""
    name = stat_path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return stat_path.stem


def _write_cluster_csv(cluster_table: list[dict], path: Path) -> None:
    """Write cluster report CSV — one row per cluster."""
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
