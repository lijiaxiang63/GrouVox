"""Brain surface plotting for GrouVox statistical maps using yabplot."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import yabplot as yab
    from yabplot.mesh import load_bmesh as _load_bmesh
    from yabplot.mesh import extract_polydata as _extract_polydata
except ImportError:
    yab = None  # type: ignore[assignment]
    _load_bmesh = None  # type: ignore[assignment]
    _extract_polydata = None  # type: ignore[assignment]

_YABPLOT_MISSING = (
    "yabplot is required for plotting. Install it with: "
    "pip install grouvox[plot]"
)

# Standard multi-view layouts for neuroimaging figures
VIEW_PRESETS: dict[str, list[str]] = {
    "default": ["left_lateral", "right_lateral", "left_medial", "right_medial"],
    "lateral": ["left_lateral", "right_lateral"],
    "medial": ["left_medial", "right_medial"],
    "all": [
        "left_lateral", "right_lateral",
        "left_medial", "right_medial",
        "superior", "inferior",
    ],
    "dorsal": ["superior", "inferior"],
}


def _require_yabplot() -> None:
    if yab is None:
        raise ImportError(_YABPLOT_MISSING)


def _resolve_views(views: str | Sequence[str]) -> list[str]:
    """Resolve a view preset name or list into a concrete view list."""
    if isinstance(views, str):
        view_list = VIEW_PRESETS.get(views)
        if view_list is None:
            raise ValueError(
                f"Unknown view preset {views!r}. "
                f"Choose from {list(VIEW_PRESETS)} or pass a list of view names."
            )
        return view_list
    return list(views)


def _infer_layout(n_views: int) -> tuple[int, int]:
    """Pick a (rows, cols) grid for *n_views* panels."""
    if n_views <= 2:
        return (1, n_views)
    if n_views <= 4:
        return (2, 2)
    return (2, (n_views + 1) // 2)


def _compute_vminmax(
    values: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    symmetric: bool,
) -> list[float]:
    """Derive colorbar bounds from projected vertex data."""
    finite = values[np.isfinite(values) & (values != 0)]
    if finite.size == 0:
        return [0.0, 1.0]

    data_min, data_max = float(finite.min()), float(finite.max())

    if vmin is not None and vmax is not None:
        return [vmin, vmax]
    if symmetric and data_min < 0 < data_max:
        bound = vmax if vmax is not None else max(abs(data_min), abs(data_max))
        return [-bound, bound]
    return [
        vmin if vmin is not None else data_min,
        vmax if vmax is not None else data_max,
    ]


def _finalize(plotter, output: str | Path | None):
    """Screenshot or save a plotter that was created with display_type='object'."""
    if output is None:
        return
    out_str = str(output)
    ext = Path(out_str).suffix.lower()
    if ext in (".svg", ".eps", ".ps", ".pdf", ".tex"):
        plotter.save_graphic(out_str)
    else:
        plotter.screenshot(out_str, transparent_background=True)
    plotter.close()


# -----------------------------------------------------------------------
# Volume → subcortical projection
# -----------------------------------------------------------------------

def _project_vol2subcortical(
    nifti_path: str,
    atlas: str = "aseg",
    summary: str = "peak",
) -> dict[str, float]:
    """Sample a NIfTI volume at subcortical mesh vertices.

    For each subcortical structure, the volume is sampled at every mesh
    vertex (via the NIfTI inverse-affine).  A summary statistic is then
    computed per region.

    Parameters
    ----------
    nifti_path : str
        Absolute path to a 3D NIfTI file.
    atlas : str
        Subcortical atlas name (default ``"aseg"``).
    summary : str
        ``"peak"`` (max absolute value) or ``"mean"`` (mean of non-zero
        vertices).

    Returns
    -------
    dict mapping region names to scalar values.  Regions with no signal
    are mapped to ``nan``.
    """
    import nibabel as nib
    import pyvista as pv
    from scipy.ndimage import map_coordinates
    from yabplot.data import _resolve_resource_path, _find_subcortical_files

    img = nib.load(nifti_path)
    vol = img.get_fdata()
    if vol.ndim > 3:
        vol = vol[..., 0]
    inv_affine = np.linalg.inv(img.affine)

    atlas_dir = _resolve_resource_path(atlas, "subcortical", custom_path=None)
    file_map = _find_subcortical_files(atlas_dir)
    regions = yab.get_atlas_regions(atlas, "subcortical")

    result: dict[str, float] = {}
    for name in regions:
        fpath = file_map.get(name)
        if fpath is None:
            result[name] = float("nan")
            continue

        mesh = pv.read(fpath)
        verts = np.asarray(mesh.points)  # (N, 3) in MNI space

        # Transform MNI vertex coords → voxel coords
        coords_homo = np.hstack([verts, np.ones((verts.shape[0], 1))])
        vox_coords = inv_affine.dot(coords_homo.T)[:3, :]

        sampled = map_coordinates(vol, vox_coords, order=1, mode="nearest")

        nonzero = sampled[sampled != 0]
        if nonzero.size == 0:
            result[name] = float("nan")
        elif summary == "peak":
            idx = np.argmax(np.abs(nonzero))
            result[name] = float(nonzero[idx])
        else:
            result[name] = float(nonzero.mean())

    return result


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def plot_brain(
    nifti_path: str | Path,
    output: str | Path | None = None,
    cmap: str = "coolwarm",
    vmin: float | None = None,
    vmax: float | None = None,
    views: str | Sequence[str] = "default",
    layout: tuple[int, int] | None = None,
    surface: str = "midthickness",
    style: str = "default",
    figsize: tuple[int, int] = (1600, 800),
    symmetric_cbar: bool = True,
    zero_transparent: bool = True,
):
    """Project a NIfTI statistical map onto cortical surfaces and render.

    Parameters
    ----------
    nifti_path : str or Path
        Path to a 3D NIfTI file (e.g., thresholded T-map or Z-map).
    output : str or Path, optional
        If given, save the figure to this path (e.g., ``results.png``).
        If *None*, display interactively.
    cmap : str
        Colormap name (default ``"coolwarm"``).
    vmin, vmax : float, optional
        Colorbar bounds. When *symmetric_cbar* is True and only *vmax* is
        given, *vmin* is set to ``-vmax`` automatically.
    views : str or list of str
        Camera preset name (``"default"``, ``"lateral"``, ``"medial"``,
        ``"all"``, ``"dorsal"``) or an explicit list of yabplot view names.
    layout : tuple of (rows, cols), optional
        Subplot grid. Inferred from number of views if *None*.
    surface : str
        Background mesh (default ``"midthickness"``).
    style : str
        Lighting style (default ``"default"``).
    figsize : tuple of (width, height)
        Figure size in pixels (default ``(1600, 800)``).
    symmetric_cbar : bool
        If True and the data spans positive and negative values, mirror
        the colorbar around zero for balanced display (default True).
    zero_transparent : bool
        If True, zero-valued vertices are set to NaN so they render as
        background (white). Essential for thresholded maps where zero
        means "not significant" (default True).

    Returns
    -------
    pyvista.Plotter
        The plotter instance (useful for further customisation).
    """
    _require_yabplot()

    nifti_path = str(Path(nifti_path).resolve())
    view_list = _resolve_views(views)
    if layout is None:
        layout = _infer_layout(len(view_list))

    # --- project volume onto cortical surface ---
    lh_data, rh_data = yab.project_vol2surf(
        nii_path=nifti_path,
        bmesh=surface,
        mask_medial_wall=True,
        interpolation="linear",
    )

    if zero_transparent:
        lh_data = lh_data.astype(np.float64, copy=True)
        rh_data = rh_data.astype(np.float64, copy=True)
        lh_data[lh_data == 0] = np.nan
        rh_data[rh_data == 0] = np.nan

    # Build PyVista meshes
    loaded = _load_bmesh(surface)
    lh_v, lh_f = _extract_polydata(loaded["L"])
    rh_v, rh_f = _extract_polydata(loaded["R"])
    lh_mesh = yab.make_cortical_mesh(lh_v, lh_f, lh_data)
    rh_mesh = yab.make_cortical_mesh(rh_v, rh_f, rh_data)

    # --- colorbar bounds ---
    all_vals = np.concatenate([lh_data, rh_data])
    vminmax = _compute_vminmax(all_vals, vmin, vmax, symmetric_cbar)

    # --- render ---
    display = "object" if output else "interactive"
    plotter = yab.plot_vertexwise(
        lh=lh_mesh,
        rh=rh_mesh,
        views=view_list,
        layout=layout,
        figsize=figsize,
        cmap=cmap,
        vminmax=vminmax,
        style=style,
        display_type=display,
    )
    _finalize(plotter, output)
    return plotter


def plot_subcortical(
    nifti_path: str | Path,
    output: str | Path | None = None,
    cmap: str = "coolwarm",
    vmin: float | None = None,
    vmax: float | None = None,
    views: str | Sequence[str] = "default",
    layout: tuple[int, int] | None = None,
    atlas: str = "aseg",
    surface: str = "midthickness",
    style: str = "default",
    figsize: tuple[int, int] = (1600, 800),
    symmetric_cbar: bool = True,
    summary: str = "peak",
    nan_alpha: float = 0.15,
):
    """Project a NIfTI statistical map onto subcortical structures and render.

    Each subcortical structure is sampled from the volume and coloured by
    a summary statistic (peak or mean).  Structures with no signal are
    shown as translucent grey.

    Parameters
    ----------
    nifti_path : str or Path
        Path to a 3D NIfTI file.
    output : str or Path, optional
        Save path.  If *None*, display interactively.
    cmap : str
        Colormap name (default ``"coolwarm"``).
    vmin, vmax : float, optional
        Colorbar bounds.
    views : str or list of str
        View preset or explicit list (same presets as ``plot_brain``).
    layout : tuple of (rows, cols), optional
        Subplot grid.
    atlas : str
        Subcortical atlas (default ``"aseg"``).
    surface : str
        Context cortical mesh (default ``"midthickness"``).
    style : str
        Lighting style (default ``"default"``).
    figsize : tuple of (width, height)
        Figure size in pixels (default ``(1600, 800)``).
    symmetric_cbar : bool
        Mirror colorbar around zero (default True).
    summary : str
        Per-structure summary: ``"peak"`` (max abs) or ``"mean"``.
    nan_alpha : float
        Opacity for structures with no signal (default 0.15).

    Returns
    -------
    pyvista.Plotter
    """
    _require_yabplot()

    nifti_path = str(Path(nifti_path).resolve())
    view_list = _resolve_views(views)
    if layout is None:
        layout = _infer_layout(len(view_list))

    # --- project volume onto subcortical structures ---
    region_data = _project_vol2subcortical(nifti_path, atlas=atlas, summary=summary)

    # --- colorbar bounds ---
    vals = np.array([v for v in region_data.values() if np.isfinite(v)])
    vminmax = _compute_vminmax(vals, vmin, vmax, symmetric_cbar)

    # --- render ---
    display = "object" if output else "interactive"
    plotter = yab.plot_subcortical(
        data=region_data,
        atlas=atlas,
        views=view_list,
        layout=layout,
        figsize=figsize,
        cmap=cmap,
        vminmax=vminmax,
        nan_alpha=nan_alpha,
        style=style,
        bmesh=surface,
        bmesh_alpha=0.1,
        display_type=display,
    )
    _finalize(plotter, output)
    return plotter
