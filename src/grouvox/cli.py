"""Command-line interface for GrouVox."""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
def main():
    """GrouVox: Voxel-wise group-level statistical analysis for neuroimaging."""


@main.command()
@click.option("--group1", required=True, type=click.Path(exists=True),
              help="Directory or 4D NIfTI for group 1.")
@click.option("--group2", required=True, type=click.Path(exists=True),
              help="Directory or 4D NIfTI for group 2.")
@click.option("--output", required=True, type=click.Path(),
              help="Output path prefix (e.g., results/ttest).")
@click.option("--mask", default=None, type=click.Path(exists=True),
              help="Brain mask NIfTI file.")
@click.option("--covariates", default=None, type=click.Path(exists=True),
              help="CSV file with covariates (rows=subjects, group1 first).")
@click.option("--contrast", default=None, type=float, nargs=2,
              help="Contrast for group terms, e.g., --contrast 1 -1.")
def ttest2(group1, group2, output, mask, covariates, contrast):
    """Run a two-sample t-test with optional covariates."""
    from grouvox.ttest import two_sample_ttest

    contrast_list = list(contrast) if contrast else None

    result = two_sample_ttest(
        group1=group1,
        group2=group2,
        output=output,
        mask=mask,
        covariates=covariates,
        contrast=contrast_list,
    )

    click.echo(f"T-map saved: {output}_T.nii.gz")
    click.echo(f"DOF: {result.dof}")
    click.echo(f"FWHM: {result.fwhm[0]:.2f} x {result.fwhm[1]:.2f} x {result.fwhm[2]:.2f} mm")


def _parse_views(views_str):
    """Resolve --views CLI string into a preset name or explicit list."""
    from grouvox.plot import VIEW_PRESETS
    if views_str in VIEW_PRESETS:
        return views_str
    return [v.strip() for v in views_str.split(",")]


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="NIfTI statistical map to plot (T-map, Z-map, or thresholded map).")
@click.option("--output", default=None, type=click.Path(),
              help="Save figure to this path (e.g., results.png). If omitted, show interactively.")
@click.option("--cmap", default="coolwarm", help="Colormap name (default: coolwarm).")
@click.option("--vmin", default=None, type=float, help="Colorbar minimum.")
@click.option("--vmax", default=None, type=float, help="Colorbar maximum.")
@click.option("--views", default="default",
              help="View preset (default, lateral, medial, all, dorsal) or comma-separated view names.")
@click.option("--surface", default="midthickness", help="Surface mesh (default: midthickness).")
@click.option("--style", default="default",
              help="Lighting style: default, matte, sculpted, glossy, flat.")
@click.option("--no-symmetric", is_flag=True, default=False,
              help="Disable symmetric colorbar (by default, colorbar is mirrored around zero).")
def plot(input_path, output, cmap, vmin, vmax, views, surface, style, no_symmetric):
    """Plot a statistical map on the cortical surface using yabplot."""
    from grouvox.plot import plot_brain

    plot_brain(
        nifti_path=input_path,
        output=output,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        views=_parse_views(views),
        surface=surface,
        style=style,
        symmetric_cbar=not no_symmetric,
    )

    if output:
        click.echo(f"Figure saved: {output}")
    else:
        click.echo("Interactive viewer closed.")


@main.command("plot-subcortical")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="NIfTI statistical map to plot.")
@click.option("--output", default=None, type=click.Path(),
              help="Save figure to this path. If omitted, show interactively.")
@click.option("--cmap", default="coolwarm", help="Colormap name (default: coolwarm).")
@click.option("--vmin", default=None, type=float, help="Colorbar minimum.")
@click.option("--vmax", default=None, type=float, help="Colorbar maximum.")
@click.option("--views", default="default",
              help="View preset or comma-separated view names.")
@click.option("--surface", default="midthickness", help="Context cortical mesh (default: midthickness).")
@click.option("--style", default="default",
              help="Lighting style: default, matte, sculpted, glossy, flat.")
@click.option("--no-symmetric", is_flag=True, default=False,
              help="Disable symmetric colorbar.")
@click.option("--summary", default="peak", type=click.Choice(["peak", "mean"]),
              help="Per-structure summary: peak (max abs) or mean (default: peak).")
def plot_sub(input_path, output, cmap, vmin, vmax, views, surface, style, no_symmetric, summary):
    """Plot a statistical map on subcortical structures using yabplot."""
    from grouvox.plot import plot_subcortical

    plot_subcortical(
        nifti_path=input_path,
        output=output,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        views=_parse_views(views),
        surface=surface,
        style=style,
        symmetric_cbar=not no_symmetric,
        summary=summary,
    )

    if output:
        click.echo(f"Figure saved: {output}")
    else:
        click.echo("Interactive viewer closed.")


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="T-statistic NIfTI file from ttest2.")
@click.option("--method", required=True, type=click.Choice(["grf", "fdr"]),
              help="Correction method: grf or fdr.")
@click.option("--mask", default=None, type=click.Path(exists=True),
              help="Brain mask NIfTI file.")
@click.option("--voxel-p", default=0.001, type=float,
              help="Voxel-level p threshold (GRF only, default 0.001).")
@click.option("--cluster-p", default=0.05, type=float,
              help="Cluster-level p threshold (GRF only, default 0.05).")
@click.option("--q", "q_value", default=0.05, type=float,
              help="FDR q threshold (FDR only, default 0.05).")
@click.option("--two-tailed/--one-tailed", default=True,
              help="Two-tailed test (default) or one-tailed.")
@click.option("--reestimate", is_flag=True, default=False,
              help="Re-estimate smoothness from the Z-map instead of using header values (GRF only).")
def correct(input_path, method, mask, voxel_p, cluster_p, q_value, two_tailed, reestimate):
    """Apply multiple comparison correction to a T-statistic map."""
    from grouvox.correction import grf_correction, fdr_correction

    if method == "grf":
        result = grf_correction(
            stat_path=input_path,
            mask_path=mask,
            voxel_p=voxel_p,
            cluster_p=cluster_p,
            two_tailed=two_tailed,
            reestimate=reestimate,
        )
        click.echo("GRF correction applied:")
        click.echo(f"  Z threshold: {result.z_threshold:.4f}")
        click.echo(f"  Cluster size threshold: {result.cluster_size_threshold} voxels")
        click.echo(f"  Surviving clusters: {result.n_clusters}")
        for c in result.cluster_table:
            mni = c.get("peak_coords_mni", c["peak_coords"])
            click.echo(
                f"\n  Cluster {c['label']}: {c['size']} voxels, "
                f"peak Z={c['peak_value']:.3f} at MNI ({mni[0]:.0f}, {mni[1]:.0f}, {mni[2]:.0f})"
            )
            peak_atlas = c.get("peak_atlas", {})
            peak_parts = [f"{k}: {v}" for k, v in peak_atlas.items() if v != "\u2014"]
            if peak_parts:
                click.echo(f"    Peak location:  {' | '.join(peak_parts)}")
            for atlas_name, regions in c.get("atlas_regions", {}).items():
                if regions:
                    short = atlas_name.replace("HarvardOxford-", "HO-")
                    click.echo(f"    {short}:")
                    for r in regions:
                        click.echo(f"       {r['pct']:5.1f}%  {r['name']} ({r['voxels']} voxels)")
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
            peak_parts = [f"{k}: {v}" for k, v in peak_atlas.items() if v != "\u2014"]
            if peak_parts:
                click.echo(f"    Peak location:  {' | '.join(peak_parts)}")
            for atlas_name, regions in c.get("atlas_regions", {}).items():
                if regions:
                    short = atlas_name.replace("HarvardOxford-", "HO-")
                    click.echo(f"    {short}:")
                    for r in regions:
                        click.echo(f"       {r['pct']:5.1f}%  {r['name']} ({r['voxels']} voxels)")
