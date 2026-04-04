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
def correct(input_path, method, mask, voxel_p, cluster_p, q_value, two_tailed):
    """Apply multiple comparison correction to a T-statistic map."""
    from grouvox.correction import grf_correction, fdr_correction

    if method == "grf":
        result = grf_correction(
            stat_path=input_path,
            mask_path=mask,
            voxel_p=voxel_p,
            cluster_p=cluster_p,
            two_tailed=two_tailed,
        )
        click.echo(f"GRF correction applied:")
        click.echo(f"  Z threshold: {result.z_threshold:.4f}")
        click.echo(f"  Cluster size threshold: {result.cluster_size_threshold} voxels")
        click.echo(f"  Surviving clusters: {result.n_clusters}")
        for c in result.cluster_table:
            click.echo(
                f"    Cluster {c['label']}: {c['size']} voxels, "
                f"peak Z={c['peak_value']:.3f} at {c['peak_coords']}"
            )
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
