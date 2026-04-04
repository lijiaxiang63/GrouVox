import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from pathlib import Path

from grouvox.cli import main


@pytest.fixture
def cli_runner():
    return CliRunner()


class TestTTest2Command:
    def test_basic_invocation(self, cli_runner, synthetic_two_groups, tmp_path):
        g1_dir, g2_dir, mask_path = synthetic_two_groups
        output = str(tmp_path / "cli_results" / "ttest")

        result = cli_runner.invoke(main, [
            "ttest2",
            "--group1", str(g1_dir),
            "--group2", str(g2_dir),
            "--output", output,
            "--mask", str(mask_path),
        ])

        assert result.exit_code == 0, result.output
        assert (tmp_path / "cli_results" / "ttest_T.nii.gz").exists()

    def test_with_covariates(self, cli_runner, synthetic_two_groups, tmp_path):
        g1_dir, g2_dir, mask_path = synthetic_two_groups
        output = str(tmp_path / "cli_results" / "ttest_cov")

        csv_path = tmp_path / "covars.csv"
        pd.DataFrame({
            "age": np.random.default_rng(0).normal(30, 5, 16),
            "sex": [0, 1] * 8,
        }).to_csv(csv_path, index=False)

        result = cli_runner.invoke(main, [
            "ttest2",
            "--group1", str(g1_dir),
            "--group2", str(g2_dir),
            "--output", output,
            "--mask", str(mask_path),
            "--covariates", str(csv_path),
        ])

        assert result.exit_code == 0, result.output


class TestCorrectCommand:
    def test_fdr(self, cli_runner, synthetic_two_groups, tmp_path):
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

    def test_grf(self, cli_runner, synthetic_two_groups, tmp_path):
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
