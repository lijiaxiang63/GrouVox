"""Tests for grouvox.plot module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from grouvox.plot import (
    VIEW_PRESETS,
    _compute_vminmax,
    _infer_layout,
    _resolve_views,
    plot_brain,
    plot_subcortical,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_yab():
    """Return a mock yabplot module with minimal stubs."""
    mock = MagicMock()
    mock.project_vol2surf.return_value = (
        np.array([0.0, 1.0, -2.0, np.nan]),
        np.array([3.0, -1.0, 0.0, np.nan]),
    )
    mock.make_cortical_mesh.side_effect = lambda v, f, d: MagicMock()
    mock.plot_vertexwise.return_value = MagicMock()
    mock.plot_subcortical.return_value = MagicMock()
    return mock


def _make_mock_mesh():
    """Return a mock for yabplot.mesh helpers."""
    mock = MagicMock()
    mock.load_bmesh.return_value = {"L": MagicMock(), "R": MagicMock()}
    mock.extract_polydata.return_value = (
        np.zeros((4, 3)),
        np.zeros((2, 3), dtype=int),
    )
    return mock


def _apply_mock_yab(mock_yab_module, real_mock):
    """Patch a module-level ``yab`` mock with concrete return values."""
    mock_yab_module.__bool__ = lambda self: True
    for attr in dir(real_mock):
        if not attr.startswith("_"):
            setattr(mock_yab_module, attr, getattr(real_mock, attr))


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------

class TestResolveViews:
    def test_known_preset(self):
        assert _resolve_views("lateral") == ["left_lateral", "right_lateral"]

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown view preset"):
            _resolve_views("bogus")

    def test_explicit_list_passthrough(self):
        assert _resolve_views(["superior"]) == ["superior"]


class TestInferLayout:
    def test_single(self):
        assert _infer_layout(1) == (1, 1)

    def test_pair(self):
        assert _infer_layout(2) == (1, 2)

    def test_quad(self):
        assert _infer_layout(4) == (2, 2)

    def test_six(self):
        assert _infer_layout(6) == (2, 3)


class TestComputeVminmax:
    def test_symmetric(self):
        vals = np.array([-2.0, 0.0, 3.0])
        assert _compute_vminmax(vals, None, None, symmetric=True) == [-3.0, 3.0]

    def test_asymmetric(self):
        vals = np.array([-2.0, 0.0, 3.0])
        assert _compute_vminmax(vals, None, None, symmetric=False) == [-2.0, 3.0]

    def test_explicit_overrides(self):
        vals = np.array([-2.0, 3.0])
        assert _compute_vminmax(vals, -10, 10, symmetric=True) == [-10, 10]

    def test_all_nan(self):
        vals = np.array([np.nan, np.nan])
        assert _compute_vminmax(vals, None, None, symmetric=True) == [0.0, 1.0]

    def test_all_zero(self):
        vals = np.array([0.0, 0.0])
        assert _compute_vminmax(vals, None, None, symmetric=True) == [0.0, 1.0]


class TestViewPresets:
    def test_all_presets_are_nonempty_lists(self):
        for name, views in VIEW_PRESETS.items():
            assert isinstance(views, list) and len(views) > 0, name


# ---------------------------------------------------------------------------
# plot_brain
# ---------------------------------------------------------------------------

class TestPlotBrain:
    @patch("grouvox.plot.yab")
    def test_export_uses_object_mode(self, mock_yab_module, tmp_nifti_factory, tmp_path):
        _apply_mock_yab(mock_yab_module, _make_mock_yab())
        nii = tmp_nifti_factory(np.ones((5, 5, 5)), "stat.nii.gz")
        out = tmp_path / "brain.png"

        with patch("grouvox.plot._load_bmesh", _make_mock_mesh().load_bmesh), \
             patch("grouvox.plot._extract_polydata", _make_mock_mesh().extract_polydata):
            plot_brain(nii, output=out)

        kw = mock_yab_module.plot_vertexwise.call_args[1]
        assert kw["display_type"] == "object"
        assert "export_path" not in kw

    @patch("grouvox.plot.yab")
    def test_symmetric_colorbar(self, mock_yab_module, tmp_nifti_factory):
        _apply_mock_yab(mock_yab_module, _make_mock_yab())
        nii = tmp_nifti_factory(np.ones((5, 5, 5)), "stat.nii.gz")

        with patch("grouvox.plot._load_bmesh", _make_mock_mesh().load_bmesh), \
             patch("grouvox.plot._extract_polydata", _make_mock_mesh().extract_polydata):
            plot_brain(nii, output="/dev/null", symmetric_cbar=True)

        vminmax = mock_yab_module.plot_vertexwise.call_args[1]["vminmax"]
        assert vminmax[0] == -vminmax[1]
        assert vminmax[1] == 3.0

    @patch("grouvox.plot.yab")
    def test_no_symmetric_colorbar(self, mock_yab_module, tmp_nifti_factory):
        _apply_mock_yab(mock_yab_module, _make_mock_yab())
        nii = tmp_nifti_factory(np.ones((5, 5, 5)), "stat.nii.gz")

        with patch("grouvox.plot._load_bmesh", _make_mock_mesh().load_bmesh), \
             patch("grouvox.plot._extract_polydata", _make_mock_mesh().extract_polydata):
            plot_brain(nii, output="/dev/null", symmetric_cbar=False)

        vminmax = mock_yab_module.plot_vertexwise.call_args[1]["vminmax"]
        assert vminmax[0] == -2.0
        assert vminmax[1] == 3.0

    @patch("grouvox.plot.yab")
    def test_explicit_vmin_vmax(self, mock_yab_module, tmp_nifti_factory):
        _apply_mock_yab(mock_yab_module, _make_mock_yab())
        nii = tmp_nifti_factory(np.ones((5, 5, 5)), "stat.nii.gz")

        with patch("grouvox.plot._load_bmesh", _make_mock_mesh().load_bmesh), \
             patch("grouvox.plot._extract_polydata", _make_mock_mesh().extract_polydata):
            plot_brain(nii, output="/dev/null", vmin=-5, vmax=5)

        assert mock_yab_module.plot_vertexwise.call_args[1]["vminmax"] == [-5, 5]

    @patch("grouvox.plot.yab")
    def test_layout_inferred_from_views(self, mock_yab_module, tmp_nifti_factory):
        _apply_mock_yab(mock_yab_module, _make_mock_yab())
        nii = tmp_nifti_factory(np.ones((5, 5, 5)), "stat.nii.gz")

        with patch("grouvox.plot._load_bmesh", _make_mock_mesh().load_bmesh), \
             patch("grouvox.plot._extract_polydata", _make_mock_mesh().extract_polydata):
            plot_brain(nii, output="/dev/null", views="lateral")

        kw = mock_yab_module.plot_vertexwise.call_args[1]
        assert kw["layout"] == (1, 2)
        assert kw["views"] == ["left_lateral", "right_lateral"]


# ---------------------------------------------------------------------------
# plot_subcortical
# ---------------------------------------------------------------------------

class TestPlotSubcortical:
    @patch("grouvox.plot._project_vol2subcortical")
    @patch("grouvox.plot.yab")
    def test_delegates_to_yab(self, mock_yab_module, mock_proj, tmp_nifti_factory):
        _apply_mock_yab(mock_yab_module, _make_mock_yab())
        mock_proj.return_value = {"Right-Putamen": -4.0, "Left-Thalamus": float("nan")}

        nii = tmp_nifti_factory(np.ones((5, 5, 5)), "stat.nii.gz")
        plot_subcortical(nii, output="/dev/null")

        mock_yab_module.plot_subcortical.assert_called_once()
        kw = mock_yab_module.plot_subcortical.call_args[1]
        assert kw["data"]["Right-Putamen"] == -4.0
        assert np.isnan(kw["data"]["Left-Thalamus"])
        assert kw["display_type"] == "object"

    @patch("grouvox.plot._project_vol2subcortical")
    @patch("grouvox.plot.yab")
    def test_symmetric_colorbar(self, mock_yab_module, mock_proj, tmp_nifti_factory):
        _apply_mock_yab(mock_yab_module, _make_mock_yab())
        mock_proj.return_value = {"A": -4.0, "B": 2.0, "C": float("nan")}

        nii = tmp_nifti_factory(np.ones((5, 5, 5)), "stat.nii.gz")
        plot_subcortical(nii, output="/dev/null", symmetric_cbar=True)

        vminmax = mock_yab_module.plot_subcortical.call_args[1]["vminmax"]
        assert vminmax == [-4.0, 4.0]

    @patch("grouvox.plot._project_vol2subcortical")
    @patch("grouvox.plot.yab")
    def test_summary_kwarg_forwarded(self, mock_yab_module, mock_proj, tmp_nifti_factory):
        _apply_mock_yab(mock_yab_module, _make_mock_yab())
        mock_proj.return_value = {"A": 1.0}

        nii = tmp_nifti_factory(np.ones((5, 5, 5)), "stat.nii.gz")
        plot_subcortical(nii, output="/dev/null", summary="mean")

        mock_proj.assert_called_once()
        assert mock_proj.call_args[1]["summary"] == "mean"


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

class TestImportGuard:
    def test_missing_yabplot_raises_on_call(self, tmp_nifti_factory):
        nii = tmp_nifti_factory(np.zeros((5, 5, 5)), "empty.nii.gz")
        with patch("grouvox.plot.yab", None):
            with pytest.raises(ImportError, match="yabplot is required"):
                plot_brain(nii)

    def test_missing_yabplot_raises_subcortical(self, tmp_nifti_factory):
        nii = tmp_nifti_factory(np.zeros((5, 5, 5)), "empty.nii.gz")
        with patch("grouvox.plot.yab", None):
            with pytest.raises(ImportError, match="yabplot is required"):
                plot_subcortical(nii)


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------

class TestPlotCLI:
    def test_plot_command_exists(self):
        from grouvox.cli import main
        assert "plot" in main.commands

    def test_plot_subcortical_command_exists(self):
        from grouvox.cli import main
        assert "plot-subcortical" in main.commands

    @patch("grouvox.plot.plot_brain")
    def test_plot_cli_invocation(self, mock_plot_brain, tmp_nifti_factory, tmp_path):
        from click.testing import CliRunner
        from grouvox.cli import main

        mock_plot_brain.return_value = MagicMock()
        nii = tmp_nifti_factory(np.ones((5, 5, 5)), "stat.nii.gz")
        out = tmp_path / "out.png"

        runner = CliRunner()
        result = runner.invoke(main, [
            "plot",
            "--input", str(nii),
            "--output", str(out),
            "--cmap", "hot",
            "--views", "lateral",
            "--style", "matte",
        ])

        assert result.exit_code == 0, result.output
        mock_plot_brain.assert_called_once()
        kw = mock_plot_brain.call_args[1]
        assert kw["cmap"] == "hot"
        assert kw["views"] == "lateral"
        assert kw["style"] == "matte"

    @patch("grouvox.plot.plot_subcortical")
    def test_plot_subcortical_cli(self, mock_plot_sub, tmp_nifti_factory, tmp_path):
        from click.testing import CliRunner
        from grouvox.cli import main

        mock_plot_sub.return_value = MagicMock()
        nii = tmp_nifti_factory(np.ones((5, 5, 5)), "stat.nii.gz")
        out = tmp_path / "sub.png"

        runner = CliRunner()
        result = runner.invoke(main, [
            "plot-subcortical",
            "--input", str(nii),
            "--output", str(out),
            "--summary", "mean",
        ])

        assert result.exit_code == 0, result.output
        mock_plot_sub.assert_called_once()
        assert mock_plot_sub.call_args[1]["summary"] == "mean"
