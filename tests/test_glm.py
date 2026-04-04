import numpy as np
from scipy import stats
import pytest

from grouvox.glm import ols_fit, compute_contrast, OLSResult


class TestOLSFit:
    def test_simple_regression(self):
        """OLS beta should match known values for y = 2*x + 3 + noise."""
        rng = np.random.default_rng(42)
        n = 50
        x = rng.normal(0, 1, n)
        noise = rng.normal(0, 0.1, n)
        y = 2 * x + 3 + noise

        X = np.column_stack([x, np.ones(n)])
        Y = y.reshape(-1, 1)

        result = ols_fit(Y, X)
        assert isinstance(result, OLSResult)
        assert result.beta.shape == (2, 1)
        assert result.dof == n - 2
        np.testing.assert_allclose(result.beta[0, 0], 2.0, atol=0.1)
        np.testing.assert_allclose(result.beta[1, 0], 3.0, atol=0.1)

    def test_matches_scipy_ttest(self):
        """For two-group cell-means model without covariates, T should match scipy."""
        rng = np.random.default_rng(99)
        n1, n2 = 15, 15
        n_voxels = 100
        g1 = rng.normal(5, 1, (n1, n_voxels))
        g2 = rng.normal(3, 1, (n2, n_voxels))

        Y = np.vstack([g1, g2])
        G1 = np.array([1]*n1 + [0]*n2, dtype=float)
        G2 = np.array([0]*n1 + [1]*n2, dtype=float)
        X = np.column_stack([G1, G2])

        result = ols_fit(Y, X)
        contrast = np.array([1.0, -1.0])
        t_vals = compute_contrast(result, X, contrast)

        scipy_t, _ = stats.ttest_ind(g1, g2, axis=0, equal_var=True)
        np.testing.assert_allclose(t_vals, scipy_t, atol=1e-10)

    def test_residuals_orthogonal_to_design(self):
        """Residuals should be orthogonal to design matrix columns."""
        rng = np.random.default_rng(7)
        n, p, v = 20, 3, 10
        X = rng.normal(size=(n, p))
        Y = rng.normal(size=(n, v))

        result = ols_fit(Y, X)
        product = X.T @ result.residuals
        np.testing.assert_allclose(product, 0, atol=1e-10)

    def test_multiple_voxels(self):
        """Verify vectorization works across many voxels."""
        rng = np.random.default_rng(0)
        n, v = 30, 500
        X = np.column_stack([rng.normal(size=n), np.ones(n)])
        Y = rng.normal(size=(n, v))

        result = ols_fit(Y, X)
        assert result.beta.shape == (2, v)
        assert result.residuals.shape == (n, v)
        assert result.t_values.shape == (2, v)
        assert result.sse.shape == (v,)


class TestComputeContrast:
    def test_contrast_direction(self):
        """[1, -1] should give positive T when group1 > group2."""
        rng = np.random.default_rng(42)
        n1 = n2 = 20
        g1 = rng.normal(10, 1, (n1, 1))
        g2 = rng.normal(5, 1, (n2, 1))
        Y = np.vstack([g1, g2])
        G1 = np.array([1]*n1 + [0]*n2, dtype=float)
        G2 = np.array([0]*n1 + [1]*n2, dtype=float)
        X = np.column_stack([G1, G2])

        result = ols_fit(Y, X)
        t_pos = compute_contrast(result, X, np.array([1.0, -1.0]))
        t_neg = compute_contrast(result, X, np.array([-1.0, 1.0]))

        assert t_pos[0] > 0
        assert t_neg[0] < 0
        np.testing.assert_allclose(t_pos, -t_neg, atol=1e-10)
