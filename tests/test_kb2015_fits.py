"""The jaxoplanet polynomial representations reproduce the K&B (2015) tables."""

import numpy as np
import pytest

import transit_polarization as tp
from transit_polarization.profiles import DEFAULT_INTENSITY_DEGREE, DEFAULT_POLARIZATION_DEGREE

# Tolerances, set at ~2x the worst case found over all 625 models at the default
# degrees (6, 6) on 2026-11-06. Table 1 is rounded to 1e-4, so |dI| ~ 1e-4 is
# at the rounding level.
MAX_ABS_INTENSITY_RESIDUAL = 5e-4  # in units of I(1); worst case 2.5e-4
MAX_REL_POLARIZED_RESIDUAL = 5e-3  # |d QQ| / max(QQ) per model; worst case 2.5e-3


@pytest.fixture(scope="module")
def grid():
    return tp.load_kb2015(download=False)


def test_grid_shape_and_ranges(grid):
    assert grid.intensity.shape == (5, 25, 5, 10)
    assert grid.pol_degree.shape == (5, 25, 5, 10)
    np.testing.assert_array_equal(grid.wavelength, [4000, 4500, 5000, 6000, 7000])
    assert grid.teff[0] == 4500 and grid.teff[-1] == 6900
    np.testing.assert_array_equal(grid.logg, [3.0, 3.5, 4.0, 4.5, 5.0])
    np.testing.assert_array_equal(grid.mu, tp.MU_TABLE)
    # Physical sanity of the published numbers.
    assert np.all((grid.intensity > 0) & (grid.intensity < 1))
    assert np.all(grid.pol_degree > 0)  # parallel to the limb everywhere


def test_first_rows_match_paper(grid):
    # First rows of Tables 1 and 2 as printed in the paper (4000 A, 4500 K, log g 3.0).
    mu, intensity, pol = grid.profile(4000, 4500, 3.0)
    np.testing.assert_allclose(intensity[[0, -1]], [0.1565, 0.9095])
    np.testing.assert_allclose(pol[[0, -1]], [0.0248575, 0.0002178])


def test_interpolation_is_exact_at_nodes(grid):
    _, intensity, pol = grid.profile(5000, 5800, 4.5)
    i, j, k = 2, list(grid.teff).index(5800), 3
    np.testing.assert_array_equal(intensity, grid.intensity[i, j, k])
    np.testing.assert_array_equal(pol, grid.pol_degree[i, j, k])


def test_out_of_grid_raises(grid):
    with pytest.raises(ValueError, match="outside the K&B 2015 grid"):
        grid.profile(3600, 5800, 4.5)


def test_fit_intensity_matches_plain_lstsq(grid):
    mu, intensity, _ = grid.profile(5000, 5800, 4.5)
    u = tp.fit_intensity(mu, intensity, degree=4)
    design = -np.vander(1 - mu, 5, increasing=True)[:, 1:]
    u_ref, *_ = np.linalg.lstsq(design, intensity - 1, rcond=None)
    np.testing.assert_allclose(u, u_ref, rtol=1e-8, atol=1e-10)


def test_polynomials_reproduce_every_table_model(grid):
    """The test the README refers to: polynomial vs table for all 625 models."""
    mu = grid.mu
    worst_i, worst_q = 0.0, 0.0
    for a, wl in enumerate(grid.wavelength):
        for b, teff in enumerate(grid.teff):
            for c, logg in enumerate(grid.logg):
                intensity = grid.intensity[a, b, c]
                qq = grid.pol_degree[a, b, c] * intensity
                u = tp.fit_intensity(mu, intensity, DEFAULT_INTENSITY_DEGREE)
                g1, v = tp.fit_polarized_intensity(mu, qq, DEFAULT_POLARIZATION_DEGREE)
                res_i = np.abs(np.asarray(tp.intensity_profile(mu, u)) - intensity).max()
                res_q = np.abs(np.asarray(tp.polarized_intensity_profile(mu, g1, v)) - qq).max() / qq.max()
                worst_i, worst_q = max(worst_i, res_i), max(worst_q, res_q)
                assert res_i < MAX_ABS_INTENSITY_RESIDUAL, (wl, teff, logg, res_i)
                assert res_q < MAX_REL_POLARIZED_RESIDUAL, (wl, teff, logg, res_q)
    print(f"worst |dI| = {worst_i:.2e}, worst |dQQ|/max QQ = {worst_q:.2e}")


@pytest.mark.parametrize("params", [(4000, 4500, 3.0), (4377, 5778, 4.44), (6500, 6900, 5.0)])
def test_fitted_profiles_are_physical(grid, params):
    """Positive, monotonic intensity; non-negative polarised intensity; exact at mu = 1."""
    model = tp.LimbPolarizationModel.from_kb2015(params[1], params[2], params[0], grid=grid)
    mu = np.linspace(0, 1, 2001)
    intensity = np.asarray(model.intensity(mu))
    qq = np.asarray(model.polarized_intensity(mu))
    assert intensity.min() > 0
    assert np.all(np.diff(intensity) > 0)
    assert qq.min() > -1e-3 * qq.max()
    assert intensity[-1] == pytest.approx(1.0)
    assert qq[-1] == pytest.approx(0.0, abs=1e-15)


def test_total_flux_matches_numerical_integral(grid):
    model = tp.LimbPolarizationModel.from_kb2015(5800, 4.5, 4377, grid=grid)
    mu = np.linspace(0, 1, 200001)
    # dA = 2 pi r dr = 2 pi mu dmu
    numeric = np.trapz(2 * np.pi * mu * np.asarray(model.intensity(mu)), mu)
    assert model.total_flux == pytest.approx(numeric, rel=1e-8)
