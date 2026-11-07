"""Sky-frame Stokes maps and their occultation by a planet."""

import numpy as np
import pytest
from jaxoplanet.orbits import TransitOrbit
from jaxoplanet.starry.light_curves import surface_light_curve
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.utils import ortho_grid
from jaxoplanet.starry.ylm import Ylm
from numpy.polynomial.legendre import leggauss

import transit_polarization as tp


@pytest.fixture(scope="module")
def model():
    # A cool, low-gravity star in the blue: large and steep limb polarisation.
    return tp.LimbPolarizationModel.from_kb2015(4500, 3.5, 4000)


def _analytic_maps(model, x, y):
    r2 = x**2 + y**2
    mu = np.sqrt(np.clip(1 - r2, 0, None))
    g = model.g1 * np.asarray(tp.intensity_profile(mu, model.v))  # same 1 - sum c (1-mu)^n form
    return np.asarray(model.intensity(mu)), -g * (x**2 - y**2), -2 * g * x * y


def _brute_force_occulted(model, xp, yp, rp, n=300):
    """Integrate the analytic I, Q, U maps over (planet disk) ∩ (star disk)."""
    s, ws = leggauss(n)
    rr = 0.5 * rp * (s + 1)
    wr = 0.5 * rp * ws
    phi = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    R, P = np.meshgrid(rr, phi, indexing="ij")
    X, Y = xp + R * np.cos(P), yp + R * np.sin(P)
    weights = R * wr[:, None] * (2 * np.pi / (2 * n)) * (X**2 + Y**2 < 1)
    return [np.sum(m * weights) for m in _analytic_maps(model, X, Y)]


@pytest.mark.parametrize("lm, coeff, poly", [((2, 2), tp.STARRY_Y22, lambda x, y: x**2 - y**2),
                                              ((2, -2), tp.STARRY_Y2M2, lambda x, y: x * y),
                                              ((0, 0), 1 / np.pi, lambda x, y: np.ones_like(x))])
def test_starry_ylm_normalisation(lm, coeff, poly):
    res = 41
    _, (x, y, _) = ortho_grid(res)
    img = np.asarray(Surface(y=Ylm({lm: 1.0}), normalize=False).render(res=res)).ravel()
    ok = np.isfinite(img)
    ref = coeff * poly(np.asarray(x).ravel(), np.asarray(y).ravel())
    np.testing.assert_allclose(img[ok], ref[ok], atol=1e-12)


def test_stokes_surfaces_render_the_analytic_maps(model):
    res = 41
    _, (x, y, _) = ortho_grid(res)
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    surf_q, surf_u = tp.stokes_surfaces(model)
    _, q_ref, u_ref = _analytic_maps(model, x, y)
    for surf, ref in [(surf_q, q_ref), (surf_u, u_ref)]:
        img = np.asarray(surf.render(res=res)).ravel()
        ok = np.isfinite(img)
        np.testing.assert_allclose(img[ok], ref[ok], atol=1e-10 * np.abs(ref[ok]).max())


def test_unocculted_star_is_unpolarised(model):
    surf_q, surf_u = tp.stokes_surfaces(model)
    assert abs(float(surface_light_curve(surf_q))) < 1e-12
    assert abs(float(surface_light_curve(surf_u))) < 1e-12
    lc = tp.occultation_stokes(model, x=[2.0, 0.0], y=[0.0, 0.3], r=0.1, z=[1.0, -1.0])
    np.testing.assert_allclose(np.asarray(lc.I), 1.0)
    np.testing.assert_allclose(np.asarray(lc.Q), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(lc.U), 0.0, atol=1e-15)


@pytest.mark.parametrize("xp, yp", [(0.0, 0.5), (0.5, 0.0), (0.4, 0.4), (-0.3, 0.7),
                                    (0.93, 0.1), (1.02, -0.05), (0.2, -0.6)])
def test_occultation_matches_brute_force(model, xp, yp):
    """starry + our normalisation equals direct integration of the maps."""
    rp = 0.1
    lc = tp.occultation_stokes(model, xp, yp, rp)
    occ_i, occ_q, occ_u = _brute_force_occulted(model, xp, yp, rp)
    f0 = model.total_flux
    scale = np.pi * rp**2 * model.g1 / f0  # natural size of the Q, U signals
    assert float(lc.I) == pytest.approx(1 - occ_i / f0, abs=1e-6)
    assert float(lc.Q) == pytest.approx(-occ_q / f0, abs=1e-3 * scale)
    assert float(lc.U) == pytest.approx(-occ_u / f0, abs=1e-3 * scale)


def test_central_transit_has_no_U_and_symmetric_Q(model):
    orbit = TransitOrbit(period=3.0, duration=0.12, impact_param=0.0, radius_ratio=0.1)
    t = np.linspace(-0.07, 0.07, 101)
    lc = tp.transit_stokes(model, orbit, t)
    np.testing.assert_allclose(np.asarray(lc.U), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(lc.Q), np.asarray(lc.Q)[::-1], atol=1e-14)
    # Along the x axis the hidden light is polarised along y, so the residual is +Q.
    assert np.asarray(lc.Q).min() >= -1e-15


def test_off_centre_transit_U_is_antisymmetric(model):
    orbit = TransitOrbit(period=3.0, duration=0.12, impact_param=0.5, radius_ratio=0.1)
    t = np.linspace(-0.07, 0.07, 101)
    lc = tp.transit_stokes(model, orbit, t)
    np.testing.assert_allclose(np.asarray(lc.U), -np.asarray(lc.U)[::-1], atol=1e-14)
    np.testing.assert_allclose(np.asarray(lc.Q), np.asarray(lc.Q)[::-1], atol=1e-14)
    assert np.abs(np.asarray(lc.U)).max() > 0


def test_rotate_stokes():
    q, u = np.array([1e-6, -2e-6]), np.array([3e-6, 0.5e-6])
    q2, u2 = tp.rotate_stokes(q, u, 0.3)
    np.testing.assert_allclose(np.hypot(q2, u2), np.hypot(q, u))
    q3, u3 = tp.rotate_stokes(q, u, np.pi / 2)  # 90 deg flips both signs
    np.testing.assert_allclose(q3, -q, atol=1e-20)
    np.testing.assert_allclose(u3, -u, atol=1e-20)
    q4, _ = tp.rotate_stokes(np.array([0.0]), np.array([1.0]), np.pi / 4)  # U -> Q at 45 deg
    np.testing.assert_allclose(q4, 1.0)
