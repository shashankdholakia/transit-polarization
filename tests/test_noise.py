"""Photon-rate and noise bookkeeping."""

import numpy as np
import pytest
from jaxoplanet.orbits import TransitOrbit

import transit_polarization as tp


def test_photon_rate_zero_mag_V_per_cm2():
    # 1 cm^2 collecting area, perfect throughput, V = 0 (Vega system).
    diameter_m = 2 / np.sqrt(np.pi) / 100
    rate = tp.photon_rate(0.0, "V", diameter_m)
    expected = 3631e-23 * 10 ** (0.4 * 0.023) / 6.62607015e-27 * 830 / 5488
    assert rate == pytest.approx(expected, rel=1e-12)
    # Textbook value: ~1000 photons / s / cm^2 / Angstrom in V, times ~850 Angstrom.
    assert 7.5e5 < rate < 9.5e5


def test_photon_rate_scalings():
    base = tp.photon_rate(8.0, "B", 4.0, throughput=0.2)
    assert tp.photon_rate(13.0, "B", 4.0, throughput=0.2) == pytest.approx(base / 100)
    assert tp.photon_rate(8.0, "B", 8.0, throughput=0.2) == pytest.approx(4 * base)
    assert tp.photon_rate(8.0, "B", 4.0, central_obstruction=0.5, throughput=0.2) == pytest.approx(0.75 * base)


def test_polarimetric_sigma():
    assert tp.polarimetric_sigma(1e10, fraction=0.5) == pytest.approx(1 / np.sqrt(5e9))
    assert tp.polarimetric_sigma(1e10, fraction=1.0, modulation_efficiency=0.5) == pytest.approx(2e-5)


def test_matched_filter_snr():
    sigma = np.full(100, 2.0)
    assert tp.matched_filter_snr(np.full(100, 5.0), sigma, fit_offset=False) == pytest.approx(25.0)
    assert tp.matched_filter_snr(np.full(100, 5.0), sigma) == pytest.approx(0.0, abs=1e-12)


def test_simulation_noise_statistics():
    model = tp.LimbPolarizationModel.from_kb2015(5000, 4.5, 4377)
    orbit = TransitOrbit(period=2.2, duration=0.075, impact_param=0.66, radius_ratio=0.155)
    kwargs = dict(vega_mag=8.6, band="B", aperture_diameter=4.0, exposure_time=60.0,
                  t_start=-0.1, t_end=0.1, throughput=0.1, seed=1)
    sim = tp.simulate_transit_observation(model, orbit, **kwargs)
    again = tp.simulate_transit_observation(model, orbit, **kwargs)
    np.testing.assert_array_equal(sim["q_obs"], again["q_obs"])  # seeded
    for key in "Iqu":
        z = (sim[f"{key}_obs"] - sim[key]) / sim[f"sigma_{key}"]
        assert abs(z.std() - 1) < 0.15
    assert np.all(sim["sigma_q"] > sim["sigma_I"])  # only half the photons measure q
