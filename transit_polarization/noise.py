"""Photon-noise-limited simulations of transits in Stokes I, Q and U.

Photon rates
------------
Magnitudes are **Vega-system** Johnson-Cousins B, V or R (Bessell). They are
converted to AB with the offsets of Bessell & Murphy (2012, PASP 124, 140;
Table 3, ``AB_nu`` row, ``m_Vega = AB - ZP``), so ``m_AB = m_Vega + ZP``. The
photon flux in the band is then approximated with a top-hat of the band's FWHM
centred on its pivot wavelength (Bessell & Murphy 2012, Table 5)::

    N_dot = (3631 Jy * 10^(-0.4 m_AB) / h) * (FWHM / lambda_pivot) * A * T

with ``A`` the collecting area (with central obstruction) and ``T`` the total
throughput (atmosphere x telescope x instrument x detector QE). The further
<= 0.01 mag zero-point corrections in their Table 5 are neglected.

Polarimetric noise
------------------
For a dual-beam polarimeter, every detected photon counts towards Stokes I, so
``sigma(I/F0) = 1 / sqrt(N)`` for ``N`` photons in an exposure. The normalised
Stokes parameter ``q = Q/I`` is estimated from the photons collected while the
instrument is set up to measure Q (a fraction ``fraction_q`` of the exposure),
and a modulation efficiency ``epsilon <= 1`` inflates the noise::

    sigma_q = 1 / (epsilon * sqrt(fraction_q * N))

and likewise for ``u``. This ignores read noise, sky, scintillation (which
affects ``I`` from the ground but largely cancels in a dual-beam ratio),
seeing-dependent systematics and any instrumental polarisation floor. Real
polarimeters (e.g. HIPPI-class instruments) reach a systematic floor at the
ppm level, so treat these numbers as best-case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .profiles import LimbPolarizationModel
from .stokes import transit_stokes

__all__ = [
    "Band",
    "BESSELL_BANDS",
    "photon_rate",
    "polarimetric_sigma",
    "simulate_transit_observation",
    "matched_filter_snr",
    "bin_data",
]

PLANCK_ERG_S = 6.62607015e-27
JANSKY_CGS = 1.0e-23  # erg s^-1 cm^-2 Hz^-1
AB_ZERO_POINT_JY = 3631.0


@dataclass(frozen=True)
class Band:
    """A photometric band approximated as a top-hat.

    Attributes:
        name: Band name.
        pivot_wavelength: Pivot wavelength in Angstrom.
        fwhm: Full width at half maximum in Angstrom.
        ab_minus_vega: ``m_AB - m_Vega`` in mag.
    """

    name: str
    pivot_wavelength: float
    fwhm: float
    ab_minus_vega: float


#: Bessell B, V, R from Bessell & Murphy (2012), Tables 3 and 5. U and I lie
#: outside the 4000-7000 Angstrom range of the K&B (2015) tables.
BESSELL_BANDS = {
    "B": Band("B", pivot_wavelength=4377.0, fwhm=890.0, ab_minus_vega=-0.138),
    "V": Band("V", pivot_wavelength=5488.0, fwhm=830.0, ab_minus_vega=-0.023),
    "R": Band("R", pivot_wavelength=6515.0, fwhm=1443.0, ab_minus_vega=0.160),
}


def _band(band: Band | str) -> Band:
    return BESSELL_BANDS[band] if isinstance(band, str) else band


def photon_rate(
    vega_mag: float,
    band: Band | str,
    aperture_diameter: float,
    central_obstruction: float = 0.0,
    throughput: float = 1.0,
) -> float:
    """Detected photon rate (photons per second) for a star of given magnitude.

    Args:
        vega_mag: Apparent Vega magnitude in ``band``.
        band: A :class:`Band` or one of ``"B"``, ``"V"``, ``"R"``.
        aperture_diameter: Primary mirror diameter in metres.
        central_obstruction: Secondary obstruction as a fraction of the diameter.
        throughput: Total system throughput (0--1).
    """
    band = _band(band)
    ab_mag = vega_mag + band.ab_minus_vega
    f_nu = AB_ZERO_POINT_JY * JANSKY_CGS * 10.0 ** (-0.4 * ab_mag)  # erg s^-1 cm^-2 Hz^-1
    # photons s^-1 cm^-2 = integral f_nu / (h nu) dnu ~= (f_nu / h) * dlnnu, and dlnnu = dlnlambda.
    photon_flux = f_nu / PLANCK_ERG_S * (band.fwhm / band.pivot_wavelength)
    area_cm2 = np.pi / 4.0 * (aperture_diameter * 100.0) ** 2 * (1.0 - central_obstruction**2)
    return float(photon_flux * area_cm2 * throughput)


def polarimetric_sigma(n_photons, fraction: float = 0.5, modulation_efficiency: float = 1.0):
    """Photon-noise uncertainty on a normalised Stokes parameter (q or u).

    Args:
        n_photons: Total detected photons in the exposure.
        fraction: Fraction of the exposure spent measuring this Stokes parameter.
        modulation_efficiency: Polarimetric (modulation) efficiency epsilon.
    """
    return 1.0 / (modulation_efficiency * np.sqrt(fraction * np.asarray(n_photons, float)))


def simulate_transit_observation(
    model: LimbPolarizationModel,
    orbit,
    *,
    vega_mag: float,
    band: Band | str,
    aperture_diameter: float,
    exposure_time: float,
    t_start: float,
    t_end: float,
    throughput: float,
    central_obstruction: float = 0.0,
    modulation_efficiency: float = 1.0,
    fraction_q: float = 0.5,
    fraction_u: float = 0.5,
    seed: int = 0,
) -> dict:
    """Simulate a photon-noise-limited transit observation in I, q = Q/I and u = U/I.

    The model is evaluated at exposure mid-times (no integration smearing).
    Noise is Gaussian with the photon-noise sigma (photon counts are large).

    Args:
        model: Stellar centre-to-limb model, ideally built at ``band``'s pivot wavelength.
        orbit: jaxoplanet orbit (times in days).
        vega_mag: Apparent Vega magnitude of the host star in ``band``.
        band: :class:`Band` or ``"B"``/``"V"``/``"R"``.
        aperture_diameter: Telescope diameter in metres.
        exposure_time: Exposure (cadence) in seconds; no dead time.
        t_start, t_end: Observation window in days, relative to the orbit's times.
        throughput: Total system throughput (0--1).
        central_obstruction: Linear obstruction fraction.
        modulation_efficiency: Polarimetric efficiency epsilon.
        fraction_q, fraction_u: Fractions of each exposure spent on Q and U.
        seed: Random seed.

    Returns:
        Dict with ``t`` (days), noise-free ``I``, ``q``, ``u`` (and ``Q``, ``U``),
        their uncertainties ``sigma_I``, ``sigma_q``, ``sigma_u``, noisy
        ``I_obs``, ``q_obs``, ``u_obs``, and ``photons_per_exposure``.
    """
    band = _band(band)
    dt_days = exposure_time / 86400.0
    t = np.arange(t_start + 0.5 * dt_days, t_end, dt_days)
    lc = transit_stokes(model, orbit, t)
    n_photons = photon_rate(vega_mag, band, aperture_diameter, central_obstruction, throughput) * exposure_time
    # Photon counts scale with the instantaneous flux I(t).
    n_t = n_photons * np.asarray(lc.I)
    sigma_I = np.sqrt(n_t) / n_photons  # sigma of I/F0
    sigma_q = polarimetric_sigma(n_t, fraction_q, modulation_efficiency)
    sigma_u = polarimetric_sigma(n_t, fraction_u, modulation_efficiency)
    rng = np.random.default_rng(seed)
    I, q, u = np.asarray(lc.I), np.asarray(lc.q), np.asarray(lc.u)
    return {
        "t": t,
        "I": I,
        "Q": np.asarray(lc.Q),
        "U": np.asarray(lc.U),
        "q": q,
        "u": u,
        "sigma_I": sigma_I,
        "sigma_q": sigma_q,
        "sigma_u": sigma_u,
        "I_obs": I + sigma_I * rng.standard_normal(t.size),
        "q_obs": q + sigma_q * rng.standard_normal(t.size),
        "u_obs": u + sigma_u * rng.standard_normal(t.size),
        "photons_per_exposure": n_photons,
    }


def matched_filter_snr(signal, sigma, fit_offset: bool = True) -> float:
    """Signal-to-noise ratio of a known template in white Gaussian noise.

    ``SNR^2 = sum (s_i / sigma_i)^2``. Real polarimetric data always carry an
    unknown constant offset (interstellar and instrumental polarisation), so by
    default the template is first made orthogonal to a constant (its weighted
    mean is subtracted), which is the SNR after fitting that offset.

    Args:
        signal: Noise-free template (e.g. ``q`` from :func:`simulate_transit_observation`).
        sigma: Per-point uncertainties.
        fit_offset: Marginalise over a constant offset.
    """
    s = np.asarray(signal, float)
    w = 1.0 / np.asarray(sigma, float) ** 2
    if fit_offset:
        s = s - np.sum(w * s) / np.sum(w)
    return float(np.sqrt(np.sum(w * s**2)))


def bin_data(t, y, sigma, n: int):
    """Average consecutive groups of ``n`` points (a trailing partial bin is dropped).

    Returns:
        ``(t_bin, y_bin, sigma_bin)`` with ``sigma_bin = sqrt(sum sigma^2) / n``.
    """
    t, y, sigma = (np.asarray(a, float) for a in (t, y, sigma))
    k = (t.size // n) * n
    shape = (-1, n)
    return (
        t[:k].reshape(shape).mean(1),
        y[:k].reshape(shape).mean(1),
        np.sqrt((sigma[:k].reshape(shape) ** 2).sum(1)) / n,
    )
