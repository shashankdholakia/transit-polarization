"""Polynomial representations of the centre-to-limb intensity and polarisation.

Conventions (see README.md for the full discussion)
---------------------------------------------------
``mu = cos(theta) = sqrt(1 - x^2 - y^2)``, where ``theta`` is the angle between
the local surface normal and the line of sight and ``(x, y)`` are sky
coordinates in units of the stellar radius. ``mu = 1`` at disk centre and
``mu = 0`` at the limb. This is the ``mu`` (= ``z``) of Agol, Luger &
Foreman-Mackey (2020), exoplanet, starry and jaxoplanet, and it is the same
``mu`` used by Kostogryz & Berdyugina (2015).

Intensity -- the jaxoplanet / exoplanet polynomial law (Agol et al. 2020, eq. 1)::

    I(mu) / I(1) = 1 - sum_{n=1}^{N} u_n (1 - mu)^n

Everything is in units of the disk-centre intensity ``I(1)``. The
disk-integrated flux of the unocculted star is then (in units of ``I(1) R_*^2``)::

    F_I = pi * (1 - sum_n 2 u_n / ((n + 1)(n + 2)))

Polarised intensity -- the limb-frame Stokes Q ("script Q") of Kostogryz &
Berdyugina, again in units of ``I(1)``::

    QQ(mu) = [Q/I](mu) * [I/I(1)](mu)          (positive = parallel to the limb)

We represent it as::

    QQ(mu) = (1 - mu^2) * g(mu),   g(mu) = g_1 * (1 - sum_{n=1}^{M} v_n (1 - mu)^n)

The prefactor ``1 - mu^2 = x^2 + y^2 = r^2`` is not optional: a limb-parallel
polarisation field is only smooth at disk centre if its amplitude vanishes like
``r^2`` there, and it makes the sky-frame Stokes maps exact polynomials in
``(x, y, z)``, which starry can integrate (see :mod:`transit_polarization.stokes`).
Writing ``g`` with the same ``1 - sum v_n (1 - mu)^n`` form as the intensity law
lets us pass ``v`` to starry in its limb-darkening slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from jaxoplanet.experimental import calc_poly_coeffs

from .kb2015 import KB2015Grid, load_kb2015

__all__ = [
    "fit_intensity",
    "fit_polarized_intensity",
    "intensity_profile",
    "polarized_intensity_profile",
    "radial_poly_disk_integral",
    "LimbPolarizationModel",
]

#: Default polynomial degrees (chosen 2026-11-06, see README "Decisions").
DEFAULT_INTENSITY_DEGREE = 6
DEFAULT_POLARIZATION_DEGREE = 6


def fit_intensity(mu, intensity, degree: int = DEFAULT_INTENSITY_DEGREE) -> np.ndarray:
    """Least-squares fit of the jaxoplanet polynomial limb-darkening law.

    This wraps :func:`jaxoplanet.experimental.calc_poly_coeffs`, which fits
    ``I/I(1) - 1 = -sum u_n (1-mu)^n``. The constant term is fixed at 1, so the
    input **must** already be normalised to ``I(1) = 1`` (K&B Table 1 is).
    The implicit table point ``mu = 1`` is therefore satisfied exactly.

    Args:
        mu: Disk positions.
        intensity: ``I(mu) / I(1)`` at ``mu``.
        degree: Polynomial degree N.

    Returns:
        ``u``, shape ``(N,)``, ordered ``u_1 ... u_N`` -- pass directly to
        :func:`jaxoplanet.light_curves.limb_dark.light_curve`.
    """
    mu = jnp.asarray(mu, dtype=float)
    intensity = jnp.asarray(intensity, dtype=float)
    return np.asarray(calc_poly_coeffs(mu, intensity, poly_degree=degree))


def fit_polarized_intensity(mu, polarized_intensity, degree: int = DEFAULT_POLARIZATION_DEGREE):
    """Least-squares fit of ``QQ(mu) = (1 - mu^2) g_1 (1 - sum v_n (1-mu)^n)``.

    The fit is linear in ``c_n`` for ``QQ = (1 - mu^2) sum_{n=0}^{M} c_n (1-mu)^n``.
    We then set ``g_1 = c_0`` and ``v_n = -c_n / c_0``. ``QQ(1) = 0`` holds
    exactly by construction.

    Args:
        mu: Disk positions (should exclude or may include ``mu = 1``).
        polarized_intensity: ``QQ(mu)`` in units of ``I(1)``, that is
            ``(Q/I) * (I/I(1))``.
        degree: Degree M of the polynomial ``g``.

    Returns:
        ``(g1, v)`` with ``g1`` a float and ``v`` of shape ``(M,)``.
    """
    mu = np.asarray(mu, dtype=float)
    qq = np.asarray(polarized_intensity, dtype=float)
    design = (1.0 - mu**2)[:, None] * np.vander(1.0 - mu, degree + 1, increasing=True)
    coeffs, *_ = np.linalg.lstsq(design, qq, rcond=None)
    g1 = float(coeffs[0])
    if not g1 > 0:
        raise ValueError(
            f"Fitted g(mu=1) = {g1:.3g} is not positive; the (1 - sum v_n (1-mu)^n) form needs g(1) > 0"
        )
    return g1, -coeffs[1:] / g1


def _radial_poly(mu, coeffs):
    """``1 - sum_n coeffs_n (1 - mu)^n`` (works on numpy or jax arrays)."""
    mu = jnp.asarray(mu)
    out = jnp.ones_like(mu)
    for n, c in enumerate(coeffs, start=1):
        out = out - c * (1.0 - mu) ** n
    return out


def intensity_profile(mu, u):
    """``I(mu) / I(1) = 1 - sum u_n (1 - mu)^n``."""
    return _radial_poly(mu, u)


def polarized_intensity_profile(mu, g1, v):
    """``QQ(mu) / I(1) = (1 - mu^2) g_1 (1 - sum v_n (1 - mu)^n)``."""
    mu = jnp.asarray(mu)
    return (1.0 - mu**2) * g1 * _radial_poly(mu, v)


def radial_poly_disk_integral(coeffs) -> float:
    """``(1/pi) * integral over the unit disk of (1 - sum c_n (1-mu)^n) dA``.

    Uses ``integral (1-mu)^n dA = 2 pi / ((n+1)(n+2))`` (since ``dA = 2 pi mu dmu``).
    For the intensity law this is ``F_I / pi``. For the ``g`` polynomial it is
    the normalisation that starry divides out of every limb-darkened map (see
    :mod:`transit_polarization.stokes`).
    """
    total = 1.0
    for n, c in enumerate(coeffs, start=1):
        total = total - 2.0 * c / ((n + 1) * (n + 2))
    return total


@dataclass(frozen=True)
class LimbPolarizationModel:
    """Polynomial centre-to-limb intensity and limb-frame polarisation of a star.

    Attributes:
        u: Intensity limb-darkening coefficients ``u_1 ... u_N`` (jaxoplanet form).
        g1: ``g(mu=1) = lim_{mu->1} QQ(mu)/(1-mu^2)`` in units of ``I(1)``.
        v: Shape coefficients ``v_1 ... v_M`` of ``g``.
        wavelength, teff, logg: Stellar parameters the model was built for
            (metadata only; ``None`` if built by hand).
    """

    u: np.ndarray
    g1: float
    v: np.ndarray
    wavelength: float | None = None
    teff: float | None = None
    logg: float | None = None
    table: dict = field(default=None, repr=False, compare=False)

    @classmethod
    def from_kb2015(
        cls,
        teff: float,
        logg: float,
        wavelength: float,
        grid: KB2015Grid | None = None,
        intensity_degree: int = DEFAULT_INTENSITY_DEGREE,
        polarization_degree: int = DEFAULT_POLARIZATION_DEGREE,
    ) -> "LimbPolarizationModel":
        """Fit the polynomial representation to (interpolated) K&B 2015 tables.

        Args:
            teff: Effective temperature in K.
            logg: log10 surface gravity (cgs).
            wavelength: Wavelength in Angstrom.
            grid: A loaded :class:`KB2015Grid`; loaded from the bundled data if None.
            intensity_degree: Degree N of the intensity polynomial.
            polarization_degree: Degree M of ``g``.
        """
        grid = load_kb2015() if grid is None else grid
        mu, intensity, pol_degree = grid.profile(wavelength, teff, logg)
        u = fit_intensity(mu, intensity, intensity_degree)
        g1, v = fit_polarized_intensity(mu, pol_degree * intensity, polarization_degree)
        table = {"mu": mu, "intensity": intensity, "pol_degree": pol_degree}
        return cls(u=u, g1=g1, v=v, wavelength=wavelength, teff=teff, logg=logg, table=table)

    def intensity(self, mu):
        """``I(mu) / I(1)``."""
        return intensity_profile(mu, self.u)

    def polarized_intensity(self, mu):
        """Limb-frame ``QQ(mu) / I(1)`` (positive = parallel to the limb)."""
        return polarized_intensity_profile(mu, self.g1, self.v)

    def pol_degree(self, mu):
        """Local degree of polarisation ``QQ(mu) / I(mu)`` (the K&B Table 2 quantity)."""
        return self.polarized_intensity(mu) / self.intensity(mu)

    @property
    def total_flux(self) -> float:
        """Unocculted disk-integrated flux ``F_I`` in units of ``I(1) R_*^2``."""
        return float(np.pi * radial_poly_disk_integral(self.u))
