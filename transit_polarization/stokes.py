"""Sky-frame Stokes maps of a limb-polarised star and their transit light curves.

Frames
------
We work in the *transit frame* used by :class:`jaxoplanet.orbits.TransitOrbit`:
``x`` along the planet's sky-projected direction of motion, ``y`` perpendicular
to it on the sky (the planet crosses at ``y = b``), ``z`` towards the observer.
All lengths are in stellar radii.

Sky-frame Stokes parameters are defined with respect to the ``x`` axis::

    Q = I_x - I_y,      U = I_(+45) - I_(-45)

with angles measured from ``+x`` towards ``+y``. A pixel at position angle
``phi`` (``x = r cos phi``, ``y = r sin phi``) whose light is polarised parallel
to the limb (polarisation angle ``phi + 90 deg``) with limb-frame polarised
intensity ``QQ`` contributes::

    Q_sky = QQ cos(2 phi + 180 deg) = -QQ cos(2 phi) = -g(mu) (x^2 - y^2)
    U_sky = QQ sin(2 phi + 180 deg) = -QQ sin(2 phi) = -g(mu) 2 x y

using ``QQ = (1 - mu^2) g = r^2 g``. Both maps are a single degree-2 spherical
harmonic times the radial polynomial ``g(mu)`` -- exactly the structure of a
limb-darkened starry map. The disk integral of either map is zero (the
unocculted star is unpolarised); a transiting planet hides part of it and the
remainder is the signal.

Normalisation traps (all handled below, all unit-tested)
--------------------------------------------------------
1. starry's real spherical harmonics are ``2/sqrt(pi)`` times the orthonormal
   ones, so ``Y_00 = 1/pi`` (unit-flux uniform disk),
   ``Y_22 = sqrt(15)/(2 pi) (x^2 - y^2)`` and ``Y_2,-2 = sqrt(15)/pi x y``.
2. :func:`jaxoplanet.starry.light_curves.surface_light_curve` divides every map
   by ``(1/pi) * disk integral of its limb-darkening polynomial`` so that limb
   darkening never changes the flux of ``Y_00``. Our "limb darkening" is ``g``,
   so we multiply that factor back in (:func:`radial_poly_disk_integral`).
3. ``jaxoplanet.core.limb_dark.light_curve`` returns ``F/F_0 - 1`` (zero out of
   transit), already divided by the full-disk limb-darkened flux ``F_0``.
4. We report ``Q`` and ``U`` divided by the same ``F_0`` as the intensity, so
   ``I``, ``Q``, ``U`` share units and ``q = Q/I`` is the observable that a
   polarimeter measures.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxoplanet.core.limb_dark import light_curve as _limb_dark_light_curve
from jaxoplanet.starry.light_curves import surface_light_curve
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.ylm import Ylm

from .profiles import LimbPolarizationModel, radial_poly_disk_integral

__all__ = [
    "STARRY_Y22",
    "STARRY_Y2M2",
    "StokesLightCurve",
    "stokes_surfaces",
    "stokes_images",
    "occultation_stokes",
    "transit_stokes",
    "rotate_stokes",
]

#: starry's Y_{2,2}(x, y, z) = STARRY_Y22 * (x^2 - y^2)
STARRY_Y22 = np.sqrt(15.0) / (2.0 * np.pi)
#: starry's Y_{2,-2}(x, y, z) = STARRY_Y2M2 * x * y
STARRY_Y2M2 = np.sqrt(15.0) / np.pi


class StokesLightCurve(NamedTuple):
    """Disk-integrated Stokes parameters, all divided by the out-of-transit flux."""

    I: jax.Array  # noqa: E741  -- Stokes I, = 1 out of transit
    Q: jax.Array  #: Stokes Q (transit frame unless rotated)
    U: jax.Array  #: Stokes U

    @property
    def q(self):
        """Normalised Stokes ``Q/I`` -- what a polarimeter measures."""
        return self.Q / self.I

    @property
    def u(self):
        """Normalised Stokes ``U/I``."""
        return self.U / self.I

    @property
    def p(self):
        """Degree of linear polarisation ``sqrt(Q^2 + U^2) / I`` (frame independent)."""
        return jnp.sqrt(self.Q**2 + self.U**2) / self.I

    @property
    def angle(self):
        """Polarisation angle ``0.5 * arctan2(U, Q)`` in radians, from +x towards +y."""
        return 0.5 * jnp.arctan2(self.U, self.Q)


def stokes_surfaces(model: LimbPolarizationModel) -> tuple[Surface, Surface]:
    """starry surfaces whose intensity equals the sky-frame Q and U maps.

    The returned surfaces are viewed equator-on with no obliquity, so the map
    frame coincides with the transit frame. Note that ``surface_light_curve``
    on these surfaces returns the map flux divided by
    ``radial_poly_disk_integral(model.v)``; :func:`occultation_stokes` undoes this.
    """
    v = tuple(float(c) for c in np.atleast_1d(model.v))
    surf_q = Surface(y=Ylm({(2, 2): -model.g1 / STARRY_Y22}), u=v, normalize=False)
    surf_u = Surface(y=Ylm({(2, -2): -2.0 * model.g1 / STARRY_Y2M2}), u=v, normalize=False)
    return surf_q, surf_u


def stokes_images(model: LimbPolarizationModel, res: int = 301):
    """Sky-frame I, Q, U maps on a regular grid (NaN off the disk), in units of I(1).

    These are evaluated directly from the profile polynomials (not via starry),
    and are meant for plotting and for brute-force checks.

    Returns:
        ``(x, y, I, Q, U)``: 1D coordinate arrays of length ``res`` and 2D maps of
        shape ``(res, res)`` indexed ``[iy, ix]`` (so ``imshow(..., origin="lower")``
        puts +y up).
    """
    x = np.linspace(-1.0, 1.0, res)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    r2 = xx**2 + yy**2
    on_disk = r2 < 1.0
    mu = np.sqrt(np.where(on_disk, 1.0 - r2, 0.0))
    intensity = np.asarray(model.intensity(mu))
    g = np.asarray(model.polarized_intensity(mu)) / np.where(on_disk & (r2 > 0), r2, 1.0)
    g = np.where(r2 > 0, g, model.g1)  # g(mu=1) = g1 at the exact centre
    q_map = -g * (xx**2 - yy**2)
    u_map = -g * 2.0 * xx * yy
    nan = np.where(on_disk, 1.0, np.nan)
    return x, x.copy(), intensity * nan, q_map * nan, u_map * nan


def occultation_stokes(model: LimbPolarizationModel, x, y, r, z=1.0, order: int = 20) -> StokesLightCurve:
    """Normalised Stokes I, Q, U of the star with an occulting disk at ``(x, y)``.

    Args:
        model: Stellar centre-to-limb model.
        x, y: Occultor centre in the transit frame, in stellar radii (arrays OK).
        r: Occultor radius in stellar radii (scalar).
        z: Line-of-sight coordinate; the occultor only blocks light if ``z > 0``.
        order: Quadrature order for the numerical line integrals in jaxoplanet.

    Returns:
        :class:`StokesLightCurve` with ``I = 1`` and ``Q = U = 0`` out of transit.
    """
    x, y, z = jnp.broadcast_arrays(jnp.asarray(x, float), jnp.asarray(y, float), jnp.asarray(z, float))
    shape = x.shape
    u = jnp.asarray(model.u, float)
    surf_q, surf_u = stokes_surfaces(model)
    # Undo starry's limb-darkening normalisation (trap 2) and divide by F_0 (trap 4).
    q_scale = radial_poly_disk_integral(model.v) / model.total_flux

    def single(xi, yi, zi):
        b = jnp.sqrt(xi**2 + yi**2)
        d_intensity = _limb_dark_light_curve(u, b, r, order=order)  # trap 3: F/F0 - 1
        d_intensity = jnp.where(zi > 0, d_intensity, 0.0)
        q = surface_light_curve(surf_q, r, xi, yi, zi, order=order) * q_scale
        uu = surface_light_curve(surf_u, r, xi, yi, zi, order=order) * q_scale
        return 1.0 + d_intensity, q, uu

    intensity, q, uu = jax.jit(jax.vmap(single))(x.ravel(), y.ravel(), z.ravel())
    return StokesLightCurve(intensity.reshape(shape), q.reshape(shape), uu.reshape(shape))


def transit_stokes(model: LimbPolarizationModel, orbit, t, order: int = 20) -> StokesLightCurve:
    """Stokes I, Q, U transit light curves in the transit frame.

    Args:
        model: Stellar centre-to-limb model.
        orbit: Any jaxoplanet orbit with ``relative_position``, ``radius`` and
            ``central_radius`` (e.g. :class:`jaxoplanet.orbits.TransitOrbit`).
        t: Times (same units as the orbit, usually days).
        order: Quadrature order.

    Returns:
        :class:`StokesLightCurve`. Use :func:`rotate_stokes` to express Q, U in
        your polarimeter's frame.
    """
    t = jnp.atleast_1d(jnp.asarray(t, float))
    x, y, z = orbit.relative_position(t)
    r_star = orbit.central_radius
    return occultation_stokes(model, x / r_star, y / r_star, orbit.radius / r_star, z, order=order)


def rotate_stokes(Q, U, angle):
    """Express (Q, U) relative to a reference axis rotated by ``angle`` from +x towards +y.

    ``Q' = Q cos 2a + U sin 2a``, ``U' = -Q sin 2a + U cos 2a``. The degree of
    polarisation is unchanged. If the planet moves along position angle ``PA``
    (measured North through East, and the transit frame is right-handed with
    ``z`` towards the observer), Stokes parameters with respect to North are
    ``rotate_stokes(Q, U, -PA)``.
    """
    c, s = jnp.cos(2.0 * angle), jnp.sin(2.0 * angle)
    return Q * c + U * s, -Q * s + U * c
