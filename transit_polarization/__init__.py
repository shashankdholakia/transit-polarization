"""Transits of limb-polarised stars in Stokes I, Q and U with jaxoplanet.

See README.md for conventions and normalisations.
"""

import jax

# Polarisation signals are ~1e-6 of the stellar flux; single precision is not enough.
jax.config.update("jax_enable_x64", True)

from .kb2015 import MU_TABLE, KB2015Grid, download_tables, load_kb2015  # noqa: E402
from .noise import (  # noqa: E402
    BESSELL_BANDS,
    Band,
    bin_data,
    matched_filter_snr,
    photon_rate,
    polarimetric_sigma,
    simulate_transit_observation,
)
from .profiles import (  # noqa: E402
    LimbPolarizationModel,
    fit_intensity,
    fit_polarized_intensity,
    intensity_profile,
    polarized_intensity_profile,
    radial_poly_disk_integral,
)
from .stokes import (  # noqa: E402
    STARRY_Y22,
    STARRY_Y2M2,
    StokesLightCurve,
    occultation_stokes,
    rotate_stokes,
    stokes_images,
    stokes_surfaces,
    transit_stokes,
)

__all__ = [
    "MU_TABLE",
    "KB2015Grid",
    "download_tables",
    "load_kb2015",
    "BESSELL_BANDS",
    "Band",
    "bin_data",
    "matched_filter_snr",
    "photon_rate",
    "polarimetric_sigma",
    "simulate_transit_observation",
    "LimbPolarizationModel",
    "fit_intensity",
    "fit_polarized_intensity",
    "intensity_profile",
    "polarized_intensity_profile",
    "radial_poly_disk_integral",
    "STARRY_Y22",
    "STARRY_Y2M2",
    "StokesLightCurve",
    "occultation_stokes",
    "rotate_stokes",
    "stokes_images",
    "stokes_surfaces",
    "transit_stokes",
]
