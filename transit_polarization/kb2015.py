"""Access to the Kostogryz & Berdyugina (2015) centre-to-limb tables.

Kostogryz & Berdyugina 2015, A&A 575, A89 ("Center-to-limb polarization in
continuum spectra of F, G, K stars", arXiv:1408.5019) solved the polarised
radiative transfer equation for plane-parallel PHOENIX LTE model atmospheres and
tabulated, at ten disk positions ``mu = 0.10 ... 0.90``:

* Table 1 -- the centre-to-limb variation of intensity (CLIV), ``I(mu) / I(1)``;
* Table 2 -- the centre-to-limb variation of polarisation (CLPV), ``Q(mu) / I(mu)``.

Their ``Q`` is defined in a *local, limb-aligned* frame: positive ``Q`` means the
electric vector is parallel to the stellar limb (tangential), and Stokes U is zero
by symmetry (paper Sect. 2.2). We call this limb-frame Q "script Q" elsewhere in
this package to distinguish it from sky-frame Stokes Q.

Both tables are on a regular grid:

* wavelength 4000, 4500, 5000, 6000, 7000 Angstrom (monochromatic continuum),
* Teff 4500 ... 6900 K in 100 K steps,
* log g 3.0 ... 5.0 (cgs) in 0.5 dex steps.

Tabulated values at ``mu = 1`` are implicit: ``I/I(1) = 1`` and ``Q/I = 0``.
Nothing is tabulated below ``mu = 0.1``.

Copies of Tables 1 and 2 (VizieR ASU-TSV format, fetched 2026-11-07 from the
CfA VizieR mirror) are bundled in ``transit_polarization/data``. Use
:func:`download_tables` to refresh them.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

__all__ = [
    "MU_TABLE",
    "KB2015Grid",
    "download_tables",
    "load_kb2015",
]

CATALOGUE = "J/A+A/575/A89"

#: Disk positions mu = cos(theta) at which both tables are given.
MU_TABLE = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70, 0.90])

_INTENSITY_COLUMNS = ["LD10", "LD15", "LD20", "LD25", "LD30", "LD35", "LD40", "LD50", "LD70", "LD90"]
_POLARIZATION_COLUMNS = ["NS10", "NS15", "NS20", "NS25", "NS30", "NS35", "NS40", "NS50", "NS70", "NS90"]

#: VizieR servers tried in order. The Strasbourg server is the primary; the CfA
#: mirror serves the same ASU-TSV output and is used if Strasbourg is unreachable.
VIZIER_MIRRORS = (
    "https://vizier.cds.unistra.fr",
    "https://vizier.cfa.harvard.edu",
)

DATA_DIR = Path(__file__).parent / "data"
_FILENAMES = {1: "kb2015_table1.tsv", 2: "kb2015_table2.tsv"}


def _asu_url(mirror: str, table: int) -> str:
    source = CATALOGUE.replace("+", "%2B")
    return (
        f"{mirror}/viz-bin/asu-tsv?-source={source}/table{table}"
        "&-out.all&-out.max=unlimited&-out.form=mini"
    )


def download_tables(data_dir: Path | str = DATA_DIR, timeout: float = 30.0) -> list[Path]:
    """Download Tables 1 and 2 from VizieR as tab-separated files.

    Each mirror in :data:`VIZIER_MIRRORS` is tried in turn.

    Args:
        data_dir: Directory to write ``kb2015_table{1,2}.tsv`` into.
        timeout: Per-request timeout in seconds.

    Returns:
        Paths of the written files.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for table, name in _FILENAMES.items():
        errors = []
        for mirror in VIZIER_MIRRORS:
            url = _asu_url(mirror, table)
            try:
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    text = response.read().decode("utf-8")
            except OSError as exc:  # URLError, timeouts, connection refused
                errors.append(f"{mirror}: {exc}")
                continue
            if "LD10" not in text and "NS10" not in text:
                errors.append(f"{mirror}: response did not contain the expected columns")
                continue
            path = data_dir / name
            path.write_text(text)
            written.append(path)
            break
        else:
            raise OSError(f"Could not download table{table} from any mirror:\n" + "\n".join(errors))
    return written


def _read_asu_tsv(path: Path) -> dict[str, np.ndarray]:
    """Parse VizieR ASU-TSV output into a dict of column arrays.

    The format is: ``#`` comment lines, a header line, a units line, a dashes
    line, then data rows. Data rows are identified by an integer ``recno``.
    """
    lines = [ln.rstrip("\n") for ln in Path(path).read_text().splitlines()]
    lines = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
    header = [h.strip() for h in lines[0].split("\t")]
    rows = []
    for ln in lines[1:]:
        fields = ln.split("\t")
        if fields[0].strip().isdigit():  # skips the units and dashes lines
            rows.append([f.strip() for f in fields])
    columns = {}
    for i, name in enumerate(header):
        values = [r[i] for r in rows]
        try:
            columns[name] = np.array(values, dtype=float)
        except ValueError:
            columns[name] = np.array(values)
    return columns


@dataclass(frozen=True)
class KB2015Grid:
    """The two K&B (2015) tables reshaped onto their regular parameter grid.

    Attributes:
        wavelength: Wavelengths in Angstrom, shape ``(n_wl,)``.
        teff: Effective temperatures in K, shape ``(n_teff,)``.
        logg: Surface gravities log10(g / cm s^-2), shape ``(n_logg,)``.
        mu: Disk positions, shape ``(n_mu,)`` (equal to :data:`MU_TABLE`).
        intensity: ``I(mu) / I(1)``, shape ``(n_wl, n_teff, n_logg, n_mu)``.
        pol_degree: ``Q(mu) / I(mu)`` in the limb frame (positive = parallel to
            the limb), same shape as ``intensity``.
    """

    wavelength: np.ndarray
    teff: np.ndarray
    logg: np.ndarray
    mu: np.ndarray
    intensity: np.ndarray
    pol_degree: np.ndarray

    def _interpolator(self, values: np.ndarray) -> RegularGridInterpolator:
        return RegularGridInterpolator(
            (self.wavelength, self.teff, self.logg), values, method="linear", bounds_error=True
        )

    def profile(self, wavelength: float, teff: float, logg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tabulated profiles at arbitrary (wavelength, Teff, log g).

        The tabulated quantities ``I/I(1)`` and ``Q/I`` are interpolated linearly
        and independently in each of wavelength, Teff and log g. This is exact
        at grid points. Between wavelength nodes (up to 1000 Angstrom apart) it
        is crude, because Q/I falls steeply with wavelength; see the README.

        Args:
            wavelength: Wavelength in Angstrom (4000--7000).
            teff: Effective temperature in K (4500--6900).
            logg: log10 surface gravity in cgs (3.0--5.0).

        Returns:
            ``(mu, I/I(1), Q/I)``, each of shape ``(n_mu,)``.
        """
        point = np.array([[wavelength, teff, logg]], dtype=float)
        try:
            intensity = self._interpolator(self.intensity)(point)[0]
            pol_degree = self._interpolator(self.pol_degree)(point)[0]
        except ValueError as exc:
            raise ValueError(
                f"(wavelength={wavelength}, teff={teff}, logg={logg}) is outside the K&B 2015 grid: "
                f"wavelength {self.wavelength.min():.0f}-{self.wavelength.max():.0f} A, "
                f"Teff {self.teff.min():.0f}-{self.teff.max():.0f} K, "
                f"log g {self.logg.min():.1f}-{self.logg.max():.1f}"
            ) from exc
        return self.mu.copy(), intensity, pol_degree


def _to_grid(columns: dict[str, np.ndarray], value_columns: list[str]):
    wl = np.unique(columns["lambda"])
    teff = np.unique(columns["Teff"])
    logg = np.unique(columns["logg"])
    cube = np.full((wl.size, teff.size, logg.size, len(value_columns)), np.nan)
    i = np.searchsorted(wl, columns["lambda"])
    j = np.searchsorted(teff, columns["Teff"])
    k = np.searchsorted(logg, columns["logg"])
    cube[i, j, k, :] = np.column_stack([columns[c] for c in value_columns])
    if np.isnan(cube).any():
        raise ValueError("K&B 2015 table does not fill a complete regular grid")
    return wl, teff, logg, cube


def load_kb2015(data_dir: Path | str = DATA_DIR, download: bool = True) -> KB2015Grid:
    """Load Tables 1 and 2 of Kostogryz & Berdyugina (2015).

    Args:
        data_dir: Directory holding ``kb2015_table{1,2}.tsv``. Defaults to the
            copies bundled with this package.
        download: If the files are missing, fetch them with :func:`download_tables`.

    Returns:
        A :class:`KB2015Grid`.
    """
    data_dir = Path(data_dir)
    paths = {t: data_dir / name for t, name in _FILENAMES.items()}
    if not all(p.exists() for p in paths.values()):
        if not download:
            raise FileNotFoundError(f"K&B 2015 tables not found in {data_dir}")
        download_tables(data_dir)

    wl1, teff1, logg1, intensity = _to_grid(_read_asu_tsv(paths[1]), _INTENSITY_COLUMNS)
    wl2, teff2, logg2, pol_degree = _to_grid(_read_asu_tsv(paths[2]), _POLARIZATION_COLUMNS)
    if not (np.array_equal(wl1, wl2) and np.array_equal(teff1, teff2) and np.array_equal(logg1, logg2)):
        raise ValueError("Tables 1 and 2 are on different parameter grids")
    return KB2015Grid(
        wavelength=wl1,
        teff=teff1,
        logg=logg1,
        mu=MU_TABLE.copy(),
        intensity=intensity,
        pol_degree=pol_degree,
    )
