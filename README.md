# transit-polarization

Transits of limb-polarised stars in Stokes I, Q and U, computed with
[jaxoplanet](https://github.com/exoplanet-dev/jaxoplanet) (polynomial limb darkening
plus `jaxoplanet.starry`) and the centre-to-limb intensity and polarisation tables of
Kostogryz & Berdyugina (2015, A&A 575, A89; arXiv:1408.5019; VizieR `J/A+A/575/A89`).

**The idea.** In the continuum of an FGK star, light from near the limb is weakly
linearly polarised by scattering, parallel to the limb. Integrated over the disk this
cancels, so the star is unpolarised. A transiting planet hides part of the disk, the
symmetry breaks, and the remaining light carries a net polarisation of a few ppm. This
package predicts that signal and estimates whether a given telescope can detect it.

The worked, pedagogical version is
[notebooks/transit_polarization_mwe.ipynb](notebooks/transit_polarization_mwe.ipynb),
written for readers who know ordinary transit light curves. This README is the
reference version of the conventions.

---

## Quick start

All of this runs in the existing `jax` conda environment (Python 3.11, JAX 0.7.1,
jaxoplanet 0.1.0). No installation is needed for the tests or the notebook, which add the
repo root to `sys.path`. `pip install -e .` also works.

```python
import numpy as np
import transit_polarization as tp
from jaxoplanet.orbits import TransitOrbit

star = tp.LimbPolarizationModel.from_kb2015(teff=4600, logg=4.5, wavelength=4377)
orbit = TransitOrbit(period=3.0, duration=0.125, impact_param=0.5, radius_ratio=0.1)
lc = tp.transit_stokes(star, orbit, np.linspace(-0.09, 0.09, 500))
lc.I, lc.q, lc.u, lc.p        # I/F0, Q/I, U/I, sqrt(Q^2+U^2)/I
```

Photon-noise simulation for a telescope:

```python
sim = tp.simulate_transit_observation(
    star, orbit, vega_mag=8.6, band="B", aperture_diameter=4.0, central_obstruction=0.3,
    throughput=0.3, exposure_time=60.0, t_start=-0.1, t_end=0.1, seed=0)
snr = np.hypot(tp.matched_filter_snr(sim["q"], sim["sigma_q"]),
               tp.matched_filter_snr(sim["u"], sim["sigma_u"]))
```

## Layout

| Path | Contents |
| --- | --- |
| `transit_polarization/kb2015.py` | Load (and re-download) K&B Tables 1–2; linear interpolation in (λ, Teff, log g) |
| `transit_polarization/profiles.py` | Polynomial fits of I(μ) and 𝒬(μ); `LimbPolarizationModel` |
| `transit_polarization/stokes.py` | Sky-frame Q/U maps as starry surfaces; transit light curves in I, Q, U; frame rotation |
| `transit_polarization/noise.py` | Vega-mag photon rates (Bessell B/V/R), polarimetric noise, simulation, matched-filter SNR |
| `transit_polarization/data/` | Bundled copies of K&B Tables 1 and 2 (VizieR ASU-TSV) |
| `tests/` | Unit tests: table-vs-polynomial for all 625 models, starry-vs-brute-force occultations, symmetries, noise |
| `notebooks/transit_polarization_mwe.ipynb` | Executed MWE notebook |

---

## Can you define a limb-frame U, perpendicular to the limb?

Not as "perpendicular to the limb". Linear polarisation is a headless direction, so
Stokes parameters live on angle 2χ. In K&B's local frame (reference axis parallel to
the limb):

* 𝒬 > 0 means the electric vector is **parallel** to the limb;
* 𝒬 < 0 means it is **perpendicular** to the limb (radial);
* a limb-frame U would mean polarisation at **±45°** to the limb.

So polarisation perpendicular to the limb is just negative 𝒬, not a separate component.
A limb-frame U is well defined, but for a plane-parallel, unmagnetised atmosphere it
vanishes identically. The scattering geometry is mirror-symmetric about the plane
containing the line of sight and the local surface normal, and U is odd under that
reflection. K&B say exactly this (Sect. 2.2: "Stokes Q represents the linear
polarization in the direction parallel to the stellar limb, and this means that Stokes U
is equal to zero"). A non-zero limb-frame U needs something that breaks the mirror
symmetry: magnetic fields via the Hanle effect, horizontal inhomogeneities such as
granulation or spots, or departures from sphericity. All of K&B's tabulated Q/I values
are positive, so the continuum polarisation is everywhere parallel to the limb.

The *sky-frame* U is a different thing. It is generally non-zero during a transit
(see below), because converting the tangential polarisation field to fixed sky axes mixes
limb-frame 𝒬 into both sky Q and sky U.

---

## Conventions

### The coordinate μ

μ = cos θ, where θ is the angle between the local surface normal and the line of
sight. A point at projected distance r (stellar radii) from disk centre has
μ = √(1 − r²): μ = 1 at disk centre and μ = 0 at the limb. **This is the same μ in K&B,
exoplanet, starry and jaxoplanet** (Agol, Luger & Foreman-Mackey 2020, arXiv:1908.03222,
call it μ = z). No conversion is needed.

### Intensity: the jaxoplanet / exoplanet polynomial law

    I(μ) / I(1) = 1 − Σ_{n=1..N} u_n (1 − μ)^n          (ALFM20 eq. 1)

The coefficients are passed as `u` to `jaxoplanet.light_curves.limb_dark.light_curve` or
`jaxoplanet.core.limb_dark.light_curve`. Default N = 6.

### K&B (2015) tables

| Table | Quantity | Notes |
| --- | --- | --- |
| 1 | I(μ)/I(1) | at μ = 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70, 0.90; 1 at μ = 1 by definition; rounded to 10⁻⁴ |
| 2 | Q(μ)/I(μ), limb frame | same μ; 0 at μ = 1; positive = parallel to the limb |
| 3 | fit coefficients a₁…a₄ | f(μ) = a₁ + a₂μ^½ + a₃μ + a₄μ^{3/2}. **Not used**: it is not polynomial in μ (so jaxoplanet can't use it) and deviates from Table 1 by up to 6×10⁻³. The paper writes Σ_{i=0}^N a_i μ^{i/2}; CDS labels the four coefficients a₁…a₄. |

The grid is λ = 4000, 4500, 5000, 6000, 7000 Å (monochromatic continuum), Teff = 4500–6900 K
in 100 K steps, and log g = 3.0–5.0 in 0.5 dex steps, giving 625 models. The models are
PHOENIX LTE plane-parallel atmospheres with no magnetic fields; the opacities include
Rayleigh scattering (H I, H₂, …) and Thomson scattering. **Nothing is tabulated below
μ = 0.1** (r > 0.995).

### Polarised intensity 𝒬 ("script Q") and the function g(μ)

Degrees of polarisation do not add across the disk; Stokes intensities do. So the
fitted quantity is the limb-frame polarised intensity in units of the disk-centre
intensity:

    𝒬(μ) = [Q/I](μ) × [I/I(1)](μ)              (Table 2 × Table 1)

It is represented as

    𝒬(μ) = (1 − μ²) g(μ),     g(μ) = g₁ [1 − Σ_{n=1..M} v_n (1 − μ)^n]

The (1 − μ²) = r² factor is required, not cosmetic. At disk centre the direction
"parallel to the limb" is undefined, so a smooth polarisation field must vanish there
like r². The factor also makes the sky maps exact polynomials (next section). The
g polynomial has the same form as the intensity law, so v can go into starry's
limb-darkening slot. Default M = 6. Large alternating v_n (|v_n| ~ 10²) are normal:
g rises steeply towards the limb.

### Frames and sky-frame Stokes parameters

We use the **transit frame** of `jaxoplanet.orbits.TransitOrbit`: x along the planet's
sky-projected direction of motion, y perpendicular (the planet crosses at y = b), and
z towards the observer, all in stellar radii. Sky-frame Stokes parameters are

    Q = I_x − I_y,    U = I_(+45°) − I_(−45°),    angles measured from +x towards +y.

A point at position angle φ (x = r cos φ, y = r sin φ) has limb-parallel polarisation at
angle φ + 90°, so

    Q_sky = −𝒬 cos 2φ = −g(μ) (x² − y²)
    U_sky = −𝒬 sin 2φ = −2 g(μ) x y

On the unit sphere x² − y² and xy are the l = 2, m = ±2 spherical harmonics, so **each
Stokes map is one Y₂,±₂ coefficient times a limb-darkening-like polynomial g**. That is
exactly the structure of a limb-darkened `jaxoplanet.starry.Surface`, which jaxoplanet
occults analytically. Each map integrates to zero over the disk.

To express Q, U in a polarimeter's frame, use `rotate_stokes(Q, U, a)` (the reference
axis rotated by a from +x towards +y). If the planet moves along position angle PA,
measured North through East, the Stokes parameters relative to North are
`rotate_stokes(Q, U, -PA)`. Here the transit frame is taken as right-handed with z
towards the observer; check the handedness of your instrument's angle convention before
relying on the sign of U. The degree of polarisation p = √(Q² + U²)/I does not depend on
the frame.

### Normalisations: every quantity and its units

| Symbol / code | Definition | Units |
| --- | --- | --- |
| `model.intensity(μ)` | I(μ)/I(1) | – |
| `model.polarized_intensity(μ)` | 𝒬(μ) | I(1) |
| `model.g1` | g(1) = lim_{μ→1} 𝒬/(1 − μ²) | I(1) |
| `model.total_flux` = F₀ | π [1 − Σ 2u_n / ((n+1)(n+2))] | I(1) R★² |
| `radial_poly_disk_integral(v)` = F_v | 1 − Σ 2v_n / ((n+1)(n+2)) = (1/π)∫ g/g₁ dA | – |
| starry Y₀,₀ | 1/π (a uniform unit-flux disk) | – |
| starry Y₂,₂ | √15/(2π) (x² − y²) = `STARRY_Y22` (x² − y²) | – |
| starry Y₂,₋₂ | √15/π · xy = `STARRY_Y2M2` · xy | – |
| `lc.I` | I(t)/F₀ (= 1 out of transit) | – |
| `lc.Q`, `lc.U` | Q(t)/F₀, U(t)/F₀ (= 0 out of transit) | – |
| `lc.q`, `lc.u`, `lc.p` | Q/I, U/I, √(Q²+U²)/I: what a polarimeter measures | – |

The disk integrals use ∫(1 − μ)^n dA = 2π ∫₀¹ (1 − μ)^n μ dμ = 2π/((n+1)(n+2)).

### The normalisation traps (probably what tripped you up before)

1. **`calc_poly_coeffs` assumes I(1) = 1.** It fits I − 1 = −Σ u_n(1−μ)^n with the
   constant pinned to 1. K&B Table 1 is already normalised this way. An intensity in
   absolute units, or normalised to the mean, must be divided by I(1) first.
2. **The light curve is normalised by F₀, not by π I(1).**
   `jaxoplanet.core.limb_dark.light_curve` returns F/F₀ − 1, with
   F₀ ≈ 2.4 I(1) R★² for a typical star rather than π I(1) R★². Dividing an occulted
   polarised flux by π I(1) instead of F₀ overestimates it by ~30 %.
3. **starry divides every limb-darkened map by F_v.**
   `jaxoplanet.starry.light_curves.surface_light_curve` multiplies by
   π / (p_u · rT) = 1/F_v so that limb darkening never changes the flux of a Y₀,₀ map.
   For intensity that's what you want. For the Q map the "limb darkening" is g, so the
   result must be **multiplied by F_v** to get ∫ Q_sky dA. Multiplying by π F_v (the
   natural-looking guess) is wrong by exactly π. This was found and pinned numerically
   against brute-force integration.
4. **starry's Ylms are 2/√π times orthonormal.** So Y₀,₀ = 1/π, not 1/(2√π), and the
   Y₂,±₂ coefficients must be divided by `STARRY_Y22` and `STARRY_Y2M2`. When a map's
   Y₀,₀ is zero (as for Q and U), `Surface` must be built with `normalize=False`, or it
   divides by zero.
5. **Fit 𝒬, not Q/I.** Q/I is a ratio and does not integrate over the disk.
6. **Q/F₀ vs Q/I.** They differ by I(t), about 1 % in transit. A polarimeter measures q = Q/I.

Traps 1–4 are covered by unit tests: `test_starry_ylm_normalisation`,
`test_stokes_surfaces_render_the_analytic_maps`, `test_occultation_matches_brute_force`
and `test_total_flux_matches_numerical_integral`.

---

## How the calculation works

1. **Tables to profiles.** `KB2015Grid.profile(λ, Teff, log g)` linearly interpolates
   Tables 1 and 2 (exact at grid nodes).
2. **Profiles to polynomials.** `fit_intensity` (wrapping
   `jaxoplanet.experimental.calc_poly_coeffs`) gives u. `fit_polarized_intensity` gives
   (g₁, v) by linear least squares with design columns (1 − μ²)(1 − μ)^n, n = 0..M.
   Both satisfy the μ = 1 condition exactly by construction.
3. **Polynomials to starry maps.** `stokes_surfaces` builds
   `Surface(y=Ylm({(2, 2): −g₁/STARRY_Y22}), u=v, normalize=False)` for Q, and the
   same with `(2, −2)` and −2g₁/`STARRY_Y2M2` for U. The surface is equator-on with
   no obliquity, so the map frame is the transit frame.
4. **Occultation.** `occultation_stokes` computes, for each planet position,
   I/F₀ = 1 + (jaxoplanet limb-dark light curve), and
   Q/F₀ = `surface_light_curve(surf_Q, r, x, y, z)` × F_v / F₀, likewise for U.
   `transit_stokes` takes positions from any jaxoplanet orbit.

### Validation

* **All 625 table models:** the degree-6 polynomials reproduce Table 1 to
  ≤ 2.5×10⁻⁴ I(1) (the table rounding is 10⁻⁴) and the polarised intensity to
  ≤ 0.25 % of its peak. The fitted I(μ) is positive and monotonic on [0, 1].
* **starry vs brute force:** occulted Q, U agree with direct quadrature over the
  planet disk to 10⁻³ of the signal scale, including a planet straddling the limb and
  the steepest (cool, low-gravity, 4000 Å) model. The notebook repeats this with a
  naive pixel sum.
* **Symmetries:** a central transit (b = 0) has U ≡ 0 and Q ≥ 0. For any b, Q is
  symmetric and U antisymmetric about mid-transit. The unocculted star has Q = U = 0.

---

## Noise model (`noise.py`)

* **Photon rate.** The host's Vega magnitude in Bessell B, V or R is converted to AB
  with Bessell & Murphy (2012, PASP 124, 140) Table 3 (`m_Vega = AB − ZP`, so
  `m_AB = m_Vega + ZP`: B −0.138, V −0.023, R +0.160). The band is approximated as a
  top-hat of the Table 5 FWHM at the pivot wavelength (B: 4377/890 Å, V: 5488/830 Å,
  R: 6515/1443 Å):
  `Ṅ = (3631 Jy × 10^(−0.4 m_AB) / h) × (FWHM/λ_pivot) × A × T`.
  As a check, V = 0 gives 8.5×10⁵ photons s⁻¹ cm⁻², matching the textbook
  ~1000 photons s⁻¹ cm⁻² Å⁻¹ over ~850 Å. U and I are not offered, because they fall
  outside the 4000–7000 Å range of the tables.
* **Polarimeter.** Dual-beam: every photon counts towards I, so σ(I/F₀) = 1/√N.
  q is measured during a fraction f_q of each exposure (default 0.5, with the other
  half on U), so σ_q = 1/(ε √(f_q N)), with modulation efficiency ε (default 1).
* **Detection metric.** Matched-filter SNR² = Σ(s/σ)², after projecting out a constant
  offset (interstellar and instrumental polarisation are always present). q and u are
  independent, so SNR_tot = √(SNR_q² + SNR_u²). Transits needed for 5σ: (5/SNR)².
* **Not modelled:** read noise, sky, scintillation (important for ground-based I, but it
  largely cancels in a dual-beam q), instrumental systematics and the ppm-level
  polarimetric floor of real instruments, and exposure smearing (the model is evaluated
  at mid-exposure). Throughput is a free parameter: **use your instrument's measured
  value**. The notebook's 0.3 is a placeholder.

## Headline numbers (from the notebook)

* Rp/R★ = 0.1, b = 0.5, at 4377 Å: peak p ≈ 1.4 ppm (F dwarf, 6500 K), 2.0 ppm
  (G dwarf, 5800 K), 4.0 ppm (K dwarf, 4600 K) and 15.6 ppm (K giant, 4500 K,
  log g 3.0, shown only to illustrate the trend). Transit depths are all ≈ 1.2 %.
* The signal peaks around 2nd and 3rd contact. Q is symmetric in time and changes sign;
  U is antisymmetric.
* HD 189733-like system (approximate literature parameters, B ≈ 8.6), B band, throughput
  0.3, photon noise only: SNR per transit is ≈ 0.36 at 2 m, 0.71 at 4 m, 1.4 at 8 m and
  6.9 at 39 m. That is roughly 200, 50, 13 and 1 transits for 5σ.
* **Model uncertainty:** changing the degree of g (4–6) moves the peak p by < 1 %, so
  the untabulated μ < 0.1 limb barely matters. Mid-transit p moves by about 20 %, because
  the tables are sparse at μ > 0.5 (only 0.5, 0.7, 0.9).

---

## Decisions and assumptions log

Recorded 2026-11-06. Items marked *(user)* were chosen by Shashank when offered options.

1. **Data source.** VizieR ASU-TSV output. CDS Strasbourg (`cdsarc`/`vizier.cds.unistra.fr`)
   was unreachable from this machine on 2026-11-06, so the tables were fetched from
   the CfA mirror (`vizier.cfa.harvard.edu`), which serves the identical catalogue.
   Tables 1 and 2 are bundled so the tests run offline; `tp.download_tables()` refreshes
   them, trying Strasbourg first.
2. **Fitting method** *(user)*: direct linear least squares to the ten tabulated points,
   degree 6 for I and degree 6 for g. Alternatives considered were a monotone spline plus
   explicit extrapolation followed by a high-degree fit, and regularised least squares.
   Degree ≥ 7 for g overfits (g(1) < 0) and raises an error. Odd degrees can produce
   tiny negative 𝒬 near μ ≈ 1.
3. **Grid lookup** *(user)*: linear interpolation in (λ, Teff, log g) of the tabulated
   quantities I/I(1) and Q/I, exact at nodes. Caveat: the wavelength nodes are up to
   1000 Å apart and Q/I falls steeply with λ, so between 5000–6000 and 6000–7000 Å the
   linear interpolation can be off by ~10 % (log-linear would differ). The B band
   (4377 Å) sits between the 4000 and 4500 Å nodes, where the error is smaller.
4. **Monochromatic approximation.** A broadband observation is modelled with the K&B
   profile at the band's pivot wavelength. Spectral line (de)polarisation, absent from
   K&B's continuum calculation, is ignored.
5. **Noise model** *(user)*: Vega magnitudes, Bessell B/V/R, Bessell & Murphy (2012)
   zero points. Their further Table 5 corrections (≤ 0.01 mag) are neglected.
6. **Polarimeter defaults:** dual-beam, half of each exposure on Q and half on U,
   ε = 1. These are arguments, not hidden constants.
7. **Plot style** *(user)*: notebook-readable (Matplotlib defaults, 12.5 pt fonts,
   gridlines, tab10). Residual panels use symmetric limits about zero. Q/U maps use a
   diverging RdBu_r colour map with a symmetric-log norm centred on zero, and grey for
   off-disk pixels. The images are in the transit frame (x along the planet's motion),
   not North-up/East-left.
8. **Physics left out.** The planet is opaque, with no atmosphere, reflected light or
   intrinsic polarisation (reflected-light polarisation peaks near quadrature, not in
   transit). The star is spherical and non-rotating: no gravity darkening, oblateness,
   spots or faculae (spots also produce polarisation; see Kostogryz et al. 2015). No
   magnetic fields or Hanle effect. The plane-parallel atmosphere misrepresents the
   extreme limb.
9. **Environment.** Developed and tested in the `jax` conda env: Python 3.11.9,
   JAX 0.7.1, jaxoplanet 0.1.0. jaxoplanet 0.2.0 exists but is untested; the starry
   API is still evolving, so `pyproject.toml` pins `<0.2`.

---

## Context: is this still an open question?

The idea is not new, but no detection has been reported to date (based on a quick
literature search on 2026-11-06; worth a proper ADS search before writing anything up).

* Carciofi & Magalhães 2005, ApJ 635, 570 (astro-ph/0508343): first detailed
  predictions for G, K, M and T dwarfs; signal peaks at ingress and egress; possible
  eccentricity signature in asymmetry.
* Kostogryz et al. 2011, MNRAS 415, 695: transit polarisation adopting solar
  centre-to-limb polarisation. Frantseva et al. (2012) found results differing by up to
  two orders of magnitude depending on the assumed stellar limb polarisation, which is
  exactly why the K&B (2015) tables matter.
* Kostogryz, Yakobchuk & Berdyugina 2015, ApJ 806, 97 (arXiv:1504.02943),
  "Polarization in exoplanetary systems caused by transits, grazing transits, and
  starspots": applies the K&B 2015 profiles to known planets, including spotted hosts.
* Wiktorowicz & Laughlin 2014 (arXiv:1409.2504), "Toward the Detection of Exoplanet
  Transits with Polarimetry".

What this package adds is an exact, differentiable (JAX) forward model inside the
jaxoplanet ecosystem, with every normalisation pinned by tests. It can be fitted to
data, or wrapped in an information-content or Fisher analysis.

## References

* Agol, Luger & Foreman-Mackey 2020, AJ 159, 123 (arXiv:1908.03222): polynomial limb darkening.
* Bessell & Murphy 2012, PASP 124, 140 (arXiv:1112.2698): UBVRI passbands and zero points.
* Kostogryz & Berdyugina 2015, A&A 575, A89 (arXiv:1408.5019); VizieR J/A+A/575/A89.
* Luger et al. 2019, AJ 157, 64: starry.

## Running the tests and the notebook

Tests (about 30 s):

```
/Users/uqsdhola/opt/anaconda3/envs/jax/bin/python -m pytest -q
```

The notebook needs a kernel for the `jax` env. Either pick that interpreter in VS Code's
kernel picker, or register it once with
`/Users/uqsdhola/opt/anaconda3/envs/jax/bin/python -m ipykernel install --user --name jax`.
