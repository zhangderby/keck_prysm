"""LGS sodium beacon: sodium profile and spot-elongation kernels.

AI use disclaimer: written with Claude Opus 5.5 (claude-opus-5-5, Anthropic)
at the request of the project owner; design in docs/design/design_notes.md,
sections 2.2 and 2.3.

Units: altitudes and heights in m, angles on sky in arcsec, pupil
coordinates in mm (prysm convention).  Heights are measured from the
telescope at zenith, as in atmosphere.los_coordinates.

References
----------
Brown, Fedrigo & van der Werf 2004, arXiv:astro-ph/0407445 (sodium beacon)
COMPASS ``shesha/init/lgs_init.py`` (elongation kernels)
    https://github.com/ANR-COMPASS/shesha
Pfrommer & Hickson 2014, A&A 565, A102 (sodium centroid statistics)
"""
import math

from prysm.conf import config
from prysm.fttools import fftrange
from prysm.mathops import np

from keck_prysm.atmosphere import ARCSEC

# W. M. Keck Observatory, Maunakea summit, m above sea level
KECK_ALTITUDE = 4145.0


def sodium_profile(centroids=(90e3,), fwhms=(10e3,), weights=(1.0,),
                   altitudes=None, n=201):
    """Sodium density profile as a sum of Gaussians.

    The default is a single Gaussian at 90 km above sea level with 10 km
    FWHM, typical of long-term lidar statistics (mean centroid 90.8 km;
    Pfrommer & Hickson 2014).  No Maunakea-specific peer-reviewed profile was
    found; pass measured arrays directly where available.

    Parameters
    ----------
    centroids : sequence of float
        center altitude of each component, m above sea level
    fwhms : sequence of float
        FWHM of each component, m
    weights : sequence of float
        relative peak density of each component
    altitudes : ndarray, optional
        altitudes to evaluate at, m above sea level; by default n samples
        spanning 2 FWHM beyond the outermost components
    n : int
        number of samples when altitudes is None

    Returns
    -------
    altitudes : ndarray
        m above sea level
    densities : ndarray
        relative density, sums to 1

    """
    if altitudes is None:
        lo = min(c - 2 * f for c, f in zip(centroids, fwhms))
        hi = max(c + 2 * f for c, f in zip(centroids, fwhms))
        altitudes = np.linspace(lo, hi, n, dtype=config.precision)

    densities = np.zeros_like(altitudes)
    for c, f, w in zip(centroids, fwhms, weights):
        sigma = f / (2 * math.sqrt(2 * math.log(2)))
        densities += w * np.exp(-(altitudes - c) ** 2 / (2 * sigma ** 2))
    return altitudes, densities / densities.sum()


def _photon_weights(altitudes, densities, site_altitude):
    """Heights above the site and the fraction of WFS photons from each.

    Return flux per sample is proportional to density / height^2 (geometric
    dilution; Brown et al. 2004 eq. 3).  Two-way transmission is common to
    all samples and drops out after normalization.
    """
    h = altitudes - site_altitude
    w = densities / h ** 2
    return h, w / w.sum()


def sodium_centroid_height(altitudes, densities, site_altitude=KECK_ALTITUDE):
    """Height of the sodium beacon above the telescope at zenith, m.

    Defined by 1/H = sum(w / h) with photon weights w, so that defocus,
    which goes as 1/H, has no residual from the profile.  This is the
    height to give los_coordinates and the true range behind the focus
    term (design 2.4); elongation_kernels is centered on the same value.
    Setup only: syncs device to host.
    """
    h, w = _photon_weights(altitudes, densities, site_altitude)
    return 1 / float((w / h).sum())


def elongation_kernels(sub_x, sub_y, altitudes, densities, n, dtheta,
                       spot_fwhm, launch=(0.0, 0.0), zenith=0.0,
                       site_altitude=KECK_ALTITUDE):
    """LGS spot elongation kernel of each subaperture.

    Each sodium sample at range R appears at angle b (1/R_c - 1/R) from the
    beacon centroid, with b the subaperture position relative to the launch
    point and R_c the centroid range (sodium_centroid_height / cos z).  The
    kernel is the photon-weighted sum of Gaussian uplink spots at those
    angles.  This is COMPASS's kernel (profile convolved with a Gaussian
    along the elongation, Gaussian across, rotated to the subaperture
    azimuth), evaluated directly on the grid as a separable sum so no
    rotation or interpolation is needed.

    Angles are on sky with the orientation of the pupil coordinates.  Each
    kernel sums to 1 and its centroid is at zero, apart from truncation at
    the window edge.

    Parameters
    ----------
    sub_x, sub_y : ndarray
        (n_sub,) subaperture centers in the pupil, mm
    altitudes, densities : ndarray
        sodium profile, from sodium_profile
    n : int
        samples across each kernel (the oversampled detector window)
    dtheta : float
        kernel sample spacing, arcsec
    spot_fwhm : float
        FWHM of the Gaussian uplink spot at the sodium layer, arcsec
    launch : (float, float) or sequence of them
        launch position(s) in the pupil, mm; give one per LGS only when they
        differ, otherwise one kernel set serves all LGS
    zenith : float
        zenith angle, deg
    site_altitude : float
        telescope altitude, m above sea level

    Returns
    -------
    ndarray
        (n_sub, n, n) for a single launch position, else
        (n_launch, n_sub, n, n); axes (..., y, x), center at n // 2

    """
    h, w = _photon_weights(altitudes, densities, site_altitude)
    secz = 1 / math.cos(math.radians(zenith))
    r = h * secz
    inv_rc = (w / r).sum()  # 1 / R_c
    per_baseline = (inv_rc - 1 / r) / ARCSEC  # arcsec per m of baseline

    sub_x = np.asarray(sub_x, dtype=config.precision)
    sub_y = np.asarray(sub_y, dtype=config.precision)
    launch = np.asarray(launch, dtype=config.precision)
    single = launch.ndim == 1
    launch = launch.reshape(-1, 2)
    bx = (sub_x[None, :] - launch[:, :1]) * 1e-3  # (n_launch, n_sub), m
    by = (sub_y[None, :] - launch[:, 1:]) * 1e-3

    theta = fftrange(n, dtype=config.precision) * dtheta
    sigma = spot_fwhm / (2 * math.sqrt(2 * math.log(2)))
    # (n_launch, n_sub, n_sodium, n): Gaussian profile of each sodium sample
    # along each axis; the 2D kernel is sum_j w_j gy_j(theta_y) gx_j(theta_x)
    gx = np.exp(-(theta - (bx[..., None] * per_baseline)[..., None]) ** 2
                / (2 * sigma ** 2))
    gy = np.exp(-(theta - (by[..., None] * per_baseline)[..., None]) ** 2
                / (2 * sigma ** 2)) * w[:, None]
    k = np.swapaxes(gy, -1, -2) @ gx
    k /= k.sum(axis=(-2, -1), keepdims=True)
    return k[0] if single else k
