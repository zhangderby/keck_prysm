import numpy as np

from prysm.coordinates import cart_to_polar
from prysm.geometry import (regular_polygon_sdf,
                            circle,
                            circle_sdf,
                            spider,
                            spider_sdf,
                            antialias,)
from prysm.polynomials.fitting import sum_of_2d_modes, orthogonalize_modes
from prysm.polynomials.zernike import zernike_nm_seq
from prysm.segmented import hex_ring, hex_to_xy

# piston, tip, tilt, defocus as Zernike (n, m)
SEGMENT_NMS = [(0, 0), (1, 1), (1, -1), (2, 0)]


def keck_primary(x, y, segment_size=1800, gap=3, rotation=0):
    """Keck primary mirror pupil mask and per-segment basis, no secondary.

    Parameters
    ----------
    x, y : ndarray
        spatial coordinates, mm, as from make_xy_grid (ascending)
    segment_size : float
        vertex-to-vertex diameter of one hexagonal segment, mm
    gap : float
        edge-to-edge gap between adjacent segments, mm
    rotation : float
        clocking of the whole pupil, degrees (same sense as regular_polygon)

    Returns
    -------
    mask : ndarray
        boolean mask, True on the 36 segments, if the sample spacing resolves
        the gaps; otherwise float coverage in [0, 1] so sub-pixel gaps still
        dim the pixels they cross
    windows : list of tuple of slice
        for each segment, the (y, x) slices of the region around it
    bases : list of ndarray
        for each segment, shape (4, *window shape); piston, tip, tilt, and
        defocus, built from Zernikes normalized to the segment circumradius
        then orthonormalized over the segment (unit RMS, mutually orthogonal),
        zero off the segment.  Tip and tilt are along the segment's own x and
        y, which turn with rotation.  Segments are numbered from the inner
        ring outward, each ring starting at 12 o'clock (before rotation) and
        proceeding clockwise

    """
    seg_radius = segment_size / 2
    pitch = np.sqrt(3) * seg_radius + gap  # segment center-to-center distance
    rot = np.radians(rotation)
    dx = float(x[0, 1] - x[0, 0])
    grayscale = dx > gap
    # window half-width, enough to hold the gap and the antialiased edge
    half = seg_radius + max(gap, dx)
    x1, y1 = x[0, :], y[:, 0]
    mask = np.zeros(x.shape, dtype=float if grayscale else bool)
    # each pixel belongs to the segment it is nearest, for the basis support
    nearest_d = np.full(x.shape, np.inf)
    owner = np.full(x.shape, -1)
    centers = []
    windows = []
    # three rings of flat-topped hexagons around an empty central position;
    # hex_ring starts each ring at 12 o'clock and steps clockwise
    for ring in (1, 2, 3):
        for h in hex_ring(ring):
            cx, cy = hex_to_xy(h, pitch / np.sqrt(3))
            cx, cy = (cx*np.cos(rot) + cy*np.sin(rot),
                      -cx*np.sin(rot) + cy*np.cos(rot))
            win = (slice(np.searchsorted(y1, cy - half),
                         np.searchsorted(y1, cy + half, side='right')),
                   slice(np.searchsorted(x1, cx - half),
                         np.searchsorted(x1, cx + half, side='right')))
            d = regular_polygon_sdf(6, seg_radius, x[win], y[win],
                                    center=(cx, cy), rotation=90+rotation)
            if grayscale:
                # ramp each segment separately and sum, so a pixel straddling
                # a gap gets the coverage of the segments on either side
                mask[win] += antialias(d, dx)
            else:
                mask[win] |= d <= 0
            closer = d < nearest_d[win]
            nearest_d[win][closer] = d[closer]
            owner[win][closer] = len(centers)
            centers.append((cx, cy))
            windows.append(win)

    if grayscale:
        mask = np.minimum(mask, 1)

    bases = []
    for i, ((cx, cy), win) in enumerate(zip(centers, windows)):
        r, t = cart_to_polar(x[win] - cx, y[win] - cy)
        # rotation is clockwise, so the segment's own azimuth is t + rot
        modes = zernike_nm_seq(SEGMENT_NMS, r / seg_radius, t + rot)
        support = (owner[win] == i) & (mask[win] > 0)
        # Zernikes are not orthogonal on a hexagon (defocus carries piston);
        # orthogonalize_modes returns unit-norm modes, rescale to unit RMS
        modes = orthogonalize_modes(modes, support) * np.sqrt(support.sum())
        bases.append(modes)

    return mask, windows, bases


def compose_segments(coefs, windows, bases, dim):
    """Sum per-segment modes into a full-size map, e.g. an OPD.

    Parameters
    ----------
    coefs : ndarray
        shape (36, 4), coefficients for each segment and mode
    windows : list of tuple of slice
        from keck_primary
    bases : list of ndarray
        from keck_primary
    dim : tuple of int
        dimensions of the output, (dim, dim)

    Returns
    -------
    ndarray
        sum of the weighted modes

    """
    out = np.zeros((dim, dim))
    for win, base, c in zip(windows, bases, coefs):
        out[win] += sum_of_2d_modes(base, c)
    return out


def keck_secondary(x, y, obscuration_diameter=2600, spider_width=26,
                   rotation=0):
    """Keck secondary obscuration and spider mask, multiply with keck_primary.

    Parameters
    ----------
    x, y : ndarray
        spatial coordinates, mm
    obscuration_diameter : float
        diameter of the central obscuration, mm
    spider_width : float
        full width of each of the six spider vanes, mm
    rotation : float
        clocking of the whole pupil, degrees; use the same value passed to
        keck_primary so the vanes stay registered to the segments

    Returns
    -------
    ndarray
        boolean mask, False under the obscuration and vanes, if the sample
        spacing resolves the vanes; otherwise float transmission in [0, 1]

    """
    r, _ = cart_to_polar(x, y)
    dx = float(x[0, 1] - x[0, 0])
    # rotation=0 puts the vanes at 0, 60, ... 300 deg, as in hcipy's
    # make_keck_aperture; spider turns opposite to regular_polygon, so negate
    rotation = -rotation
    if dx > spider_width:
        d = spider_sdf(6, spider_width, x, y, rotation=rotation)
        # box-filter coverage of a vane narrower than a pixel: ramp in at the
        # near edge, ramp out at the far edge
        vanes = antialias(d, dx) - antialias(d + spider_width, dx)
        obsc = antialias(circle_sdf(obscuration_diameter / 2, r), dx)
        return 1 - np.maximum(vanes, obsc)

    return ~(circle(obscuration_diameter / 2, r)
             | spider(6, spider_width, x, y, rotation=rotation))
