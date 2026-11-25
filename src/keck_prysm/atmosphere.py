"""Turbulent layers and line-of-sight sampling.

AI use disclaimer: written with Claude Opus 5.5 (claude-opus-5-5, Anthropic)
at the request of the project owner; design in docs/design/design_notes.md,
section 4.  The sodium beacon is in lgs.py.

Units follow prysm: pupil and layer coordinates in mm, OPD in nm, wavelength
in um.  Altitudes and heights are in m, wind speed in m/s, angles on sky in
arcsec, r0 and L0 in m.  Layer heights and source heights are measured from
the telescope at zenith; the zenith angle is applied here.

Every source samples the same set of frozen-flow layers (design 4.1).  Layers
stay aligned with the pupil grid axes, so the line-of-sight map

    x' = x (1 - h/H) + h theta - v t

is separable and is stored as one x and one y vector per (source, layer).
Both screen providers are addressed modulo the screen size: an FFT screen is
periodic, and an infinite screen is a ring buffer whose window follows the
wind.

References
----------
Assemat, Wilson & Gendron 2006, Opt. Express 14, 988 (infinite screens);
    extrusion translated from aotools ``infinitephasescreen.PhaseScreenVonKarman``
    https://github.com/AOtools/aotools, extended to extrude along x and y as in
    COMPASS ``sutra_tscreen`` https://github.com/ANR-COMPASS/shesha
Johansson & Gavel 1994, Proc. SPIE 2200, 372; Lane, Glindemann & Dainty 1992,
    Waves Random Media 2, 209 (FFT screens and subharmonics); translated from
    Schmidt 2010, *Numerical Simulation of Optical Wave Propagation*, SPIE
    Press, listings 9.2-9.3, via aotools ``phasescreen.ft_sh_phase_screen``
Sasiela 1994, *Electromagnetic Wave Propagation in Turbulence*, Springer
    (von Karman phase PSD and covariance)
"""
import math
from dataclasses import dataclass

from scipy import special

from prysm.conf import config
from prysm.fttools import fftfreq, fftrange
from prysm.mathops import np, fft

ARCSEC = math.pi / 180 / 3600  # rad per arcsec

# von Karman phase PSD, Phi(f) = _VK_PSD r0^(-5/3) (f^2 + 1/L0^2)^(-11/6), and
# phase variance, _VK_VAR (L0/r0)^(5/3); f in cycles/m, phase in rad
_VK_K = (24 / 5 * math.gamma(6 / 5)) ** (5 / 6)
_VK_PSD = _VK_K * math.gamma(11 / 6) ** 2 / (2 * math.pi ** (11 / 3))  # 0.0229
_VK_VAR = (_VK_K * math.gamma(11 / 6) * math.gamma(5 / 6)
           / (2 * math.pi ** (8 / 3)))  # 0.0863


@dataclass(frozen=True, slots=True)
class Extrusion:
    """Assemat & Wilson extrusion operators shared by all infinite layers.

    A block X of new lines next to a stencil Z of the nearest n_stencil lines
    is X = A Z + B b, with b white noise.  A does not depend on r0; B is for
    r0 = 1 m and is scaled per layer.  By isotropy the same A and B serve
    lines added on any of the four sides of the screen.
    """
    A: object        # (block * n, n_stencil * n)
    B: object        # (block * n, block * n)
    scale: object    # (n_layer,), r0_layer^(-5/6), multiplies the noise
    n_stencil: int
    block: int       # lines added per extrusion
    along: object    # arange(n), positions along a line


@dataclass(frozen=True, slots=True)
class Layers:
    """A set of frozen-flow layers sampled by every source."""
    screens: object       # (n_layer, n, n) OPD, nm; ring buffer if infinite
    dx: float             # screen sample spacing, mm
    heights: tuple        # m above the telescope, at zenith
    wind: tuple           # ((vx, vy), ...) m/s, host side
    wind_px: object       # (n_layer, 2) float64, screen px/s
    zenith: float         # deg
    periodic: bool        # sampling may wrap around the screen edges
    extrusion: Extrusion | None  # None for FFT screens


@dataclass(frozen=True, slots=True)
class LineOfSight:
    """Static line-of-sight coordinates, screen px, before the wind shift."""
    x: object  # (n_src, n_layer, nx)
    y: object  # (n_src, n_layer, ny)


@dataclass(frozen=True, slots=True)
class SampleBuffers:
    """Preallocated work buffers for sample_atmosphere."""
    offsets: object  # (n_layer, 2) float64, wind shift, screen px
    floor_x: object  # (n_src, n_layer, nx)
    floor_y: object  # (n_src, n_layer, ny)
    ix: object       # (2, n_src, n_layer, nx), lower and upper neighbour
    iy: object       # (2, n_src, n_layer, ny)
    wx: object       # (2, n_src, n_layer, nx), their weights
    wy: object       # (2, n_src, n_layer, ny)
    base: object     # (n_layer, 1), first flat index of each layer
    idx: object      # (2, 2, n_src, n_layer, ny, nx), flat corner indices
    vals: object     # (2, 2, n_src, n_layer, ny, nx), weighted corners
    opd: object      # (n_src, ny, nx), nm


# ---------------------------------------------------------------- screens

def _layer_r0(r0, fractions, zenith):
    """Per-layer r0 along the line of sight from the total r0 at zenith."""
    total = sum(fractions)
    airmass_scale = math.cos(math.radians(zenith)) ** (3 / 5)
    return tuple(r0 * airmass_scale * (f / total) ** (-3 / 5) for f in fractions)


def _make_layers(screens, dx, heights, wind_speed, wind_direction, zenith,
                 periodic, extrusion):
    n_layer = screens.shape[0]
    if not len(heights) == len(wind_speed) == len(wind_direction) == n_layer:
        raise ValueError('fractions, heights, wind_speed and wind_direction '
                         'must all have one entry per layer')

    wind = tuple((s * math.cos(math.radians(d)), s * math.sin(math.radians(d)))
                 for s, d in zip(wind_speed, wind_direction))
    wind_px = np.asarray([(vx * 1e3 / dx, vy * 1e3 / dx) for vx, vy in wind],
                         dtype=np.float64)
    return Layers(screens=screens, dx=dx, heights=tuple(heights), wind=wind,
                  wind_px=wind_px, zenith=zenith, periodic=periodic,
                  extrusion=extrusion)


def von_karman_screen(n, dx, r0, L0, rng, r0_wvl=0.5, subharmonics=False):
    """One von Karman OPD screen by the FFT method.

    Fourier coefficients are complex Gaussian with variance set by the von
    Karman PSD (Johansson & Gavel 1994; Schmidt 2010 listing 9.2).  Without
    subharmonics the screen is periodic.  Subharmonics restore the low
    frequencies below 1 / (n dx) (Lane et al. 1992; Schmidt 2010 listing 9.3),
    but the screen is then no longer periodic and must not be wrapped.

    Parameters
    ----------
    n : int
        samples across the square screen
    dx : float
        sample spacing, mm
    r0 : float
        Fried parameter of this layer at r0_wvl, m
    L0 : float
        outer scale, m
    rng : Generator
        from np.random.default_rng, with np from prysm.mathops
    r0_wvl : float
        wavelength at which r0 is given, um
    subharmonics : bool
        if True, add three levels of 3x3 subharmonics

    Returns
    -------
    ndarray
        shape (n, n), OPD, nm

    """
    dx_m = dx * 1e-3
    df = 1 / (n * dx_m)
    nm_per_rad = r0_wvl * 1e3 / (2 * math.pi)
    psd_scale = _VK_PSD * r0 ** (-5 / 3)

    f = fftfreq(n, dx_m)
    f2 = f[None, :] ** 2 + f[:, None] ** 2
    amp = np.sqrt(psd_scale * (f2 + L0 ** -2) ** (-11 / 6)) * (df * nm_per_rad)
    amp[0, 0] = 0  # piston
    w = rng.standard_normal((2, n, n), dtype=config.precision)
    # sum of cn exp(-2 pi i f.x); the sign of the exponent is immaterial for
    # random coefficients, and fft2 needs no 1/n^2 correction
    screen = fft.fft2((w[0] + 1j * w[1]) * amp).real

    if subharmonics:
        x = fftrange(n, dtype=config.precision) * dx_m
        low = np.zeros((n, n), dtype=config.precision)
        for p in (1, 2, 3):
            dfp = df / 3 ** p
            fp = np.asarray([-dfp, 0, dfp], dtype=config.precision)
            fp2 = fp[None, :] ** 2 + fp[:, None] ** 2
            amp = np.sqrt(psd_scale * (fp2 + L0 ** -2) ** (-11 / 6)) * (dfp * nm_per_rad)
            amp[1, 1] = 0
            w = rng.standard_normal((2, 3, 3), dtype=config.precision)
            # separable: sum over (a, b) of cn[a, b] exp(2 pi i (fp_a y + fp_b x))
            e = np.exp(2j * np.pi * fp[:, None] * x[None, :])
            low += (e.T @ ((w[0] + 1j * w[1]) * amp) @ e).real
        screen += low - low.mean()

    return screen.astype(config.precision, copy=False)


def fft_layers(n, dx, r0, L0, fractions, heights, wind_speed, wind_direction,
               zenith=0.0, r0_wvl=0.5, subharmonics=False, rng=None):
    """Fixed-size frozen-flow layers from FFT screens.

    Parameters
    ----------
    n : int
        samples across each square screen
    dx : float
        screen sample spacing, mm
    r0 : float
        total Fried parameter at zenith and r0_wvl, m
    L0 : float
        outer scale, m, shared by all layers
    fractions : sequence of float
        fraction of the Cn^2 integral in each layer; normalized to sum to 1
    heights : sequence of float
        layer heights above the telescope at zenith, m
    wind_speed : sequence of float
        m/s
    wind_direction : sequence of float
        direction each layer moves toward, deg counterclockwise from +x
    zenith : float
        zenith angle, deg
    r0_wvl : float
        wavelength at which r0 is given, um
    subharmonics : bool
        add subharmonics; the screens are then not periodic, so the
        footprint must not travel off the screen (see check_coverage)
    rng : Generator, optional
        from np.random.default_rng; a fresh unseeded one if None

    Returns
    -------
    Layers

    """
    rng = np.random.default_rng() if rng is None else rng
    r0s = _layer_r0(r0, fractions, zenith)
    screens = np.empty((len(r0s), n, n), dtype=config.precision)
    for i, r0_layer in enumerate(r0s):
        screens[i] = von_karman_screen(n, dx, r0_layer, L0, rng, r0_wvl,
                                       subharmonics)

    return _make_layers(screens, dx, heights, wind_speed, wind_direction,
                        zenith, periodic=not subharmonics, extrusion=None)


def von_karman_covariance(r, r0, L0, r0_wvl=0.5):
    """OPD covariance of von Karman turbulence at separations r.

    C(r) = C(0) 2^(1/6) / Gamma(5/6) x^(5/6) K_5/6(x), x = 2 pi r / L0, with
    C(0) = 0.0863 (L0/r0)^(5/3) rad^2 (Sasiela 1994; as in aotools
    ``phasecovariance``).  Evaluated on the host; used only at setup.

    Parameters
    ----------
    r : sequence of float
        separations, m
    r0 : float
        Fried parameter at r0_wvl, m
    L0 : float
        outer scale, m
    r0_wvl : float
        wavelength at which r0 is given, um

    Returns
    -------
    list of float
        covariance, nm^2

    """
    c0 = _VK_VAR * (L0 / r0) ** (5 / 3) * (r0_wvl * 1e3 / (2 * math.pi)) ** 2
    shape = 2 ** (1 / 6) / math.gamma(5 / 6)

    def cov(ri):
        if ri == 0:
            return c0
        x = 2 * math.pi * ri / L0
        return c0 * shape * x ** (5 / 6) * float(special.kv(5 / 6, x))

    return [cov(ri) for ri in r]


def assemat_operators(n, dx, L0, n_stencil=2, block=1, r0_wvl=0.5):
    """Extrusion operators A and B for r0 = 1 m (Assemat et al. 2006).

    The stencil is the n_stencil full lines nearest the new ones, as in
    aotools ``PhaseScreenVonKarman``.  With block > 1 the block of new lines
    is drawn jointly from its Gaussian distribution conditioned on the
    stencil.  Built in float64, returned in config.precision.

    Parameters
    ----------
    n : int
        samples along a line (the screen size)
    dx : float
        sample spacing, mm
    L0 : float
        outer scale, m
    n_stencil : int
        number of existing lines the new lines are conditioned on
    block : int
        number of new lines per draw; see infinite_layers
    r0_wvl : float
        wavelength at which r0 is given, um

    Returns
    -------
    A : ndarray
        (block * n, n_stencil * n)
    B : ndarray
        (block * n, block * n)

    """
    dx_m = dx * 1e-3
    n_dist = block + n_stencil
    # covariance vs. (distance between lines, separation along the line)
    table = von_karman_covariance(
        [dx_m * math.hypot(d, j) for d in range(n_dist) for j in range(n)],
        1.0, L0, r0_wvl)
    table = np.asarray(table, dtype=np.float64).reshape(n_dist, n)
    along = np.arange(n)
    sep = np.abs(along[:, None] - along[None, :])

    def cov(rows, cols):
        """Covariance between lines at normal positions rows and cols."""
        out = np.empty((len(rows) * n, len(cols) * n), dtype=np.float64)
        for a, p in enumerate(rows):
            for b, q in enumerate(cols):
                out[a * n:(a + 1) * n, b * n:(b + 1) * n] = table[abs(p - q)][sep]
        return out

    # normal positions: new line a at -a (a = 0 touches the stencil), stencil
    # line i at i + 1
    new = [-a for a in range(block)]
    stencil = [i + 1 for i in range(n_stencil)]
    czz, cxz, cxx = cov(stencil, stencil), cov(new, stencil), cov(new, new)

    A = np.linalg.solve(czz, cxz.T).T  # Cxz Czz^-1, Czz symmetric
    bbt = cxx - A @ cxz.T
    w, v = np.linalg.eigh(bbt)
    B = v * np.sqrt(np.clip(w, 0, None))
    return A.astype(config.precision), B.astype(config.precision)


def infinite_layers(n, dx, r0, L0, fractions, heights, wind_speed,
                    wind_direction, zenith=0.0, r0_wvl=0.5, n_stencil=2,
                    block=1, rng=None):
    """Frozen-flow layers that extrude new turbulence as they move.

    Each layer is an n x n ring buffer, initialized with an FFT screen with
    subharmonics.  extrude_layers adds rows or columns on the upwind sides
    (Assemat et al. 2006), so memory stays constant for any run length.  Start
    the extrusion state from ``window = ((0, 0),) * n_layer``.

    Parameters are as for fft_layers, plus

    n_stencil : int
        number of existing lines each new block is conditioned on
    block : int
        number of lines added per extrusion, drawn jointly in one
        conditional Gaussian draw (one gather, two matrix multiplies and one
        scatter per block, batched over layers).  Statistics are the same
        kind of approximation for any value: each block is conditioned on
        the n_stencil lines next to it.

        The trade-off is per-line work against the number of operations.
        B is (block n)^2, so the work and memory traffic per line grow
        roughly in proportion to block, while the number of extrusion
        rounds (each about 20 small array operations plus Python) falls as
        1 / block.

        - CPU: the matrix multiplies dominate and are limited by memory
          bandwidth, so keep block = 1 (the default).  Measured on an Apple
          M4, n = 1024, 7 layers: 0.38 ms per line at block = 1, 0.74 at 4,
          3.0 at 16.  At block = 1 the operators are small (12 MB in float32
          at n = 1024), so they fit in the Ryzen 9950X3D's L3 cache.
        - GPU: reading the operators takes microseconds, and the cost is
          kernel launches and Python overhead per round, so a larger block
          (e.g. 4-16) cuts that cost by about block.  Choose it from an RTX
          5090 benchmark.  Limits: B takes (block n)^2 4 bytes (1 GB at
          block = 16, n = 1024), and the setup eigendecomposition scales as
          (block n)^3.

        The window can be up to block - 1 lines away from the wind shift, so
        the footprint needs block - 1 extra lines of margin on each side of
        the screen (enforced by check_coverage).

    Returns
    -------
    Layers

    """
    rng = np.random.default_rng() if rng is None else rng
    r0s = _layer_r0(r0, fractions, zenith)
    A, B = assemat_operators(n, dx, L0, n_stencil, block, r0_wvl)
    screens = np.empty((len(r0s), n, n), dtype=config.precision)
    for i, r0_layer in enumerate(r0s):
        screens[i] = von_karman_screen(n, dx, r0_layer, L0, rng, r0_wvl,
                                       subharmonics=True)

    extrusion = Extrusion(
        A=A, B=B,
        scale=np.asarray([r ** (-5 / 6) for r in r0s], dtype=config.precision),
        n_stencil=n_stencil, block=block, along=np.arange(n))
    return _make_layers(screens, dx, heights, wind_speed, wind_direction,
                        zenith, periodic=True, extrusion=extrusion)


def extrude_layers(layers, window, t, rng):
    """Extrude infinite layers so their windows cover the footprint at time t.

    Screen pixel u (logical, unbounded) is stored at ring index u mod n.  The
    window of layer l is [w, w + n) on each axis.  It follows
    floor(-v t / dx) in steps of layers.extrusion.block lines, staying within
    block - 1 lines of it.  The footprint (logical
    coordinates in [block - 1, n - 1 - block], see check_coverage) plus the
    wind shift therefore always lies inside it.  Blocks are added in rounds,
    each round batched over every layer that still needs one; x lines first,
    then y lines.  Writes layers.screens in place.

    The schedule depends only on host-side t, so there is no device sync.
    This runs outside a captured CUDA graph (design 1.5).

    Parameters
    ----------
    layers : Layers
        from infinite_layers
    window : tuple of (int, int)
        per layer (wx, wy), as returned by the previous call
    t : float
        simulation time, s, host side (e.g. k * dt)
    rng : Generator
        from np.random.default_rng

    Returns
    -------
    tuple of (int, int)
        the new window

    """
    ext = layers.extrusion
    n = layers.screens.shape[-1]
    m, k = ext.block, ext.n_stencil
    flat = layers.screens.reshape(-1)
    # line offsets from the first new line: outward for the new block,
    # inward for the stencil
    depth = np.concatenate([np.arange(m), -np.arange(1, k + 1)])
    window = [list(w) for w in window]

    for axis in (0, 1):
        targets = [math.floor(-v[axis] * t * 1e3 / layers.dx) for v in layers.wind]
        while True:
            # (layer, first new line, side: -1 below the window, +1 above it)
            # for every layer whose window is more than m - 1 lines from its
            # target; the tolerance stops a block that overshoots the target
            # from triggering one on the other side
            batch = []
            for i, (w, target) in enumerate(zip(window, targets)):
                if target < w[axis] - (m - 1):
                    batch.append((i, w[axis] - 1, -1))
                    w[axis] -= m
                elif target > w[axis] + (m - 1):
                    batch.append((i, w[axis] + n, 1))
                    w[axis] += m
            if not batch:
                break

            ids = np.asarray([b[0] for b in batch])
            edge = np.asarray([b[1] for b in batch])
            side = np.asarray([b[2] for b in batch])
            other = np.asarray([window[b[0]][1 - axis] for b in batch])

            # flat indices of the new block, then its stencil, (P, m + k, n)
            line = (edge[:, None] + side[:, None] * depth[None, :]) % n
            across = (other[:, None] + ext.along[None, :]) % n
            if axis == 0:  # columns
                idx = across[:, None, :] * n + line[:, :, None]
            else:          # rows
                idx = line[:, :, None] * n + across[:, None, :]
            idx += ids[:, None, None] * (n * n)

            z = flat[idx[:, m:]].reshape(len(batch), k * n)
            b = rng.standard_normal((len(batch), m * n), dtype=flat.dtype)
            b *= ext.scale[ids][:, None]
            x = z @ ext.A.T + b @ ext.B.T
            flat[idx[:, :m]] = x.reshape(len(batch), m, n)

    return tuple(tuple(w) for w in window)


# ---------------------------------------------------------- line of sight

def los_coordinates(x, y, layers, directions, heights):
    """Geometric line-of-sight coordinates of every source in every layer.

    x' = x (1 - h/H) + h theta (design 4.2), with h and H scaled by 1/cos z.
    The pupil center maps to screen pixel n // 2.  The wind shift is added by
    sample_atmosphere.

    Parameters
    ----------
    x, y : ndarray
        pupil coordinates, mm, from make_xy_grid (2D or 1D)
    layers : Layers
        the layers to sample
    directions : sequence of (float, float)
        (theta_x, theta_y) of each source, arcsec
    heights : sequence of float
        height of each source above the telescope at zenith, m; math.inf for
        stars.  For an LGS use lgs.sodium_centroid_height

    Returns
    -------
    LineOfSight

    """
    x = x[0] if x.ndim == 2 else x
    y = y[:, 0] if y.ndim == 2 else y
    n = layers.screens.shape[-1]
    secz = 1 / math.cos(math.radians(layers.zenith))

    scale, shift_x, shift_y = [], [], []
    for (tx, ty), H in zip(directions, heights):
        if any(h >= H for h in layers.heights):
            raise ValueError(f'a layer is at or above a source at {H} m')
        # cone effect: secz cancels in h/H
        scale.append([1 - h / H for h in layers.heights])
        shift_x.append([h * secz * 1e3 * tx * ARCSEC for h in layers.heights])
        shift_y.append([h * secz * 1e3 * ty * ARCSEC for h in layers.heights])

    scale, shift_x, shift_y = (np.asarray(a, dtype=config.precision)[..., None]
                               for a in (scale, shift_x, shift_y))
    c = n // 2
    return LineOfSight(x=(x * scale + shift_x) / layers.dx + c,
                       y=(y * scale + shift_y) / layers.dx + c)


def check_coverage(layers, los, duration=0.0):
    """Raise if the footprint does not fit the screens.

    Periodic FFT screens need the footprint to be narrower than the screen.
    FFT screens with subharmonics need it to stay on the screen for the whole
    run.  Infinite screens need it inside [block - 1, n - 1 - block], i.e.
    [0, n - 2] for block = 1, because the window can be up to block - 1
    lines away from the wind shift.  Setup only: syncs device to host.

    Parameters
    ----------
    layers : Layers
        the layers to sample
    los : LineOfSight
        from los_coordinates
    duration : float
        run length, s; used for FFT screens with subharmonics

    """
    n = layers.screens.shape[-1]
    margin = 0 if layers.extrusion is None else layers.extrusion.block - 1
    for axis, c in enumerate((los.x, los.y)):
        for i, v in enumerate(layers.wind):
            lo, hi = float(c[:, i].min()), float(c[:, i].max())
            if layers.extrusion is None and layers.periodic:
                lo, hi = 0, hi - lo  # only the width matters
            elif layers.extrusion is None:
                travel = -v[axis] * duration * 1e3 / layers.dx
                lo, hi = min(lo, lo + travel), max(hi, hi + travel)
            if lo < margin or hi > n - 2 - margin:
                raise ValueError(
                    f'layer {i} axis {"xy"[axis]}: footprint needs screen px '
                    f'[{lo:.1f}, {hi:.1f}], usable range is '
                    f'[{margin}, {n - 2 - margin}]')


def layer_offsets(t, layers, out=None):
    """Wind shift of each layer at time t, screen px, modulo the screen size.

    Parameters
    ----------
    t : float or ndarray
        simulation time, s; a device scalar keeps the hot loop sync-free
    layers : Layers
        the layers
    out : ndarray, optional
        (n_layer, 2) float64 buffer

    Returns
    -------
    ndarray
        (n_layer, 2), (x, y) shift

    """
    n = layers.screens.shape[-1]
    out = np.multiply(layers.wind_px, -t, out=out)
    return np.remainder(out, n, out=out)


def make_sample_buffers(layers, los):
    """Preallocate the work buffers for sample_atmosphere."""
    n_layer, n, _ = layers.screens.shape
    n_src, _, nx = los.x.shape
    ny = los.y.shape[-1]
    real = config.precision
    index = np.int32 if n_layer * n * n < 2 ** 31 else np.int64
    corners = (2, 2, n_src, n_layer, ny, nx)
    return SampleBuffers(
        offsets=np.empty((n_layer, 2), dtype=np.float64),
        floor_x=np.empty((n_src, n_layer, nx), dtype=real),
        floor_y=np.empty((n_src, n_layer, ny), dtype=real),
        ix=np.empty((2, n_src, n_layer, nx), dtype=index),
        iy=np.empty((2, n_src, n_layer, ny), dtype=index),
        wx=np.empty((2, n_src, n_layer, nx), dtype=real),
        wy=np.empty((2, n_src, n_layer, ny), dtype=real),
        base=(np.arange(n_layer, dtype=index) * (n * n))[:, None],
        idx=np.empty(corners, dtype=index),
        vals=np.empty(corners, dtype=real),
        opd=np.empty((n_src, ny, nx), dtype=real))


def _neighbours(c, offset, n, floor, i, w):
    """Lower/upper screen indices and bilinear weights along one axis."""
    np.add(c, offset[:, None], out=w[1])  # sample position, screen px
    np.floor(w[1], out=floor)
    np.subtract(w[1], floor, out=w[1])    # weight of the upper neighbour
    np.subtract(1, w[1], out=w[0])
    np.remainder(floor, n, out=floor)
    np.copyto(i[0], floor, casting='unsafe')
    np.add(i[0], 1, out=i[1])
    np.remainder(i[1], n, out=i[1])


def sample_atmosphere(layers, los, t, buf):
    """OPD of every source, summed over layers along each line of sight.

    Separable bilinear interpolation: the four corner indices and weights are
    outer sums and products of per-axis vectors, gathered in one take over
    the flattened layer stack.  Writes only into buf (no allocation of
    per-step arrays) and does not sync device to host.

    Parameters
    ----------
    layers : Layers
        the layers
    los : LineOfSight
        from los_coordinates
    t : float or ndarray
        simulation time, s
    buf : SampleBuffers
        from make_sample_buffers

    Returns
    -------
    ndarray
        buf.opd, (n_src, ny, nx), nm

    """
    n = layers.screens.shape[-1]
    offset = layer_offsets(t, layers, out=buf.offsets)
    _neighbours(los.x, offset[:, 0], n, buf.floor_x, buf.ix, buf.wx)
    _neighbours(los.y, offset[:, 1], n, buf.floor_y, buf.iy, buf.wy)

    # row starts in the flattened (n_layer, n, n) stack: layer n^2 + y n
    np.multiply(buf.iy, n, out=buf.iy)
    np.add(buf.iy, buf.base, out=buf.iy)

    # (y corner, x corner, source, layer, y, x)
    np.add(buf.iy[:, None, ..., None], buf.ix[None, :, ..., None, :], out=buf.idx)
    np.take(layers.screens.reshape(-1), buf.idx, out=buf.vals)
    np.multiply(buf.vals, buf.wy[:, None, ..., None], out=buf.vals)
    np.multiply(buf.vals, buf.wx[None, :, ..., None, :], out=buf.vals)
    return np.sum(buf.vals, axis=(0, 1, 3), out=buf.opd)
