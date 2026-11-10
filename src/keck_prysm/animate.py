"""Animate the Keck OPD and PSF while the pupil rotates and segments drift.

Usage: python -m keck_prysm.animate [--out keck.gif] [--frames 360] [--fps 24]
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

from prysm.coordinates import make_xy_grid
from prysm.propagation.dft import (prepare_executor,
                                   focus_dft,)
from prysm.propagation.wavefront import Wavefront

from keck_prysm.utils import (keck_primary,
                              compose_segments,
                              keck_secondary,)

n_pupil = 512 # pix
d_pupil = 11e3 # mm
dx_pupil = d_pupil / n_pupil # mm/pix
fno = 15 # unitless
n_focal = 256 # pix
wvl = 1 # um
n_psf = 4 # pix/lamD
dx_focal = fno * wvl / n_psf # um/pix
fov = n_focal / n_psf # lamD

n_segments = 36
mode_names = ['piston', 'tip', 'tilt', 'defocus']
mode_rms = np.array([50, 30, 30, 20]) * 2 # nm, RMS of each mode's drift over time


def segment_drift(n_frames, rng):
    """Per-segment coefficients over time, shape (n_frames, 36, 4), nm.

    Each coefficient is a sum of two sinusoids with a whole number of cycles
    per animation, so the drift is smooth and the animation loops cleanly.
    """
    phase = 2 * np.pi * np.arange(n_frames) / n_frames
    coefs = np.zeros((n_frames, n_segments, len(mode_names)))
    for cycles in (1, 2):
        amp = rng.normal(0, 1 / cycles, (n_segments, len(mode_names)))
        offset = rng.uniform(0, 2 * np.pi, (n_segments, len(mode_names)))
        coefs += amp * np.sin(cycles * phase[:, None, None] + offset)
    # expected RMS of the sum above is sqrt((1 + 1/4) / 2)
    return coefs * mode_rms / np.sqrt(0.625)


def frame(rotation, coefs, x, y, ceo):
    """Pupil, OPD, and PSF (normalized to the unaberrated peak) for one frame."""
    pupil, segment_windows, segment_bases = keck_primary(x, y, rotation=rotation)
    pupil = pupil * keck_secondary(x, y, rotation=rotation)
    opd = compose_segments(coefs=coefs, windows=segment_windows,
                           bases=segment_bases, dim=n_pupil)

    wf = Wavefront.from_amp_and_phase(amplitude=pupil, phase=opd, wavelength=wvl, dx=dx_pupil)
    img = np.abs(focus_dft(wavefunction=wf.data, executor=ceo)) ** 2
    ref = np.abs(focus_dft(wavefunction=pupil.astype(complex), executor=ceo)) ** 2
    return pupil, opd, img / ref.max()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='keck_opd_psf.gif', help='.gif, or .mp4 if ffmpeg is installed')
    parser.add_argument('--frames', type=int, default=360, help='one full turn of the pupil')
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    x, y = make_xy_grid(shape=n_pupil, diameter=d_pupil)
    ceo = prepare_executor(pupil_dx=dx_pupil, pupil_samples=n_pupil, focal_dx=dx_focal,
                           focal_samples=n_focal, wavelength=wvl, efl=fno * d_pupil,
                           focal_shift=(0, 0), kind='mdft')

    rotations = 360 * np.arange(args.frames) / args.frames # deg
    coefs = segment_drift(args.frames, np.random.default_rng(args.seed))
    lim = 3 * np.sqrt(np.sum(mode_rms ** 2)) # nm, fixed OPD color scale

    fig, (ax_opd, ax_psf) = plt.subplots(1, 2, figsize=(12, 5.2))
    pupil, opd, img = frame(rotations[0], coefs[0], x, y, ceo)

    im_opd = ax_opd.imshow(np.where(pupil > 0, opd, np.nan),
                           extent=[x.min(), x.max(), y.min(), y.max()], origin='lower',
                           cmap='coolwarm', vmin=-lim, vmax=lim)
    fig.colorbar(im_opd, ax=ax_opd, pad=0.04, fraction=0.046, label='OPD (nm)')
    ax_opd.set_xlabel('mm'); ax_opd.set_ylabel('mm')

    im_psf = ax_psf.imshow(img, extent=[-fov / 2, fov / 2, -fov / 2, fov / 2],
                           origin='lower', norm='log', cmap='magma', vmin=1e-7, vmax=1)
    fig.colorbar(im_psf, ax=ax_psf, pad=0.04, fraction=0.046, label='Intensity / unaberrated peak')
    ax_psf.set_xlabel(r'$\lambda/D$'); ax_psf.set_ylabel(r'$\lambda/D$')
    fig.tight_layout()

    def update(i):
        pupil, opd, img = frame(rotations[i], coefs[i], x, y, ceo)
        rms = np.sqrt(np.mean(opd[pupil > 0] ** 2))
        im_opd.set_data(np.where(pupil > 0, opd, np.nan))
        im_psf.set_data(img)
        ax_opd.set_title(f'OPD, rotation {rotations[i]:5.1f} deg, {rms:5.1f} nm RMS')
        ax_psf.set_title(rf'PSF, $\lambda$ = {wvl} $\mu m$, Strehl {img.max():.3f}')
        print(f'\rframe {i + 1}/{args.frames}', end='', flush=True)
        return im_opd, im_psf

    anim = FuncAnimation(fig, update, frames=args.frames)
    writer = FFMpegWriter(fps=args.fps) if args.out.endswith('.mp4') else PillowWriter(fps=args.fps)
    anim.save(args.out, writer=writer, dpi=80)
    print(f'\nsaved {args.out}')


if __name__ == '__main__':
    main()
