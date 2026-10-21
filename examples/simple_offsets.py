#!/usr/bin/env python3
"""
A self-contained example of pycuampcor: estimate the offsets between two synthetic images with a
known shift

The reference image is band-limited complex white noise (like an SLC); the secondary image is
the reference shifted by a known sub-pixel offset, with some decorrelation noise. The offsets of
a grid of windows are estimated on the GPU (or the CPU, with --cpu or without CUDA support),
and compared with the known shift.

usage: simple_offsets.py [--cpu] [--outdir DIR]
needs: numpy
"""

import argparse
import os

import numpy as np
import pycuampcor


def make_images(outdir, shape, shift, noise=0.3, seed=1):
    """
    Write a reference/secondary pair of raw complex64 images of shape (down, across), the
    secondary shifted by shift (down, across) pixels
    @return the file names
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    # band-limited (80% of the sampling rate) complex white noise, and its shifted copy
    spectrum = np.fft.fft2(rng.standard_normal(shape) + 1j*rng.standard_normal(shape))
    spectrum *= (np.abs(ky) < 0.4) & (np.abs(kx) < 0.4)
    reference = np.fft.ifft2(spectrum)
    secondary = np.fft.ifft2(spectrum*np.exp(-2j*np.pi*(ky*shift[0] + kx*shift[1])))
    secondary += noise*np.std(secondary)*(rng.standard_normal(shape) + 1j*rng.standard_normal(shape))
    os.makedirs(outdir, exist_ok=True)
    files = os.path.join(outdir, 'reference.slc'), os.path.join(outdir, 'secondary.slc')
    for image, fname in zip((reference, secondary), files):
        image.astype(np.complex64).tofile(fname)
    return files


def main():
    """Make the images, estimate the offsets, and compare them with the known shift"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--cpu', action='store_true', help='use the CPU implementation')
    parser.add_argument('--outdir', default='simple_offsets', help='the directory of the images and outputs')
    args = parser.parse_args()

    shape, shift = (1024, 1024), (2.3, -1.6)
    reference, secondary = make_images(args.outdir, shape, shift)

    # the windows: 64 x 64, every 64 pixels, searched +/- 16 pixels, starting after the search range
    window, skip, search = 64, 64, 16
    n_down = (shape[0] - 2*search - window)//skip
    n_across = (shape[1] - 2*search - window)//skip
    out = os.path.join(args.outdir, 'offset')
    params = dict(
        referenceImageName=reference, referenceImageHeight=shape[0], referenceImageWidth=shape[1],
        secondaryImageName=secondary, secondaryImageHeight=shape[0], secondaryImageWidth=shape[1],
        windowSizeHeight=window, windowSizeWidth=window,
        halfSearchRangeDown=search, halfSearchRangeAcross=search,
        skipSampleDown=skip, skipSampleAcross=skip,
        referenceStartPixelDownStatic=search, referenceStartPixelAcrossStatic=search,
        numberWindowDown=n_down, numberWindowAcross=n_across,
        derampMethod='complex', corrSurfaceOverSamplingFactor=64,
        offsetImageName=out + '.bip', grossOffsetImageName=out + '_gross.bip',
        snrImageName=out + '_snr.bip', covImageName=out + '_cov.bip',
        peakValueImageName=out + '_peak.bip',
        # no gross offset: the secondary windows are searched around the reference windows
        grossOffsetDown=0, grossOffsetAcross=0)
    # the keyword constructor sets the parameters and sets the object up to run
    if args.cpu or not pycuampcor.has_cuda:
        ampcor = pycuampcor.PyCPUAmpcor(**params)
    else:
        ampcor = pycuampcor.PyCuAmpcor(**params)
    ampcor.runAmpcor()

    # the offsets (down, across) of the windows, and the SNR
    offsets = np.fromfile(out + '.bip', np.float32).reshape(n_down, n_across, 2)
    snr = np.fromfile(out + '_snr.bip', np.float32).reshape(n_down, n_across)
    print(f'{type(ampcor).__name__}: {n_down} x {n_across} windows')
    print(f'known shift (down, across): {shift}')
    print(f'median offset (down, across): {tuple(np.round(np.median(offsets, axis=(0, 1)), 3))}')
    print(f'max |error|: {np.abs(offsets - np.array(shift)).max():.3f} pixels, median SNR {np.median(snr):.1f}')


if __name__ == '__main__':
    main()
