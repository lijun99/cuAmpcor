"""
Shared fixtures and helpers for the pycuampcor tests
"""

import os

import numpy as np
import pytest

import pycuampcor

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "ampcor"))

# output files (name, number of bands)
OUTPUTS = {"dense_offsets": 2, "gross_offsets": 2, "snr": 1,
           "covariance": 3, "correlation_peak": 1}


def gpu_available():
    """Whether the CUDA implementation is built and a device is available"""
    if not pycuampcor.has_cuda:
        return False
    try:
        pycuampcor.PyCuAmpcor.get_sm_count(0)
        return True
    except RuntimeError:
        return False


IMPLS = ["cpu"] + (["gpu"] if gpu_available() else [])


def new_ampcor(impl):
    """Create an ampcor object for the implementation ('cpu' or 'gpu')"""
    return pycuampcor.PyCuAmpcor() if impl == "gpu" else pycuampcor.PyCPUAmpcor()


def make_images(path, shape, shift, noise=0.3, seed=12345, dtype=np.complex64, bandwidth=0.8):
    """
    Create a reference/secondary image pair with a known (sub-pixel) shift

    The reference is band-limited (by default, 80% of the sampling rate, as SAR
    images are oversampled) complex white noise; the secondary is the reference shifted by
    `shift` (down, across) with the Fourier shift theorem, with decorrelation
    noise of relative amplitude `noise`. Real (amplitude) images are written if
    `dtype` is float32.
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape
    noise_ref = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    band = (np.abs(ky) < bandwidth / 2) & (np.abs(kx) < bandwidth / 2)
    spectrum = np.fft.fft2(noise_ref) * band
    ref = np.fft.ifft2(spectrum)
    ramp = np.exp(-2j * np.pi * (ky * shift[0] + kx * shift[1]))
    sec = np.fft.ifft2(spectrum * ramp)
    sec += noise * np.std(sec) * (rng.standard_normal(shape)
                                  + 1j * rng.standard_normal(shape))
    if dtype == np.float32:
        ref, sec = np.abs(ref), np.abs(sec)
    os.makedirs(path, exist_ok=True)
    ref_file = os.path.join(str(path), "reference.slc")
    sec_file = os.path.join(str(path), "secondary.slc")
    ref.astype(dtype).tofile(ref_file)
    sec.astype(dtype).tofile(sec_file)
    return ref_file, sec_file


def run_ampcor(impl, images, shape, outdir, n_windows=None, start=None,
               gross_offset=(0, 0), **params):
    """
    Run ampcor with the default test parameters, updated with `params`

    Parameters
    ----------
    impl : 'cpu' or 'gpu'
    images : (reference, secondary) file names
    shape : (height, width) of the images
    outdir : directory for the outputs
    n_windows : (down, across) number of windows; default: fill the image
    start : (down, across) starting pixel of the first reference window;
        default: the half search range
    gross_offset : (down, across) constant gross offset, or a pair of arrays
        with varying gross offsets for each window
    params : other ampcor parameters (python names)

    Returns
    -------
    dict of output arrays with shape (down, across, bands)
    """
    ampcor = new_ampcor(impl)
    p = dict(windowSizeHeight=32, windowSizeWidth=64,
             halfSearchRangeDown=8, halfSearchRangeAcross=10,
             skipSampleDown=32, skipSampleAcross=32,
             numberWindowDownInChunk=2, numberWindowAcrossInChunk=4,
             rawDataOversamplingFactor=2, derampMethod=1,
             corrStatWindowSize=21, corrSurfaceZoomInWindow=8,
             corrSurfaceOverSamplingMethod=0, corrSurfaceOverSamplingFactor=64)
    p.update(params)
    for key, value in p.items():
        setattr(ampcor, key, value)

    height, width = shape
    ampcor.referenceImageName, ampcor.secondaryImageName = images
    ampcor.referenceImageHeight = ampcor.secondaryImageHeight = height
    ampcor.referenceImageWidth = ampcor.secondaryImageWidth = width

    hs = (ampcor.halfSearchRangeDown, ampcor.halfSearchRangeAcross)
    if start is None:
        start = hs
    ampcor.referenceStartPixelDownStatic, ampcor.referenceStartPixelAcrossStatic = start
    if n_windows is None:
        n_windows = ((height - 2 * hs[0] - ampcor.windowSizeHeight) // ampcor.skipSampleDown,
                     (width - 2 * hs[1] - ampcor.windowSizeWidth) // ampcor.skipSampleAcross)
    ampcor.numberWindowDown, ampcor.numberWindowAcross = n_windows

    os.makedirs(outdir, exist_ok=True)
    ampcor.offsetImageName = os.path.join(str(outdir), "dense_offsets")
    ampcor.grossOffsetImageName = os.path.join(str(outdir), "gross_offsets")
    ampcor.snrImageName = os.path.join(str(outdir), "snr")
    ampcor.covImageName = os.path.join(str(outdir), "covariance")
    ampcor.peakValueImageName = os.path.join(str(outdir), "correlation_peak")

    ampcor.setupParams()
    if np.isscalar(gross_offset[0]):
        ampcor.setConstantGrossOffset(*gross_offset)
    else:
        ampcor.setVaryingGrossOffset([int(v) for v in gross_offset[0]],
                                     [int(v) for v in gross_offset[1]])
    ampcor.runAmpcor()
    return read_outputs(outdir, n_windows, ampcor.isDoublePrecision())


def read_outputs(outdir, n_windows, double=False):
    """Read the output files as arrays with shape (down, across, bands)"""
    dtype = np.float64 if double else np.float32
    return {name: np.fromfile(os.path.join(str(outdir), name), dtype=dtype)
            .reshape(*n_windows, bands)
            for name, bands in OUTPUTS.items()}


@pytest.fixture(params=IMPLS)
def impl(request):
    """The available implementations: 'cpu', and 'gpu' if a device is found"""
    return request.param


needs_gpu = pytest.mark.skipif(not gpu_available(), reason="no CUDA device")
