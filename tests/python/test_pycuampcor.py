"""
Tests for pycuampcor with synthetic images

A complex white-noise reference image is shifted by a known sub-pixel offset
(with the Fourier shift theorem) to create the secondary image, and the
offsets estimated by the CPU/GPU implementations are compared to the truth.
"""

import os

import numpy as np
import pytest

import pycuampcor

# image size and the true offsets (down/azimuth, across/range) in pixels
N = 512
SHIFT = (1.3, -2.6)


def gpu_available():
    if not pycuampcor.has_cuda:
        return False
    try:
        pycuampcor.PyCuAmpcor.get_sm_count(0)
        return True
    except RuntimeError:
        return False


IMPLS = ["cpu"] + (["gpu"] if gpu_available() else [])


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    """Create a reference/secondary image pair with a known shift."""
    path = tmp_path_factory.mktemp("images")
    rng = np.random.default_rng(12345)
    noise = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    # band-limited (80% of the sampling rate) as SAR images are oversampled
    ky = np.fft.fftfreq(N)[:, None]
    kx = np.fft.fftfreq(N)[None, :]
    band = (np.abs(ky) < 0.4) & (np.abs(kx) < 0.4)
    spectrum = np.fft.fft2(noise) * band
    ref = np.fft.ifft2(spectrum)
    ramp = np.exp(-2j * np.pi * (ky * SHIFT[0] + kx * SHIFT[1]))
    sec = np.fft.ifft2(spectrum * ramp)
    # add some decorrelation noise
    sec += 0.3 * np.std(sec) * (rng.standard_normal((N, N))
                                + 1j * rng.standard_normal((N, N)))
    ref_file = str(path / "reference.slc")
    sec_file = str(path / "secondary.slc")
    ref.astype(np.complex64).tofile(ref_file)
    sec.astype(np.complex64).tofile(sec_file)
    return ref_file, sec_file


def run_ampcor(impl, images, outdir, workflow=0, ovs_method=0, algorithm=0,
               start=None, n_windows=None, workers=0):
    """Run ampcor and return the outputs as a dict of numpy arrays."""
    cls = pycuampcor.PyCuAmpcor if impl == "gpu" else pycuampcor.PyCPUAmpcor
    ampcor = cls()
    ampcor.workflow = workflow
    ampcor.algorithm = algorithm
    ampcor.referenceImageName, ampcor.secondaryImageName = images
    ampcor.referenceImageHeight = ampcor.referenceImageWidth = N
    ampcor.secondaryImageHeight = ampcor.secondaryImageWidth = N

    ampcor.windowSizeHeight = 32
    ampcor.windowSizeWidth = 64
    ampcor.halfSearchRangeDown = 8
    ampcor.halfSearchRangeAcross = 10
    ampcor.skipSampleDown = 32
    ampcor.skipSampleAcross = 32

    if start is None:
        start = (8, 10)
    ampcor.referenceStartPixelDownStatic, ampcor.referenceStartPixelAcrossStatic = start
    if n_windows is None:
        n_windows = ((N - 2 * 8 - 32) // 32, (N - 2 * 10 - 64) // 32)
    ampcor.numberWindowDown, ampcor.numberWindowAcross = n_windows
    ampcor.numberWindowDownInChunk = 2
    ampcor.numberWindowAcrossInChunk = 4

    ampcor.rawDataOversamplingFactor = 2
    ampcor.derampMethod = 1
    ampcor.corrStatWindowSize = 21
    ampcor.corrSurfaceZoomInWindow = 8
    ampcor.corrSurfaceOverSamplingMethod = ovs_method
    ampcor.corrSurfaceOverSamplingFactor = 64
    if workers:
        if impl == "gpu":
            ampcor.nStreams = workers
        else:
            ampcor.nThreads = workers

    outputs = {"dense_offsets": 2, "gross_offsets": 2, "snr": 1,
               "covariance": 3, "correlation_peak": 1}
    os.makedirs(outdir, exist_ok=True)
    ampcor.offsetImageName = os.path.join(outdir, "dense_offsets")
    ampcor.grossOffsetImageName = os.path.join(outdir, "gross_offsets")
    ampcor.snrImageName = os.path.join(outdir, "snr")
    ampcor.covImageName = os.path.join(outdir, "covariance")
    ampcor.peakValueImageName = os.path.join(outdir, "correlation_peak")

    ampcor.setupParams()
    ampcor.setConstantGrossOffset(0, 0)
    ampcor.runAmpcor()

    shape = n_windows
    return {name: np.fromfile(os.path.join(outdir, name), dtype=np.float32)
            .reshape(*shape, bands)
            for name, bands in outputs.items()}


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("workflow", [0, 1])
@pytest.mark.parametrize("ovs_method", [0, 1])
def test_known_shift(impl, workflow, ovs_method, images, tmp_path):
    out = run_ampcor(impl, images, str(tmp_path), workflow=workflow,
                     ovs_method=ovs_method)
    offsets = out["dense_offsets"]
    err = offsets - np.array(SHIFT)
    assert np.all(np.abs(err) < 0.1)
    assert np.all(np.abs(np.median(err, axis=(0, 1))) < 0.02)
    assert np.all(out["correlation_peak"] > 0.5)
    assert np.all(out["snr"] > 1)


@pytest.mark.parametrize("impl", IMPLS)
def test_time_domain(impl, images, tmp_path):
    out = run_ampcor(impl, images, str(tmp_path), algorithm=1)
    err = out["dense_offsets"] - np.array(SHIFT)
    assert np.all(np.abs(err) < 0.1)


@pytest.mark.skipif(not gpu_available(), reason="no CUDA device")
@pytest.mark.parametrize("workflow", [0, 1])
@pytest.mark.parametrize("ovs_method", [0, 1])
def test_cpu_gpu_consistency(workflow, ovs_method, images, tmp_path):
    cpu = run_ampcor("cpu", images, str(tmp_path / "cpu"), workflow, ovs_method)
    gpu = run_ampcor("gpu", images, str(tmp_path / "gpu"), workflow, ovs_method)
    # the offsets may differ by an oversampled grid spacing in rare cases
    diff = np.abs(cpu["dense_offsets"] - gpu["dense_offsets"])
    assert diff.max() <= 1 / 64 + 1e-6
    assert diff.mean() < 1e-3
    np.testing.assert_allclose(cpu["correlation_peak"], gpu["correlation_peak"],
                               atol=1e-4)
    np.testing.assert_allclose(cpu["snr"], gpu["snr"], rtol=1e-3)
    np.testing.assert_allclose(cpu["covariance"], gpu["covariance"],
                               rtol=1e-3, atol=1e-8)


def test_cpu_threads_deterministic(images, tmp_path):
    one = run_ampcor("cpu", images, str(tmp_path / "one"), workers=1)
    many = run_ampcor("cpu", images, str(tmp_path / "many"), workers=4)
    for name in one:
        np.testing.assert_array_equal(one[name], many[name])


@pytest.mark.parametrize("impl", IMPLS)
def test_windows_outside_image(impl, images, tmp_path):
    """Windows partially outside the image are zero-padded, not a crash."""
    # start before the image and extend beyond the bottom/right edges
    n_windows = (17, 17)
    out = run_ampcor(impl, images, str(tmp_path), start=(-20, -30),
                     n_windows=n_windows)
    offsets = out["dense_offsets"]
    # windows fully inside the image are still correct
    err = offsets[2:-3, 2:-3] - np.array(SHIFT)
    assert np.all(np.abs(err) < 0.1)


@pytest.mark.parametrize("impl", IMPLS)
def test_isce3_alias(impl):
    cls = pycuampcor.PyCuAmpcor if impl == "gpu" else pycuampcor.PyCPUAmpcor
    ampcor = cls()
    ampcor.corrImageName = "peak"
    assert ampcor.peakValueImageName == "peak"


def test_errors(images, tmp_path):
    ampcor = pycuampcor.PyCPUAmpcor()
    ampcor.numberWindowDown = 0
    with pytest.raises(ValueError):
        ampcor.setupParams()

    # image size larger than the file
    ampcor = pycuampcor.PyCPUAmpcor()
    ampcor.referenceImageName, ampcor.secondaryImageName = images
    ampcor.referenceImageHeight = ampcor.referenceImageWidth = 2 * N
    ampcor.secondaryImageHeight = ampcor.secondaryImageWidth = 2 * N
    ampcor.setupParams()
    ampcor.setConstantGrossOffset(0, 0)
    with pytest.raises(RuntimeError):
        ampcor.runAmpcor()
