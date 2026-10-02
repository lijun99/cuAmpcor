"""
Accuracy and robustness tests with synthetic images of known shifts

A band-limited complex white-noise reference is shifted by a known sub-pixel
offset (with the Fourier shift theorem) to create the secondary image.
"""

import numpy as np
import pytest

from conftest import IMPLS, make_images, needs_gpu, run_ampcor

N = 512
SHAPE = (N, N)
# true offsets (down/azimuth, across/range) in pixels
SHIFT = (1.3, -2.6)
# offsets close to the edge of the across search range (10)
EDGE_SHIFT = (0.4, -9.3)


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    return make_images(tmp_path_factory.mktemp("images"), SHAPE, SHIFT)


@pytest.fixture(scope="module")
def edge_images(tmp_path_factory):
    return make_images(tmp_path_factory.mktemp("edge_images"), SHAPE, EDGE_SHIFT)


@pytest.mark.parametrize("workflow", [0, 1])
@pytest.mark.parametrize("ovs_method", [0, 1])
def test_known_shift(impl, workflow, ovs_method, images, tmp_path):
    out = run_ampcor(impl, images, SHAPE, tmp_path, workflow=workflow,
                     corrSurfaceOverSamplingMethod=ovs_method)
    err = out["dense_offsets"] - np.array(SHIFT)
    assert np.all(np.abs(err) < 0.1)
    assert np.all(np.abs(np.median(err, axis=(0, 1))) < 0.02)
    assert np.all(out["correlation_peak"] > 0.5)
    assert np.all(out["snr"] > 1)
    assert np.all(out["gross_offsets"] == 0)
    # covariance: positive variances, flagged (99) only at margins
    cov = out["covariance"]
    assert np.all(cov[..., :2] > 0)
    assert np.all(cov[..., :2] < 99)


@pytest.mark.parametrize("ovs_method", [0, 1])
def test_search_edge(impl, ovs_method, edge_images, tmp_path):
    """Peaks close to the search range edge (the zoom-in window is shifted)"""
    out = run_ampcor(impl, edge_images, SHAPE, tmp_path,
                     corrSurfaceOverSamplingMethod=ovs_method)
    err = out["dense_offsets"] - np.array(EDGE_SHIFT)
    assert np.all(np.abs(err) < 0.1)


@pytest.mark.parametrize("workflow", [0, 1])
def test_time_domain(impl, workflow, images, tmp_path):
    out = run_ampcor(impl, images, SHAPE, tmp_path, algorithm=1, workflow=workflow)
    err = out["dense_offsets"] - np.array(SHIFT)
    assert np.all(np.abs(err) < 0.1)


@pytest.fixture(scope="module")
def narrowband_images(tmp_path_factory):
    # the amplitudes of speckle have twice the bandwidth of the complex signal;
    # a bandwidth <= 50% of the sampling rate avoids their aliasing (which biases
    # the offsets towards integer pixels)
    return make_images(tmp_path_factory.mktemp("narrowband"), SHAPE, SHIFT, bandwidth=0.5)


def test_deramp_no_deramping(impl, images, tmp_path):
    """no deramping (2) for images without phase ramps"""
    out = run_ampcor(impl, images, SHAPE, tmp_path, derampMethod=2)
    err = out["dense_offsets"] - np.array(SHIFT)
    assert np.all(np.abs(err) < 0.1)


@pytest.mark.parametrize("workflow", [0, 1])
def test_deramp_magnitude(impl, workflow, narrowband_images, tmp_path):
    """correlate the amplitudes (deramp method 0), as for TOPS"""
    out = run_ampcor(impl, narrowband_images, SHAPE, tmp_path, derampMethod=0,
                     workflow=workflow)
    err = out["dense_offsets"] - np.array(SHIFT)
    assert np.all(np.abs(err) < 0.1)
    assert np.all(np.abs(np.median(err, axis=(0, 1))) < 0.02)


def test_window_sizes(impl, images, tmp_path):
    """Other window/search/oversampling sizes, and uneven chunks"""
    out = run_ampcor(impl, images, SHAPE, tmp_path,
                     windowSizeHeight=48, windowSizeWidth=40,
                     halfSearchRangeDown=6, halfSearchRangeAcross=12,
                     skipSampleDown=40, skipSampleAcross=24,
                     numberWindowDownInChunk=3, numberWindowAcrossInChunk=5,
                     rawDataOversamplingFactor=3, corrSurfaceZoomInWindow=12,
                     corrSurfaceOverSamplingFactor=32)
    err = out["dense_offsets"] - np.array(SHIFT)
    assert np.all(np.abs(err) < 0.1)


@pytest.mark.parametrize("workflow", [0, 1])
def test_noisy_images(impl, workflow, tmp_path_factory, tmp_path):
    """Decorrelated images: most offsets are still close to the truth"""
    images = make_images(tmp_path_factory.mktemp("noisy"), SHAPE, SHIFT, noise=0.8)
    out = run_ampcor(impl, images, SHAPE, tmp_path, workflow=workflow)
    err = np.abs(out["dense_offsets"] - np.array(SHIFT))
    good = np.all(err < 0.2, axis=-1)
    assert good.mean() > 0.9
    # the correlation is lower than for the default noise level
    assert np.median(out["correlation_peak"]) < 0.5


@needs_gpu
@pytest.mark.parametrize("workflow", [0, 1])
@pytest.mark.parametrize("ovs_method", [0, 1])
def test_cpu_gpu_consistency(workflow, ovs_method, images, tmp_path):
    kw = dict(workflow=workflow, corrSurfaceOverSamplingMethod=ovs_method)
    cpu = run_ampcor("cpu", images, SHAPE, tmp_path / "cpu", **kw)
    gpu = run_ampcor("gpu", images, SHAPE, tmp_path / "gpu", **kw)
    # the offsets may differ by an oversampled grid spacing in rare cases
    diff = np.abs(cpu["dense_offsets"] - gpu["dense_offsets"])
    assert diff.max() <= 1 / 64 + 1e-6
    assert diff.mean() < 1e-3
    np.testing.assert_allclose(cpu["correlation_peak"], gpu["correlation_peak"], atol=1e-4)
    np.testing.assert_allclose(cpu["snr"], gpu["snr"], rtol=1e-3)
    np.testing.assert_allclose(cpu["covariance"], gpu["covariance"], rtol=1e-3, atol=1e-8)


def test_cpu_threads_deterministic(images, tmp_path):
    one = run_ampcor("cpu", images, SHAPE, tmp_path / "one", nThreads=1)
    many = run_ampcor("cpu", images, SHAPE, tmp_path / "many", nThreads=4)
    for name in one:
        np.testing.assert_array_equal(one[name], many[name])


@needs_gpu
def test_gpu_streams(images, tmp_path):
    one = run_ampcor("gpu", images, SHAPE, tmp_path / "one", nStreams=1)
    many = run_ampcor("gpu", images, SHAPE, tmp_path / "many", nStreams=3)
    for name in one:
        np.testing.assert_array_equal(one[name], many[name])


def test_windows_outside_image(impl, images, tmp_path):
    """Windows partially outside the image are zero-padded, not a crash"""
    # start before the image and extend beyond the bottom/right edges
    out = run_ampcor(impl, images, SHAPE, tmp_path, start=(-20, -30), n_windows=(17, 17))
    # windows fully inside the image are still correct
    err = out["dense_offsets"][2:-3, 2:-3] - np.array(SHIFT)
    assert np.all(np.abs(err) < 0.1)


@pytest.mark.parametrize("deramp, ratio_range", [
    # magnitude: both workflows correlate the amplitudes before (or without) anti-aliasing,
    # so the snr of the one-pass workflow is close to the two-pass one
    (0, (0.8, 2.0)),
    # complex: the one-pass workflow oversamples the complex images before taking the amplitudes,
    # a correlation surface with a lower background (the two-pass snr is on raw amplitudes)
    (1, (1.0, 10.0)),
])
def test_snr_onepass_vs_twopass(impl, deramp, ratio_range, images, tmp_path):
    """The one-pass snr is estimated as the two-pass snr (same window at the raw pixel spacing)"""
    snr = {wf: run_ampcor(impl, images, SHAPE, tmp_path / f"wf{wf}", workflow=wf, derampMethod=deramp)["snr"][..., 0]
           for wf in (0, 1)}
    ratio = np.median(snr[1] / snr[0])
    assert ratio_range[0] < ratio < ratio_range[1], ratio
