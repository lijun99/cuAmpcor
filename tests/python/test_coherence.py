"""
The coherent cross-correlation (algorithm "coherence"): the complex windows correlated, with the
squared coherence as the correlation surfaces
"""

import numpy as np
import pytest

import pycuampcor
from conftest import IMPLS, configure_ampcor, make_images, run_ampcor
from pycuampcor import dense_offsets

SHAPE = (512, 512)
SHIFT = (1.37, -2.61)
COHERENCE = dict(algorithm="coherence", derampMethod="complex")


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    """A pair of synthetic SLCs with a known shift, highly coherent"""
    return make_images(tmp_path_factory.mktemp("images"), SHAPE, SHIFT, noise=0.3)


def test_self(impl, images, tmp_path):
    """An image against itself: no offset (to the oversampling step), the coherence 1"""
    out = run_ampcor(impl, (images[0], images[0]), SHAPE, tmp_path, **COHERENCE)
    assert np.all(np.abs(out["dense_offsets"]) <= 1 / 64)
    assert np.all(out["correlation_peak"] > 0.95)


def test_known_shift(impl, images, tmp_path):
    """A known shift; the peak value is the squared coherence (here about 1/(1 + 0.3^2)^2)"""
    out = run_ampcor(impl, images, SHAPE, tmp_path, **COHERENCE)
    err = out["dense_offsets"] - np.array(SHIFT)
    assert np.all(np.abs(err) < 0.03)
    assert np.median(np.abs(err)) < 0.01
    assert abs(np.median(out["correlation_peak"]) - 1 / (1 + 0.3 ** 2) ** 2) < 0.1


@pytest.mark.skipif(len(IMPLS) < 2, reason="needs both implementations")
def test_cpu_gpu(images, tmp_path):
    """The CPU and GPU implementations agree (to the oversampling step)"""
    cpu = run_ampcor("cpu", images, SHAPE, tmp_path / "cpu", **COHERENCE)
    gpu = run_ampcor("gpu", images, SHAPE, tmp_path / "gpu", **COHERENCE)
    np.testing.assert_allclose(cpu["dense_offsets"], gpu["dense_offsets"], atol=1 / 64)
    np.testing.assert_allclose(cpu["correlation_peak"], gpu["correlation_peak"], atol=1e-3)


def test_low_coherence(impl, tmp_path):
    """At a low coherence, the coherent correlation still finds the shift, the amplitudes do not"""
    images = make_images(tmp_path / "images", SHAPE, SHIFT, noise=2.0)
    coherent = run_ampcor(impl, images, SHAPE, tmp_path / "coherence", **COHERENCE)
    amplitude = run_ampcor(impl, images, SHAPE, tmp_path / "amplitude", derampMethod="complex")
    err_c = np.abs(coherent["dense_offsets"] - np.array(SHIFT))
    err_a = np.abs(amplitude["dense_offsets"] - np.array(SHIFT))
    assert np.all(err_c < 0.2) and np.median(err_c) < 0.05
    assert np.mean(err_a > 0.5) > 0.05


def test_dense_offsets_tool(impl, images, tmp_path):
    """The dense offsets tool with --alg 2 (and --drmp 1)"""
    n = SHAPE[0]
    args = ["-r", images[0], "-s", images[1], "--op", str(tmp_path / "offset"), "--length", str(n),
            "--width", str(n), "--ww", "64", "--wh", "64", "--sw", "16", "--sh", "16", "--kw", "64",
            "--kh", "64", "--alg", "2", "--drmp", "1"] + (["--cpu", "--nthreads", "2"] if impl == "cpu" else [])
    dense_offsets.main(args)
    n_windows = (n - 2 * 16 - 64) // 64
    offsets = np.fromfile(tmp_path / "offset.bip", np.float32).reshape(n_windows, n_windows, 2)
    assert np.all(np.abs(offsets - np.array(SHIFT)) < 0.05)


@pytest.mark.parametrize("params, match", [
    (dict(workflow="one_pass"), "two-pass"),
    (dict(derampMethod="magnitude"), "complex windows"),
    (dict(referenceImageDataType="float32", secondaryImageDataType="float32"), "complex images"),
])
def test_unsupported(images, tmp_path, params, match):
    """The coherence needs the two-pass workflow, the complex windows, complex images"""
    with pytest.raises(ValueError, match=match):
        configure_ampcor("cpu", images, SHAPE, tmp_path, **{**COHERENCE, **params})


def test_option():
    """The option values"""
    assert pycuampcor.Algorithm.COHERENCE == 2
    a = pycuampcor.PyCPUAmpcor()
    a.algorithm = "coherence"
    assert a.algorithm == pycuampcor.Algorithm.COHERENCE
