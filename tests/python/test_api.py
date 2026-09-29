"""
Tests for the python interface: parameters, gross offsets, image types and errors
"""

import numpy as np
import pytest

import pycuampcor
from conftest import make_images, needs_gpu, new_ampcor, run_ampcor

N = 384
SHAPE = (N, N)
SHIFT = (1.3, -2.6)


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    return make_images(tmp_path_factory.mktemp("images"), SHAPE, SHIFT)


def test_module():
    assert isinstance(pycuampcor.__version__, str)
    assert isinstance(pycuampcor.has_cuda, bool)
    assert hasattr(pycuampcor, "PyCPUAmpcor")
    assert hasattr(pycuampcor, "PyCuAmpcor") == pycuampcor.has_cuda


# python name -> (value to set, default value or None if not checked)
INT_PARAMS = {
    "algorithm": (1, 0), "deviceID": (1, 0), "nStreams": (3, 1), "nThreads": (2, 0),
    "derampMethod": (1, 0), "derampAxis": (1, 2), "workflow": (1, 0),
    "referenceImageHeight": (11, 1000), "referenceImageWidth": (12, 1000),
    "referenceImageDataType": (1, 2),
    "secondaryImageHeight": (13, 1000), "secondaryImageWidth": (14, 1000),
    "secondaryImageDataType": (1, 2),
    "numberWindowDown": (5, 1), "numberWindowAcross": (6, 1),
    "windowSizeHeight": (48, 64), "windowSizeWidth": (40, 64),
    "mergeGrossOffset": (1, 0),
    "rawDataOversamplingFactor": (3, 2), "corrStatWindowSize": (11, 21),
    "numberWindowDownInChunk": (2, 1), "numberWindowAcrossInChunk": (8, 1),
    "useMmap": (0, 1), "mmapSize": (4, 1),
    "halfSearchRangeDown": (7, 20), "halfSearchRangeAcross": (9, 20),
    "referenceStartPixelDownStatic": (15, 0), "referenceStartPixelAcrossStatic": (16, 0),
    "corrSurfaceOverSamplingMethod": (1, 0), "corrSurfaceOverSamplingFactor": (32, 16),
    "skipSampleDown": (17, 64), "skipSampleAcross": (18, 64),
    "corrSurfaceZoomInWindow": (12, 16),
}
STR_PARAMS = ["referenceImageName", "secondaryImageName", "offsetImageName",
              "grossOffsetImageName", "snrImageName", "covImageName", "peakValueImageName"]


def test_parameters(impl):
    ampcor = new_ampcor(impl)
    for name, (value, default) in INT_PARAMS.items():
        assert getattr(ampcor, name) == default, name
        setattr(ampcor, name, value)
        assert getattr(ampcor, name) == value, name
    for name in STR_PARAMS:
        setattr(ampcor, name, name + ".bin")
        assert getattr(ampcor, name) == name + ".bin"
    assert isinstance(ampcor.isDoublePrecision(), bool)


def test_isce3_alias(impl):
    ampcor = new_ampcor(impl)
    ampcor.corrImageName = "peak"
    assert ampcor.peakValueImageName == "peak"
    assert ampcor.corrImageName == "peak"


def test_constant_gross_offset(impl, images, tmp_path):
    """A constant gross offset is subtracted from (or merged into) the dense offsets"""
    gross = (1, -3)
    out = run_ampcor(impl, images, SHAPE, tmp_path / "separate", gross_offset=gross,
                     start=(20, 20))
    assert np.all(out["gross_offsets"] == np.array(gross))
    err = out["dense_offsets"] - (np.array(SHIFT) - np.array(gross))
    assert np.all(np.abs(err) < 0.1)

    merged = run_ampcor(impl, images, SHAPE, tmp_path / "merged", gross_offset=gross,
                        start=(20, 20), mergeGrossOffset=1)
    err = merged["dense_offsets"] - np.array(SHIFT)
    assert np.all(np.abs(err) < 0.1)


def test_varying_gross_offset(impl, tmp_path_factory, tmp_path):
    """Varying gross offsets allow offsets beyond the search range"""
    shift = (12.3, -15.6)
    images = make_images(tmp_path_factory.mktemp("large_shift"), SHAPE, shift)
    n_windows = (8, 8)
    # alternate between the right gross offsets and a slightly wrong one
    gd = np.full(n_windows, 12, dtype=int)
    ga = np.full(n_windows, -16, dtype=int)
    ga[::2, ::2] = -14
    out = run_ampcor(impl, images, SHAPE, tmp_path, n_windows=n_windows, start=(40, 40),
                     gross_offset=(gd.ravel(), ga.ravel()), mergeGrossOffset=1)
    np.testing.assert_array_equal(out["gross_offsets"][..., 0], gd)
    np.testing.assert_array_equal(out["gross_offsets"][..., 1], ga)
    err = out["dense_offsets"] - np.array(shift)
    assert np.all(np.abs(err) < 0.1)


def test_real_images(impl, tmp_path_factory, tmp_path):
    """Real (amplitude) images"""
    # a bandwidth <= 50% avoids the aliasing of the amplitudes (see test_synthetic.py)
    images = make_images(tmp_path_factory.mktemp("real"), SHAPE, SHIFT,
                         dtype=np.float32, bandwidth=0.5)
    out = run_ampcor(impl, images, SHAPE, tmp_path,
                     referenceImageDataType=1, secondaryImageDataType=1)
    err = out["dense_offsets"] - np.array(SHIFT)
    assert np.all(np.abs(err) < 0.1)
    assert np.all(np.abs(np.median(err, axis=(0, 1))) < 0.02)


def test_check_pixel_in_image_range(impl, images):
    ampcor = new_ampcor(impl)
    ampcor.referenceImageName, ampcor.secondaryImageName = images
    ampcor.setupParams()
    ampcor.setConstantGrossOffset(0, 0)
    ampcor.checkPixelInImageRange()


def test_setup_errors(impl, images):
    ampcor = new_ampcor(impl)
    # run before setting up
    with pytest.raises(RuntimeError, match="setupParams"):
        ampcor.runAmpcor()
    with pytest.raises(RuntimeError, match="setupParams"):
        ampcor.setConstantGrossOffset(0, 0)
    ampcor.setupParams()
    # run before setting the starting pixels
    with pytest.raises(RuntimeError, match="GrossOffset"):
        ampcor.runAmpcor()
    # wrong size of varying gross offsets
    with pytest.raises(ValueError):
        ampcor.setVaryingGrossOffset([0, 0], [0, 0])

    ampcor = new_ampcor(impl)
    ampcor.numberWindowDown = 0
    with pytest.raises(ValueError):
        ampcor.setupParams()
    ampcor = new_ampcor(impl)
    ampcor.workflow = 3
    with pytest.raises(ValueError):
        ampcor.setupParams()


def test_image_errors(impl, images, tmp_path):
    # image size larger than the file
    with pytest.raises(RuntimeError, match="smaller than the image size"):
        run_ampcor(impl, images, (2 * N, 2 * N), tmp_path, n_windows=(2, 2))
    # missing file
    with pytest.raises(RuntimeError, match="Failed to open"):
        run_ampcor(impl, (images[0] + ".missing", images[1]), SHAPE, tmp_path, n_windows=(2, 2))
    # errors raised while processing chunks (in parallel) are propagated:
    # with zero mmap buffer size, loading any image tile fails
    with pytest.raises(RuntimeError, match="mmap"):
        run_ampcor(impl, images, SHAPE, tmp_path, mmapSize=0, nThreads=4)


@needs_gpu
def test_gpu_device_errors(images, tmp_path):
    with pytest.raises(RuntimeError, match="not a valid GPU device"):
        run_ampcor("gpu", images, SHAPE, tmp_path, deviceID=99, n_windows=(2, 2))
    assert pycuampcor.PyCuAmpcor.get_sm_count(0) > 0
