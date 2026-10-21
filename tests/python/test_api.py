"""
Tests for the python interface: parameters, gross offsets, image types and errors
"""

import numpy as np
import pytest

import pycuampcor
from conftest import (configure_ampcor, make_images, needs_gpu, new_ampcor, read_outputs,
                      run_ampcor)

N = 384
SHAPE = (N, N)
SHIFT = (1.3, -2.6)


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    """A pair of synthetic images with a known shift"""
    return make_images(tmp_path_factory.mktemp("images"), SHAPE, SHIFT)


def test_module():
    """The module attributes: version, CUDA availability, classes"""
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
    "numberWindowDownInChunk": (2, 0), "numberWindowAcrossInChunk": (8, 0),
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
    """The parameters have their defaults, and can be set and read back"""
    ampcor = new_ampcor(impl)
    for name, (value, default) in INT_PARAMS.items():
        assert getattr(ampcor, name) == default, name
        setattr(ampcor, name, value)
        assert getattr(ampcor, name) == value, name
    for name in STR_PARAMS:
        setattr(ampcor, name, name + ".bin")
        assert getattr(ampcor, name) == name + ".bin"
    assert isinstance(ampcor.isDoublePrecision(), bool)


def test_docstrings(impl):
    """The class, its parameters (with their defaults) and methods have docstrings"""
    cls = type(new_ampcor(impl))
    assert cls.__doc__
    names = list(INT_PARAMS) + STR_PARAMS
    for name in names:
        doc = getattr(cls, name).__doc__
        assert doc and "\n\nDefault: " in doc, name
    assert "Default: Workflow.TWO_PASS" in cls.workflow.__doc__
    assert "Default: 64" in cls.windowSizeHeight.__doc__
    for method in ["runAmpcor", "setupParams", "setConstantGrossOffset", "setVaryingGrossOffset",
                   "checkPixelInImageRange", "isDoublePrecision", "runAmpcorLayers"]:
        assert getattr(cls, method).__doc__, method
    assert "grossOffsetDown" in cls.setConstantGrossOffset.__doc__


def test_isce3_alias(impl):
    """corrImageName (the isce3 name) is an alias of peakValueImageName"""
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


@pytest.mark.parametrize("workflow", [0, 1])
def test_constructor(impl, images, tmp_path, workflow):
    """Parameters given to the constructor give the same results as set one by one"""
    n_windows = (8, 8)
    gd = np.arange(64) % 3 - 1
    for gross in [(1, -2), (gd, -gd), (gd, 1)]:
        out = {constructor: run_ampcor(impl, images, SHAPE, tmp_path / str(constructor),
                                       n_windows=n_windows, start=(20, 20), gross_offset=gross,
                                       workflow=workflow, constructor=constructor)
               for constructor in (False, True)}
        for name in out[False]:
            np.testing.assert_array_equal(out[True][name], out[False][name], err_msg=name)
        np.testing.assert_array_equal(out[True]["gross_offsets"][..., 0],
                                      np.broadcast_to(gross[0], 64).reshape(n_windows))
        np.testing.assert_array_equal(out[True]["gross_offsets"][..., 1],
                                      np.broadcast_to(gross[1], 64).reshape(n_windows))


def test_constructor_parameters(impl, images, tmp_path):
    """The keyword constructor sets the parameters and sets up the object to run"""
    cls = pycuampcor.PyCuAmpcor if impl == "gpu" else pycuampcor.PyCPUAmpcor
    ampcor = cls(referenceImageName=tmp_path / "ref.slc", corrImageName="peak",
                 windowSizeHeight=np.int32(48), numberWindowDown=3, numberWindowAcross=2)
    assert ampcor.referenceImageName == str(tmp_path / "ref.slc")
    assert ampcor.peakValueImageName == "peak"
    assert ampcor.windowSizeHeight == 48
    # the other parameters keep their default values
    assert ampcor.windowSizeWidth == INT_PARAMS["windowSizeWidth"][1]
    # set up: ready to run
    with pytest.raises(RuntimeError, match="smaller than the image size|No such file|open"):
        ampcor.runAmpcor()

    with pytest.raises(TypeError, match="windowSize"):
        cls(windowSize=32)
    with pytest.raises(TypeError, match="windowSizeHeight"):
        cls(windowSizeHeight="32")
    with pytest.raises(TypeError, match="referenceImageName"):
        cls(referenceImageName=1)
    with pytest.raises(TypeError, match="windowSizeWidth"):
        ampcor.windowSizeWidth = 1.5
    with pytest.raises(ValueError, match="number of windows"):
        cls(numberWindowDown=2, grossOffsetDown=[0, 0, 0])
    with pytest.raises(TypeError, match="gross offsets"):
        cls(grossOffsetDown=1.5)
    with pytest.raises(TypeError, match="gross offsets"):
        cls(grossOffsetAcross=[0.5])


def test_setup_repeated(impl, images, tmp_path):
    """setupParams can be called again (e.g., after the constructor) without changes"""
    for workflow in (0, 1):
        once = run_ampcor(impl, images, SHAPE, tmp_path / "once", workflow=workflow)
        ampcor, n_windows = configure_ampcor(impl, images, SHAPE, tmp_path / "again",
                                             workflow=workflow, constructor=True)
        ampcor.setupParams()
        ampcor.setupParams()
        assert (ampcor.halfSearchRangeDown, ampcor.halfSearchRangeAcross) == (8, 10)
        ampcor.setConstantGrossOffset(0, 0)
        ampcor.runAmpcor()
        again = read_outputs(tmp_path / "again", n_windows, ampcor.isDoublePrecision())
        for name in once:
            np.testing.assert_array_equal(again[name], once[name], err_msg=name)


def test_log_level(impl, images, tmp_path, capfd):
    """The messages written, by level (set for both backends)"""
    assert pycuampcor.get_log_level() is pycuampcor.LogLevel.INFO
    try:
        for level, progress in [("quiet", False), (pycuampcor.LogLevel.WARNING, False),
                                (2, True), ("debug", True)]:
            pycuampcor.set_log_level(level)
            assert pycuampcor.get_log_level() == pycuampcor.LogLevel.from_value(level)
            capfd.readouterr()
            run_ampcor(impl, images, SHAPE, tmp_path / str(level), n_windows=(2, 2))
            out = capfd.readouterr().out
            assert ("Processing chunks" in out) == progress, level
            assert ("all objects in chunk are created" in out) == (level == "debug"), level
        with pytest.raises(ValueError):
            pycuampcor.set_log_level("verbose")
    finally:
        pycuampcor.set_log_level("info")


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
    """checkPixelInImageRange runs once set up"""
    ampcor = new_ampcor(impl)
    ampcor.referenceImageName, ampcor.secondaryImageName = images
    ampcor.setupParams()
    ampcor.setConstantGrossOffset(0, 0)
    ampcor.checkPixelInImageRange()


def test_setup_errors(impl, images):
    """Running before the setup or the gross offsets, and invalid parameters raise errors"""
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
    with pytest.raises(ValueError, match="workflow"):
        ampcor.workflow = 3


# option -> (member, integer value, accepted names)
OPTIONS = {
    "algorithm": (pycuampcor.Algorithm.TIME, 1, ["time", "TIME"]),
    "workflow": (pycuampcor.Workflow.ONE_PASS, 1, ["one_pass", "One-Pass"]),
    "derampMethod": (pycuampcor.DerampMethod.NONE, 2, ["none", "skip"]),
    "derampAxis": (pycuampcor.DerampAxis.ACROSS, 1, ["across", "range"]),
    "corrSurfaceOverSamplingMethod": (pycuampcor.OversamplingMethod.SINC, 1, ["sinc"]),
    "referenceImageDataType": (pycuampcor.DataType.FLOAT32, 1, ["float32", "real"]),
    "secondaryImageDataType": (pycuampcor.DataType.COMPLEX64, 2, ["complex64", "complex"]),
}


def test_options(impl):
    """The options are set by the member, the integer value or the name, and read as the member"""
    ampcor = new_ampcor(impl)
    for name, (member, value, names) in OPTIONS.items():
        other = next(m for m in type(member) if m != member)
        for v in [member, value, np.int32(value)] + names:
            setattr(ampcor, name, other)
            setattr(ampcor, name, v)
            # read as the member, equal to the integer value
            assert getattr(ampcor, name) is member, (name, v)
            assert getattr(ampcor, name) == value
    # in the constructor
    cls = type(ampcor)
    assert cls(workflow="one_pass", derampAxis="azimuth").workflow is pycuampcor.Workflow.ONE_PASS
    # invalid options
    with pytest.raises(ValueError, match="derampMethod.*'magnitude'"):
        ampcor.derampMethod = "deramp"
    with pytest.raises(ValueError, match="algorithm"):
        ampcor.algorithm = 2
    with pytest.raises(TypeError, match="workflow"):
        ampcor.workflow = 1.0
    with pytest.raises(ValueError, match="derampAxis"):
        cls(derampAxis="diagonal")


def test_flags(impl):
    """The flags are set by a bool or an integer, and read as a bool"""
    ampcor = new_ampcor(impl)
    for name in ["useMmap", "mergeGrossOffset"]:
        for value in [True, 1, np.int64(1), np.bool_(True)]:
            setattr(ampcor, name, False)
            setattr(ampcor, name, value)
            assert getattr(ampcor, name) is True
        setattr(ampcor, name, 0)
        assert getattr(ampcor, name) is False
        with pytest.raises(TypeError, match=name):
            setattr(ampcor, name, "yes")
        with pytest.raises(TypeError, match=name):
            setattr(ampcor, name, 0.5)


def test_image_errors(impl, images, tmp_path):
    # image size larger than the file
    """Images smaller than their size, or missing, raise errors"""
    with pytest.raises(RuntimeError, match="smaller than the image size"):
        run_ampcor(impl, images, (2 * N, 2 * N), tmp_path, n_windows=(2, 2))
    # missing file
    with pytest.raises(RuntimeError, match="Failed to open"):
        run_ampcor(impl, (images[0] + ".missing", images[1]), SHAPE, tmp_path, n_windows=(2, 2))
    # (errors raised while processing chunks in parallel: see test_hdf5_corrupt_chunk)


def test_mmap_buffer_smaller_than_chunk(impl, images, tmp_path):
    """The mmap window is enlarged to the chunks that need more than mmapSize"""
    expected = run_ampcor(impl, images, SHAPE, tmp_path / "default")
    out = run_ampcor(impl, images, SHAPE, tmp_path / "zero", mmapSize=0)
    for name in expected:
        np.testing.assert_array_equal(out[name], expected[name], err_msg=name)


def test_auto_chunk_size(impl, images, tmp_path):
    """numberWindow*InChunk = 0: chosen by the backend in setupParams, limited by the numbers of windows"""
    n_windows = (9, 30)
    ampcor, _ = configure_ampcor(impl, images, SHAPE, tmp_path, n_windows=n_windows,
                                 numberWindowDownInChunk=0, numberWindowAcrossInChunk=0)
    if impl == "cpu":
        expected = (1, 1)
    else:
        sms = pycuampcor.PyCuAmpcor.get_sm_count(ampcor.deviceID)
        expected = (min(max(sms // 4, 1), n_windows[0]), min(8, n_windows[1]))
    assert (ampcor.windowsDownInChunk, ampcor.windowsAcrossInChunk) == expected
    # the inputs are kept (automatic): a new setup uses the new numbers of windows
    assert (ampcor.numberWindowDownInChunk, ampcor.numberWindowAcrossInChunk) == (0, 0)
    ampcor.numberWindowDown, ampcor.numberWindowAcross = 1, 2
    ampcor.setupParams()
    assert (ampcor.windowsDownInChunk, ampcor.windowsAcrossInChunk) == (1, min(expected[1], 2))
    # explicit values are used as given
    ampcor.numberWindowDownInChunk, ampcor.numberWindowAcrossInChunk = 3, 5
    ampcor.setupParams()
    assert (ampcor.windowsDownInChunk, ampcor.windowsAcrossInChunk) == (3, 5)
    assert new_ampcor(impl).windowsDownInChunk == 0
    # same results as with an explicit chunk size
    auto = run_ampcor(impl, images, SHAPE, tmp_path / "auto", n_windows=n_windows,
                      numberWindowDownInChunk=0, numberWindowAcrossInChunk=0)
    explicit = run_ampcor(impl, images, SHAPE, tmp_path / "explicit", n_windows=n_windows)
    np.testing.assert_allclose(auto["dense_offsets"], explicit["dense_offsets"], atol=0.05)


@needs_gpu
def test_gpu_device_errors(images, tmp_path):
    """An invalid CUDA device raises an error"""
    with pytest.raises(RuntimeError, match="Invalid CUDA device 99"):
        run_ampcor("gpu", images, SHAPE, tmp_path, deviceID=99, n_windows=(2, 2))
    # invalid device when choosing the chunk size automatically
    with pytest.raises(RuntimeError, match="cudaErrorInvalidDevice"):
        run_ampcor("gpu", images, SHAPE, tmp_path, deviceID=99, n_windows=(2, 2),
                   numberWindowDownInChunk=0)
    assert pycuampcor.PyCuAmpcor.get_sm_count() == pycuampcor.PyCuAmpcor.get_sm_count(0) > 0
    # static, callable from the class or an instance
    devices = pycuampcor.PyCuAmpcor.device_list()
    assert devices == pycuampcor.PyCuAmpcor().device_list()
    assert [d["id"] for d in devices] == list(range(len(devices)))
    assert devices[0]["sm_count"] == pycuampcor.PyCuAmpcor.get_sm_count(0)
    assert devices[0]["compute_capability"][0] > 0
    assert devices[0]["total_memory"] > 0 and devices[0]["name"]
