"""
Regression tests with the ovs128-rho0.8 accuracy test data (from isce3)

* against the outputs of the isce3 (v1) pycuampcor (two-pass workflow), with
  the tolerances used in the isce3 test;
* against reference outputs of this package for all workflows and correlation
  surface oversampling methods (regenerate with make_golden.py when a change
  of the results is intended).
"""

import os

import numpy as np
import pytest

from conftest import DATA_DIR, run_ampcor

DATASET = os.path.join(DATA_DIR, "ovs128-rho0.8")
IMAGES = (os.path.join(DATASET, "img1_WN_512x512_1x1_128"),
          os.path.join(DATASET, "img2_WN_512x512_1x1_128"))
SHAPE = (512, 512)
N_WINDOWS = (13, 12)
# parameters used by the isce3 test
PARAMS = dict(windowSizeHeight=32, windowSizeWidth=64,
              halfSearchRangeDown=20, halfSearchRangeAcross=20,
              skipSampleDown=32, skipSampleAcross=32,
              numberWindowDownInChunk=1, numberWindowAcrossInChunk=2,
              algorithm=0, derampMethod=1, derampAxis=0,
              corrStatWindowSize=21, corrSurfaceZoomInWindow=8,
              rawDataOversamplingFactor=2, corrSurfaceOverSamplingFactor=64)
# outputs compared to the golden files
GOLDEN_OUTPUTS = ("dense_offsets", "snr", "covariance", "correlation_peak")


def run_dataset(impl, outdir, workflow, ovs_method):
    return run_ampcor(impl, IMAGES, SHAPE, outdir, n_windows=N_WINDOWS,
                      workflow=workflow, corrSurfaceOverSamplingMethod=ovs_method,
                      **PARAMS)


def golden_dir(workflow, ovs_method):
    return os.path.join(DATASET, "golden", f"wf{workflow}_ovs{ovs_method}")


def read_golden(path, name, shape):
    return np.fromfile(os.path.join(path, name), dtype=np.float32).reshape(shape)


@pytest.mark.parametrize("ovs_method", [0, 1])
def test_isce3_golden(impl, ovs_method, tmp_path):
    """Two-pass results agree with the isce3 (v1) outputs"""
    out = run_dataset(impl, tmp_path, 0, ovs_method)
    for name in GOLDEN_OUTPUTS:
        got = out[name]
        expected = read_golden(os.path.join(DATASET, "golden_isce3"), name, got.shape)
        if name == "dense_offsets":
            tol, meantol = 1e-1, 2e-2
        elif name == "correlation_peak":
            tol, meantol = 5e-2, 2e-2
        else:
            tol, meantol = 1 / 64, 1 / 64 / 5
        diff = np.abs(got - expected)
        assert diff.max() < tol, name
        assert diff.mean() < meantol, name


@pytest.mark.parametrize("workflow", [0, 1])
@pytest.mark.parametrize("ovs_method", [0, 1])
def test_golden(impl, workflow, ovs_method, tmp_path):
    """Results agree with the reference outputs of this package"""
    out = run_dataset(impl, tmp_path, workflow, ovs_method)
    path = golden_dir(workflow, ovs_method)
    for name in GOLDEN_OUTPUTS:
        got = out[name]
        expected = read_golden(path, name, got.shape)
        if name == "dense_offsets":
            # may differ by an oversampled grid spacing in rare cases
            diff = np.abs(got - expected)
            assert diff.max() <= 1 / 64 + 1e-6
            assert diff.mean() < 1e-3
        elif name == "correlation_peak":
            np.testing.assert_allclose(got, expected, atol=1e-4)
        else:
            np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-9)
    # the offsets of this dataset are close to 0
    assert np.all(np.abs(out["dense_offsets"]) < 0.1)
