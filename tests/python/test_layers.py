"""
Several layers (e.g., window sizes) run together with runAmpcorLayers, sharing the loaded chunks
"""

import numpy as np
import pytest

from conftest import configure_ampcor, make_images, read_outputs, run_ampcor

N = 512
SHAPE = (N, N)
SHIFT = (1.3, -2.6)
N_WINDOWS = (11, 13)
SKIP = 32
# window sizes and search ranges of the layers
LAYERS = [dict(windowSizeHeight=32, windowSizeWidth=32, halfSearchRangeDown=8, halfSearchRangeAcross=8),
          dict(windowSizeHeight=48, windowSizeWidth=64, halfSearchRangeDown=12, halfSearchRangeAcross=10),
          dict(windowSizeHeight=64, windowSizeWidth=64, halfSearchRangeDown=10, halfSearchRangeAcross=16)]


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    return make_images(tmp_path_factory.mktemp("images"), SHAPE, SHIFT)


def layer_kwargs(layer, margin=0):
    """The windows of all layers centered on a common grid (as in isce3 offsets_product);
    the larger windows start before the image with margin=0"""
    start = []
    for axis, window, search in (("Height", "windowSizeHeight", "halfSearchRangeDown"),
                                 ("Width", "windowSizeWidth", "halfSearchRangeAcross")):
        wmin = min(lay[window] for lay in LAYERS)
        smin = min(lay[search] for lay in LAYERS)
        start.append(margin + smin + wmin // 2 - layer[window] // 2)
    return dict(n_windows=N_WINDOWS, start=tuple(start), skipSampleDown=SKIP, skipSampleAcross=SKIP,
                **layer)


def gross_offsets(kind):
    if kind == "constant":
        return (2, -1)
    rng = np.random.default_rng(7)
    n = N_WINDOWS[0] * N_WINDOWS[1]
    return rng.integers(-3, 4, n), rng.integers(-3, 4, n)


@pytest.mark.parametrize("workflows", [(0, 0, 0), (1, 1, 1), (0, 1, 0)])
@pytest.mark.parametrize("gross", ["constant", "varying"])
def test_layers_match_separate_runs(impl, workflows, gross, images, tmp_path):
    go = gross_offsets(gross)
    kwargs = [dict(layer_kwargs(layer), workflow=wf, gross_offset=go) for layer, wf in zip(LAYERS, workflows)]
    separate = [run_ampcor(impl, images, SHAPE, tmp_path / f"separate{k}", **kw) for k, kw in enumerate(kwargs)]

    layers = [configure_ampcor(impl, images, SHAPE, tmp_path / f"layer{k}", **kw)[0] for k, kw in enumerate(kwargs)]
    type(layers[0]).runAmpcorLayers(layers)

    for k, expected in enumerate(separate):
        out = read_outputs(tmp_path / f"layer{k}", N_WINDOWS, layers[k].isDoublePrecision())
        for name in expected:
            np.testing.assert_array_equal(out[name], expected[name], err_msg=f"layer {k} {name}")
    # offsets (plus gross offsets) of windows fully inside the image are correct for all layers
    go = np.stack(np.broadcast_arrays(*[np.reshape(g, N_WINDOWS) if np.ndim(g) else g for g in go]), axis=-1)
    for k in range(len(LAYERS)):
        err = (separate[k]["dense_offsets"] + go - np.array(SHIFT))[2:-2, 2:-2]
        assert np.all(np.abs(err) < 0.1), k


def test_layers_errors(impl, images, tmp_path):
    def layer(k, **kw):
        return configure_ampcor(impl, images, SHAPE, tmp_path / f"layer{k}", **dict(layer_kwargs(LAYERS[k]), **kw))[0]

    cls = type(layer(0))
    with pytest.raises(ValueError, match="No layers"):
        cls.runAmpcorLayers([])
    with pytest.raises(ValueError, match="same numbers of windows"):
        cls.runAmpcorLayers([layer(0), layer(1, n_windows=(N_WINDOWS[0], N_WINDOWS[1] - 1))])
    with pytest.raises(ValueError, match="same numbers of windows"):
        cls.runAmpcorLayers([layer(0), layer(1, numberWindowAcrossInChunk=3)])
    # both layers write to the same files
    a = configure_ampcor(impl, images, SHAPE, tmp_path / "same", **layer_kwargs(LAYERS[0]))[0]
    b = configure_ampcor(impl, images, SHAPE, tmp_path / "same", **layer_kwargs(LAYERS[1]))[0]
    with pytest.raises(ValueError, match="used more than once"):
        cls.runAmpcorLayers([a, b])
