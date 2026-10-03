"""
Images read from 2D datasets in HDF5 files (HDF5:<file>:<dataset>) give the same results as
the same images in raw binary files, whether the chunks are decoded directly or read with the
HDF5 library
"""

import numpy as np
import pytest

import pycuampcor
from conftest import configure_ampcor, make_images, read_outputs, run_ampcor

h5py = pytest.importorskip("h5py")
pytestmark = pytest.mark.skipif(not pycuampcor.has_hdf5, reason="pycuampcor is built without HDF5")

SHAPE = (300, 400)  # not a multiple of the chunk sizes
SHIFT = (1.3, -2.6)
DATASET = "/science/LSAR/RSLC/swaths/frequencyA/HH"

# dataset layouts (h5py create_dataset options) and the expected reading path
LAYOUTS = {
    # as in the NISAR products
    "gzip_shuffle": (dict(chunks=(64, 96), compression="gzip", compression_opts=4, shuffle=True), "direct"),
    "gzip": (dict(chunks=(64, 96), compression="gzip"), "direct"),
    "chunked": (dict(chunks=(128, 128)), "direct"),
    "contiguous": (dict(), "library"),
    # a filter not decoded directly
    "fletcher32": (dict(chunks=(64, 96), compression="gzip", shuffle=True, fletcher32=True), "library"),
}


def write_h5(path, data, name=DATASET, userblock_size=0, write=None, **options):
    """Write data as a dataset in a new HDF5 file; write: the region to write (default all)"""
    with h5py.File(path, "w", userblock_size=userblock_size) as f:
        if write is None:
            f.create_dataset(name, data=data, **options)
        else:
            dset = f.create_dataset(name, shape=data.shape, dtype=data.dtype, **options)
            dset[write] = data[write]
    return f"HDF5:{path}:{name}"


def read_raw(file, dtype=np.complex64):
    return np.fromfile(file, dtype=dtype).reshape(SHAPE)


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    return make_images(tmp_path_factory.mktemp("images"), SHAPE, SHIFT)


@pytest.fixture(scope="module")
def raw_results(images, tmp_path_factory):
    """Results with the raw binary images, for each implementation"""
    cache = {}

    def get(impl):
        if impl not in cache:
            cache[impl] = run_ampcor(impl, images, SHAPE, tmp_path_factory.mktemp(f"raw_{impl}"))
        return cache[impl]
    return get


def assert_same(out, expected):
    for name in expected:
        np.testing.assert_array_equal(out[name], expected[name], err_msg=name)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_hdf5_matches_raw(impl, layout, images, raw_results, tmp_path, capfd):
    options, path = LAYOUTS[layout]
    h5 = tuple(write_h5(tmp_path / f"{k}.h5", read_raw(f), **options) for k, f in zip(("ref", "sec"), images))
    out = run_ampcor(impl, h5, SHAPE, tmp_path / "out")
    assert_same(out, raw_results(impl))
    log = capfd.readouterr().out
    expected = "decoded directly" if path == "direct" else "read with the HDF5 library"
    assert log.count(expected) == 2, log


@pytest.mark.parametrize("cache_gb", [0, 1])
def test_hdf5_cache_size(impl, cache_gb, images, raw_results, tmp_path):
    """The smallest cache (one row of chunks) gives the same results"""
    options = dict(chunks=(32, 48), compression="gzip", shuffle=True)
    h5 = tuple(write_h5(tmp_path / f"{k}.h5", read_raw(f), **options) for k, f in zip(("ref", "sec"), images))
    out = run_ampcor(impl, h5, SHAPE, tmp_path / "out", mmapSize=cache_gb)
    assert_same(out, raw_results(impl))


@pytest.mark.parametrize("threads", [1, 3])
def test_hdf5_decode_threads(impl, threads, images, raw_results, tmp_path, capfd, monkeypatch):
    """The number of threads to decode chunks (PYCUAMPCOR_HDF5_THREADS) doesn't change the results"""
    monkeypatch.setenv("PYCUAMPCOR_HDF5_THREADS", str(threads))
    options = dict(chunks=(32, 48), compression="gzip", shuffle=True)
    h5 = tuple(write_h5(tmp_path / f"{k}.h5", read_raw(f), **options) for k, f in zip(("ref", "sec"), images))
    out = run_ampcor(impl, h5, SHAPE, tmp_path / "out")
    assert_same(out, raw_results(impl))
    assert capfd.readouterr().out.count(f"up to {threads} threads") == 2


def test_hdf5_unallocated_chunks(impl, images, tmp_path, capfd):
    """Chunks never written are read as zeros (the default fill value), as in the raw file"""
    ref, sec = (read_raw(f) for f in images)
    region = np.s_[64:256, :]
    sec_zeros = np.zeros_like(sec)
    sec_zeros[region] = sec[region]
    raw_sec = tmp_path / "sec_zeros.slc"
    sec_zeros.tofile(raw_sec)
    expected = run_ampcor(impl, (images[0], str(raw_sec)), SHAPE, tmp_path / "raw")

    options = dict(chunks=(64, 96), compression="gzip", shuffle=True)
    h5 = (write_h5(tmp_path / "ref.h5", ref, **options),
          write_h5(tmp_path / "sec.h5", sec, write=region, **options))
    out = run_ampcor(impl, h5, SHAPE, tmp_path / "out")
    assert_same(out, expected)
    assert capfd.readouterr().out.count("decoded directly") == 2


def test_hdf5_userblock(impl, images, raw_results, tmp_path, capfd):
    """A file with a user block is read with the library"""
    options = dict(chunks=(64, 96), compression="gzip", shuffle=True)
    h5 = tuple(write_h5(tmp_path / f"{k}.h5", read_raw(f), userblock_size=512, **options)
               for k, f in zip(("ref", "sec"), images))
    out = run_ampcor(impl, h5, SHAPE, tmp_path / "out")
    assert_same(out, raw_results(impl))
    assert capfd.readouterr().out.count("read with the HDF5 library") == 2


def test_hdf5_real_images(impl, tmp_path):
    """Real (amplitude) float32 images"""
    images = make_images(tmp_path / "images", SHAPE, SHIFT, dtype=np.float32, bandwidth=0.5)
    params = dict(referenceImageDataType=1, secondaryImageDataType=1)
    expected = run_ampcor(impl, images, SHAPE, tmp_path / "raw", **params)
    options = dict(chunks=(64, 96), compression="gzip", shuffle=True)
    h5 = tuple(write_h5(tmp_path / f"{k}.h5", read_raw(f, np.float32), **options)
               for k, f in zip(("ref", "sec"), images))
    out = run_ampcor(impl, h5, SHAPE, tmp_path / "out", **params)
    assert_same(out, expected)


def test_hdf5_layers(impl, images, tmp_path):
    """Several layers (runAmpcorLayers) sharing the chunks loaded from HDF5"""
    options = dict(chunks=(64, 96), compression="gzip", shuffle=True)
    h5 = tuple(write_h5(tmp_path / f"{k}.h5", read_raw(f), **options) for k, f in zip(("ref", "sec"), images))
    layers = [dict(windowSizeHeight=32, windowSizeWidth=64, halfSearchRangeDown=8, halfSearchRangeAcross=10),
              dict(windowSizeHeight=48, windowSizeWidth=48, halfSearchRangeDown=10, halfSearchRangeAcross=8)]
    common = dict(n_windows=(6, 8), start=(16, 16))
    objs = [configure_ampcor(impl, h5, SHAPE, tmp_path / f"h5_{k}", **common, **layer)[0]
            for k, layer in enumerate(layers)]
    type(objs[0]).runAmpcorLayers(objs)
    for k, layer in enumerate(layers):
        expected = run_ampcor(impl, images, SHAPE, tmp_path / f"raw{k}", **common, **layer)
        out = read_outputs(tmp_path / f"h5_{k}", common["n_windows"], objs[k].isDoublePrecision())
        assert_same(out, expected)


@pytest.mark.parametrize("form", ["plain", "double_slash", "quoted", "relative_dataset"])
def test_hdf5_name_forms(impl, form, images, raw_results, tmp_path):
    """The forms of HDF5 image names, e.g. as built by isce3 (HDF5:<file>://<dataset>)"""
    options = dict(chunks=(64, 96), compression="gzip", shuffle=True)
    names = []
    for k, f in zip(("ref", "sec"), images):
        path = tmp_path / f"{k}.h5"
        write_h5(path, read_raw(f), **options)
        names.append({"plain": f"HDF5:{path}:{DATASET}",
                      "double_slash": f"HDF5:{path}:/{DATASET}",
                      "quoted": f'HDF5:"{path}":{DATASET}',
                      "relative_dataset": f"HDF5:{path}:{DATASET.lstrip('/')}"}[form])
    out = run_ampcor(impl, tuple(names), SHAPE, tmp_path / "out")
    assert_same(out, raw_results(impl))


def test_reader_option(impl, images, raw_results, tmp_path):
    """The image reader chosen explicitly: hdf5 without the HDF5: prefix, raw for a raw file"""
    names = []
    for k, f in zip(("ref", "sec"), images):
        path = tmp_path / f"{k}.h5"
        write_h5(path, read_raw(f), chunks=(64, 96), compression="gzip", shuffle=True)
        names.append(f"{path}:{DATASET}")
    out = run_ampcor(impl, tuple(names), SHAPE, tmp_path / "h5",
                     referenceImageReader="hdf5", secondaryImageReader="hdf5")
    assert_same(out, raw_results(impl))
    # the reference from HDF5, the secondary from the raw file
    out = run_ampcor(impl, (names[0], images[1]), SHAPE, tmp_path / "mixed",
                     referenceImageReader="hdf5", secondaryImageReader="raw")
    assert_same(out, raw_results(impl))
    ampcor, _ = configure_ampcor(impl, images, SHAPE, tmp_path / "out", referenceImageReader="gdal")
    assert ampcor.referenceImageReader == "gdal" and ampcor.secondaryImageReader == "auto"
    with pytest.raises(ValueError, match="Unknown image reader gdal"):
        ampcor.runAmpcor()


def test_hdf5_corrupt_chunk(impl, images, tmp_path):
    """An error raised while decoding a chunk in a worker (CUDA stream or CPU thread) is propagated"""
    path = tmp_path / "sec.h5"
    name = write_h5(path, read_raw(images[1]), chunks=(64, 96), compression="gzip", shuffle=True)
    with h5py.File(path, "r") as f:
        info = f[DATASET].id.get_chunk_info_by_coord((128, 192))
    with open(path, "r+b") as f:
        f.seek(info.byte_offset)
        f.write(b"\xff" * info.size)
    with pytest.raises(RuntimeError, match="decompress"):
        run_ampcor(impl, (images[0], name), SHAPE, tmp_path / "out", nThreads=4)


def test_hdf5_errors(impl, images, tmp_path):
    ref, sec = (read_raw(f) for f in images)
    good = write_h5(tmp_path / "ref.h5", ref)
    cases = {
        "missing file": f"HDF5:{tmp_path / 'missing.h5'}:{DATASET}",
        "missing dataset": f"HDF5:{tmp_path / 'ref.h5'}:/no/such/dataset",
        "wrong size": write_h5(tmp_path / "small.h5", sec[:-1]),
        "wrong type": write_h5(tmp_path / "c128.h5", sec.astype(np.complex128)),
        "no dataset": f"HDF5:{tmp_path / 'ref.h5'}:/",
    }
    for case, name in cases.items():
        ampcor, _ = configure_ampcor(impl, (good, name), SHAPE, tmp_path / "out")
        with pytest.raises((RuntimeError, ValueError), match=r"HDF5|dataset|size|type"):
            ampcor.runAmpcor()
