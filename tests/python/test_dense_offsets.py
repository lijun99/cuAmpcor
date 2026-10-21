"""
Tests of the dense offsets tool (python -m pycuampcor.dense_offsets) with synthetic images
"""

import os

import numpy as np
import pytest

import pycuampcor
from conftest import make_images
from pycuampcor import dense_offsets

N = 512
SHAPE = (N, N)
# true offsets (down/azimuth, across/range) in pixels
SHIFT = (1.3, -2.6)
# the windows: 64 x 64, every 64 pixels, searched +/- 16 pixels
ARGS = ["--ww", "64", "--wh", "64", "--sw", "16", "--sh", "16", "--kw", "64", "--kh", "64", "--drmp", "1"]
# the number of windows (down, across): (N - 2*search - window)//skip
N_WINDOWS = ((N - 2 * 16 - 64) // 64,) * 2
# the outputs (suffix, number of bands)
OUTPUTS = {"": 2, "_gross": 2, "_snr": 1, "_cov": 3, "_peak": 1}


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    """A pair of synthetic images with a known shift"""
    return make_images(tmp_path_factory.mktemp("images"), SHAPE, SHIFT)


def run(impl, images, prefix, *args):
    """Run the tool on the images (on the implementation impl) and read the offsets"""
    ref, sec = images
    dense_offsets.main(["-r", ref, "-s", sec, "--op", str(prefix), *ARGS, *args]
                       + (["--cpu", "--nthreads", "2"] if impl == "cpu" else []))
    return np.fromfile(f"{prefix}.bip", np.float32).reshape(*N_WINDOWS, 2)


def test_explicit_size(impl, images, tmp_path):
    """With the size and type given, the offsets match the known shift, and the outputs have headers"""
    offsets = run(impl, images, tmp_path / "out" / "offset",
                  "--length", str(N), "--width", str(N), "--data-type", "complex64")
    assert np.all(np.abs(offsets - np.array(SHIFT)) < 0.1)
    for suffix, bands in OUTPUTS.items():
        fname = tmp_path / "out" / f"offset{suffix}.bip"
        assert os.path.getsize(fname) == N_WINDOWS[0] * N_WINDOWS[1] * bands * 4
        if not dense_offsets.isce_available():
            header = open(f"{fname}.hdr").read()
            assert f"bands = {bands}" in header and f"samples = {N_WINDOWS[1]}" in header
            assert os.path.exists(f"{fname}.vrt")


def test_gdal_metadata(impl, images, tmp_path):
    """The size and type of the images are read from their .vrt (GDAL), and the output .vrt read back"""
    gdal = pytest.importorskip("osgeo.gdal")
    gdal.UseExceptions()
    for fname in images:
        with open(fname + ".vrt", "w") as f:
            f.write(f'<VRTDataset rasterXSize="{N}" rasterYSize="{N}">\n'
                    '  <VRTRasterBand dataType="CFloat32" band="1" subClass="VRTRawRasterBand">\n'
                    f'    <SourceFilename relativeToVRT="1">{os.path.basename(fname)}</SourceFilename>\n'
                    f'    <ImageOffset>0</ImageOffset><PixelOffset>8</PixelOffset><LineOffset>{8 * N}</LineOffset>\n'
                    '  </VRTRasterBand>\n</VRTDataset>\n')
    try:
        offsets = run(impl, images, tmp_path / "offset")
    finally:
        for fname in images:
            os.remove(fname + ".vrt")
    expected = run(impl, images, tmp_path / "explicit",
                   "--length", str(N), "--width", str(N), "--data-type", "complex64")
    np.testing.assert_array_equal(offsets, expected)
    if not dense_offsets.isce_available():
        ds = gdal.Open(str(tmp_path / "offset.bip.vrt"))
        np.testing.assert_array_equal(ds.ReadAsArray().transpose(1, 2, 0), offsets)


def test_varying_gross_offsets(impl, images, tmp_path):
    """Varying gross offsets from a file (int32 pairs) of zeros give the results of no gross offset"""
    size = ["--length", str(N), "--width", str(N)]
    gross_file = tmp_path / "gross.bin"
    np.zeros((*N_WINDOWS, 2), np.int32).tofile(gross_file)
    varying = run(impl, images, tmp_path / "varying", *size, "--gross", "1", "--gross-file", str(gross_file))
    constant = run(impl, images, tmp_path / "constant", *size)
    np.testing.assert_array_equal(varying, constant)
    # a file of the wrong size
    np.zeros(3, np.int32).tofile(gross_file)
    with pytest.raises(RuntimeError, match="do not match the number of windows"):
        run(impl, images, tmp_path / "wrong", *size, "--gross", "1", "--gross-file", str(gross_file))


def test_wrong_size(images, tmp_path):
    """An image of another size than given is an error (it must be a raw image of that size)"""
    with pytest.raises(RuntimeError, match="not a raw complex64 image"):
        run("cpu", images, tmp_path / "offset", "--length", str(N), "--width", str(N + 1))


@pytest.mark.parametrize("name, reader", [("tif", []), ("img", ["--reader", "gdal"])])
def test_gdal_rasters(impl, images, tmp_path, name, reader):
    """GeoTIFF images (by the extension, or --reader gdal) give the results of the raw images"""
    gdal = pytest.importorskip("osgeo.gdal")
    if not pycuampcor.has_gdal:
        pytest.skip("pycuampcor is built without GDAL")
    gdal.UseExceptions()
    rasters = []
    for k, fname in zip(("ref", "sec"), images):
        rasters.append(str(tmp_path / f"{k}.{name}"))
        ds = gdal.GetDriverByName("GTiff").Create(rasters[-1], N, N, 1, gdal.GDT_CFloat32, ["TILED=YES"])
        ds.GetRasterBand(1).WriteArray(np.fromfile(fname, np.complex64).reshape(N, N))
        ds = None
    expected = run(impl, images, tmp_path / "raw", "--length", str(N), "--width", str(N))
    np.testing.assert_array_equal(run(impl, rasters, tmp_path / "gdal", *reader), expected)


def test_hdf5_datasets(impl, images, tmp_path):
    """HDF5 datasets (HDF5:<file>:<dataset>, or <file>:<dataset> with --reader hdf5) give the results of the raw images"""
    h5py = pytest.importorskip("h5py")
    if not pycuampcor.has_hdf5:
        pytest.skip("pycuampcor is built without HDF5")
    dataset = "/science/LSAR/RSLC/swaths/frequencyA/HH"
    names = []
    for k, fname in zip(("ref", "sec"), images):
        with h5py.File(tmp_path / f"{k}.h5", "w") as f:
            f.create_dataset(dataset, data=np.fromfile(fname, np.complex64).reshape(N, N),
                             chunks=(64, 64), compression="gzip", shuffle=True)
        names.append(f"{tmp_path / k}.h5:{dataset}")
    expected = run(impl, images, tmp_path / "raw", "--length", str(N), "--width", str(N))
    np.testing.assert_array_equal(run(impl, [f"HDF5:{n}" for n in names], tmp_path / "h5"), expected)
    np.testing.assert_array_equal(run(impl, names, tmp_path / "h5r", "--reader", "hdf5"), expected)
    with pytest.raises(RuntimeError, match="No dataset"):
        run(impl, [f"HDF5:{names[0]}", f"HDF5:{tmp_path / 'sec.h5'}:/missing"], tmp_path / "bad")
