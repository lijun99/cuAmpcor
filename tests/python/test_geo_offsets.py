"""
Tests of the displacement tool for georeferenced images (python -m pycuampcor.geo_offsets) with
synthetic optical-like images on shifted map grids
"""

import numpy as np
import pytest

import pycuampcor
from pycuampcor import geo_offsets

gdal = pytest.importorskip("osgeo.gdal")
osr = pytest.importorskip("osgeo.osr")
gdal.UseExceptions()
pytestmark = pytest.mark.skipif(not pycuampcor.has_gdal, reason="pycuampcor is built without GDAL")

WORLD = (640, 720)            # the texture, in pixels
RES = 10.0                    # pixel size (m)
X0, Y0 = 500000.0, 4000000.0  # the origin of the texture
REF = ((40, 60), (400, 480))  # the reference: origin (down, across) in the texture, size
SEC = ((47.4, 54.75), (420, 500))  # the secondary: origin (fractional), size
MOTION = (0.3, -0.45)         # the ground displacement (down, across) in pixels
ARGS = ["--ww", "64", "--wh", "64", "--sw", "8", "--sh", "8", "--kw", "32", "--kh", "32"]


def texture(shift=(0.0, 0.0)):
    """An optical-like texture (positive, Gaussian blurred noise), shifted by shift (down, across)"""
    rng = np.random.default_rng(7)
    ky, kx = np.fft.fftfreq(WORLD[0])[:, None], np.fft.fftfreq(WORLD[1])[None, :]
    spec = np.fft.fft2(rng.standard_normal(WORLD)) * np.exp(-2 * (np.pi * 0.8) ** 2 * (kx ** 2 + ky ** 2))
    t = np.fft.ifft2(spec * np.exp(-2j * np.pi * (ky * shift[0] + kx * shift[1]))).real
    return 1000 + 200 * t / t.std()


def write_raster(path, bands, origin, epsg=32611, res=RES, nodata=None, dtype=gdal.GDT_UInt16):
    """Write the bands as a GeoTIFF with its upper-left corner at the texture pixel origin (down, across)"""
    ds = gdal.GetDriverByName("GTiff").Create(str(path), bands[0].shape[1], bands[0].shape[0], len(bands), dtype)
    ds.SetGeoTransform((X0 + origin[1] * RES, res, 0, Y0 - origin[0] * RES, 0, -res))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetSpatialRef(srs)
    for k, b in enumerate(bands):
        ds.GetRasterBand(k + 1).WriteArray(np.round(b))
        if nodata is not None:
            ds.GetRasterBand(k + 1).SetNoDataValue(nodata)
    ds = None
    return str(path)


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    """The reference and the secondary images (after the ground motion), on shifted grids"""
    path = tmp_path_factory.mktemp("geo")
    (ry, rx), (rl, rw) = REF
    ref = texture()[ry:ry + rl, rx:rx + rw]
    (sy, sx), (sl, sw) = SEC
    iy, ix = int(np.floor(sy)), int(np.floor(sx))
    sec = texture((MOTION[0] - (sy - iy), MOTION[1] - (sx - ix)))[iy:iy + sl, ix:ix + sw]
    return write_raster(path / "ref.tif", [ref], REF[0]), write_raster(path / "sec.tif", [sec], SEC[0]), ref, sec


def run(impl, ref, sec, out, *args):
    """Run the tool and read the output bands"""
    geo_offsets.main(["-r", ref, "-s", sec, "--op", str(out), *ARGS, *args]
                     + (["--cpu", "--nthreads", "2"] if impl == "cpu" else []))
    ds = gdal.Open(f"{out}.tif")
    bands = {ds.GetRasterBand(k + 1).GetDescription(): ds.GetRasterBand(k + 1).ReadAsArray()
             for k in range(ds.RasterCount)}
    return bands, ds.GetGeoTransform()


def test_displacement(impl, images, tmp_path):
    """The east and north displacements are the ground motion, without the shift of the grids"""
    bands, gt = run(impl, images[0], images[1], tmp_path / "disp")
    east, north = bands["east"], bands["north"]
    assert np.all(np.isfinite(east))
    np.testing.assert_allclose(east, MOTION[1] * RES, atol=0.02 * RES)
    np.testing.assert_allclose(north, -MOTION[0] * RES, atol=0.02 * RES)
    assert np.all(bands["snr"] > 5)
    # the grid: the window centers every 32 pixels, the first window at the search range minus the
    # gross offset (the shift of the grids, -7.4, 5.25 to the pixel: -7, 5) in the reference
    start = (max(0, 8 + 7), max(0, 8 - 5))
    x0 = X0 + REF[0][1] * RES + (start[1] + 32) * RES
    y0 = Y0 - REF[0][0] * RES - (start[0] + 32) * RES
    np.testing.assert_allclose(gt, (x0 - 16 * RES, 32 * RES, 0, y0 + 16 * RES, 0, -32 * RES))
    # the last windows: their search areas in the secondary, their windows in the reference
    n_down, n_across = east.shape
    assert start[0] + (n_down - 1) * 32 + 64 + 8 - 7 <= SEC[1][0] < start[0] + n_down * 32 + 64 + 8 - 7
    assert start[1] + (n_across - 1) * 32 + 64 <= REF[1][1] < start[1] + n_across * 32 + 64


def test_nodata(impl, images, tmp_path):
    """The windows with no-data pixels are NaN, the others unchanged"""
    ref = images[2].copy()
    ref[100:150, 200:260] = 0
    name = write_raster(tmp_path / "ref.tif", [ref], REF[0], nodata=0)
    bands, gt = run(impl, name, images[1], tmp_path / "nodata")
    expected, _ = run(impl, images[0], images[1], tmp_path / "all")
    # the windows (64 x 64 every 32 pixels, from (15, 3)) overlapping the block
    rows = (15 + np.arange(bands["east"].shape[0]) * 32)[:, None]
    cols = (3 + np.arange(bands["east"].shape[1]) * 32)[None, :]
    hit = (rows < 150) & (rows + 64 > 100) & (cols < 260) & (cols + 64 > 200)
    assert np.all(np.isnan(bands["east"][hit])) and np.all(np.isnan(bands["snr"][hit]))
    np.testing.assert_array_equal(bands["east"][~hit], expected["east"][~hit])


def test_band_and_bbox(impl, images, tmp_path):
    """The second band of 2-band images, in a bounding box"""
    ref = write_raster(tmp_path / "ref2.tif", [np.zeros_like(images[2]), images[2]], REF[0])
    sec = write_raster(tmp_path / "sec2.tif", [np.zeros_like(images[3]), images[3]], SEC[0])
    expected, egt = run(impl, images[0], images[1], tmp_path / "all")
    # the window centers in a box around the 2nd to 4th centers down and across
    xs = egt[0] + egt[1] * (np.arange(5) + 0.5)
    ys = egt[3] + egt[5] * (np.arange(5) + 0.5)
    bbox = [str(v) for v in (xs[1] - 1, ys[3] - 1, xs[3] + 1, ys[1] + 1)]
    bands, gt = run(impl, ref, sec, tmp_path / "box", "--band", "2", "--bbox", *bbox)
    assert bands["east"].shape == (3, 3)
    np.testing.assert_allclose(gt, (egt[0] + egt[1], egt[1], 0, egt[3] + egt[5], 0, egt[5]))
    np.testing.assert_array_equal(bands["east"], expected["east"][1:4, 1:4])


def test_errors(images, tmp_path):
    """Images of different coordinate systems or pixel sizes are errors"""
    utm12 = write_raster(tmp_path / "utm12.tif", [images[3]], SEC[0], epsg=32612)
    with pytest.raises(RuntimeError, match="coordinate systems"):
        run("cpu", images[0], utm12, tmp_path / "out")
    coarse = write_raster(tmp_path / "coarse.tif", [images[3]], SEC[0], res=20.0)
    with pytest.raises(RuntimeError, match="pixel sizes"):
        run("cpu", images[0], coarse, tmp_path / "out")
