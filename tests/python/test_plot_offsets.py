"""
Tests of the plotting tool (python -m pycuampcor.plot_offsets) on dense_offsets and geo_offsets
outputs written here
"""

import os

import numpy as np
import pytest

pytest.importorskip("matplotlib")
gdal = pytest.importorskip("osgeo.gdal")
osr = pytest.importorskip("osgeo.osr")
gdal.UseExceptions()

from pycuampcor import dense_offsets, geo_offsets, plot_offsets  # noqa: E402

SHAPE = (20, 30)


@pytest.fixture(autouse=True)
def headless(monkeypatch):
    """No display: the figures are saved"""
    import matplotlib
    matplotlib.use("Agg")


@pytest.fixture
def radar(tmp_path):
    """A dense_offsets output: offset.bip (down, across) and offset_snr.bip, with ENVI .hdr and .vrt"""
    rng = np.random.default_rng(0)
    off = np.stack([np.full(SHAPE, 1.5), np.linspace(-2, 2, SHAPE[1])[None, :].repeat(SHAPE[0], 0)], axis=-1)
    off[3, 4] = np.nan
    snr = rng.uniform(1, 20, SHAPE)
    fname = str(tmp_path / "offset.bip")
    off.astype(np.float32).tofile(fname)
    snr.astype(np.float32).tofile(str(tmp_path / "offset_snr.bip"))
    for f, bands in ((fname, 2), (str(tmp_path / "offset_snr.bip"), 1)):
        dense_offsets.write_header(f, SHAPE[1], SHAPE[0], bands, False) if not dense_offsets.isce_available() \
            else pytest.skip("ISCE2 headers")
    return fname, off, snr


@pytest.fixture
def geo(tmp_path):
    """A geo_offsets output GeoTIFF (UTM 11N, 160 m windows)"""
    north = np.zeros(SHAPE)
    north[:, SHAPE[1] // 2:] = 2.0
    bands = dict(east=np.ones(SHAPE), north=north, snr=np.full(SHAPE, 10.0), peak=np.full(SHAPE, 0.5),
                 var_east=np.zeros(SHAPE), var_north=np.zeros(SHAPE), cov_east_north=np.zeros(SHAPE))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32611)
    fname = str(tmp_path / "disp.tif")
    geo_offsets.write_geotiff(fname, bands, (440000.0, 160.0, 0, 3960000.0, 0, -160.0), srs)
    return fname, bands


@pytest.mark.parametrize("name", ["offset.bip", "offset.bip.vrt", "offset.bip.hdr"])
def test_radar_read(radar, name):
    """A dense_offsets output is read by its data file, VRT or ENVI header, with its SNR file"""
    fname, off, snr = radar
    field = plot_offsets.OffsetField(os.path.join(os.path.dirname(fname), name))
    assert not field.georeferenced and field.units == "pixels"
    np.testing.assert_array_equal(field.components[0], off[..., 0].astype(np.float32))
    np.testing.assert_array_equal(field.components[1], off[..., 1].astype(np.float32))
    np.testing.assert_allclose(field.snr, snr.astype(np.float32))
    field.mask(10)
    assert np.isnan(field.components[0][snr < 10]).all()


@pytest.mark.parametrize("mode", [[], ["--vectors"], ["--vectors", "4"], ["--3d"],
                                  ["--min-snr", "5", "--vmax", "1", "--subset", "2", "10", "5", "20"],
                                  ["--pixel-size", "2.3", "14.1", "--center", "zero"]])
def test_radar_plots(radar, tmp_path, mode):
    """The modes write the figure, by default next to the file"""
    fname = radar[0]
    plot_offsets.main([fname, *mode])
    assert os.path.getsize(str(tmp_path / "offset.png")) > 0


def test_geo_plots(geo, tmp_path):
    """A geo_offsets output: the map grid, crops by an area or longitude/latitude, the vectors"""
    fname, bands = geo
    field = plot_offsets.OffsetField(fname)
    assert field.georeferenced and field.names == ("east", "north") and field.units == "m"
    assert field.extent() == [440.0, 440.0 + 0.16 * SHAPE[1], 3960.0 - 0.16 * SHAPE[0], 3960.0]
    # an area of 5 x 4 windows
    rows, cols = field.area(440000 + 160 * 10, 3960000 - 160 * 9, 440000 + 160 * 14, 3960000 - 160 * 4)
    assert (rows, cols) == (slice(4, 9), slice(10, 14))
    field.crop(rows, cols)
    assert field.components[0].shape == (5, 4) and field.gt[0] == 440000 + 160 * 10
    out = str(tmp_path / "geo.png")
    plot_offsets.main([fname, "--bbox", "441000", "3958000", "443000", "3959500", "--vectors", "-o", out])
    plot_offsets.main([fname, "--lonlat", "-117.7", "35.7", "-117.6", "35.78", "-o", out])
    assert os.path.getsize(out) > 0
    with pytest.raises(RuntimeError, match="outside"):
        plot_offsets.main([fname, "--bbox", "0", "0", "1", "1", "-o", out])


def test_errors(radar, tmp_path):
    """No SNR to mask with; an area of a file without georeferencing"""
    fname = radar[0]
    os.remove(str(tmp_path / "offset_snr.bip"))
    with pytest.raises(RuntimeError, match="No SNR"):
        plot_offsets.main([fname, "--min-snr", "5"])
    with pytest.raises(RuntimeError, match="georeferenced"):
        plot_offsets.main([fname, "--bbox", "0", "0", "1", "1"])
