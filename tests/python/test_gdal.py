"""
Images read from rasters with GDAL (GeoTIFF, VRT) give the same results as the same images in raw
binary files, with the pixels converted by GDAL to the type of the image
"""

import os

import numpy as np
import pytest

import pycuampcor
from conftest import configure_ampcor, make_images, run_ampcor

gdal = pytest.importorskip("osgeo.gdal")
gdal.UseExceptions()
pytestmark = pytest.mark.skipif(not pycuampcor.has_gdal, reason="pycuampcor is built without GDAL")

SHAPE = (300, 400)  # not a multiple of the block sizes
SHIFT = (1.3, -2.6)

# GeoTIFF layouts (creation options)
LAYOUTS = {
    "striped": [],
    "tiled": ["TILED=YES", "BLOCKXSIZE=64", "BLOCKYSIZE=32"],
    "deflate": ["TILED=YES", "COMPRESS=DEFLATE"],
}


def read_raw(file, dtype=np.complex64):
    """A raw binary image as an array"""
    return np.fromfile(file, dtype=dtype).reshape(SHAPE)


def write_tif(path, data, gdal_type, options=()):
    """Write data as a single band GeoTIFF of the GDAL type"""
    ds = gdal.GetDriverByName("GTiff").Create(str(path), SHAPE[1], SHAPE[0], 1, gdal_type, list(options))
    ds.GetRasterBand(1).WriteArray(data)
    ds = None
    return str(path)


def write_vrt(path, raw, gdal_type="CFloat32", pixel_size=8):
    """Write a VRT of a raw binary image"""
    with open(path, "w") as f:
        f.write(f'<VRTDataset rasterXSize="{SHAPE[1]}" rasterYSize="{SHAPE[0]}">\n'
                f'  <VRTRasterBand dataType="{gdal_type}" band="1" subClass="VRTRawRasterBand">\n'
                f'    <SourceFilename relativeToVRT="0">{raw}</SourceFilename>\n'
                f'    <ImageOffset>0</ImageOffset><PixelOffset>{pixel_size}</PixelOffset>'
                f'<LineOffset>{pixel_size * SHAPE[1]}</LineOffset>\n'
                '  </VRTRasterBand>\n</VRTDataset>\n')
    return str(path)


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    """A pair of synthetic raw images with a known shift"""
    return make_images(tmp_path_factory.mktemp("images"), SHAPE, SHIFT)


@pytest.fixture(scope="module")
def raw_results(images, tmp_path_factory):
    """Results with the raw binary images, for each implementation"""
    cache = {}

    def get(impl):
        """The results of the implementation impl (computed once)"""
        if impl not in cache:
            cache[impl] = run_ampcor(impl, images, SHAPE, tmp_path_factory.mktemp(f"raw_{impl}"))
        return cache[impl]
    return get


def assert_same(out, expected):
    """The outputs are equal to the expected ones"""
    for name in expected:
        np.testing.assert_array_equal(out[name], expected[name], err_msg=name)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_geotiff_matches_raw(impl, layout, images, raw_results, tmp_path):
    """Complex GeoTIFF images of each layout (read by the auto reader) give the results of the raw images"""
    tifs = tuple(write_tif(tmp_path / f"{k}.tif", read_raw(f), gdal.GDT_CFloat32, LAYOUTS[layout])
                 for k, f in zip(("ref", "sec"), images))
    assert_same(run_ampcor(impl, tifs, SHAPE, tmp_path / "out", nThreads=4), raw_results(impl))


def test_vrt_matches_raw(impl, images, raw_results, tmp_path):
    """The VRTs of the raw images give the results of the raw images"""
    vrts = tuple(write_vrt(tmp_path / f"{k}.vrt", f) for k, f in zip(("ref", "sec"), images))
    assert_same(run_ampcor(impl, vrts, SHAPE, tmp_path / "out"), raw_results(impl))


# rasters of other pixel types: (the numpy type written, the image type, the scale of the values)
CONVERSIONS = {
    "UInt16": (np.uint16, np.float32, 1000),
    "Byte": (np.uint8, np.float32, 50),
    "Float64": (np.float64, np.float32, 1),
    "CInt16": (np.complex64, np.complex64, 1000),  # values rounded to integers
}


@pytest.mark.parametrize("gdal_type", CONVERSIONS)
def test_pixel_conversion(impl, gdal_type, tmp_path):
    """Rasters of other pixel types give the results of the raw images of the same values"""
    dtype, image_type, scale = CONVERSIONS[gdal_type]
    real = image_type == np.float32
    images = make_images(tmp_path / "images", SHAPE, SHIFT, dtype=image_type, bandwidth=0.5 if real else 0.8)
    raws, tifs = [], []
    for k, f in zip(("ref", "sec"), images):
        # the values representable in the raster type
        values = read_raw(f, image_type) * scale
        if gdal_type != "Float64":
            values = np.round(values.real) + 1j * np.round(values.imag) if not real else np.round(values)
        raws.append(str(tmp_path / f"{k}.raw"))
        values.astype(image_type).tofile(raws[-1])
        tifs.append(write_tif(tmp_path / f"{k}.tif", values.astype(dtype), gdal.GetDataTypeByName(gdal_type)))
    params = dict(referenceImageDataType=1, secondaryImageDataType=1) if real else {}
    expected = run_ampcor(impl, tuple(raws), SHAPE, tmp_path / "raw", **params)
    out = run_ampcor(impl, tuple(tifs), SHAPE, tmp_path / "tif", **params)
    assert_same(out, expected)


def test_reader_option(impl, images, raw_results, tmp_path):
    """The gdal reader chosen explicitly for a name without a known extension, and mixed readers"""
    names = [write_tif(tmp_path / f"{k}.img", read_raw(f), gdal.GDT_CFloat32) for k, f in zip(("ref", "sec"), images)]
    out = run_ampcor(impl, tuple(names), SHAPE, tmp_path / "gdal",
                     referenceImageReader="gdal", secondaryImageReader="gdal")
    assert_same(out, raw_results(impl))
    # the reference from GDAL, the secondary from the raw file
    out = run_ampcor(impl, (names[0], images[1]), SHAPE, tmp_path / "mixed", referenceImageReader="gdal")
    assert_same(out, raw_results(impl))


def test_gdal_errors(impl, images, tmp_path):
    """Invalid rasters (missing, wrong size, complex for real or real for complex images) raise errors"""
    ref = read_raw(images[0])
    good = write_tif(tmp_path / "ref.tif", ref, gdal.GDT_CFloat32)
    small = tmp_path / "small.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(small), SHAPE[1] - 1, SHAPE[0], 1, gdal.GDT_CFloat32)
    ds = None
    cases = {
        "missing": (str(tmp_path / "missing.tif"), {}, "open"),
        "wrong size": (str(small), {}, "size"),
        "real for complex": (write_tif(tmp_path / "real.tif", np.abs(ref), gdal.GDT_Float32), {}, "real"),
        "complex for real": (good, dict(referenceImageDataType=1, secondaryImageDataType=1), "complex"),
    }
    for case, (name, params, match) in cases.items():
        ampcor, _ = configure_ampcor(impl, (good, name), SHAPE, tmp_path / "out", **params)
        with pytest.raises(RuntimeError, match=match):
            ampcor.runAmpcor()
