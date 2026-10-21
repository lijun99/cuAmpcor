"""
pycuampcor.geo_offsets: the displacement field between two georeferenced images with pycuampcor

For images on map grids, e.g., optical images (Sentinel-2, Landsat, Planet) or geocoded SAR
amplitudes, of an earthquake or a glacier: the images are read with GDAL (any format, one band,
any pixel type), and must share the coordinate system and the pixel size, but may have different
extents. The shift between their grids is computed from their georeferencing and used as the
gross offset; the windows cover the area of the reference image where both images are available
(or a part of it, --bbox), on a regular grid.

The output is a GeoTIFF on the grid of the window centers (<prefix>.tif) with the bands:
east and north displacements (in the units of the coordinate system, e.g., meters; the shift of
the grids removed), SNR, correlation peak value, and the variances of the east and north
displacements and their covariance. Windows with no-data pixels (--nodata, by default the
no-data value of the images) are set to NaN.

GDAL (the Python bindings) is required, and pycuampcor built with GDAL support.
"""

import argparse
import os
import sys
import time

import numpy as np

import pycuampcor

EXAMPLE = '''example:
  # Sentinel-2 NIR bands (10 m) before and after an earthquake: 64 x 64 windows every 16 pixels
  python -m pycuampcor.geo_offsets -r T11SMV_20190628_B08.tif -s T11SMV_20190708_B08.tif --op offsets/ridgecrest

  # the third band of 4-band images, in an area (in the coordinates of the images)
  python -m pycuampcor.geo_offsets -r ref.tif -s sec.tif --band 3 --bbox 440000 3940000 480000 3980000 --op out/disp
'''

def create_parser():
    """The command-line parser"""
    parser = argparse.ArgumentParser(prog='python -m pycuampcor.geo_offsets',
                                     description='Estimate the displacement field between two georeferenced images',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)
    # input/output
    parser.add_argument('-r', '--reference', required=True, help='Reference image (a raster read with GDAL).')
    parser.add_argument('-s', '--secondary', required=True, help='Secondary image (a raster read with GDAL).')
    parser.add_argument('--band', type=int, default=1, help='Band of the images (default: %(default)s).')
    parser.add_argument('--sec-band', type=int, dest='sec_band', default=None,
                        help='Band of the secondary image, if different (default: --band).')
    parser.add_argument('--op', '--outprefix', dest='outprefix', required=True,
                        help='Output prefix: the displacements in <prefix>.tif.')
    parser.add_argument('--bbox', type=float, nargs=4, metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'), default=None,
                        help='Area of the window centers, in the coordinates of the images (default: all).')
    parser.add_argument('--nodata', type=float, default=None,
                        help='No-data value of the images (default: from the images, if any).')
    parser.add_argument('--max-nodata', type=float, dest='max_nodata', default=0.0,
                        help='Largest fraction of no-data pixels in a window (and its search area)\n'
                             'for a valid result (default: %(default)s).')
    parser.add_argument('--keep', action='store_true',
                        help='Keep the intermediate files (the raw outputs of ampcor, in pixels).')

    # windows
    win = parser.add_argument_group('Windows')
    win.add_argument('--ww', type=int, dest='winwidth', default=64, help='Window width (default: %(default)s).')
    win.add_argument('--wh', type=int, dest='winhgt', default=64, help='Window height (default: %(default)s).')
    win.add_argument('--sw', type=int, dest='srcwidth', default=8,
                     help='Half search range across (columns) (default: %(default)s).')
    win.add_argument('--sh', type=int, dest='srchgt', default=8,
                     help='Half search range down (rows) (default: %(default)s).')
    win.add_argument('--kw', type=int, dest='skipwidth', default=16,
                     help='Step between windows across (default: %(default)s).')
    win.add_argument('--kh', type=int, dest='skiphgt', default=16,
                     help='Step between windows down (default: %(default)s).')

    # correlation
    corr = parser.add_argument_group('Correlation')
    corr.add_argument('--raw-osf', type=int, dest='raw_oversample', default=2,
                      help='Oversampling factor of the windows (default: %(default)s).')
    corr.add_argument('--oo', type=int, dest='corr_oversample', default=64,
                      help='Oversampling factor of the correlation surface (default: %(default)s).')
    corr.add_argument('--corr-srch-size', type=int, dest='corr_srch_size', default=4,
                      help='Half size of the correlation surface oversampled (default: %(default)s).')
    corr.add_argument('--corr-stat-size', type=int, dest='corr_stat_win_size', default=21,
                      help='Size of the area of the correlation surface for the SNR (default: %(default)s).')

    # processing
    proc = parser.add_argument_group('Processing')
    proc.add_argument('--cpu', action='store_true',
                      help='Use the CPU implementation (also used if pycuampcor has no CUDA support).')
    proc.add_argument('--nthreads', type=int, default=0,
                      help='Number of CPU threads, 0 for the OpenMP default (default: %(default)s).')
    proc.add_argument('--gpuid', type=int, default=0, help='GPU ID (default: %(default)s).')
    proc.add_argument('--nstreams', type=int, default=2, help='Number of CUDA streams (default: %(default)s).')
    proc.add_argument('--nwac', type=int, dest='numWinAcrossInChunk', default=0,
                      help='Number of windows across in a chunk, 0 for automatic (default: %(default)s).')
    proc.add_argument('--nwdc', type=int, dest='numWinDownInChunk', default=0,
                      help='Number of windows down in a chunk, 0 for automatic (default: %(default)s).')
    return parser


def cmd_line_parse(iargs=None):
    """The command-line options"""
    return create_parser().parse_args(args=iargs)


def gdal_module():
    """The GDAL Python bindings, with exceptions"""
    try:
        from osgeo import gdal
    except ImportError:
        raise RuntimeError('geo_offsets requires GDAL (the Python bindings, osgeo)') from None
    gdal.UseExceptions()
    return gdal


class Raster:
    """A band of a georeferenced raster: its size, georeferencing, pixel type and no-data value"""

    def __init__(self, fname, band, nodata=None):
        """Open the band of the raster fname; nodata: the no-data value (default: the band's)"""
        gdal = gdal_module()
        self.fname, self.band = fname, band
        self.ds = gdal.Open(fname, gdal.GA_ReadOnly)
        if not 1 <= band <= self.ds.RasterCount:
            raise RuntimeError(f'{fname} has no band {band} (it has {self.ds.RasterCount})')
        b = self.ds.GetRasterBand(band)
        self.length, self.width = self.ds.RasterYSize, self.ds.RasterXSize
        self.gt = self.ds.GetGeoTransform()
        self.crs = self.ds.GetSpatialRef()
        self.complex = bool(gdal.DataTypeIsComplex(b.DataType))
        self.nodata = nodata if nodata is not None else b.GetNoDataValue()
        if self.crs is None or self.gt == (0, 1, 0, 0, 0, 1):
            raise RuntimeError(f'{fname} is not georeferenced')
        if self.gt[2] != 0 or self.gt[4] != 0:
            raise RuntimeError(f'{fname} is not north-up (rotated geotransform); warp it first (gdalwarp)')

    def source(self, outdir):
        """The name of the band for the GDAL reader of pycuampcor (a VRT of the band if not the first)"""
        if self.band == 1:
            return self.fname
        gdal = gdal_module()
        vrt = os.path.join(outdir, f'{os.path.splitext(os.path.basename(self.fname))[0]}_band{self.band}.vrt')
        gdal.Translate(vrt, self.ds, format='VRT', bandList=[self.band])
        return vrt

    def nodata_mask(self):
        """The no-data pixels of the band (a boolean array), or None if no no-data value"""
        if self.nodata is None:
            return None
        data = self.ds.GetRasterBand(self.band).ReadAsArray()
        return np.isnan(data) if np.isnan(self.nodata) else data == self.nodata


def check_grids(ref, sec):
    """Check that the images share the coordinate system and the pixel size"""
    if not ref.crs.IsSame(sec.crs):
        raise RuntimeError(f'{ref.fname} and {sec.fname} have different coordinate systems; warp one of them '
                           'to the other (gdalwarp -t_srs ... -tr ...)')
    for k, name in ((1, 'x'), (5, 'y')):
        if abs(ref.gt[k] - sec.gt[k]) > 1e-6 * abs(ref.gt[k]):
            raise RuntimeError(f'{ref.fname} and {sec.fname} have different pixel sizes in {name} '
                               f'({ref.gt[k]}, {sec.gt[k]}); resample one of them (gdalwarp -tr ...)')


def grid_shift(ref, sec):
    """
    The shift (down, across) of the secondary grid from the reference grid, in pixels: the pixel
    (i, j) of the reference is at the location of the pixel (i + down, j + across) of the secondary
    """
    return (ref.gt[3] - sec.gt[3]) / ref.gt[5], (ref.gt[0] - sec.gt[0]) / ref.gt[1]


def window_grid(ref, sec, gross, inps):
    """
    The windows on the reference image whose search areas are in the secondary image, with
    their centers in the bounding box if given
    @return the first window (start down, start across) and the numbers of windows (down, across)
    """
    axes = []
    for n_ref, n_sec, win, srch, step, g, origin, res, k in (
            (ref.length, sec.length, inps.winhgt, inps.srchgt, inps.skiphgt, gross[0], ref.gt[3], ref.gt[5], 1),
            (ref.width, sec.width, inps.winwidth, inps.srcwidth, inps.skipwidth, gross[1], ref.gt[0], ref.gt[1], 0)):
        # the first pixels of the windows: in the reference, and the search area in the secondary
        lo = max(0, srch - g)
        hi = min(n_ref - win, n_sec - win - srch - g)
        if inps.bbox is not None:
            # the window centers (start + win/2) in the bounding box
            c0, c1 = (inps.bbox[k] - origin) / res, (inps.bbox[k + 2] - origin) / res
            c0, c1 = min(c0, c1), max(c0, c1)
            lo = max(lo, int(np.ceil(c0 - win / 2)))
            hi = min(hi, int(np.floor(c1 - win / 2)))
        if hi < lo:
            raise RuntimeError('No window fits in the area common to the images (and the bounding box)')
        axes.append((lo, (hi - lo) // step + 1))
    (start_down, n_down), (start_across, n_across) = axes
    return (start_down, start_across), (n_down, n_across)


def window_nodata(mask, start, n, step, size):
    """
    The fraction of no-data pixels in each window of a grid
    @param mask the no-data pixels of the image (or None)
    @param start,n,step,size the first pixel, the number, the step and the size of the windows
        (down, across); the windows may extend beyond the image (counted as no-data)
    """
    if mask is None:
        return np.zeros(n)
    # the integral image, padded for the windows beyond the image
    pad = [(max(0, -start[k]), max(0, start[k] + (n[k] - 1) * step[k] + size[k] - mask.shape[k])) for k in (0, 1)]
    m = np.pad(mask, pad, constant_values=True)
    s = np.zeros((m.shape[0] + 1, m.shape[1] + 1), np.int64)
    s[1:, 1:] = m.cumsum(0).cumsum(1)
    r = start[0] + pad[0][0] + np.arange(n[0]) * step[0]
    c = start[1] + pad[1][0] + np.arange(n[1]) * step[1]
    r0, c0 = r[:, None], c[None, :]
    r1, c1 = r0 + size[0], c0 + size[1]
    return (s[r1, c1] - s[r0, c1] - s[r1, c0] + s[r0, c0]) / (size[0] * size[1])


def write_geotiff(fname, bands, gt, crs):
    """Write the bands (a dict name -> 2D array) as a float32 GeoTIFF with NaN as no-data"""
    gdal = gdal_module()
    first = next(iter(bands.values()))
    ds = gdal.GetDriverByName('GTiff').Create(fname, first.shape[1], first.shape[0], len(bands),
                                              gdal.GDT_Float32, ['COMPRESS=DEFLATE', 'TILED=YES'])
    ds.SetGeoTransform(gt)
    ds.SetSpatialRef(crs)
    for k, (name, data) in enumerate(bands.items()):
        b = ds.GetRasterBand(k + 1)
        b.SetDescription(name)
        b.SetNoDataValue(float('nan'))
        b.WriteArray(data.astype(np.float32))
    ds = None


def estimate_displacements(inps):
    """Estimate the displacement field with the options inps; return the output bands and the grid"""
    gdal_module()
    if not pycuampcor.has_gdal:
        raise RuntimeError('pycuampcor is built without GDAL support')
    ref = Raster(inps.reference, inps.band, inps.nodata)
    sec = Raster(inps.secondary, inps.sec_band or inps.band, inps.nodata)
    check_grids(ref, sec)
    if ref.complex != sec.complex:
        raise RuntimeError('One image is complex, the other real')
    data_type = 'complex64' if ref.complex else 'float32'

    # the shift of the grids, as the gross offset (to the pixel), and the windows
    shift = grid_shift(ref, sec)
    gross = tuple(int(round(s)) for s in shift)
    start, n = window_grid(ref, sec, gross, inps)
    size, step = (inps.winhgt, inps.winwidth), (inps.skiphgt, inps.skipwidth)
    search = (inps.srchgt, inps.srcwidth)
    print(f'image sizes (length x width): {ref.length} x {ref.width}, {sec.length} x {sec.width}')
    print(f'shift of the secondary grid (down, across): ({shift[0]:.3f}, {shift[1]:.3f}) pixels')
    print(f'windows: {n[0]} x {n[1]}, from pixel {start} of the reference, every {step} pixels')

    # the ampcor run, with the intermediate outputs (in pixels) next to the output
    work = f'{inps.outprefix}_ampcor'
    os.makedirs(work, exist_ok=True)
    raw = {name: os.path.join(work, f'{name}.bip') for name in ('offset', 'gross', 'snr', 'cov', 'peak')}
    params = dict(
        referenceImageName=ref.source(work), referenceImageReader='gdal',
        referenceImageHeight=ref.length, referenceImageWidth=ref.width, referenceImageDataType=data_type,
        secondaryImageName=sec.source(work), secondaryImageReader='gdal',
        secondaryImageHeight=sec.length, secondaryImageWidth=sec.width, secondaryImageDataType=data_type,
        windowSizeHeight=size[0], windowSizeWidth=size[1],
        halfSearchRangeDown=search[0], halfSearchRangeAcross=search[1],
        skipSampleDown=step[0], skipSampleAcross=step[1],
        referenceStartPixelDownStatic=start[0], referenceStartPixelAcrossStatic=start[1],
        numberWindowDown=n[0], numberWindowAcross=n[1],
        numberWindowDownInChunk=inps.numWinDownInChunk, numberWindowAcrossInChunk=inps.numWinAcrossInChunk,
        derampMethod='magnitude', rawDataOversamplingFactor=inps.raw_oversample,
        corrSurfaceOverSamplingFactor=inps.corr_oversample,
        corrSurfaceZoomInWindow=2 * inps.corr_srch_size * inps.raw_oversample,
        corrStatWindowSize=inps.corr_stat_win_size, mergeGrossOffset=True,
        offsetImageName=raw['offset'], grossOffsetImageName=raw['gross'], snrImageName=raw['snr'],
        covImageName=raw['cov'], peakValueImageName=raw['peak'],
        grossOffsetDown=gross[0], grossOffsetAcross=gross[1])
    if inps.cpu or not pycuampcor.has_cuda:
        ampcor = pycuampcor.PyCPUAmpcor(nThreads=inps.nthreads, **params)
    else:
        ampcor = pycuampcor.PyCuAmpcor(deviceID=inps.gpuid, nStreams=inps.nstreams, **params)
    ampcor.checkPixelInImageRange()
    ampcor.runAmpcor()

    # the outputs, in pixels
    dtype = np.float64 if ampcor.isDoublePrecision() else np.float32
    read = lambda name, bands: np.fromfile(raw[name], dtype).reshape(n[0], n[1], bands)
    offset, cov = read('offset', 2), read('cov', 3)
    snr, peak = read('snr', 1)[..., 0], read('peak', 1)[..., 0]

    # the displacements (the offsets without the shift of the grids), in the map units
    xres, yres = ref.gt[1], ref.gt[5]
    bands = dict(east=(offset[..., 1] - shift[1]) * xres, north=(offset[..., 0] - shift[0]) * yres,
                 snr=snr, peak=peak,
                 var_east=cov[..., 1] * xres ** 2, var_north=cov[..., 0] * yres ** 2,
                 cov_east_north=cov[..., 2] * xres * yres)
    # invalid covariances (99: a flat correlation surface)
    for name in ('var_east', 'var_north', 'cov_east_north'):
        bands[name][cov[..., 0] >= 99] = np.nan

    # the windows with no-data pixels: the reference windows and the secondary search areas
    nodata = np.maximum(
        window_nodata(ref.nodata_mask(), start, n, step, size),
        window_nodata(sec.nodata_mask(), tuple(start[k] + gross[k] - search[k] for k in (0, 1)), n, step,
                      tuple(size[k] + 2 * search[k] for k in (0, 1))))
    invalid = nodata > inps.max_nodata
    for data in bands.values():
        data[invalid] = np.nan
    print(f'windows with no-data pixels (set to NaN): {invalid.sum()} of {invalid.size}')

    # the grid of the window centers
    x0 = ref.gt[0] + (start[1] + size[1] / 2) * xres
    y0 = ref.gt[3] + (start[0] + size[0] / 2) * yres
    gt = (x0 - step[1] * xres / 2, step[1] * xres, 0.0, y0 - step[0] * yres / 2, 0.0, step[0] * yres)
    if not inps.keep:
        for fname in os.listdir(work):
            os.remove(os.path.join(work, fname))
        os.rmdir(work)
    return bands, gt, ref.crs


def main(iargs=None):
    """Estimate the displacement field with the command-line options iargs"""
    inps = cmd_line_parse(iargs)
    start_time = time.time()
    out_dir = os.path.dirname(inps.outprefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    bands, gt, crs = estimate_displacements(inps)
    write_geotiff(f'{inps.outprefix}.tif', bands, gt, crs)
    print(f'output: {inps.outprefix}.tif (bands: {", ".join(bands)})')
    m, s = divmod(time.time() - start_time, 60)
    print(f'time used: {m:02.0f} mins {s:02.1f} secs.\n')


if __name__ == '__main__':
    main(sys.argv[1:])
