"""
pycuampcor.plot_offsets: a first look at an offset field

Plots the outputs of pycuampcor.dense_offsets (the two-band offset file, in pixels: down/azimuth
and across/range; with its SNR file <prefix>_snr.bip if found) or of pycuampcor.geo_offsets (the
GeoTIFF of the east/north displacements and SNR, on map coordinates): maps of the two components
and the SNR (default), the magnitude with the vectors (--vectors), or 3D surfaces (--3d). The
color scales are centered on the medians, with a robust range unless given (--vmax).

The figure is shown if a display is available, or saved to <file>.png (or --output).
GDAL (the Python bindings) and matplotlib are required.
"""

import argparse
import glob
import os
import sys

import numpy as np

EXAMPLE = '''example:
  python -m pycuampcor.plot_offsets offsets/offset.bip                 # dense_offsets output (+ _snr.bip)
  python -m pycuampcor.plot_offsets offsets/offset.bip --pixel-size 2.3 14.1 --min-snr 5 --vectors
  python -m pycuampcor.plot_offsets ridgecrest.tif --lonlat -117.85 35.5 -117.3 35.95 --vmax 3 -o ridgecrest.png
'''


def create_parser():
    """The command-line parser"""
    parser = argparse.ArgumentParser(prog='python -m pycuampcor.plot_offsets',
                                     description='Plot an offset field (dense_offsets or geo_offsets output)',
                                     formatter_class=argparse.RawTextHelpFormatter, epilog=EXAMPLE)
    parser.add_argument('file', help='the offset file: a dense_offsets output (.bip, .vrt, .hdr; GDAL-readable)\n'
                                     'or a geo_offsets GeoTIFF')
    parser.add_argument('--snr', default=None,
                        help='the SNR file of a dense_offsets output (default: <prefix>_snr.bip, if found)')
    parser.add_argument('--min-snr', type=float, dest='min_snr', default=None,
                        help='mask the windows with an SNR below this (default: none)')
    parser.add_argument('--vmax', type=float, default=None,
                        help='the half range of the color scales around the medians (default: 3 times the\n'
                             'median absolute deviation)')
    parser.add_argument('--center', choices=['median', 'zero'], default='median',
                        help='the center of the color scales (default: %(default)s)')
    parser.add_argument('--pixel-size', type=float, nargs=2, dest='pixel_size', metavar=('DOWN', 'ACROSS'),
                        default=None, help='scale the offsets in pixels (dense_offsets) to these units, e.g., meters')
    parser.add_argument('--subset', type=int, nargs=4, metavar=('ROW0', 'ROW1', 'COL0', 'COL1'), default=None,
                        help='plot these rows and columns of the offset field')
    parser.add_argument('--bbox', type=float, nargs=4, metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'), default=None,
                        help='plot this area, in the coordinates of a georeferenced file')
    parser.add_argument('--lonlat', type=float, nargs=4, metavar=('LON0', 'LAT0', 'LON1', 'LAT1'), default=None,
                        help='plot this area, in longitude/latitude, of a georeferenced file')
    parser.add_argument('--vectors', nargs='?', type=int, const=0, default=None, metavar='STEP',
                        help='plot the magnitude with the vectors, every STEP windows (default: about 30 across)')
    parser.add_argument('--3d', dest='surface', action='store_true', help='plot the two components as 3D surfaces')
    parser.add_argument('--cmap', default=None, help='the colormap of the components (default: RdBu_r)')
    parser.add_argument('--title', default=None, help='the title of the figure (default: the file name)')
    parser.add_argument('-o', '--output', default=None,
                        help='save the figure to this file (default: show it, or save to <file>.png without a display)')
    return parser


def gdal_module():
    """The GDAL Python bindings, with exceptions"""
    try:
        from osgeo import gdal
    except ImportError:
        raise RuntimeError('plot_offsets requires GDAL (the Python bindings, osgeo)') from None
    gdal.UseExceptions()
    return gdal


def open_raster(fname):
    """Open an offset file with GDAL: the .vrt next to it if any, the data file of an ENVI .hdr"""
    gdal = gdal_module()
    if fname.endswith('.hdr'):
        stem = fname[:-len('.hdr')]
        candidates = [stem] if os.path.isfile(stem) else [
            f for f in glob.glob(glob.escape(stem) + '.*') if not f.endswith(('.hdr', '.aux.xml', '.vrt', '.xml'))]
        if len(candidates) != 1:
            raise RuntimeError(f'Could not find a unique data file for the ENVI header {fname}')
        fname = candidates[0]
    elif not fname.endswith('.vrt') and os.path.isfile(fname + '.vrt'):
        fname = fname + '.vrt'
    return gdal.Open(fname)


def data_file(fname):
    """The data file of an offset file name (without .vrt, .hdr or .xml)"""
    for ext in ('.vrt', '.hdr', '.xml'):
        if fname.endswith(ext):
            return fname[:-len(ext)]
    return fname


class OffsetField:
    """The two components of an offset field, its SNR (or None), and its grid"""

    def __init__(self, fname, snr=None, pixel_size=None):
        """Read a dense_offsets output (with its SNR file snr, or <prefix>_snr.bip if found) or a geo_offsets GeoTIFF"""
        ds = open_raster(fname)
        names = [ds.GetRasterBand(k + 1).GetDescription() for k in range(ds.RasterCount)]
        bands = {name: ds.GetRasterBand(k + 1).ReadAsArray().astype(np.float64) for k, name in enumerate(names) if name}
        gt = ds.GetGeoTransform(can_return_null=True)
        self.georeferenced = 'east' in bands and 'north' in bands
        if self.georeferenced:
            # geo_offsets: east, north, snr on map coordinates
            self.names = ('east', 'north')
            self.components = [bands['east'], bands['north']]
            self.snr = bands.get('snr')
            self.units = 'm' if ds.GetSpatialRef() is None or ds.GetSpatialRef().IsProjected() else 'map units'
            self.gt, self.crs = gt, ds.GetSpatialRef()
        else:
            # dense_offsets: down (azimuth), across (range) in pixels; the SNR file next to it
            if ds.RasterCount < 2:
                raise RuntimeError(f'{fname} has {ds.RasterCount} band; expect the two bands of an offset field')
            self.names = ('down (azimuth)', 'across (range)')
            self.components = [ds.GetRasterBand(k + 1).ReadAsArray().astype(np.float64) for k in (0, 1)]
            self.units = 'pixels'
            if pixel_size is not None:
                self.components = [c * s for c, s in zip(self.components, pixel_size)]
                self.units = 'scaled'
            if snr is None:
                base = data_file(fname)
                stem, ext = os.path.splitext(base)
                candidate = f'{stem}_snr{ext}'
                snr = candidate if os.path.isfile(candidate) else None
            self.snr = None
            if snr:
                snr_ds = open_raster(snr)   # kept while its band is read
                self.snr = snr_ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
            self.gt, self.crs = None, None
        # the invalid values of dense_offsets (e.g., no-data windows) as NaN
        for c in self.components:
            c[~np.isfinite(c)] = np.nan

    def mask(self, min_snr):
        """Set the windows with an SNR below min_snr to NaN"""
        if self.snr is None:
            raise RuntimeError('No SNR to mask with: give --snr')
        low = ~(self.snr >= min_snr)
        for c in self.components:
            c[low] = np.nan

    def crop(self, rows, cols):
        """Keep the rows and columns (slices) of the field"""
        self.components = [c[rows, cols] for c in self.components]
        if self.snr is not None:
            self.snr = self.snr[rows, cols]
        if self.gt is not None:
            r0, c0 = rows.start or 0, cols.start or 0
            gt = self.gt
            self.gt = (gt[0] + c0 * gt[1], gt[1], gt[2], gt[3] + r0 * gt[5], gt[4], gt[5])

    def area(self, xmin, ymin, xmax, ymax):
        """The rows and columns (slices) of an area in the coordinates of a georeferenced field"""
        gt = self.gt
        c0, c1 = sorted(((xmin - gt[0]) / gt[1], (xmax - gt[0]) / gt[1]))
        r0, r1 = sorted(((ymin - gt[3]) / gt[5], (ymax - gt[3]) / gt[5]))
        n_rows, n_cols = self.components[0].shape
        rows = slice(max(int(np.floor(r0)), 0), min(int(np.ceil(r1)), n_rows))
        cols = slice(max(int(np.floor(c0)), 0), min(int(np.ceil(c1)), n_cols))
        if rows.start >= rows.stop or cols.start >= cols.stop:
            raise RuntimeError('The area is outside of the offset field')
        return rows, cols

    def lonlat_area(self, lon0, lat0, lon1, lat1):
        """The rows and columns (slices) of an area in longitude/latitude"""
        from osgeo import osr
        wgs = osr.SpatialReference()
        wgs.ImportFromEPSG(4326)
        wgs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        crs = self.crs.Clone()
        crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        tr = osr.CoordinateTransformation(wgs, crs)
        pts = [tr.TransformPoint(lon, lat)[:2] for lon in (lon0, lon1) for lat in (lat0, lat1)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return self.area(min(xs), min(ys), max(xs), max(ys))

    def extent(self):
        """The extent for imshow: in km of the map coordinates, or in rows/columns of the field"""
        n_rows, n_cols = self.components[0].shape
        if self.gt is None:
            return None
        gt, k = self.gt, 1e-3 if self.units == 'm' else 1
        return [gt[0] * k, (gt[0] + gt[1] * n_cols) * k, (gt[3] + gt[5] * n_rows) * k, gt[3] * k]


def robust_spread(data):
    """
    A robust half range of the data around their median: 3 times the median absolute deviation
    (scaled to the standard deviation; about 2 standard deviations), not inflated by outliers
    (decorrelated windows); the largest deviation if most values are equal, or 1 if none
    """
    dev = np.abs(data - np.nanmedian(data))
    dev = dev[np.isfinite(dev)]
    if dev.size == 0:
        return 1.0
    spread = 3 * 1.4826 * np.median(dev)
    if spread <= 0:
        spread = dev.max()
    return spread if spread > 0 else 1.0


def color_range(data, center, vmax):
    """The color range of a component: around its median (or zero), of a half range vmax or robust"""
    c = np.nanmedian(data) if center == 'median' else 0.0
    if vmax is None:
        vmax = robust_spread(data) + abs(np.nanmedian(data) - c)
    return c - vmax, c + vmax


def plot(field, args):
    """Plot the field with the options args, and return the figure"""
    import matplotlib.pyplot as plt
    extent = field.extent()
    xlabel, ylabel = ('easting (km)', 'northing (km)') if extent and field.units == 'm' else \
        (('x', 'y') if extent else ('across (window)', 'down (window)'))
    cmap = args.cmap or 'RdBu_r'
    title = args.title or os.path.basename(args.file)

    if args.surface:
        fig = plt.figure(figsize=(14, 7))
        n_rows, n_cols = field.components[0].shape
        y, x = np.mgrid[0:n_rows, 0:n_cols]
        for k, (name, data) in enumerate(zip(field.names, field.components)):
            lo, hi = color_range(data, args.center, args.vmax)
            ax = fig.add_subplot(1, 2, k + 1, projection='3d')
            shown = np.where((data < lo) | (data > hi), np.nan, data)
            surf = ax.plot_surface(x, y, shown, cmap=cmap, vmin=lo, vmax=hi, edgecolor='none')
            ax.set_zlim(lo, hi)
            ax.set_box_aspect((n_cols / max(n_rows, n_cols), n_rows / max(n_rows, n_cols), 0.5))
            ax.invert_yaxis()
            ax.set_title(f'{name} ({field.units})')
            fig.colorbar(surf, ax=ax, shrink=0.5)
        fig.suptitle(title)
        return fig

    if args.vectors is not None:
        first, second = field.components
        magnitude = np.hypot(first, second)
        fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
        hi = np.nanmedian(magnitude) + robust_spread(magnitude) if args.vmax is None else args.vmax
        im = ax.imshow(magnitude, extent=extent, cmap=args.cmap or 'viridis', vmin=0, vmax=hi,
                       interpolation='nearest')
        fig.colorbar(im, ax=ax, shrink=0.8, label=f'magnitude ({field.units})')
        n_rows, n_cols = magnitude.shape
        step = args.vectors or max(1, n_cols // 30)
        rows, cols = np.mgrid[step // 2:n_rows:step, step // 2:n_cols:step]
        if extent:
            # east/north on map coordinates (y up)
            xs = extent[0] + (cols + 0.5) * (extent[1] - extent[0]) / n_cols
            ys = extent[3] - (rows + 0.5) * (extent[3] - extent[2]) / n_rows
            u, v = first[rows, cols], second[rows, cols]
            spacing = step * (extent[1] - extent[0]) / n_cols
        else:
            # across/down on the rows and columns (the rows down)
            xs, ys = cols, rows
            u, v = second[rows, cols], first[rows, cols]
            spacing = step
        # an arrow of the magnitude hi spans the spacing of the arrows; the arrows of the outliers
        # (longer than twice that) are not drawn
        long = ~(np.hypot(u, v) <= 2 * hi)
        u, v = np.where(long, np.nan, u), np.where(long, np.nan, v)
        ax.quiver(xs, ys, u, v, color='r', angles='xy', scale_units='xy', scale=hi / spacing)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return fig

    panels = list(zip(field.names, field.components)) + ([('SNR', field.snr)] if field.snr is not None else [])
    # the panels 6 inches wide, as high as the field (within limits), with room for the titles
    n_rows, n_cols = field.components[0].shape
    aspect = abs((extent[3] - extent[2]) / (extent[1] - extent[0])) if extent else n_rows / n_cols
    height = min(max(6 * aspect, 2.5), 9) + 1
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), height), constrained_layout=True,
                             squeeze=False)
    for ax, (name, data) in zip(axes[0], panels):
        if name == 'SNR':
            hi = np.nanpercentile(data, 98) if np.isfinite(data).any() else 1
            im = ax.imshow(data, extent=extent, cmap='viridis', vmin=0, vmax=hi, interpolation='nearest')
            ax.set_title('SNR')
        else:
            lo, hi = color_range(data, args.center, args.vmax)
            im = ax.imshow(data, extent=extent, cmap=cmap, vmin=lo, vmax=hi, interpolation='nearest')
            ax.set_title(f'{name} ({field.units}), median {np.nanmedian(data):.3g}')
        ax.set_xlabel(xlabel)
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes[0][0].set_ylabel(ylabel)
    fig.suptitle(title)
    return fig


def main(iargs=None):
    """Plot the offset field with the command-line options iargs"""
    args = create_parser().parse_args(iargs)
    try:
        import matplotlib
    except ImportError:
        raise RuntimeError('plot_offsets requires matplotlib') from None
    field = OffsetField(args.file, args.snr, args.pixel_size)
    if args.min_snr is not None:
        field.mask(args.min_snr)
    if args.bbox or args.lonlat:
        if not field.georeferenced:
            raise RuntimeError('--bbox and --lonlat need a georeferenced file (geo_offsets); use --subset')
        field.crop(*(field.area(*args.bbox) if args.bbox else field.lonlat_area(*args.lonlat)))
    if args.subset:
        field.crop(slice(args.subset[0], args.subset[1]), slice(args.subset[2], args.subset[3]))

    fig = plot(field, args)
    interactive = matplotlib.get_backend().lower() not in ('agg', 'pdf', 'ps', 'svg', 'cairo', 'template')
    if args.output or not interactive:
        output = args.output or data_file(args.file).rsplit('.', 1)[0] + '.png'
        fig.savefig(output, dpi=100)
        print(f'figure: {output}')
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
