#!/usr/bin/env python3
"""
Gross offsets from a velocity model, and velocities from the offsets, for pycuampcor.dense_offsets
on SAR images in radar geometry (e.g., a NISAR RSLC and a coregistered secondary SLC)

The geometry is given by the longitude and latitude of each pixel of the reference image (e.g., x.rdr
and y.rdr of isce3 rdr2geo, or lon.rdr.full and lat.rdr.full of ISCE2), and the velocity model by its
two horizontal components (e.g., vx and vy of an ITS_LIVE mosaic, in m/yr, on a projected grid).
The windows are on the grid of dense_offsets: the first window at (--start), every (--skip) pixels,
(--number) windows, of (--window) pixels (all down, across); the offsets are taken at their centers.

At each window center, the local mapping of the pixel offsets (down, across) to displacements in the
projected coordinates of the model (x, y) is computed from the longitude and latitude of the pixels
(--step pixels apart); it converts the displacements of the model into offsets (gross), and the
offsets into velocities (velocity).

  gross:    the gross offsets (int32 pairs, down and across, per window) for dense_offsets --gross 1
            --gross-file, from the model and the time between the images (--days); clipped so that
            the search areas (--search) are in the image
  velocity: the velocities (in the units of the model coordinates per day) of the offsets of
            dense_offsets; with the model, their difference, and maps of the speeds

usage:
  velocity_model.py gross --lon x.rdr --lat y.rdr --vx 'NETCDF:"v.nc":vx' --vy 'NETCDF:"v.nc":vy' --days 12 \\
      --start 250 450 --skip 64 128 --window 64 128 --number 870 390 --search 16 32 -o gross.bin
  velocity_model.py velocity offset.bip --lon x.rdr --lat y.rdr --days 12 --start 250 450 --skip 64 128 \\
      --window 64 128 --number 870 390 [--vx ... --vy ...] [--min-snr 5] -o velocity
needs: numpy, GDAL (osgeo); matplotlib for the figures
"""

import argparse
import os

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()


def add_grid_options(parser):
    """The options of the geometry and of the window grid"""
    parser.add_argument('--lon', required=True, help='the longitude of the pixels of the reference image (GDAL raster)')
    parser.add_argument('--lat', required=True, help='the latitude of the pixels of the reference image (GDAL raster)')
    parser.add_argument('--days', type=float, required=True, help='the time between the images, in days')
    parser.add_argument('--start', type=int, nargs=2, required=True, metavar=('DOWN', 'ACROSS'),
                        help='the first pixel of the first window')
    parser.add_argument('--skip', type=int, nargs=2, required=True, metavar=('DOWN', 'ACROSS'),
                        help='the step between the windows')
    parser.add_argument('--window', type=int, nargs=2, required=True, metavar=('DOWN', 'ACROSS'),
                        help='the size of the windows')
    parser.add_argument('--number', type=int, nargs=2, required=True, metavar=('DOWN', 'ACROSS'),
                        help='the number of windows')
    parser.add_argument('--step', type=int, default=16,
                        help='the pixel distance of the differences for the local mapping (default: %(default)s)')
    parser.add_argument('--vx', help='the x component of the velocity model (GDAL raster, per year)')
    parser.add_argument('--vy', help='the y component of the velocity model (GDAL raster, per year)')


class Geometry:
    """The window centers: their coordinates in the projection of the model, and the local mapping"""

    def __init__(self, args, crs):
        """The geometry of the windows of args, in the coordinate system crs (of the model)"""
        lon_ds, lat_ds = gdal.Open(args.lon), gdal.Open(args.lat)
        self.length, self.width = lon_ds.RasterYSize, lon_ds.RasterXSize
        rows = args.start[0] + np.arange(args.number[0]) * args.skip[0] + args.window[0] // 2
        cols = args.start[1] + np.arange(args.number[1]) * args.skip[1] + args.window[1] // 2
        h = args.step
        if rows[-1] + h >= self.length or cols[-1] + h >= self.width:
            raise RuntimeError('The windows (and --step) exceed the geometry rasters')
        wgs = osr.SpatialReference()
        wgs.ImportFromEPSG(4326)
        for s in (wgs, crs):
            s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        tr = osr.CoordinateTransformation(wgs, crs)

        def xy(row):
            """The projected coordinates of the window centers of a row, and of their neighbors across"""
            lon = lon_ds.GetRasterBand(1).ReadAsArray(0, int(row), self.width, 1)[0]
            lat = lat_ds.GetRasterBand(1).ReadAsArray(0, int(row), self.width, 1)[0]
            idx = np.concatenate([cols, cols + h])
            pts = np.array(tr.TransformPoints(np.stack([lon[idx], lat[idx]], axis=1).astype(float)))
            return pts[:len(cols), :2], pts[len(cols):, :2]

        n = len(cols)
        self.x = np.empty((len(rows), n))
        self.y = np.empty((len(rows), n))
        # the mapping: d(x, y) = J (d down, d across), J = [[dx/ddown, dx/dacross], [dy/ddown, dy/dacross]]
        self.jacobian = np.empty((len(rows), n, 2, 2))
        for i, r in enumerate(rows):
            center, across = xy(r)
            down, _ = xy(r + h)
            self.x[i], self.y[i] = center[:, 0], center[:, 1]
            self.jacobian[i, :, :, 0] = (down - center) / h
            self.jacobian[i, :, :, 1] = (across - center) / h

    def to_offsets(self, dx, dy):
        """The pixel offsets (down, across) of the displacements (dx, dy)"""
        d = np.stack([dx, dy], axis=-1)[..., None]
        with np.errstate(invalid='ignore'):
            off = np.linalg.solve(self.jacobian, d)[..., 0]
        return off[..., 0], off[..., 1]

    def to_displacements(self, down, across):
        """The displacements (dx, dy) of the pixel offsets (down, across)"""
        d = self.jacobian @ np.stack([down, across], axis=-1)[..., None]
        return d[..., 0, 0], d[..., 1, 0]


def model_crs(args):
    """The coordinate system of the velocity model"""
    return gdal.Open(args.vx).GetSpatialRef()


def sample_model(name, x, y):
    """The model raster at the coordinates (x, y) (bilinear; NaN outside or no data), per year"""
    ds = gdal.Open(name)
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    # the area of the points
    col = (x - gt[0]) / gt[1] - 0.5
    row = (y - gt[3]) / gt[5] - 0.5
    ok = np.isfinite(col) & np.isfinite(row)
    c0 = max(int(np.floor(col[ok].min())), 0)
    r0 = max(int(np.floor(row[ok].min())), 0)
    c1 = min(int(np.ceil(col[ok].max())) + 2, ds.RasterXSize)
    r1 = min(int(np.ceil(row[ok].max())) + 2, ds.RasterYSize)
    data = band.ReadAsArray(c0, r0, c1 - c0, r1 - r0).astype(np.float64)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        data[data == nodata] = np.nan
    data = data * (band.GetScale() or 1) + (band.GetOffset() or 0)
    # bilinear interpolation (NaN if a neighbor is missing or outside)
    c, r = col - c0, row - r0
    ci, ri = np.floor(c).astype(int), np.floor(r).astype(int)
    inside = ok & (ci >= 0) & (ri >= 0) & (ci < data.shape[1] - 1) & (ri < data.shape[0] - 1)
    ci, ri = np.where(inside, ci, 0), np.where(inside, ri, 0)
    fc, fr = c - ci, r - ri
    v = (data[ri, ci] * (1 - fc) * (1 - fr) + data[ri, ci + 1] * fc * (1 - fr)
         + data[ri + 1, ci] * (1 - fc) * fr + data[ri + 1, ci + 1] * fc * fr)
    return np.where(inside, v, np.nan)


def gross(args):
    """Write the gross offsets of the velocity model"""
    geom = Geometry(args, model_crs(args))
    years = args.days / 365.25
    vx, vy = sample_model(args.vx, geom.x, geom.y), sample_model(args.vy, geom.x, geom.y)
    down, across = geom.to_offsets(vx * years, vy * years)
    # no model: no gross offset
    down, across = np.nan_to_num(down), np.nan_to_num(across)
    g = np.rint(np.stack([down, across], axis=-1)).astype(np.int32)
    # clip so that the search areas are in the image
    for k, n in ((0, geom.length), (1, geom.width)):
        start = args.start[k] + np.arange(args.number[k]) * args.skip[k]
        start = start[:, None] if k == 0 else start[None, :]
        lo = args.search[k] - start
        hi = n - args.window[k] - args.search[k] - start
        clipped = np.clip(g[..., k], lo, hi)
        if (clipped != g[..., k]).any():
            print(f'{(clipped != g[..., k]).sum()} gross offsets {("down", "across")[k]} clipped to the image')
        g[..., k] = clipped
    g.tofile(args.output)
    print(f'gross offsets: {args.output} ({g.shape[0]} x {g.shape[1]} windows; model at {np.isfinite(vx).mean():.0%})')
    for k, name in enumerate(('down', 'across')):
        print(f'  {name}: {g[..., k].min()} to {g[..., k].max()} pixels, median |.| {np.median(np.abs(g[..., k])):.0f}')


def velocity(args):
    """Write the velocities of the offsets (and their difference with the model), with maps"""
    crs = model_crs(args) if args.vx else None
    if crs is None:
        if not args.epsg:
            raise RuntimeError('Give the velocity model (--vx, --vy) or the coordinate system (--epsg)')
        crs = osr.SpatialReference()
        crs.ImportFromEPSG(args.epsg)
    geom = Geometry(args, crs)
    shape = tuple(args.number)
    ds = gdal.Open(args.offsets + '.vrt' if os.path.exists(args.offsets + '.vrt') else args.offsets)
    down, across = (ds.GetRasterBand(k + 1).ReadAsArray().astype(np.float64) for k in (0, 1))
    if down.shape != shape:
        raise RuntimeError(f'The offsets are {down.shape[0]} x {down.shape[1]} windows, not {shape}')
    snr = None
    stem, ext = os.path.splitext(args.offsets)
    if os.path.exists(f'{stem}_snr{ext}'):
        snr_ds = gdal.Open(f'{stem}_snr{ext}.vrt' if os.path.exists(f'{stem}_snr{ext}.vrt') else f'{stem}_snr{ext}')
        snr = snr_ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    dx, dy = geom.to_displacements(down, across)
    vx, vy = dx / args.days, dy / args.days
    if snr is not None and args.min_snr:
        vx[snr < args.min_snr] = np.nan
        vy[snr < args.min_snr] = np.nan
    bands = {'vx': vx, 'vy': vy}
    if args.vx:
        mx = sample_model(args.vx, geom.x, geom.y) / 365.25
        my = sample_model(args.vy, geom.x, geom.y) / 365.25
        bands.update(model_vx=mx, model_vy=my)
        diff = np.hypot(vx - mx, vy - my)
        speed = np.hypot(mx, my)
        ok = np.isfinite(diff)
        print(f'velocities: {ok.mean():.0%} of the windows; |v - model| median {np.median(diff[ok]):.2f}, '
              f'90th percentile {np.percentile(diff[ok], 90):.2f} (per day)')
        for lo, hi in ((0, 1), (1, 5), (5, 20), (20, 1e9)):
            sel = ok & (speed >= lo) & (speed < hi)
            if sel.any():
                print(f'  model speed {lo}-{hi if hi < 1e9 else "":>2} per day: {sel.sum():7d} windows, '
                      f'|v - model| median {np.median(diff[sel]):.2f}, speed ratio median '
                      f'{np.median(np.hypot(vx, vy)[sel] / speed[sel]):.3f}')
    # the velocities, as a multi-band raster in radar geometry (one pixel per window)
    out = gdal.GetDriverByName('GTiff').Create(f'{args.output}.tif', shape[1], shape[0], len(bands), gdal.GDT_Float32,
                                               ['COMPRESS=DEFLATE'])
    for k, (name, data) in enumerate(bands.items()):
        b = out.GetRasterBand(k + 1)
        b.SetDescription(name)
        b.SetNoDataValue(float('nan'))
        b.WriteArray(data.astype(np.float32))
    out = None
    print(f'velocities: {args.output}.tif (bands: {", ".join(bands)}; radar geometry, one pixel per window)')
    if not args.no_plot:
        plot(args, geom, bands)


def plot(args, geom, bands):
    """Maps (in the model coordinates) of the speed, of the model speed and of their difference"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    speed = np.hypot(bands['vx'], bands['vy'])
    panels = [('speed', speed)]
    if 'model_vx' in bands:
        model = np.hypot(bands['model_vx'], bands['model_vy'])
        panels += [('model speed', model), ('|v - model|', np.hypot(bands['vx'] - bands['model_vx'],
                                                                    bands['vy'] - bands['model_vy']))]
    vmax = args.vmax or np.nanpercentile(speed, 99)
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 7), constrained_layout=True, squeeze=False)
    for ax, (name, data) in zip(axes[0], panels):
        ok = np.isfinite(data)
        difference = name.startswith('|')
        im = ax.scatter(geom.x[ok] / 1e3, geom.y[ok] / 1e3, c=data[ok], s=0.5, cmap='magma' if difference else 'viridis',
                        vmin=0, vmax=vmax / 5 if difference else vmax, rasterized=True)
        ax.set_aspect('equal')
        ax.set_title(f'{name} (per day)')
        ax.set_xlabel('x (km)')
        fig.colorbar(im, ax=ax, shrink=0.7)
    axes[0][0].set_ylabel('y (km)')
    fig.suptitle(args.title or args.offsets)
    fig.savefig(f'{args.output}.png', dpi=90)
    print(f'figure: {args.output}.png')


def main():
    """The gross and velocity commands"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    g = sub.add_parser('gross', help='the gross offsets of a velocity model')
    add_grid_options(g)
    g.add_argument('--search', type=int, nargs=2, required=True, metavar=('DOWN', 'ACROSS'),
                   help='the half search range of dense_offsets (to keep the search areas in the image)')
    g.add_argument('-o', '--output', required=True, help='the gross offset file (int32 pairs, down and across)')
    v = sub.add_parser('velocity', help='the velocities of the offsets')
    v.add_argument('offsets', help='the offsets of dense_offsets (a two-band .bip, down and across)')
    add_grid_options(v)
    v.add_argument('--epsg', type=int, default=None, help='the projection of the velocities without a model')
    v.add_argument('--min-snr', type=float, default=None, help='mask the windows of a lower SNR (<offsets>_snr.bip)')
    v.add_argument('--vmax', type=float, default=None, help='the color scale of the speeds (default: 99th percentile)')
    v.add_argument('--title', default=None, help='the title of the figure')
    v.add_argument('--no-plot', action='store_true', help='no figure')
    v.add_argument('-o', '--output', required=True, help='the output prefix (<prefix>.tif, <prefix>.png)')
    args = parser.parse_args()
    if args.command == 'gross':
        if not (args.vx and args.vy):
            parser.error('gross needs the velocity model (--vx, --vy)')
        gross(args)
    else:
        velocity(args)


if __name__ == '__main__':
    main()
