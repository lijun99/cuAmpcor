#!/usr/bin/env python3
"""
Maps of the east and north displacements and the SNR of a pycuampcor.geo_offsets output

usage:
  plot_displacements.py ridgecrest.tif --lonlat -117.85 35.50 -117.30 35.95 --vmax 3
needs: numpy, matplotlib, GDAL (osgeo)
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from osgeo import gdal, osr

gdal.UseExceptions()


def read(fname):
    """The bands (by their names) and the geotransform and coordinate system of the output"""
    ds = gdal.Open(fname)
    bands = {ds.GetRasterBand(k + 1).GetDescription(): ds.GetRasterBand(k + 1).ReadAsArray()
             for k in range(ds.RasterCount)}
    return bands, ds.GetGeoTransform(), ds.GetSpatialRef()


def main():
    """Plot the maps to <file>.png (or --output)"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('file', help='the output GeoTIFF of geo_offsets')
    parser.add_argument('--lonlat', type=float, nargs=4, metavar=('LON0', 'LAT0', 'LON1', 'LAT1'),
                        help='the area to plot, in longitude/latitude (default: all)')
    parser.add_argument('--vmax', type=float, default=3, help='color scale of the displacements (default: +/- %(default)s)')
    parser.add_argument('--title', default=None, help='the title of the figure')
    parser.add_argument('--output', default=None, help='the figure (default: <file>.png)')
    args = parser.parse_args()

    bands, gt, crs = read(args.file)
    rows, cols = slice(None), slice(None)
    if args.lonlat:
        # the area in the coordinates of the output
        wgs = osr.SpatialReference()
        wgs.ImportFromEPSG(4326)
        wgs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        tr = osr.CoordinateTransformation(wgs, crs)
        corners = [tr.TransformPoint(lon, lat)[:2] for lon in args.lonlat[0::2] for lat in args.lonlat[1::2]]
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        c0, c1 = sorted(int((v - gt[0]) / gt[1]) for v in (min(xs), max(xs)))
        r0, r1 = sorted(int((v - gt[3]) / gt[5]) for v in (min(ys), max(ys)))
        n_rows, n_cols = bands['east'].shape
        rows, cols = slice(max(r0, 0), min(r1, n_rows)), slice(max(c0, 0), min(c1, n_cols))
    r = range(bands['east'].shape[0])[rows]
    c = range(bands['east'].shape[1])[cols]
    extent = [(gt[0] + gt[1] * c.start) / 1e3, (gt[0] + gt[1] * c.stop) / 1e3,
              (gt[3] + gt[5] * r.stop) / 1e3, (gt[3] + gt[5] * r.start) / 1e3]
    print(f'median east/north: {np.nanmedian(bands["east"]):.3f}, {np.nanmedian(bands["north"]):.3f}; '
          f'median SNR {np.nanmedian(bands["snr"]):.1f}')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax, name in zip(axes, ('east', 'north', 'snr')):
        data = bands[name][rows, cols]
        if name == 'snr':
            im = ax.imshow(data, extent=extent, cmap='viridis', vmin=0, vmax=np.nanpercentile(data, 98))
            ax.set_title('SNR')
        else:
            im = ax.imshow(data, extent=extent, cmap='RdBu_r', vmin=-args.vmax, vmax=args.vmax)
            ax.set_title(f'{name} displacement')
        ax.set_xlabel('easting (km)')
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes[0].set_ylabel('northing (km)')
    fig.suptitle(args.title or args.file)
    output = args.output or args.file.rsplit('.', 1)[0] + '.png'
    fig.savefig(output, dpi=80)
    print(f'figure: {output}')


if __name__ == '__main__':
    main()
