#!/usr/bin/env python3
"""
The strike-slip offset along a fault from a pycuampcor.geo_offsets output

The displacements are projected on the strike of the fault (fault-parallel); in sections along
strike, the profile of the fault-parallel displacement across the fault is computed (medians in
0.5-km bins, windows with SNR above --min-snr), the fault trace is located at its largest step,
and the slip is the difference of the medians on the two sides (--sides, km from the trace).
The slip is left-lateral if positive, right-lateral if negative.

usage:
  fault_slip.py ridgecrest.tif --strike 326 --center 446 3956 --sections -12 12 4 --width 6
  fault_slip.py turkey.tif --strike 62 --center 360 4171.5 --sections -40 50 10
needs: numpy, GDAL (osgeo)
"""

import argparse

import numpy as np
from osgeo import gdal

gdal.UseExceptions()


def main():
    """Print the slip in the sections along strike"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('file', help='the output GeoTIFF of geo_offsets (projected coordinates, in meters)')
    parser.add_argument('--strike', type=float, required=True, help='the strike of the fault, degrees clockwise from north')
    parser.add_argument('--center', type=float, nargs=2, required=True, metavar=('X', 'Y'),
                        help='a point near the trace (km, in the coordinates of the file)')
    parser.add_argument('--sections', type=float, nargs=3, default=(-40, 40, 10), metavar=('FROM', 'TO', 'STEP'),
                        help='the centers of the sections along strike from the point, in km (default: %(default)s)')
    parser.add_argument('--width', type=float, default=10, help='the width of the sections along strike, km (default: %(default)s)')
    parser.add_argument('--search', type=float, default=15, help='the search of the trace across, +/- km (default: %(default)s)')
    parser.add_argument('--sides', type=float, nargs=2, default=(1, 5), metavar=('NEAR', 'FAR'),
                        help='the distances from the trace of the medians, km (default: %(default)s)')
    parser.add_argument('--min-snr', type=float, default=5, help='the smallest SNR of the windows used (default: %(default)s)')
    args = parser.parse_args()

    ds = gdal.Open(args.file)
    gt = ds.GetGeoTransform()
    b = {ds.GetRasterBand(k + 1).GetDescription(): ds.GetRasterBand(k + 1).ReadAsArray() for k in range(ds.RasterCount)}
    x = (gt[0] + gt[1] * (np.arange(ds.RasterXSize) + 0.5)) / 1e3
    y = (gt[3] + gt[5] * (np.arange(ds.RasterYSize) + 0.5)) / 1e3
    X, Y = np.meshgrid(x, y)
    s = np.radians(args.strike)
    u = np.array([np.sin(s), np.cos(s)])        # along strike
    n = np.array([np.cos(s), -np.sin(s)])       # across, to the right of the strike
    along = (X - args.center[0]) * u[0] + (Y - args.center[1]) * u[1]
    across = (X - args.center[0]) * n[0] + (Y - args.center[1]) * n[1]
    parallel = b['east'] * u[0] + b['north'] * u[1]
    ok = np.isfinite(parallel) & (b['snr'] > args.min_snr)

    near, far = args.sides
    edges = np.arange(-args.search, args.search + 0.01, 0.5)
    print('along (km)  trace across (km)  left side (m)  right side (m)  slip (m, + left-lateral)')
    for a0 in np.arange(args.sections[0], args.sections[1] + 1e-6, args.sections[2]):
        sel = ok & (np.abs(along - a0) < args.width / 2)
        prof = np.full(len(edges) - 1, np.nan)
        for k, e in enumerate(edges[:-1]):
            m = sel & (across >= e) & (across < e + 0.5)
            if m.sum() > 20:
                prof[k] = np.median(parallel[m])
        if np.isnan(prof).mean() > 0.3:
            print(f'{a0:+8.1f}    (too few windows)')
            continue
        # the trace: the largest step of the profile over 1 km
        jump = np.abs(prof[2:] - prof[:-2])
        trace = edges[np.nanargmax(jump) + 1] + 0.25
        left = parallel[sel & (across > trace - far) & (across < trace - near)]
        right = parallel[sel & (across > trace + near) & (across < trace + far)]
        print(f'{a0:+8.1f}    {trace:+10.1f}        {np.median(left):+9.2f}      {np.median(right):+9.2f}'
              f'      {np.median(right) - np.median(left):+7.2f}')


if __name__ == '__main__':
    main()
