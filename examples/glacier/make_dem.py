#!/usr/bin/env python3
"""
A DEM for ISCE2 (WGS84 ellipsoidal heights, longitude/latitude grid) from the Copernicus GLO-30 DEM,
e.g., where the SRTM DEM of ISCE2's dem.py is not available (north of 60 N)

The DEM tiles are read from the public AWS archive and stitched by dem_stitcher, with the heights
converted from the EGM2008 geoid to the WGS84 ellipsoid; the DEM is written as a float32 ENVI file.
Add the ISCE2 header afterwards with gdal2isce_xml.py -i <output>.

usage:
  make_dem.py --bounds -53 68.6 -46 69.8 -o dem/glo30.dem.wgs84
needs: dem_stitcher, GDAL
"""

import argparse
import os

import numpy as np
from dem_stitcher import stitch_dem
from osgeo import gdal

gdal.UseExceptions()


def main():
    """Stitch and write the DEM"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--bounds', type=float, nargs=4, required=True, metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'),
                        help='the area of the DEM, in degrees; cover all the bursts processed')
    parser.add_argument('-o', '--output', required=True, help='the DEM file (ENVI, float32)')
    args = parser.parse_args()
    dem, profile = stitch_dem(list(args.bounds), dem_name='glo_30', dst_ellipsoidal_height=True,
                              dst_area_or_point='Point')
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    t = profile['transform']
    ds = gdal.GetDriverByName('ENVI').Create(args.output, dem.shape[1], dem.shape[0], 1, gdal.GDT_Float32)
    ds.SetGeoTransform((t.c, t.a, t.b, t.f, t.d, t.e))
    ds.SetProjection('EPSG:4326')
    ds.GetRasterBand(1).WriteArray(np.nan_to_num(dem).astype(np.float32))
    ds = None
    print(f'DEM: {args.output} ({dem.shape[0]} x {dem.shape[1]}, heights {np.nanmin(dem):.0f} to {np.nanmax(dem):.0f} m)')


if __name__ == '__main__':
    main()
