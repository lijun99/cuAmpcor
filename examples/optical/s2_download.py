#!/usr/bin/env python3
"""
Search and download Sentinel-2 L2A bands (cloud-optimized GeoTIFFs) from the public AWS archive

The scenes are searched with the STAC API of Element84 (earth-search, collection sentinel-2-l2a),
no account needed. Without --items, the scenes over a point are listed (date, item id, cloud, snow,
no-data percentages, and the orbit, the same for the scenes of a satellite on the same relative
orbit); with --items, the band (--asset, default nir: B08, 10 m) of these items is downloaded as
<item>_<band>.tif.

usage:
  s2_download.py --lon -117.6 --lat 35.77 --start 2019-06-01 --end 2019-08-31 --max-cloud 2
  s2_download.py --items S2B_11SMV_20190628_1_L2A S2B_11SMV_20190708_1_L2A --outdir data
needs: python 3 (standard library only)
"""

import argparse
import json
import os
import urllib.request

STAC = 'https://earth-search.aws.element84.com/v1'
COLLECTION = 'sentinel-2-l2a'


def stac(path, body=None):
    """A request to the STAC API: GET path, or POST the body (a dict) as JSON"""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f'{STAC}/{path}', data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def search(lon, lat, start, end, max_cloud):
    """The scenes over the point (lon, lat) between the dates, with a cloud cover below max_cloud (%)"""
    body = {'collections': [COLLECTION], 'intersects': {'type': 'Point', 'coordinates': [lon, lat]},
            'datetime': f'{start}T00:00:00Z/{end}T23:59:59Z', 'query': {'eo:cloud_cover': {'lt': max_cloud}},
            'limit': 200}
    return sorted(stac('search', body)['features'], key=lambda f: f['properties']['datetime'])


def orbit(item):
    """The satellite and the absolute orbit modulo 143 (equal for the scenes of a relative orbit)"""
    take = item['properties'].get('s2:datatake_id', '')   # e.g., GS2A_20230110T082321_039443_N05.09
    parts = take.split('_')
    return f'{item["id"][:3]} {int(parts[2]) % 143:3d}' if len(parts) > 2 else '?'


def download(item_id, asset, outdir):
    """Download the asset (band) of the item to <outdir>/<item>_<band>.tif"""
    item = stac(f'collections/{COLLECTION}/items/{item_id}')
    href = item['assets'][asset]['href']
    fname = os.path.join(outdir, f'{item_id}_{os.path.basename(href)}')
    if os.path.exists(fname):
        print(f'{fname} exists')
        return fname
    print(f'downloading {href} -> {fname}')
    urllib.request.urlretrieve(href, fname + '.part')
    os.rename(fname + '.part', fname)
    return fname


def main():
    """List the scenes over a point, or download the band of the items given"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--lon', type=float, help='longitude of the point (search)')
    parser.add_argument('--lat', type=float, help='latitude of the point (search)')
    parser.add_argument('--start', help='first date, YYYY-MM-DD (search)')
    parser.add_argument('--end', help='last date, YYYY-MM-DD (search)')
    parser.add_argument('--max-cloud', type=float, default=10, help='largest cloud cover in %% (default: %(default)s)')
    parser.add_argument('--items', nargs='+', help='the items to download')
    parser.add_argument('--asset', default='nir', help='the band: nir (B08), red (B04), ... (default: %(default)s)')
    parser.add_argument('--outdir', default='.', help='the directory of the downloads (default: %(default)s)')
    args = parser.parse_args()

    if args.items:
        os.makedirs(args.outdir, exist_ok=True)
        for item_id in args.items:
            download(item_id, args.asset, args.outdir)
        return
    if None in (args.lon, args.lat, args.start, args.end):
        parser.error('give --lon, --lat, --start and --end to search, or --items to download')
    print(f'{"date":10}  {"item":26} {"cloud%":>6} {"snow%":>6} {"nodata%":>7}  orbit')
    for f in search(args.lon, args.lat, args.start, args.end, args.max_cloud):
        p = f['properties']
        print(f'{p["datetime"][:10]}  {f["id"]:26} {p["eo:cloud_cover"]:6.1f} '
              f'{p.get("s2:snow_ice_percentage", float("nan")):6.1f} '
              f'{p.get("s2:nodata_pixel_percentage", float("nan")):7.1f}  {orbit(f)}')


if __name__ == '__main__':
    main()
