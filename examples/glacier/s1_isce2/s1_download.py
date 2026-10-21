#!/usr/bin/env python3
"""
Download Sentinel-1 SLCs (SAFE zip files) from the ASF archive by their granule names, and their
precise orbits (POEORB, from the ASF orbit archive) with --orbits

A NASA Earthdata login is needed (free; https://urs.earthdata.nasa.gov), in ~/.netrc:
  machine urs.earthdata.nasa.gov login <user> password <password>
The granules can be found with ASF Search (https://search.asf.alaska.edu) or asf_search.

usage:
  s1_download.py --granules S1A_IW_SLC__1SDH_20250810T100031_... S1A_IW_SLC__1SDH_20250822T100031_... \
      --outdir SLC --orbits orbits
needs: asf_search
"""

import argparse
import datetime
import os
import re

import asf_search as asf


ORBITS = 'https://s1qc.asf.alaska.edu/aux_poeorb/'


def download_orbits(granules, outdir, session):
    """Download the precise orbit file of each granule: of its satellite, covering the acquisition"""
    os.makedirs(outdir, exist_ok=True)
    pattern = r'(S1[A-D])_OPER_AUX_POEORB_OPOD_(\d{8}T\d{6})_V(\d{8}T\d{6})_(\d{8}T\d{6})\.EOF'
    orbits = set(re.findall(pattern, session.get(ORBITS).text))
    parse = lambda t: datetime.datetime.strptime(t, '%Y%m%dT%H%M%S')
    for g in granules:
        mission, start, stop = g[:3], parse(g.split('_')[5]), parse(g.split('_')[6])
        # the orbits of the satellite covering the acquisition, the latest produced first
        found = sorted((o for o in orbits if o[0] == mission and parse(o[2]) <= start and parse(o[3]) >= stop),
                       key=lambda o: o[1], reverse=True)
        if not found:
            raise RuntimeError(f'No precise orbit found for {g}')
        name = f'{found[0][0]}_OPER_AUX_POEORB_OPOD_{found[0][1]}_V{found[0][2]}_{found[0][3]}.EOF'
        fname = os.path.join(outdir, name)
        if not os.path.exists(fname):
            r = session.get(ORBITS + name)
            r.raise_for_status()
            with open(fname, 'wb') as f:
                f.write(r.content)
        print(fname)


def main():
    """Download the SLCs (<outdir>/<granule>.zip), skipping those already there"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--granules', nargs='+', required=True, help='the granule names (without .zip)')
    parser.add_argument('--outdir', default='.', help='the directory of the downloads (default: %(default)s)')
    parser.add_argument('--orbits', default=None, help='download the precise orbits into this directory too')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    missing = [g for g in args.granules if not os.path.exists(os.path.join(args.outdir, f'{g}.zip'))]
    if missing:
        results = [r for r in asf.granule_search(missing) if r.properties['processingLevel'] == 'SLC']
        results = asf.ASFSearchResults(results)
        results.download(path=args.outdir, session=asf.ASFSession())
    for g in args.granules:
        print(os.path.join(args.outdir, f'{g}.zip'))
    if args.orbits:
        download_orbits(args.granules, args.orbits, asf.ASFSession())


if __name__ == '__main__':
    main()
