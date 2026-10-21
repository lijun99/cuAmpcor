#!/usr/bin/env python3
"""
Download NISAR products (e.g., RSLCs) from the ASF archive by their granule names

A NASA Earthdata login is needed (free; https://urs.earthdata.nasa.gov), in ~/.netrc:
  machine urs.earthdata.nasa.gov login <user> password <password>
The granules can be found with ASF Search (https://search.asf.alaska.edu) or asf_search.

usage:
  nisar_download.py --product RSLC --granules NISAR_L1_PR_RSLC_030_... NISAR_L1_PR_RSLC_031_... --outdir input
needs: asf_search
"""

import argparse
import os

import asf_search as asf

BASE = 'https://nisar.asf.earthdatacloud.nasa.gov/NISAR'


def main():
    """Download the granules (<outdir>/<granule>.h5), skipping those already there"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--granules', nargs='+', required=True, help='the granule names (without .h5)')
    parser.add_argument('--product', default='RSLC', help='the product type (default: %(default)s)')
    parser.add_argument('--collection', default=None,
                        help='the collection (default: NISAR_L1_<product>_PROVISIONAL_V1)')
    parser.add_argument('--outdir', default='.', help='the directory of the downloads (default: %(default)s)')
    args = parser.parse_args()
    collection = args.collection or f'NISAR_L1_{args.product}_PROVISIONAL_V1'
    os.makedirs(args.outdir, exist_ok=True)
    urls = [f'{BASE}/{collection}/{g}/{g}.h5' for g in args.granules
            if not os.path.exists(os.path.join(args.outdir, f'{g}.h5'))]
    if urls:
        asf.download_urls(urls, args.outdir, session=asf.ASFSession())
    for g in args.granules:
        print(os.path.join(args.outdir, f'{g}.h5'))


if __name__ == '__main__':
    main()
