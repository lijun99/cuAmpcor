#!/bin/bash
# Ridgecrest 2019 earthquakes (Mw 6.4 July 4, Mw 7.1 July 6), Sentinel-2B NIR (B08, 10 m), tile 11SMV,
# 2019-06-28 and 2019-07-08 (same orbit): download (~400 MB), displacements, maps, slip of the Mw 7.1
# needs: pycuampcor (built with GDAL), GDAL, numpy, matplotlib; a GPU (or add --cpu below)
set -e
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p ridgecrest && cd ridgecrest

# the scenes (to choose a pair: python $here/s2_download.py --lon -117.6 --lat 35.77 --start 2019-06-01 --end 2019-08-31 --max-cloud 2)
ref=S2B_11SMV_20190628_1_L2A; sec=S2B_11SMV_20190708_1_L2A
python "$here/s2_download.py" --items $ref $sec --outdir data

# 64 x 64 windows every 16 pixels (160 m), searched +/- 8 pixels (80 m)
python -m pycuampcor.geo_offsets -r data/${ref}_B08.tif -s data/${sec}_B08.tif --op ridgecrest \
    --ww 64 --wh 64 --sw 8 --sh 8 --kw 16 --kh 16

python "$here/plot_displacements.py" ridgecrest.tif --lonlat -117.85 35.50 -117.30 35.95 --vmax 3 \
    --title "Ridgecrest 2019, Sentinel-2B B08, 2019-06-28 / 2019-07-08"
# the Mw 7.1 rupture: strike N34W through (446, 3956) km (UTM 11N)
python "$here/fault_slip.py" ridgecrest.tif --strike 326 --center 446 3956 --sections -12 12 4 --width 6
