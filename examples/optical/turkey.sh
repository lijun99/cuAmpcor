#!/bin/bash
# Kahramanmaras 2023 earthquakes (Mw 7.8 and Mw 7.5, February 6), Sentinel-2A NIR (B08, 10 m), tile 37SCB
# (the Pazarcik - Golbasi part of the Mw 7.8 rupture), 2023-01-10 and 2023-03-21 (same orbit; the February
# scenes are partly covered by snow): download (~400 MB), displacements, maps, slip of the Mw 7.8
# needs: pycuampcor (built with GDAL), GDAL, numpy, matplotlib; a GPU (or add --cpu below)
set -e
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p turkey && cd turkey

# the scenes (to choose a pair: python $here/s2_download.py --lon 37.30 --lat 37.49 --start 2022-12-15 --end 2023-04-15 --max-cloud 10)
ref=S2A_37SCB_20230110_0_L2A; sec=S2A_37SCB_20230321_1_L2A
python "$here/s2_download.py" --items $ref $sec --outdir data

python -m pycuampcor.geo_offsets -r data/${ref}_B08.tif -s data/${sec}_B08.tif --op turkey \
    --ww 64 --wh 64 --sw 8 --sh 8 --kw 16 --kh 16

python -m pycuampcor.plot_offsets turkey.tif --center zero --vmax 5 \
    --title "Kahramanmaras 2023, Sentinel-2A B08, 2023-01-10 / 2023-03-21"
# the Mw 7.8 rupture: strike N62E, near (360, 4171.5) km (UTM 37N)
python "$here/fault_slip.py" turkey.tif --strike 62 --center 360 4171.5 --sections -40 50 10
