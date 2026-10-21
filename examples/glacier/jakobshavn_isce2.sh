#!/bin/bash
# Jakobshavn Isbrae (Greenland) velocities from a Sentinel-1 pair (C-band, IW, track 25 descending,
# Sentinel-1A 2025-03-19 and 2025-03-31, dry snow), with ISCE2 topsStack (geometry and coregistration) and
# pycuampcor.dense_offsets, without and with the ITS_LIVE velocity model for the gross offsets
# needs: ISCE2 (with contrib/stack/topsStack in PATH), pycuampcor (with GDAL), GDAL, asf_search,
#   dem_stitcher, matplotlib; a NASA Earthdata login in ~/.netrc;
#   disk: ~8 GB of SLCs, ~1.6 GB for the model, ~40 GB of topsStack files
# topsStack runs its geometry and resampling on the GPU (-useGPU) if ISCE2 is built with CUDA (the
# GPU modules of the gpuresamp branch, https://github.com/earthdef/isce2/tree/gpuresamp); USE_GPU=0 for
# the CPU. The pycuampcor commands run on the GPU if pycuampcor is built with CUDA, on the CPU otherwise.
set -e
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p jakobshavn_s1 && cd jakobshavn_s1
# (a pair of the melt season, e.g., 2025-08-10 and 2025-08-22, loses the texture of the inland ice sheet)
ref=S1A_IW_SLC__1SDH_20250319T100034_20250319T100102_058372_0737BC_C8B8
sec=S1A_IW_SLC__1SDH_20250331T100035_20250331T100103_058547_073EC2_C1BF
ref_date=20250319; sec_date=20250331; days=12
# the area processed (south north west east): the lower Jakobshavn Isbrae
bbox="68.95 69.35 -50.3 -48.6"

# 1. the data: the SLCs and their precise orbits (from ASF), a DEM in ellipsoidal heights (Copernicus
#    GLO-30, covering the bursts of the area), the velocity model (ITS_LIVE 2022, Greenland)
python "$here/s1_download.py" --granules $ref $sec --outdir SLC --orbits orbits
mkdir -p aux
dem=$PWD/dem/glo30.dem.wgs84
if [ ! -f $dem.xml ]; then
    python "$here/make_dem.py" --bounds -53 68.6 -46 69.8 -o $dem
    gdal2isce_xml.py -i $dem
fi
model=model/ITS_LIVE_velocity_120m_RGI05A_2022_v02.nc
mkdir -p model
[ -f $model ] || curl -sS -f -o $model https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2/annual/$(basename $model)

# 2. topsStack: the offset workflow, coregistered with the geometry only (the burst overlaps of the
#    NESD azimuth correction move with the ice); its run files unpack the bursts, compute the
#    geometry, coregister and merge the SLCs; the last one (ISCE2's CPU ampcor) is replaced by
#    pycuampcor below
gpu=$([ "${USE_GPU:-1}" = 1 ] && echo -useGPU)
[ -d run_files ] || stackSentinel.py -s SLC -o orbits -a aux -d $dem -w $PWD -p hh -W offset -C geometry \
    -b "$bbox" --num_proc 8 --num_proc4topo 8 $gpu
# each run file: its commands in parallel (8 at a time), waiting for all of them and stopping if one fails
# (the run files end the commands with &, but do not always wait for them)
for f in $(ls run_files/run_* | grep -v -e dense_offsets -e '\.done$'); do
    [ -f $f.done ] && continue
    sed -e 's/ *&$//' $f | grep -v -e '^wait$' -e '^$' | xargs -P 8 -I{} sh -c '{}'
    touch $f.done
done

# 3. the offsets of the merged, coregistered SLCs; windows of 64 x 256 pixels (azimuth x slant
#    range, ~0.9 x 0.95 km), every 16 x 64 pixels (~220 x 240 m); the same grid for both runs
reference=merged/SLC/$ref_date/$ref_date.slc.full
secondary=merged/SLC/$sec_date/$sec_date.slc.full
geometry="--lon merged/geom_reference/lon.rdr.full --lat merged/geom_reference/lat.rdr.full --days $days"
its_live="--vx NETCDF:\"$model\":vx --vy NETCDF:\"$model\":vy"
# the grid: the first window after the widest search range, as many windows as fit
read length width <<< $(python -c "from osgeo import gdal; d = gdal.Open('$reference.vrt'); print(d.RasterYSize, d.RasterXSize)")
start_down=48; start_across=160
n_down=$(( (length - 2 * start_down - 64) / 16 )); n_across=$(( (width - 2 * start_across - 256) / 64 ))
grid_dense="--wh 64 --ww 256 --kh 16 --kw 64 --startpixeldw $start_down --startpixelac $start_across --nwd $n_down --nwa $n_across"
grid_model="--window 64 256 --skip 16 64 --start $start_down $start_across --number $n_down $n_across"
mkdir -p out

# 3a. without a model: a search range for the fastest motion expected (+/- 48 x 160 pixels,
#     ~670 x 590 m); --drmp 0: the amplitudes, as the TOPS bursts have azimuth phase ramps
python -m pycuampcor.dense_offsets -r $reference -s $secondary $grid_dense --sh 48 --sw 160 --drmp 0 \
    --op out/nomodel

# 3b. with the model: the gross offsets from ITS_LIVE, a small search range around them
#     (+/- 8 x 32 pixels), and the gross offsets added to the outputs (--mg 1)
eval python "$here/velocity_model.py" gross $geometry $its_live $grid_model --search 8 32 -o out/gross.bin
python -m pycuampcor.dense_offsets -r $reference -s $secondary $grid_dense --sh 8 --sw 32 --drmp 0 \
    --gross 1 --gross-file out/gross.bin --mg 1 --op out/model

# 4. the velocities (m/day, polar stereographic x and y), compared with ITS_LIVE
for run in nomodel model; do
    eval python "$here/velocity_model.py" velocity out/$run.bip $geometry $its_live $grid_model --min-snr 5 \
        --vmax 30 -o out/velocity_$run --title "'Jakobshavn, Sentinel-1A 2025-03-19 / 2025-03-31, $run'"
done
python -m pycuampcor.plot_offsets out/model.bip --min-snr 5 -o out/model_offsets.png
