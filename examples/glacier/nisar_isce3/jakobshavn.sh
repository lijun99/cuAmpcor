#!/bin/bash
# Jakobshavn Isbrae (Greenland) velocities from a NISAR RSLC pair (L-band, track 170 descending,
# 2026-09-17 and 2026-09-29), with isce3 (geometry and coregistration) and pycuampcor.dense_offsets,
# without and with the ITS_LIVE velocity model for the gross offsets
# needs: isce3 (with the nisar workflows), pycuampcor (with HDF5), GDAL, h5py, asf_search, matplotlib;
#   a NASA Earthdata login in ~/.netrc; a GPU (or add --cpu to the dense_offsets commands);
#   disk: ~20 GB of RSLCs, ~1.6 GB for the model, ~130 GB of isce3 scratch
set -e
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p jakobshavn && cd jakobshavn
ref=NISAR_L1_PR_RSLC_030_170_D_053_7700_SHNA_A_20260917T235201_20260917T235237_P05023_F_P_J_001
sec=NISAR_L1_PR_RSLC_031_170_D_053_7700_SHNA_A_20260929T235201_20260929T235237_P05023_N_P_J_001
days=12

# 1. the data: the RSLCs, the DEM (NISAR DEM over the frame), the velocity model (ITS_LIVE 2022, Greenland)
python "$here/nisar_download.py" --granules $ref $sec --outdir input
[ -f dem/dem.vrt ] || python -m nisar.workflows.stage_dem -p input/$ref.h5 -o dem/dem.vrt
model=model/ITS_LIVE_velocity_120m_RGI05A_2022_v02.nc
mkdir -p model
[ -f $model ] || curl -sS -f -o $model https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2/annual/$(basename $model)

# 2. isce3: the geometry (longitude, latitude of the reference pixels), and the secondary coregistered
#    to the reference with the geometric offsets (orbits and DEM)
sed -e "s#REFERENCE_RSLC#$PWD/input/$ref.h5#" -e "s#SECONDARY_RSLC#$PWD/input/$sec.h5#" -e "s#DEM_FILE#$PWD/dem/dem.vrt#" \
    "$here/insar_runconfig.yaml" > runconfig.yaml
[ -f scratch/rdr2geo/freqA/x.rdr ] || python -m nisar.workflows.rdr2geo runconfig.yaml
[ -f scratch/geo2rdr/freqA/range.off ] || python -m nisar.workflows.geo2rdr runconfig.yaml
[ -f scratch/coarse_resample_slc/freqA/HH/coregistered_secondary.slc ] || \
    python -m nisar.workflows.resample_slc runconfig.yaml --resample-type coarse

# 3. the offsets: the reference read from the RSLC (HDF5), the coregistered secondary (ENVI);
#    windows of 64 x 128 pixels (azimuth x slant range, ~280 x 330 m), every 64 x 128 pixels
reference="HDF5:input/$ref.h5:/science/LSAR/RSLC/swaths/frequencyA/HH"
secondary=scratch/coarse_resample_slc/freqA/HH/coregistered_secondary.slc
grid_dense="--wh 64 --ww 128 --kh 64 --kw 128 --startpixeldw 88 --startpixelac 178 --nwd 875 --nwa 390"
grid_model="--window 64 128 --skip 64 128 --start 88 178 --number 875 390"
geometry="--lon scratch/rdr2geo/freqA/x.rdr --lat scratch/rdr2geo/freqA/y.rdr --days $days"
its_live="--vx NETCDF:\"$model\":vx --vy NETCDF:\"$model\":vy"
mkdir -p out

# 3a. without a model: a search range for the fastest motion expected (+/- 88 x 178 pixels, ~390 x 450 m)
python -m pycuampcor.dense_offsets -r "$reference" -s $secondary $grid_dense --sh 88 --sw 178 --drmp 1 \
    --op out/nomodel

# 3b. with the model: the gross offsets from ITS_LIVE (int32 per window), a small search range
#     around them (+/- 16 x 32 pixels), and the gross offsets added to the outputs (--mg 1)
eval python "$here/../velocity_model.py" gross $geometry $its_live $grid_model --search 16 32 -o out/gross.bin
python -m pycuampcor.dense_offsets -r "$reference" -s $secondary $grid_dense --sh 16 --sw 32 --drmp 1 \
    --gross 1 --gross-file out/gross.bin --mg 1 --op out/model

# 4. the velocities (m/day, polar stereographic x and y), compared with ITS_LIVE
for run in nomodel model; do
    eval python "$here/../velocity_model.py" velocity out/$run.bip $geometry $its_live $grid_model --min-snr 5 \
        --vmax 30 -o out/velocity_$run --title "'Jakobshavn, NISAR 2026-09-17 / 2026-09-29, $run'"
done
# a first look at the offsets in radar geometry
python -m pycuampcor.plot_offsets out/model.bip --min-snr 5 -o out/model_offsets.png
