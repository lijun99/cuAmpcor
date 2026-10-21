# Jakobshavn Isbræ velocities from NISAR, with isce3 and pycuampcor

The velocity of Jakobshavn Isbræ (Greenland) from a pair of NISAR L-band RSLCs 12 days apart
(track 170 descending, 2026-09-17 and 2026-09-29). isce3 provides the geometry and coregisters the
images; `python -m pycuampcor.dense_offsets` measures the offsets, without and with the ITS_LIVE
velocity model for the gross offsets (see `../README.md`).

| File | What it does |
|---|---|
| `jakobshavn.sh` | Runs everything: downloads, isce3, pycuampcor, velocities |
| `insar_runconfig.yaml` | A minimal isce3 InSAR runconfig for the isce3 steps |
| `nisar_download.py` | Downloads NISAR products from ASF by their granule names |
| `../velocity_model.py` | Gross offsets from a velocity model, and velocities from offsets |

## Requirements

* isce3 with the `nisar` workflows, for example the conda-forge `isce3` package.
* pycuampcor built with HDF5, to read the NISAR RSLCs directly.
* GDAL, h5py, asf_search and matplotlib.
* A NASA Earthdata login (free, https://urs.earthdata.nasa.gov) in `~/.netrc`, to download the RSLCs
  and the DEM:

  ```
  machine urs.earthdata.nasa.gov login <user> password <password>
  ```
* A GPU, or add `--cpu` to the `dense_offsets` commands.
* Disk space: about 20 GB for the RSLCs, 1.6 GB for the velocity model, and 130 GB for the isce3
  scratch files (the geometry and the coregistered secondary, at full resolution).

## The steps

```bash
bash jakobshavn.sh     # creates ./jakobshavn
```

1. **Data.**
   * The two RSLCs, from ASF.
   * The NISAR DEM over the frame, staged by `python -m nisar.workflows.stage_dem`. If your isce3 has
     the `-s https` option (isce3 PR #343), it stages the DEM from ASF with the Earthdata login,
     without AWS credentials.
   * The ITS_LIVE 2022 annual velocity mosaic of Greenland: polar stereographic, 120 m, `vx` and `vy`
     in m/yr, read by GDAL as `NETCDF:"<file>":vx`.
2. **isce3.** `rdr2geo` computes the longitude and latitude of each reference pixel (`x.rdr`,
   `y.rdr`). `geo2rdr` and `resample_slc --resample-type coarse` resample the secondary RSLC onto the
   reference grid, with the offsets predicted by the orbits and the DEM. The remaining offsets between
   the reference and the coregistered secondary are then the ground motion, plus any residual
   misregistration.
3. **Offsets.**
   * The reference is read directly from the RSLC as
     `HDF5:<file>:/science/LSAR/RSLC/swaths/frequencyA/HH`. The coregistered secondary is a raw
     complex64 file with an ENVI header.
   * The windows are 64 × 128 pixels (azimuth × slant range; about 280 × 330 m on the ground), every
     64 × 128 pixels. That gives 875 × 390 windows over the frame.
   * The same window grid is used for both runs (`--startpixeldw`, `--startpixelac`, `--nwd`, `--nwa`),
     so they can be compared window by window.
   * Deramping (`--drmp 1`) removes the phase ramp of the windows before oversampling. This is needed
     for complex SLCs with a Doppler centroid.
4. **Velocities.** `../velocity_model.py velocity` converts the offsets to velocities in the coordinates
   of the model, with the local geometry of each window. It compares them with the model and maps the
   speeds.

## Results

Jakobshavn, NISAR 2026-09-17 / 2026-09-29, 875 × 390 windows with SNR above 5, compared with ITS_LIVE
2022 (velocities in m/day):

| ITS_LIVE speed | Windows | Median \|v − model\|, without model | With model | Median speed / model, without | With |
|---|---|---|---|---|---|
| < 1 m/day | 226,000 | 0.25 | 0.25 | 1.19 | 1.19 |
| 1–5 m/day | 33,000 | 0.26 | 0.25 | 1.07 | 1.07 |
| 5–20 m/day | 2,300 | 2.8 | 2.6 | 1.27 | 1.19 |
| > 20 m/day | 1,200 | 21.2 | 5.6 | 1.35 | 1.03 |

* **The fastest ice** (the trunk of Jakobshavn, over 20 m/day) moves up to about 216 pixels in slant
  range in 12 days, beyond the ±178-pixel search without the model. That run mostly finds false peaks
  there. With the model, the speeds agree with ITS_LIVE to within a few percent.
* **Both runs** take about 20 s on a GPU for the whole frame. Here the model's benefit is the
  reliability on fast ice rather than speed.
* **On slow ice,** the 0.25 m/day difference (1–2 pixels over 12 days) is mostly:
  * stripes along the satellite track, likely from the ionosphere, common at L-band in polar regions;
  * residual misregistration, since only the coarse, orbit-based coregistration was applied;
  * real changes since the model's 2022 average.

  The speed ratios above 1 on slow ice come from noise, which always adds to the magnitude.
* **Outside the glacier,** the open water and the ice mélange in the fjord do not correlate, and the
  model has no data there.

These are checks of the method, not a validated velocity product.
