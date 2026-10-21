# Glacier velocities from SAR images, with and without a velocity model

This example measures the velocity of Jakobshavn Isbræ (Greenland), one of the fastest glaciers, from
a pair of NISAR L-band images 12 days apart. isce3 provides the geometry and coregisters the images;
`python -m pycuampcor.dense_offsets` measures the offsets. It runs twice:

* **without a velocity model,** with a search range wide enough for the fastest motion expected;
* **with a velocity model** (ITS_LIVE), which gives each window its own gross offset, so that a small
  search range around it suffices.

The offsets are then converted to velocities and compared with the model.

| File | What it does |
|---|---|
| `jakobshavn_isce3.sh` | Runs everything: downloads, isce3, pycuampcor, velocities |
| `insar_runconfig.yaml` | A minimal isce3 InSAR runconfig for the isce3 steps |
| `nisar_download.py` | Downloads NISAR products from ASF by their granule names |
| `velocity_model.py` | Gross offsets from a velocity model (`gross`), and velocities from offsets (`velocity`) |

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
bash jakobshavn_isce3.sh     # creates ./jakobshavn
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
4. **Velocities.** `velocity_model.py velocity` converts the offsets to velocities in the coordinates
   of the model, with the local geometry of each window. It compares them with the model and maps the
   speeds.

## Gross offsets from a velocity model

A fast glacier can move hundreds of meters between two images. Without prior knowledge, every window
needs a search range covering the largest motion. That is slow, and on the slow ice, a large search
range is more likely to find a false peak. With a velocity model, `velocity_model.py gross`:

1. finds the ground position of each window center (`x.rdr`, `y.rdr`), in the model's coordinates;
2. samples the model velocity there, and multiplies it by the time between the images;
3. converts this displacement to pixel offsets (down, across) with the local mapping between pixels
   and ground coordinates, from the longitude and latitude of neighboring pixels;
4. rounds the offsets and writes them as int32 pairs per window, the format `dense_offsets --gross 1
   --gross-file` reads. The offsets are clipped so that each search area stays inside the image.

`dense_offsets` then searches only a small range around each gross offset. Add `--mg 1` so that the
output offsets include the gross offsets; without it, they are only the residual within the search.

The same script works with ISCE2 outputs: give `--lon` and `--lat` as the full-resolution longitude
and latitude of the reference geometry (`lon.rdr.full`, `lat.rdr.full`). It also works with any
velocity model GDAL can read with its two horizontal components (for example MEaSUREs).

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
