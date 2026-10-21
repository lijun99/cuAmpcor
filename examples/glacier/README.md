# Glacier velocities from SAR images, with and without a velocity model

This example measures the velocity of Jakobshavn Isbræ (Greenland), one of the fastest glaciers, from
pairs of SAR images 12 days apart, in two versions:

* **isce3 and NISAR** (`jakobshavn_isce3.sh`): a pair of NISAR L-band RSLCs. isce3 provides the
  geometry and coregisters the images.
* **ISCE2 and Sentinel-1** (`jakobshavn_isce2.sh`): a pair of Sentinel-1 C-band IW SLCs. ISCE2's
  topsStack provides the geometry and coregisters the images, on the GPU if ISCE2 is built with CUDA.

In both, `python -m pycuampcor.dense_offsets` measures the offsets. It runs twice:

* **without a velocity model,** with a search range wide enough for the fastest motion expected;
* **with a velocity model** (ITS_LIVE), which gives each window its own gross offset, so that a small
  search range around it suffices.

The offsets are then converted to velocities and compared with the model.

| File | What it does |
|---|---|
| `jakobshavn_isce3.sh` | isce3 and NISAR: downloads, isce3, pycuampcor, velocities |
| `insar_runconfig.yaml` | A minimal isce3 InSAR runconfig for the isce3 steps |
| `nisar_download.py` | Downloads NISAR products from ASF by their granule names |
| `jakobshavn_isce2.sh` | ISCE2 and Sentinel-1: downloads, topsStack, pycuampcor, velocities |
| `s1_download.py` | Downloads Sentinel-1 SLCs and their precise orbits from ASF |
| `make_dem.py` | A DEM for ISCE2 in ellipsoidal heights from the Copernicus GLO-30 DEM |
| `velocity_model.py` | Gross offsets from a velocity model (`gross`), and velocities from offsets (`velocity`); used by both |

## Requirements (isce3 and NISAR)

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

## The steps (isce3 and NISAR)

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

## The steps (ISCE2 and Sentinel-1)

Requirements:
* ISCE2 with `contrib/stack/topsStack` set up as in its README: `ISCE_STACK=<isce2>/contrib/stack`,
  `$ISCE_STACK/topsStack` in `PATH`, and `$ISCE_STACK` in `PYTHONPATH`.
* For the GPU, ISCE2 built with CUDA from the `gpuresamp` branch
  (https://github.com/earthdef/isce2/tree/gpuresamp), whose topsStack runs the geometry, `geo2rdr`
  and the resampling on the GPU (`-useGPU`, on by default in the script; `USE_GPU=0` for the CPU).
  On an RTX PRO 6000, topsStack took about 3.5 minutes for this pair. On the CPU, the geometry step
  alone was still running after 35 minutes.
* pycuampcor built with GDAL.
* GDAL, asf_search, dem_stitcher and matplotlib.
* A NASA Earthdata login in `~/.netrc`.
* Disk space: about 8 GB for the SLCs, 1.6 GB for the model, and 40 GB for topsStack.

```bash
bash jakobshavn_isce2.sh     # creates ./jakobshavn_s1
```

1. **Data.**
   * The two SLCs (Sentinel-1A, track 25 descending, 2025-03-19 and 2025-03-31) and their precise
     orbits, from ASF.
   * A DEM in ellipsoidal heights from the Copernicus GLO-30 DEM. ISCE2's `dem.py` uses SRTM, which
     stops at 60°N.
   * The same ITS_LIVE model as above.
2. **topsStack.**
   * `stackSentinel.py -W offset -C geometry` over a box around the lower glacier (`-b`).
   * It coregisters with the geometry only. The NESD azimuth correction measures the misregistration
     in the burst overlaps, where fast ice would bias it.
   * The driver runs all of its run files but the last. That one is ISCE2's own dense offsets, on the
     CPU with ampcor, and is replaced by pycuampcor.
   * Each run file's commands run 8 at a time with `xargs`, which waits for all of them and stops if
     one fails. The run files end each command with `&` but do not always end with a `wait`.
   * The products used are the merged, coregistered SLCs (`merged/SLC/<date>/<date>.slc.full`) and
     the reference geometry (`merged/geom_reference/lon.rdr.full`, `lat.rdr.full`).
3. **Offsets.**
   * The pixels are 2.3 m in slant range and about 14 m in azimuth, so the windows are 64 × 256
     pixels (azimuth × slant range; about 0.9 × 0.95 km), every 16 × 64 pixels.
   * Without the model, the search is ±48 × ±160 pixels. With the model, ±8 × ±32 pixels.
   * The amplitudes are correlated (`--drmp 0`), since the bursts of TOPS SLCs have azimuth phase
     ramps.
4. **Velocities.** The same `velocity_model.py velocity` as for NISAR.

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

### isce3 and NISAR

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

### ISCE2 and Sentinel-1

Jakobshavn, Sentinel-1A 2025-03-19 / 2025-03-31, 448 × 690 windows with SNR above 5, compared with
ITS_LIVE 2022 (velocities in m/day). The whole script took 6.5 minutes with the GPUs, including the
downloads; each `dense_offsets` run took about 9 s.

| ITS_LIVE speed | Windows | Median \|v − model\|, without model | With model | Median speed / model, without | With |
|---|---|---|---|---|---|
| < 1 m/day | 117,000 | 0.08 | 0.08 | 1.16 | 1.15 |
| 1–5 m/day | 36,000 | 0.17 | 0.16 | 0.99 | 0.99 |
| 5–20 m/day | 2,700 | 1.2 | 1.2 | 0.95 | 0.93 |
| > 20 m/day | 1,600 | 40.1 | 8.2 | 1.29 | 0.96 |

* **The fastest ice** moves up to about 174 pixels in range in 12 days, beyond the ±160-pixel search
  without the model. As with NISAR, the model is what makes the trunk work.
* **Slow ice** agrees with ITS_LIVE to 0.08 m/day, about 1 m over 12 days.
* **The trunk** is about 5% slower than ITS_LIVE's 2022 average. Jakobshavn is slower in winter, and
  the 1-km windows also average across its sharp shear margins.
* **Known artifacts:**
  * a thin strip at the near-range edge, where the windows overlap the no-data border of the merged
    image;
  * a gap along the seam between subswaths IW1 and IW2, where `velocity_model.py` rejects the windows
    whose local geometry spans the seam;
  * noise over the open water of Disko Bay, which ITS_LIVE leaves out.
* **The season matters at C-band.** A pair from the melt season (2025-08-10 / 2025-08-22) loses the
  texture of the inland ice sheet: the offsets there are noise, about 5 m/day. The coast and the
  glacier still work, with slow ice within 0.03 m/day of ITS_LIVE. NISAR's L-band September pair
  above does not have this problem.

These are checks of the method, not a validated velocity product.
