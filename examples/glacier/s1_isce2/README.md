# Jakobshavn Isbræ velocities from Sentinel-1, with ISCE2 topsStack and pycuampcor

The velocity of Jakobshavn Isbræ (Greenland) from a pair of Sentinel-1 C-band IW SLCs 12 days apart
(Sentinel-1A, track 25 descending, 2025-03-19 and 2025-03-31). ISCE2's topsStack provides the
geometry and coregisters the images, on the GPU if ISCE2 is built with CUDA;
`python -m pycuampcor.dense_offsets` measures the offsets, without and with the ITS_LIVE velocity
model for the gross offsets (see `../README.md`).

| File | What it does |
|---|---|
| `jakobshavn.sh` | Runs everything: downloads, topsStack, pycuampcor, velocities |
| `s1_download.py` | Downloads Sentinel-1 SLCs and their precise orbits from ASF |
| `make_dem.py` | A DEM for ISCE2 in ellipsoidal heights from the Copernicus GLO-30 DEM |
| `../velocity_model.py` | Gross offsets from a velocity model, and velocities from offsets |

## Requirements

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

## The steps

```bash
bash jakobshavn.sh     # creates ./jakobshavn_s1
```

1. **Data.**
   * The two SLCs (Sentinel-1A, track 25 descending, 2025-03-19 and 2025-03-31) and their precise
     orbits, from ASF.
   * A DEM in ellipsoidal heights from the Copernicus GLO-30 DEM. ISCE2's `dem.py` uses SRTM, which
     stops at 60°N.
   * The ITS_LIVE 2022 annual velocity mosaic of Greenland, as in `../nisar_isce3`.
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
4. **Velocities.** `../velocity_model.py velocity` converts the offsets to velocities and compares them
   with ITS_LIVE.

## Results

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
  without the model. As with NISAR (`../nisar_isce3`), the model is what makes the trunk work.
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
  (`../nisar_isce3`) does not have this problem.

These are checks of the method, not a validated velocity product.
