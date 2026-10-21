# Glacier velocities from SAR images, with and without a velocity model

These examples measure the velocity of Jakobshavn Isbræ (Greenland), one of the fastest glaciers, from
pairs of SAR images 12 days apart:

| Folder | Images | Processor |
|---|---|---|
| `nisar_isce3` | NISAR L-band RSLCs, 2026-09-17 / 2026-09-29 | isce3: geometry and coregistration |
| `s1_isce2` | Sentinel-1 C-band IW SLCs, 2025-03-19 / 2025-03-31 | ISCE2 topsStack: geometry and coregistration, on the GPU with the `gpuresamp` branch |

In both, `python -m pycuampcor.dense_offsets` measures the offsets twice:

* **without a velocity model,** with a search range wide enough for the fastest motion expected;
* **with a velocity model** (ITS_LIVE), which gives each window its own gross offset, so that a small
  search range around it suffices.

The offsets are then converted to velocities and compared with the model, with `velocity_model.py`,
shared by both. Each folder has a `jakobshavn.sh` that runs everything, and a README with the
requirements, the steps and the results.

## Gross offsets from a velocity model

A fast glacier can move hundreds of meters between two images. Without prior knowledge, every window
needs a search range covering the largest motion. That is slow, and on the slow ice, a large search
range is more likely to find a false peak. With a velocity model, `velocity_model.py gross`:

1. finds the ground position of each window center, in the model's coordinates, from the longitude
   and latitude of the reference pixels (e.g., `x.rdr`, `y.rdr` of isce3 `rdr2geo`);
2. samples the model velocity there, and multiplies it by the time between the images;
3. converts this displacement to pixel offsets (down, across) with the local mapping between pixels
   and ground coordinates, from the longitude and latitude of neighboring pixels;
4. rounds the offsets and writes them as int32 pairs per window, the format `dense_offsets --gross 1
   --gross-file` reads. The offsets are clipped so that each search area stays inside the image.

`dense_offsets` then searches only a small range around each gross offset. Add `--mg 1` so that the
output offsets include the gross offsets; without it, they are only the residual within the search.

With ISCE2, `--lon` and `--lat` are the full-resolution longitude and latitude of the reference
geometry (`lon.rdr.full`, `lat.rdr.full`). Any velocity model GDAL can read, with its two horizontal
components, works too (for example MEaSUREs).

`velocity_model.py velocity` converts the offsets back to velocities in the coordinates of the model,
with the same local mapping, and compares them with the model. Windows without a valid geometry
(no data, or a seam between TOPS subswaths) get neither a gross offset nor a velocity.

## The two examples compared

Median |v − ITS_LIVE| in m/day, without / with the model:

| ITS_LIVE speed | NISAR (isce3) | Sentinel-1 (ISCE2) |
|---|---|---|
| < 1 m/day | 0.25 / 0.25 | 0.08 / 0.08 |
| 1–5 m/day | 0.26 / 0.25 | 0.17 / 0.16 |
| 5–20 m/day | 2.8 / 2.6 | 1.2 / 1.2 |
| > 20 m/day | 21.2 / 5.6 | 40.1 / 8.2 |

* **In both,** the fastest ice moves beyond the search range of the run without a model, and only the
  model's gross offsets make it work.
* **The two pairs differ** in date, season, wavelength and window size, so they are checks of the
  method rather than a comparison of the sensors.
* **The NISAR pair** has stripes along the track on slow ice, likely from the ionosphere at L-band.
* **The Sentinel-1 pair** needs a dry season: at C-band, melting snow loses its texture (see
  `s1_isce2/README.md`).
