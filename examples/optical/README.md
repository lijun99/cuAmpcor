# Displacements from GeoTIFF and optical images

These examples measure ground displacements between two georeferenced images with
`python -m pycuampcor.geo_offsets`. They use Sentinel-2 optical images of two earthquakes. The same
steps apply to other georeferenced rasters, such as Landsat, Planet, aerial orthophotos and geocoded
SAR amplitude images.

| File | What it does |
|---|---|
| `s2_download.py` | Searches the public Sentinel-2 archive for scenes over a point, and downloads a band of the chosen scenes |
| `plot_displacements.py` | Plots the east and north displacements and the SNR of a `geo_offsets` output |
| `fault_slip.py` | Measures the strike-slip offset in sections along a fault |
| `ridgecrest.sh` | Runs everything for the 2019 Ridgecrest earthquakes (California) |
| `turkey.sh` | Runs everything for the 2023 Kahramanmaraş earthquakes (Türkiye) |

## Requirements

* pycuampcor built with GDAL support (`python -c "import pycuampcor; print(pycuampcor.has_gdal)"` prints `True`).
* The GDAL Python bindings, numpy and matplotlib.
* A GPU, or add `--cpu` to the `geo_offsets` command. A Sentinel-2 tile takes about 25 s on a recent
  GPU and about 2 minutes on 64 CPU cores.
* About 400 MB of disk space per example for the two images. The downloads need no account.

## Which tool for which images

* **`pycuampcor.dense_offsets`** is for images in radar geometry, such as SLCs from ISCE2. They are
  raw binary files, or rasters GDAL can read: names ending in `.tif`, `.tiff` or `.vrt` are read with
  GDAL, and `--reader gdal` reads other formats. Offsets are in pixels.
* **`pycuampcor.geo_offsets`** is for georeferenced images: map-projected optical images, or
  geocoded SAR amplitude images. It reads the georeferencing and writes displacements in map units.

Both read any pixel type: real bands (for example UInt16 reflectance) are read as float32, complex
bands as complex64. Only one band is read (`--band`).

## How geo_offsets works

1. Both images must use the same coordinate system and pixel size. If they don't, resample one onto
   the other's grid first, for example `gdalwarp -t_srs EPSG:32611 -tr 10 10 -tap in.tif out.tif`.
   The extents can differ.
2. The shift between the two grids comes from the georeferencing. It is used as the gross offset and
   then removed, so the output is the ground displacement only.
3. Windows of `--ww` × `--wh` pixels, every `--kw` × `--kh` pixels, cover the area where both images
   are available, or the box given by `--bbox xmin ymin xmax ymax`. Each window is searched ±`--sw`,
   ±`--sh` pixels in the secondary image.
4. The output `<prefix>.tif` is a GeoTIFF with one pixel per window, at the window centers. Its bands:

   | Band | Content |
   |---|---|
   | `east`, `north` | displacement, in the map units (meters for UTM) |
   | `snr` | signal-to-noise ratio of the correlation peak |
   | `peak` | normalized correlation peak value |
   | `var_east`, `var_north`, `cov_east_north` | variances and covariance of the displacement |

   Windows with no-data pixels in the reference window or the secondary search area are NaN.

Choosing the parameters:

* **Window size:** 64 pixels is robust. 32 pixels gives finer detail but more noise.
* **Search range:** it must cover the largest displacement plus any misalignment between the images.
  ±8 pixels at 10 m covers ±80 m, plenty for earthquakes. For fast glaciers, increase it, or reduce
  the time between the images.
* **Step:** it sets the output resolution, and the run time grows with 1/step². With a step smaller
  than the window, neighboring windows overlap.

## Choosing a pair of scenes

Pick two scenes from the same satellite and relative orbit, so that the viewing geometry is the
same. Avoid clouds and snow, and choose dates close in time and season. `s2_download.py` lists the
candidates:

```bash
python s2_download.py --lon 37.30 --lat 37.49 --start 2023-01-01 --end 2023-03-31 --max-cloud 2
```

```
date        item                       cloud%  snow% nodata%  orbit
2023-01-05  S2B_37SCB_20230105_0_L2A      0.0    1.2     0.0  S2B   4
2023-01-10  S2A_37SCB_20230110_0_L2A      0.2    1.4     0.0  S2A 118
2023-01-20  S2A_37SCB_20230120_0_L2A      0.8    6.4     0.0  S2A 118
2023-01-25  S2B_37SCB_20230125_0_L2A      0.2    5.3     0.0  S2B   4
2023-02-09  S2A_37SCB_20230209_0_L2A      0.5   40.3     0.0  S2A 118
2023-03-21  S2A_37SCB_20230321_1_L2A      0.5    4.8     1.0  S2A 118
```

Scenes with the same value in the `orbit` column were taken from the same satellite and relative
orbit. Then download one band (`--asset nir` is B08, 10 m) of the chosen items:

```bash
python s2_download.py --items S2A_37SCB_20230110_0_L2A S2A_37SCB_20230321_1_L2A --outdir data
```

## Example: Ridgecrest 2019

`ridgecrest.sh` uses Sentinel-2B B08 of tile 11SMV, 2019-06-28 and 2019-07-08. The Mw 6.4 and Mw 7.1
earthquakes struck on July 4 and 6.

```bash
bash ridgecrest.sh      # creates ./ridgecrest: data/, ridgecrest.tif, ridgecrest.png
```

* The Mw 7.1 rupture is a sharp NW–SE break with right-lateral motion: the block west of the fault
  moved northwest, the block east of it southeast.
* `fault_slip.py` measures about 3 m of right-lateral slip near the middle of the rupture (negative
  values in its table), decreasing to about 1.4 m 12 km away along the fault.
* Sentinel-2 artifacts are also visible:
  * a uniform shift between the two dates (median about −1.3 m east, +0.4 m north), from the
    alignment of the two images;
  * stripes along the satellite track;
  * straight steps at the boundaries between the sensor's detectors.

  Remove the uniform shift by subtracting the median of a stable area.

## Example: Kahramanmaraş 2023

`turkey.sh` uses Sentinel-2A B08 of tile 37SCB, on the Pazarcık–Gölbaşı part of the Mw 7.8 rupture.
The dates are 2023-01-10 and 2023-03-21. The scenes just after the February 6 earthquakes are 30–40%
snow here, and nearly 100% on the Mw 7.5 rupture to the north.

```bash
bash turkey.sh          # creates ./turkey: data/, turkey.tif, turkey.png
```

* The Mw 7.8 rupture crosses the tile from SW to NE with left-lateral motion: the block NW of the
  fault moved southwest, the block SE of it northeast.
* `fault_slip.py` measures 4.6–6.7 m of left-lateral slip along the central 60 km, and 3–4 m further
  NE.
* Snow in the northern mountains and seasonal changes between January and March reduce the
  correlation there; about 60% of the windows have SNR above 5.

## Measuring slip across a fault

`fault_slip.py` takes the fault's strike and a point near the trace (in km, in the map coordinates):

1. It projects the displacements on the strike.
2. In each section along the fault, it finds the trace at the largest step of the across-fault
   profile.
3. It takes the difference of the medians 1–5 km on each side (`--sides`).

Positive values are left-lateral slip, negative values right-lateral. Measuring 1–5 km from the trace
includes deformation off the fault, so the values can differ from offsets measured right at the
trace.

These examples are checks of the tool, not validated results: compare with published displacement
maps before using the numbers.
