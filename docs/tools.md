# The command-line tools

The package has three command-line tools, run as Python modules:

* `python -m pycuampcor.dense_offsets`: the offset field between two images in radar geometry (or any
  two images on the same grid);
* `python -m pycuampcor.geo_offsets`: the displacement field between two georeferenced images;
* `python -m pycuampcor.plot_offsets`: a first look at the outputs of the two.

Each tool lists its options with `--help`. The offsets are computed on the GPU, or on the CPU with
`--cpu` (or if the package is built without CUDA).

## dense_offsets

`dense_offsets` estimates the offsets of windows on a regular grid of the reference image. It replaces
*cuDenseOffsets.py* of ISCE2 (*contrib/PyCuAmpcor/examples*), as used with the ISCE2 stack processors
and [MintPy](https://github.com/insarlab/MintPy) for offset time series.

```bash
python -m pycuampcor.dense_offsets -r 20151120.slc.full -s 20151214.slc.full \
    --op ./20151120_20151214/offset --ww 64 --wh 64 --sw 20 --sh 20 --kw 32 --kh 32 --oo 32
```

### Images

The images are

* raw binary files (one band of complex64 or float32 pixels, e.g., the SLCs of the ISCE2 stack
  processors). Their size and pixel type are read from their ISCE *.xml* if
  [ISCE2](https://github.com/isce-framework/isce2) is installed, or else with GDAL (from the *.vrt* next
  to the image, or the image itself), or given with `--length`, `--width` and `--data-type`
  (`complex64` or `float32`);
* 2D datasets in HDF5 files, if pycuampcor is built with HDF5, named `HDF5:<file>:<dataset>`, e.g.,
  `HDF5:rslc.h5:/science/LSAR/RSLC/swaths/frequencyA/HH` for a NISAR RSLC (or `<file>:<dataset>` with
  `--reader hdf5`). Their size and type are read with h5py (or GDAL);
* rasters read with GDAL, if pycuampcor is built with GDAL: names ending with *.tif*, *.tiff* or *.vrt*,
  or any raster with `--reader gdal`. Their size and type are read with GDAL; a real band of any type
  (e.g., GeoTIFF amplitudes of UInt16) is read as float32, a complex band as complex64.

### Windows

The windows are `--wh` x `--ww` pixels (down x across, i.e., azimuth x range), every `--kh` x `--kw`
pixels, searched ±`--sh` x ±`--sw` pixels. By default, the first window starts after the margin
(`--mm`) and the search range, and the windows cover the image, as many as fit:

    numberWindowDown = (length - 2 margin - 2 sh - wh) // kh

and the same across. To process a part of the image, give the first pixel of the first window and
the number of windows,

```bash
--startpixeldw $startPixelDown --startpixelac $startPixelAcross --nwd $numberWindowsDown --nwa $numberWindowsAcross
```

or the last pixel instead of the numbers,

```bash
--startpixeldw $startPixelDown --startpixelac $startPixelAcross --endpixeldw $endPixelDown --endpixelac $endPixelAcross
```

To compare runs with different window sizes or search ranges, fix the window centers with
`--use-center 1`: the starting (and ending) pixels are then the centers of the windows. Windows
beyond the image are padded with zeros, and their results may be wrong.

### Gross offsets

* A constant gross offset, `--aa $down --rr $across` (in pixels); the margin is then at least the
  gross offset.
* A gross offset per window, `--gross 1 --gross-file $file`: int32 pairs (down, across), one per
  window, in the order of the outputs (row major). It keeps the search range small where the offsets
  vary a lot, e.g., from a velocity model of a glacier (see
  [examples/glacier](../examples/glacier)). The reference window (i, j) starts at (startPixelDown + i kh,
  startPixelAcross + j kw); the tool prints the starting pixel and the numbers of windows.
* By default, the output offsets do not include the gross offsets; `--mg 1` adds them.

### Correlation

| Option | Default | Notes |
|:---|:---|:---|
| `--wf` | 0 | 0 = two-pass, 1 = one-pass (see [the algorithm](algorithm.md)) |
| `--alg` | 0 | 0 = frequency domain, 1 = time domain, of the amplitudes; 2 = coherence (experimental), of the complex windows, with `--drmp 1` |
| `--drmp` | 0 | before the oversampling: 0 = the amplitudes (e.g., TOPS), 1 = remove a linear phase ramp (e.g., stripmap SLCs with a Doppler centroid), 2 = no deramping |
| `--raw-osf` | 2 | the antialiasing oversampling of the windows |
| `--oo` | 32 | the oversampling of the correlation surface; the sub-pixel step is 1 / (2 x 32) pixel |
| `--corr-osm` | 0 | 0 = FFT, 1 = sinc |

### Outputs

The outputs are `<prefix><suffix>.bip` (`--op`, `--os`), band-interleaved (BIP) float32 files, one
pixel per window:

* `.bip`: the offsets (down, across), in pixels;
* `_gross.bip`: the gross offsets;
* `_snr.bip`: the SNR of the correlation peak;
* `_cov.bip`: the covariance of the offsets (down variance, across variance, covariance);
* `_peak.bip`: the correlation peak value.

Their headers are ISCE *.xml* (and *.vrt*) files if ISCE2 is installed, or else ENVI *.hdr* and *.vrt*
files; both are read by MintPy and GDAL. An existing output is not recomputed unless `--redo` is
given. With `--full-geom` and `--out-geom` (which need GDAL), the geometry (e.g., *lat.rdr.full*,
*lon.rdr.full*, *hgt.rdr.full* of the ISCE2 stack processors) is also written on the grid of the
windows: the values at the window centers.

### Processing

`--gpuid` and `--nstreams` choose the GPU and the number of CUDA streams; `--cpu` and `--nthreads`
the CPU implementation and its threads; `--mmapsize` the memory map of raw images (GB); `--nwdc` and
`--nwac` the windows in a chunk (0 for automatic).

### A shell script

To keep a record of the options, a script may set them as variables:

```bash
#!/bin/bash
reference=./merged/SLC/20151120/20151120.slc.full  # reference image
secondary=./merged/SLC/20151214/20151214.slc.full  # secondary image
ww=64      # window width
wh=64      # window height
sw=20      # half search range across
sh=20      # half search range down
kw=32      # step between windows across
kh=32      # step between windows down
mm=0       # margin
gross=0    # 1: varying gross offsets (--gross-file)
azshift=0  # constant gross offset down
rgshift=0  # constant gross offset across
deramp=0   # 0: the amplitudes (TOPS), 1: remove a linear phase ramp, 2: no deramping
oo=32      # oversampling of the correlation surface
workflow=0 # 0: two-pass, 1: one-pass
outprefix=./merged/20151120_20151214/offset
outsuffix=_ww64_wh64

python -m pycuampcor.dense_offsets --reference $reference --secondary $secondary \
    --ww $ww --wh $wh --sw $sw --sh $sh --mm $mm --kw $kw --kh $kh \
    --gross $gross --aa $azshift --rr $rgshift --drmp $deramp --oo $oo --wf $workflow \
    --op $outprefix --os $outsuffix --gpuid 0 --nstreams 2 --redo
```

## geo_offsets

`geo_offsets` estimates the displacement field between two georeferenced images: e.g., optical
images (Sentinel-2, Landsat, Planet) before and after an earthquake, or of a glacier, or geocoded
SAR amplitudes. It needs GDAL (the Python bindings) and pycuampcor built with GDAL.

```bash
python -m pycuampcor.geo_offsets -r T11SMV_20190628_B08.tif -s T11SMV_20190708_B08.tif \
    --op offsets/ridgecrest --ww 64 --wh 64 --sw 8 --sh 8 --kw 16 --kh 16
```

* The images are read with GDAL (any format; one band, `--band`; any pixel type, read as float32,
  or complex64 if complex). They must have the same coordinate system and pixel size (use `gdalwarp`
  otherwise), but may have different extents.
* The shift between the two grids is computed from the georeferencing and used as the gross offset.
  The windows cover the area of the reference image where the secondary search areas are available,
  or the part given by `--bbox xmin ymin xmax ymax` (in the coordinates of the images).
* The output `<prefix>.tif` is a GeoTIFF on the grid of the window centers (one pixel per window, of
  `--kw` x `--kh` pixels of the images), with the bands `east` and `north` (the displacements, in the
  units of the coordinate system, e.g., meters, without the shift of the grids), `snr`, `peak` (the
  correlation peak value), and `var_east`, `var_north`, `cov_east_north` (their covariance).
* The windows with no-data pixels in the reference window or the secondary search area (`--nodata`,
  by default the no-data value of the images; `--max-nodata` for the fraction allowed) are NaN.

[examples/optical](../examples/optical) shows its use on Sentinel-2 images of two earthquakes.

## plot_offsets

`plot_offsets` gives a first look at an offset field: the output of `dense_offsets` (the offset file,
*.bip*, its *.vrt*, or its ENVI *.hdr*; the SNR file *<prefix>_snr.bip* next to it is found and plotted
too), or the GeoTIFF of `geo_offsets` (with map coordinates and the units of the displacements). It
needs GDAL and matplotlib.

```bash
python -m pycuampcor.plot_offsets offsets/offset.bip                  # the two components and the SNR
python -m pycuampcor.plot_offsets offsets/offset.bip --min-snr 5 --vectors --pixel-size 2.3 14.1
python -m pycuampcor.plot_offsets ridgecrest.tif --lonlat -117.85 35.5 -117.3 35.95 --vmax 3 -o ridgecrest.png
```

* By default, it plots the maps of the two components and the SNR; `--vectors [STEP]` the magnitude
  with the vectors (every STEP windows); `--3d` the components as 3D surfaces.
* The color scales are centered on the medians (`--center zero` for zero), with a half range of three
  times the median absolute deviation, so that the outliers of decorrelated windows saturate; `--vmax`
  sets the half range.
* `--min-snr` masks the windows of a lower SNR; `--pixel-size` scales offsets in pixels (e.g., to
  meters); `--subset` plots rows and columns of the field, `--bbox` or `--lonlat` an area of a
  georeferenced field.
* The figure is shown if a display is available, or saved to *<file>.png* (or `-o`).

The outputs may also be viewed with other tools, e.g., *mdx.py* of ISCE2, or QGIS for the GeoTIFF of
`geo_offsets`.
