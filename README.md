# PyCuAmpcor - Amplitude Cross-Correlation with GPU and CPU (Version 2)

## Contents

  * [1. Introduction](#1-introduction)
  * [2. Installation](#2-installation)
  * [3. User Guide](#3-user-guide)
  * [4. List of Parameters](#4-list-of-parameters)
  * [5. List of Procedures](#5-list-of-procedures)

## 1. Introduction

Ampcor (Amplitude cross correlation) in InSAR processing offers an estimate of spatial displacements (offsets) with the feature tracking method (also called as speckle tracking or pixel tracking). The offsets are in dimensions of a pixel or sub-pixel (with additional oversampling).

In practice, we

  * choose a rectangle window, $R(x,y)$, from the reference image, serving as the template,

 * choose a series of windows of the same size, $S(x+u, y+v)$, from the search image; the search windows are shifted in location by $(u,v)$;

  * perform cross-correlation between the search windows with the reference window, to obtain the normalized correlation surface $c(u,v)$;

  * find the maximum of $c(u,v)$ while its location, $(u_m,v_m)$, provides an estimate of the offset.

A detailed formulation can be found, e.g., by J. P. Lewis with [the frequency domain approach](http://scribblethink.org/Work/nvisionInterface/nip.html).

InSAR images are band-limited, which enables the correlation surface interpolation to achieve sub-pixel resolutions on the displacements. To mitigate the aliasing in cross-correlation and improve the accuracy in displacement estimation, the input raw images are typically oversampled (2x or more) beforehand. The antialiasing oversampling increases the window sizes to be processed and therefore the computation cost. Two workflows are offered to balance the efficiency and accuracy:

1) A two-pass workflow (implemented in Version 1). In the first pass, an initial offset $(r_x^0, r_y^0)$ (in units of a pixel) is estimated from the cross-correlation over a large search range, but without the anti-aliasing oversampling. In the second pass, a much smaller search range centered $(r_x^0, r_y^0)$ may be chosen for a second cross-correlation, for which the windows are oversampled.

2) A one-pass workflow (added in Version 2). The anti-aliasing oversampling of SAR images is performed over the entire search range before cross-correlation. The first pass, to estimate an initial offset, is therefore no longer required.

In general, the two-pass workflow performs more efficiently than the one-pass workflow, and is adequate in most situations. However, when high frequency noises are present, manifested as noisy correlation surfaces, the one-pass workflow is recommended for its improved accuracy.

PyCuAmpcor offers two implementations with the same interface and results (within floating-point round-off):

* `PyCuAmpcor`, the CUDA implementation for NVIDIA GPUs;
* `PyCPUAmpcor`, the CPU implementation, which processes batches of windows (chunks) in parallel with OpenMP threads and uses FFTW for FFTs.

PyCuAmpcor follows the same procedure as the FORTRAN code, ampcor.F, from the ROIPAC package. In order to optimize the performance on GPU, some implementations are slightly different. In the [list the procedures](#5-list-of-procedures), we show the detailed steps of this workflow, as well as its difference to the Fortran code.

## 2. Installation

### 2.1 Conda packages

The `pycuampcor` conda package comes in two variants,

* `cpu`: with the CPU implementation (`PyCPUAmpcor`) only;
* `cuda`: with both the CPU and CUDA implementations (`PyCPUAmpcor` and `PyCuAmpcor`), which requires an NVIDIA driver.

Conda chooses the `cuda` variant automatically if an NVIDIA driver is detected, or the `cpu` variant otherwise. You may also select a variant explicitly,

```bash
conda install pycuampcor=*=cpu*   # CPU only
conda install pycuampcor=*=cuda*  # CPU + GPU
```

The packages are on the conda-forge channel (`conda install -c conda-forge pycuampcor`), for Linux (x86_64, aarch64) and macOS (x86_64, arm64). They are built from tagged releases by the [pycuampcor-feedstock](https://github.com/conda-forge/pycuampcor-feedstock), which maintains the recipe.

### 2.2 Installation with pip

```bash
    pip install .
```

The CUDA implementation is built if a CUDA compiler (nvcc) is found. The build requires a C++17 compiler, CMake >= 3.18, pybind11, FFTW (single precision), and optionally OpenMP, CUDA >= 11, HDF5 (>= 1.14 for the fast reading of compressed datasets) with zlib to read images from HDF5 files, and GDAL to read rasters such as GeoTIFF. CMake options can be passed with `-C`, e.g.,

```bash
    # CPU only
    pip install . -Ccmake.define.PYCUAMPCOR_CUDA=OFF
    # CUDA for a specific GPU architecture
    pip install . -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=native
```

### 2.3 Installation with CMake

```bash
    cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
      -DCMAKE_CUDA_ARCHITECTURES=native \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j && cmake --install build
```

CMake options:

* `PYCUAMPCOR_CUDA`: AUTO (default, build the CUDA implementation if a CUDA compiler is found), ON, or OFF;
* `PYCUAMPCOR_HDF5`: AUTO (default, read images from HDF5 datasets if HDF5 and zlib are found), ON, or OFF;
* `PYCUAMPCOR_GDAL`: AUTO (default, read rasters such as GeoTIFF with GDAL if GDAL is found), ON, or OFF;
* `PYCUAMPCOR_DOUBLE`: use double precision for internal computations (default OFF);
* `PYCUAMPCOR_INSTALL_DIR`: the directory to install the python package (default, `site-packages/pycuampcor` of the python in use);
* `CMAKE_CUDA_ARCHITECTURES`: the targeted GPU architectures, e.g., `native` to detect the GPU(s) on the system, or `60` for P100, `70` for V100, `80` for A100.

PyCuAmpcor offers a debug mode which outputs intermediate results, enabled with the Debug build type, *-DCMAKE_BUILD_TYPE=Debug*.

### 2.4 Tests

The tests are organized as

* *tests/cxx*: C++ unit tests (with GoogleTest) of the processing kernels and helpers (copying, statistics, offsets, FFT oversamplers and correlators, normalization, deramping, sinc interpolation, parameters, image loading), each checked against a simple reference. The test sources are backend-agnostic and compiled for each backend (`test_cpu`, `test_cuda`); tests of the CUDA backend are skipped if no GPU is available.
* *tests/python*: tests of the python package,
    * *test_synthetic.py*: accuracy and robustness with synthetic images of known sub-pixel shifts, for both workflows and correlation surface oversampling methods, and CPU/GPU consistency;
    * *test_regression.py*: regression tests with the *ovs128-rho0.8* data (from isce3), against the isce3 outputs and against reference outputs of this package (*tests/data/ampcor/ovs128-rho0.8/golden*);
    * *test_api.py*: parameters, gross offsets, real images, and error handling;
    * *test_layers.py*: several layers (window sizes) run together with `runAmpcorLayers`, identical to separate runs;
    * *test_hdf5.py*: images read from HDF5 datasets (compressed or not, chunked or contiguous), identical to the same images in raw files;
    * *test_gdal.py*: images read with GDAL (GeoTIFF, VRT; pixel types converted), identical to the same images in raw files.

To build and run the C++ unit tests,

```bash
    cmake -S . -B build -DPYCUAMPCOR_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
    cmake --build build -j && ctest --test-dir build
```

To run the python tests (after installation),

```bash
    pytest tests/python
```

The GitHub Actions CI (*.github/workflows/tests.yml*) runs the tests of the CPU backend only, as the GitHub-hosted runners have no GPU (GPU runners are not free). The CUDA backend is tested locally, with the locally installed CUDA toolkit (found by CMake, e.g., in */usr/local/cuda*; the conda CUDA compiler/libraries are not needed):

```bash
    # C++ unit tests for both backends (test_cpu, test_cuda)
    cmake -S . -B build -DPYCUAMPCOR_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
    cmake --build build -j && ctest --test-dir build
    # python tests, including the GPU and CPU/GPU consistency tests
    pip install . -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=native
    pytest tests/python
```

The GPU tests are skipped if no GPU is available. If nvcc is not in the `PATH`, specify it with `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` (or `-Ccmake.define.CMAKE_CUDA_COMPILER=...` for pip).

If a change of the results is intended, regenerate the reference outputs with `python tests/python/make_golden.py` and commit them.

The package includes type stubs (`_cpu.pyi`, `_cuda.pyi`) for type checkers and IDEs, generated from the bindings. If the bindings change (parameters, methods or docstrings), regenerate them with `python tests/python/make_stubs.py` (with pycuampcor built with CUDA) and commit them; `test_stubs.py` checks that they are up to date, and `python -m mypy.stubtest pycuampcor._cpu pycuampcor._cuda --allowlist tests/python/stubtest_allowlist.txt` checks them against the modules.

### 2.5 Installation with ISCE2

[ISCE2](https://github.com/isce-framework/isce2) includes an older version of PyCuAmpcor (*contrib/PyCuAmpcor*, GPU only), built with ISCE2 (see the [ISCE2 installation guide](https://github.com/isce-framework/isce2#building-isce)) and used by *topsApp* and *alos2App* for dense offsets. This package can be installed in the same environment; its dense offsets tool (*pycuampcor.dense_offsets*, see 3.1) works with the images of the ISCE2 stack processors.

### 2.6 Code organization

* *cxx/pycuampcor/common*: the shared code, in namespace `pycuampcor`, compiled once (library `pycuampcor_common`): the parameters and their options, logging, the image readers (raw files, HDF5 datasets and GDAL rasters), and the layers processed together (chunk footprints); and the processors of the two-pass and one-pass workflows, written once as templates on the array type (*Processor\*.h*, with the definitions in *Processor\*Impl.h*, with the types that go with the arrays in `array_traits`) and instantiated by each package (*processors.cpp*);
* *cxx/pycuampcor/cpu*: the CPU package, in namespace `pycuampcor::cpu` (library `pycuampcor_cpu`): arrays and kernels, FFTs with FFTW, and the controller and chunk loader, running chunks in parallel with OpenMP threads;
* *cxx/pycuampcor/cuda*: the CUDA package, in namespace `pycuampcor::cuda` (library `pycuampcor_cuda`): arrays and kernels, FFTs with cuFFT, and the controller and chunk loader, running chunks on CUDA streams;
* *python/extensions*: the pybind11 bindings, compiled to `pycuampcor._cuda` and `pycuampcor._cpu`;
* *python/packages/pycuampcor*: the python package;
* *tests/cxx*, *tests/python*, *tests/data*: tests and test data (see [2.4 Tests](#24-tests)).

The headers are organized as in [pyre](https://github.com/pyre/pyre): in each of *common/*, *cpu/* and *cuda/*, *externals.h* (the external headers), *forward.h* (forward declarations), *api.h* (the public type aliases) and *public.h* (everything, in dependency order); those of *cpu/* and *cuda/* include those of *common/*. Clients (the python bindings, the C++ tests) include *public.h*, with the include directories of the library `pycuampcor_cpu` or `pycuampcor_cuda` (the package's directory, then *common/*). Each header includes *forward.h* and the headers of the types it needs complete, and compiles on its own; each source includes its own header and the headers it uses, so that changing a header recompiles only the sources using it.

## 3. User Guide

The main procedures of PyCuAmpcor are implemented with CUDA/C++. A Python interface to configure and run PyCuAmpcor is offered. The package includes the dense offsets tool *pycuampcor.dense_offsets* (3.1), for images in radar geometry, the displacement tool *pycuampcor.geo_offsets* (3.2), for georeferenced images such as optical images, and *pycuampcor.plot_offsets* (3.4) to plot their outputs; the *examples* directory has a self-contained example on synthetic images (*simple_offsets.py*, 3.3), and examples of GeoTIFF and optical images (*examples/optical*: Sentinel-2 images of the Ridgecrest 2019 and Kahramanmaraş 2023 earthquakes, with scripts to download the images and measure the fault slip; see its README), and glacier velocities from SAR images in radar geometry, with and without a velocity model (*examples/glacier*: NISAR images of Jakobshavn Isbræ with isce3, Sentinel-1 images with ISCE2 topsStack, and ITS_LIVE for the gross offsets; see its README).

### 3.1 pycuampcor.dense_offsets

*pycuampcor.dense_offsets* (formerly *cuDenseOffsets.py* in ISCE2's *contrib/PyCuAmpcor/examples*, as used with [MintPy](https://github.com/insarlab/MintPy) for offset time series), run as `python -m pycuampcor.dense_offsets`, serves as a general purpose script to run PyCuAmpcor. It uses *argparse* to pass parameters, either from a command line

```bash
python -m pycuampcor.dense_offsets -r 20151120.slc.full -s 20151214.slc.full --outprefix ./20151120_20151214/offset --ww 64 --wh 64 --oo 32 --kw 32 --kh 32 --sw  20 --sh 20 --gpuid 0 --workflow 0
 ```

The images are raw binary files (one band of complex64 or float32 pixels, e.g., the SLCs of the ISCE2 stack processors), 2D datasets in HDF5 files if pycuampcor is built with HDF5 (named `HDF5:<file>:<dataset>`, e.g., `HDF5:rslc.h5:/science/LSAR/RSLC/swaths/frequencyA/HH` for a NISAR RSLC, or `<file>:<dataset>` with *--reader hdf5*), whose size and type are read with h5py, or rasters read with GDAL if pycuampcor is built with GDAL (names ending with *.tif*, *.tiff* or *.vrt*, or any raster with *--reader gdal*; e.g., GeoTIFF amplitude images of UInt16, read as float32), whose size and type are read with GDAL. The size and pixel type of raw images are read from their ISCE *.xml* files if [ISCE2](https://github.com/isce-framework/isce2) is installed, or else with GDAL (from the *.vrt* next to an image, or the image itself), or given with *--length*, *--width* and *--data-type* (*complex64* or *float32*). The headers of the outputs are ISCE *.xml* (and *.vrt*) files if ISCE2 is installed, or else ENVI *.hdr* and *.vrt* files; both are read by MintPy and GDAL. The offsets are computed on the GPU, or on the CPU with *--cpu* (or if the package is built without CUDA). The geometry options (*--full-geom*, *--out-geom*) need GDAL.

 or by a shell script which may expose more options o

 ```
#!/bin/bash
reference=./merged/SLC/20151120/20151120.slc.full # reference image name 
secondary=./merged/SLC/20151214/20151214.slc.full # secondary image name
ww=64  # template window width
wh=64  # template window height
sw=20   # (half) search range along width
sh=20   # (half) search range along height
kw=32   # skip between windows along width
kh=32   # skip between windows along height
mm=0   # margin to be neglected 
gross=0  # whether to use a varying gross offset
azshift=0 # constant gross offset along height/azimuth 
rgshift=0 # constant gross offset along width/range
deramp=0 # 0 for mag (TOPS), 1 for complex linear ramp, 2 for complex no deramping  
oo=16  # correlation surface oversampling factor 
outprefix=./merged/20151120_20151214/offset  # output prefix
outsuffix=_ww64_wh64   # output suffix
gpuid=0   # GPU device ID
nstreams=2 # number of CUDA streams
mmapsize=4 # buffer size in GB for memory map
nwac=0 # number of windows in a batch along width (0: automatic)
nwdc=0  # number of windows in a batch along height (0: automatic)
workflow=0 # 0=two pass 1=one pass

rm $outprefix$outsuffix*
python -m pycuampcor.dense_offsets --reference $reference --secondary $secondary --ww $ww --wh $wh --sw $sw --sh $sh \
    --mm $mm --kw $kw --kh $kh --gross $gross --rr $rgshift --aa $azshift --oo $oo \
    --deramp $deramp --outprefix $outprefix --outsuffix $outsuffix \
    --gpuid $gpuid --mmapsize $mmapsize \
    --nwac $nwac --nwdc $nwdc --workflow $workflow
 ```

Note that in PyCuAmpcor, the following names for directions are equivalent:
* row, height, down, azimuth, along the track.
* column, width, across, range, along the sight.

In the above script, the computation starts from the (mm+sh, mm+sw) pixel in the reference image, take a series of template windows of size (wh, ww) with a skip (sh, sw), cross-correlate with the corresponding windows in the secondary image, and iterate till the end of the images. The output offset fields are stored in *outprefix+outputsuffix+'.bip'*, which is in BIP format, i.e., each pixel has two bands of float32 data, (offsetDown, offsetAcross). The total number of pixels is given by the total number of windows (numberWindowDown, numberWindowAcross), which is computed by the script and also saved to the xml file.

If you are interested in a particular region instead of the whole image, you may specify the location of the starting pixel (in reference image) and the number of windows desired by adding

```
--startpixelac $startPixelAcross --startpixeldw $startPixelDown --nwa $numberOfWindowsAcross --nwd $numberOfWindowsDown
```

If you prefer to provide the location of the last pixel instead of the number of windows, you may use

```
--startpixelac $startPixelAcross --startpixeldw $startPixelDown --endpixelac $endPixelAcross --endpixeldw $endPixelDown
```

The program will calculate the number of windows based on the skip size and the window size. 

When comparing results from various windows sizes and search ranges, you may prefer to fix the center location for the search window.  To achieve that, use *--use-center 1* together with the *--startpixel* options, e.g., 

```
--startpixelac $startPixelAcross --startpixeldw $startPixelDown --use-center 1 --nwa $numberOfWindowsAcross --nwd $numberOfWindowsDown
```

Please be advised, by enforcing the window locations and sizes, the windows near the image edge may exceed the range of the original image (zeros are padded in this case), and their results may be incorrect.

PyCuAmpcor supports two types of gross offset fields,
* static (--gross=0), i.e., a constant shift between reference and secondary images. The static gross offsets can be passed by *--rr $rgshift --aa $azshift*. Note that the margin as well as the starting pixel may be adjusted.
* dynamic (--gross=1), i.e., shifts between reference windows and secondary windows are varying in different locations. This is helpful to reduce the search range if you have a prior knowledge of the estimated offset fields, e.g., the velocity model of glaciers. You may prepare a binary file of the varying gross offsets, int32 pairs (down, across) for each window in the order of the output offset fields, and use the option *--gross-file $grossOffsetFilename*. If you need the coordinates of reference windows, you may run *pycuampcor.dense_offsets* at first to find out the location of the starting pixel and the total number of windows. The coordinate for the starting pixel of the (iDown, iAcross) window will be (startPixelDown+iDown\*skipDown, startPixelAcross+iAcross\*skipAcross).

### 3.2 pycuampcor.geo_offsets

*pycuampcor.geo_offsets* estimates the displacement field between two georeferenced images, e.g., optical images (Sentinel-2, Landsat, Planet) before and after an earthquake, or of a glacier, or geocoded SAR amplitudes. It needs GDAL (the Python bindings) and pycuampcor built with GDAL.

```bash
python -m pycuampcor.geo_offsets -r T11SMV_20190628_B08.tif -s T11SMV_20190708_B08.tif --op offsets/ridgecrest --ww 64 --wh 64 --sw 8 --sh 8 --kw 16 --kh 16
```

* The images are read with GDAL (any format; one band, `--band`; any pixel type, read as float32, or complex64 if complex). They must have the same coordinate system and pixel size (use `gdalwarp` otherwise), but may have different extents.
* The shift between the two grids is computed from the georeferencing and used as the gross offset; the windows cover the area of the reference image where the secondary search areas are available, or the part given by `--bbox xmin ymin xmax ymax` (in the coordinates of the images).
* The output `<prefix>.tif` is a GeoTIFF on the grid of the window centers (one pixel per window, of `--kw` x `--kh` pixels of the images), with the bands `east` and `north` (the displacements in the units of the coordinate system, e.g., meters, without the shift of the grids), `snr`, `peak` (the correlation peak value), and `var_east`, `var_north`, `cov_east_north` (their covariance).
* The windows with no-data pixels in the reference window or the secondary search area (`--nodata`, by default the no-data value of the images; `--max-nodata` for the fraction allowed) are NaN.

### 3.3 Customized Python Scripts

If you need more control of the computation, or incorporate the PyCuAmpcor procedure in your InSAR processing stacks, you may follow *examples/simple_offsets.py* (offsets of synthetic images with a known shift: `python simple_offsets.py [--cpu]`) or *pycuampcor/dense_offsets.py* to create your own Python script. The general steps are
* create a PyCuAmpcor instance
```python
from pycuampcor import PyCuAmpcor, PyCPUAmpcor
# create an instance (GPU)
objOffset = PyCuAmpcor()
# or for CPU
objOffset = PyCPUAmpcor()
```

* set various parameters, e.g., (see a [list of configurable parameters](#4-list-of-parameters) below)
```python
objOffset.referenceImageName="20151120.slc.full"
...
objOffset.windowSizeWidth = 64
...
```

* ask CUDA/C++ to check/initialize parameters
```python
objOffset.setupParams()
```

* set up the starting pixel(s) and gross offsets
```python
objOffset.referenceStartPixelDownStatic = objOffset.halfSearchRangeDown 
objOffset.referenceStartPixelAcrossStatic = objOffset.halfSearchRangeDown
# if static gross offset
objOffset.setConstantGrossOffset(0, 0)
# if dynamic gross offset, computed and stored in vD, vA 
objOffset.setVaryingGrossOffset(vD, vA)
```

* and finally, run PyCuAmpcor
```python
objOffset.runAmpcor()
```

Alternatively, the parameters may be given to the constructor as keyword arguments, named as the parameters above, with the gross offsets as `grossOffsetDown` and `grossOffsetAcross` (numbers for a constant gross offset, or sequences of `numberWindowDown*numberWindowAcross` numbers for varying gross offsets; default 0). The constructor then sets up the parameters and the gross offsets, so the object is ready to run:
```python
params = dict(referenceImageName="20151120.slc.full", windowSizeWidth=64, ...)
objOffset = PyCuAmpcor(**params, grossOffsetDown=vD, grossOffsetAcross=vA)
objOffset.runAmpcor()
```
Unknown parameter names and values of a wrong type raise a `TypeError`.

The messages written while running (progress to stdout, warnings to stderr) are chosen with `pycuampcor.set_log_level`, e.g., `pycuampcor.set_log_level("warning")` for the warnings only (levels `quiet`, `warning`, `info` (default) and `debug`). File names may be given as strings or path-like objects (e.g., `pathlib.Path`).

### 3.4 Offset visualization

*pycuampcor.plot_offsets* gives a first look at an offset field: the output of *pycuampcor.dense_offsets* (the offset file, *.bip*, its *.vrt*, or its ENVI *.hdr*; the SNR file *<prefix>_snr.bip* next to it is found and plotted too), or the GeoTIFF of *pycuampcor.geo_offsets* (with map coordinates and the units of the displacements). It needs GDAL and matplotlib.

```bash
python -m pycuampcor.plot_offsets offsets/offset.bip                  # the two components and the SNR
python -m pycuampcor.plot_offsets offsets/offset.bip --min-snr 5 --vectors --pixel-size 2.3 14.1
python -m pycuampcor.plot_offsets ridgecrest.tif --lonlat -117.85 35.5 -117.3 35.95 --vmax 3 -o ridgecrest.png
```

* By default, the maps of the two components and the SNR; *--vectors [STEP]* plots the magnitude with the vectors (every STEP windows), *--3d* the components as 3D surfaces.
* The color scales are centered on the medians (*--center zero* for zero), with a half range of three times the median absolute deviation, so that the outliers of decorrelated windows saturate; *--vmax* sets the half range.
* *--min-snr* masks the windows of a lower SNR; *--pixel-size* scales offsets in pixels (e.g., to meters); *--subset* plots rows and columns of the field, *--bbox* or *--lonlat* an area of a georeferenced field.
* The figure is shown if a display is available, or saved to *<file>.png* (or *-o*). The output files may also be viewed with other tools, e.g., *mdx.py* of ISCE2, or QGIS for the GeoTIFF of *geo_offsets*.

## 4. List of Parameters (ROIPAC/ampcor workflow)

**Image Parameters**

| PyCuAmpcor                | Notes                                                                   |
|:--------------------------|:------------------------------------------------------------------------|
| referenceImageName        | The file name of the reference/template image                           |
| referenceImageReader      | How to read the reference image: auto (default), raw, hdf5 or gdal      |
| referenceImageHeight      | The height of the reference image                                       |
| referenceImageWidth       | The width of the reference image                                        |
| secondaryImageName        | The file name of the secondary/search image                             |
| secondaryImageReader      | How to read the secondary image: auto (default), raw, hdf5 or gdal      |
| secondaryImageHeight      | The height of the secondary image                                       |
| secondaryImageWidth       | The width of the secondary image                                        |
| grossOffsetImageName      | The output file name for gross offsets                                  |
| offsetImageName           | The output file name for dense offsets                                  |
| snrImageName              | The output file name for signal-noise-ratio of the correlation          |
| covImageName              | The output file name for variance of the correlation surface            |
| peakValueImageName        | The output file name for the normalized correlation surface peak values |

PyCuAmpcor reads images as raw binary files (e.g., ENVI format, or the data file of a GDAL VRT raw raster) with memory map, and image heights/widths are required as inputs. Single-precision complex (referenceImageDataType/secondaryImageDataType=2, default) or real (=1) data are supported. Multi-band is not currently supported, but can be added if desired.

If built with HDF5 (`pycuampcor.has_hdf5`), images may also be 2D datasets in HDF5 files, named as in GDAL, `HDF5:<file>:<dataset>` (or `HDF5:"<file>":<dataset>` if the file name contains ':'), e.g., `HDF5:rslc.h5:/science/LSAR/RSLC/swaths/frequencyA/HH`. The dataset must be float32 (real) or complex64 (a compound of two float32, as written by h5py and the NISAR products), with the given height and width. Chunked datasets compressed with deflate (gzip), with or without shuffle, as the NISAR products, or uncompressed, are read without the HDF5 library: the chunks are located once, read and decoded in parallel, and kept in a cache of `mmapSize` GB shared by the workers (CUDA streams or CPU threads). Other datasets (contiguous, other filters, or files with a user block) are read with the HDF5 library. The image is read directly from the HDF5 file, without an intermediate raw copy.

If built with GDAL (`pycuampcor.has_gdal`), images may also be rasters read with GDAL, e.g., GeoTIFF or VRT files, or any dataset name of GDAL. The first band is read, with the given height and width, and converted by GDAL to the pixel type of the image: a real band of any type (e.g., Byte, UInt16, Int16, Float64) to float32 (referenceImageDataType/secondaryImageDataType=1), a complex band (e.g., CInt16, CFloat32) to complex64. The reads of an image are serialized among the workers; raw files and HDF5 datasets remain the faster readers for large images.

The reader of each image is chosen by `referenceImageReader`/`secondaryImageReader`: `auto` (default) reads names starting with `HDF5:` as HDF5 datasets, names ending with `.tif`, `.tiff` or `.vrt` with GDAL, and other names as raw binary files; `raw`, `hdf5` and `gdal` choose the reader explicitly (with `hdf5`, the `HDF5:` prefix of the name is optional, e.g., `rslc.h5:/science/LSAR/RSLC/swaths/frequencyA/HH`).

The offset output is arranged in BIP format, with each pixel (azimuth offset, range offset). In addition to a static gross offset (i.e., a constant for all search windows), PyCuAmpcor supports varying gross offsets as inputs (e.g., for glaciers, users can compute the gross offsets with the velocity model for different locations and use them as inputs for PyCuAmpcor.

The offsetImage only outputs the (dense) offset values computed from the cross-correlations. Users need to add offsetImage and grossOffsetImage to obtain the total offsets.

The dimension/direction names used in PyCuAmpcor are:
* the inner-most dimension x(i): row, height, down, azimuth, along the track.
* the outer-most dimension y(j): column, width, across, range, along the sight.

Note that ampcor.F and GDAL in general use y for rows and x for columns.

Note also PyCuAmpcor parameters refer to the names used by the PyCuAmpcor Python class. They may be different from those used in C/C++/CUDA, or the arguments of pycuampcor.dense_offsets.

**Process Parameters**

| PyCuAmpcor           | Notes                                                                                                                                                           |
| :---                 |:----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| devID                | The CUDA GPU to be used for computation, usually=0, or users can use the CUDA_VISIBLE_DEVICES=n environmental variable to choose GPU                            |
| nStreams | The number of CUDA streams to be used, recommended=2, to overlap the CUDA kernels with data copying, more streams require more memory which isn't always better |
| nThreads             | (CPU only) The number of CPU threads to be used, 0 (default) for the OpenMP default (e.g., set by the OMP_NUM_THREADS environmental variable)                  |
| useMmap              | Unused (raw images are always read with a memory map); kept for compatibility                                                                                  |
| mmapSize             | The cache size used for memory map, in units of GB (enlarged if a chunk needs more). The larger the better, but not exceed 1/4 the total physical memory.        |
| numberWindowDownInChunk | The number of windows processed in a batch/chunk, along lines; 0 (default) for automatic                                                                         |
| numberWindowAcrossInChunk | The number of windows processed in a batch/chunk, along columns; 0 (default) for automatic                                                                    |

Many windows are processed together to maximize the usage of GPU cores; which is called as a Chunk. The total number of windows in a chunk is limited by the GPU memory.
By default (0), `setupParams()` chooses the chunk size for the backend, limited by the numbers of windows:
for GPU, 2x the number of SMs (streaming multiprocessors), as (SM/4) x 8 windows (down x across), which was
found to be the fastest (within a few percent) on V100 and RTX PRO 6000 Blackwell GPUs; the number of windows
matters more than the shape of the chunk. For CPU, each OpenMP thread processes a chunk (its windows one after
another), and the default is one window per chunk.


**Search Parameters**

| PyCuAmpcor           | Notes    |
| :---                 | :----                     |
| skipSampleDown       | The skip in pixels for neighboring windows along height |
| skipSampleAcross     | The skip in pixels for neighboring windows along width |
| numberWindowDown     | the number of windows along height |
| numberWindowAcross   | the number of windows along width  |
| referenceStartPixelDownStatic | the starting pixel location of the first reference window - along height component |
|referenceStartPixelAcrossStatic | the starting pixel location of the first reference window - along width component |

The C/C++/CUDA program accepts inputs with the total number of windows (numberWindowDown, numberWindowAcross) and the starting pixels of each reference window. The purpose is to establish multiple-threads/streams processing. Therefore, users are required to provide/compute these inputs, with tools available from PyCuAmpcor python class. The pycuampcor.dense_offsets tool also does the job.

We provide some examples below, assuming a PyCuAmpcor class object is created as

```python
    objOffset = PyCuAmpcor()
```

**To compute the total number of windows**

We use the line direction as an example, assuming parameters as

```
   margin # the number of pixels to neglect at edges
   halfSearchRangeDown # the half of the search range
   windowSizeHeight # the size of the reference window for feature tracking
   skipSampleDown # the skip in pixels between two reference windows
   referenceImageHeight # the reference image height, usually the same as the secondary image height
```

and the number of windows may be computed along lines as

```python
   objOffset.numberWindowDown = (referenceImageHeight-2*margin-2*halfSearchRangeDown-windowSizeHeight) // skipSampleDown
```

If there is a gross offset, you may also need to subtract it when computing the number of windows.

The output offset fields will be of size (numberWindowDown, numberWindowAcross). The total number of windows numberWindows = numberWindowDown\*numberWindowAcross.

**To compute the starting pixels of reference/secondary windows**

The starting pixel for the first reference window is usually set as

```python
   objOffset.referenceStartPixelDownStatic = margin + halfSearchRangeDown
   objOffset.referenceStartPixelAcrossStatic = margin + halfSearchRangeAcross
```

you may also choose other values, e.g., for a particular region of the image, or a certain location for debug purposes.

With a constant gross offset, call

```python
   objOffset.setConstantGrossOffset(grossOffsetDown, grossOffsetAcross)
```
 
to set the starting pixels of all reference and secondary windows.

The starting pixel for the secondary window will be (referenceStartPixelDownStatic-halfSearchRangeDown+grossOffsetDown, referenceStartPixelAcrossStatic-halfSearchRangeAcross+grossOffsetAcross).

For cases you choose a varying grossOffset, you may use two numpy arrays to pass the information to PyCuAmpcor, e.g.,

```python
    objOffset.referenceStartPixelDownStatic = objOffset.halfSearchRangeDown + margin
    objOffset.referenceStartPixelAcrossStatic = objOffset.halfSearchRangeAcross + margin
    vD = np.random.randint(0, 10, size =objOffset.numberWindows, dtype=np.int32)
    vA = np.random.randint(0, 1, size = objOffset.numberWindows, dtype=np.int32)
    objOffset.setVaryingGrossOffset(vD, vA)
```

to set all the starting pixels for reference/secondary windows.

Sometimes, adding a large gross offset may cause the windows near the edge to be out of range of the original image. To avoid memory access errors, call

```python
   objOffset.checkPixelInImageRange()
```

to verify. If an out-of-range error is reported, you may consider to increase the margin or reduce the number of windows.

## 5. List of Procedures

The following procedures apply to one pair of reference/secondary windows, which are iterated through the whole image.

### 5.1 Read a window from Reference/Secondary images

* Load a window of size (windowSizeHeight, windowSizeWidth) from a starting pixel from the reference image

* Load a larger chip of size (windowSizeHeight+2\*halfSearchRangeDown, windowSizeWidth+2\*halfSearchRangeAcross) from the secondary image, the starting position is shifted by (-halfSearchRangeDown, -halfSearchRangeAcross) from the starting position of the reference image (may also be shifted additionally by the gross offset). The secondary chip can be viewed as a set of windows of the same size as the reference window, but shifted in locations varied within the search range.

**Parameters**

| PyCuAmpcor          | CUDA variable       | ampcor.F equivalent   | Notes                     |
| :---                | :---                | :----                 | :---                      |
| windowSizeHeight    | windowSizeHeightRaw | i_wsyi                |Reference window height     |
| windowSizeWidth     | windowSizeWidthRaw  | i_wsxi                |Reference window width      |
| halfSearchRangeDown | halfSearchRangeDownRaw | i_srchy            | half of the search range along lines |
| halfSearchRangeAcross | halfSearchRangeAcrossRaw | i_srchx            | half of the search range along  |


**Difference to ROIPAC**
No major difference


### 5.2 Perform cross-correlation and obtain an offset in units of the pixel size

* Take amplitudes (real) of the signals (complex or real) in reference/secondary windows
* Compute the normalized correlation surface between reference and secondary windows: the resulting correlation surface is of size (2\*halfSearchRangeDown+1, 2\*halfSearchRangeAcross+1); two cross-correlation methods are offered, time domain or frequency domain algorithms.
* Find the location of the maximum/peak in correlation surface.
* Around the peak position, extract a smaller window from the correlation surface for statistics, such as signal-noise-ratio (SNR), variance.

This step provides an initial estimate of the offset, usually with a large search range. In the following, we will zoom in around the peak, and oversample the windows with a smaller search range.


**Parameters**

| PyCuAmpcor         | CUDA variable      | ampcor.F equivalent   | Notes                                                                                         |
|:-------------------|:-------------------| :----                 |:----------------------------------------------------------------------------------------------|
| algorithm          | algorithm          | N/A                   | the cross-correlation computation method 0=Freq (default) 1=time, of the amplitudes; 2=coherence, of the complex windows (see 5.7) |
| workflow           | workflow           | N/A                   | the offset estimation workflow 0=Two-Pass (default) 1=One-Pass                                |
| corrStatWindowSize | corrStatWindowSize | 21               | the size of correlation surface around the peak position used for statistics, may be adjusted |


**Difference to ROIPAC**

* ROIPAC only offers the time-domain algorithm. The frequency-domain algorithm is faster and is set as default in PyCuAmpcor.
* ROIPAC proceeds from here only for windows with *good* match, or with high coherence. To maintain parallelism, PyCuAmpcor proceeds anyway while leaving the *filtering* to users in post processing.


### 5.3 Extract a smaller window from the secondary window for oversampling

* From the secondary window, we extract a smaller window of size (windowSizeHeightRaw+2\*halfZoomWindowSizeRaw, windowSizeWidthRaw+2\*halfZoomWindowSizeRaw) with the center determined by the peak position. If the peak position, e.g., along height, is OffsetInit (taking values in \[0, 2\*halfSearchRangeDownRaw\]), the starting position to extract will be OffsetInit+halfSearchRangeDownRaw-halfZoomWindowSizeRaw.

**Parameters**

| PyCuAmpcor          | CUDA variable       | ampcor.F equivalent   | Notes                     |
| :---                | :---                | :----                 | :---                      |
| N/A                 | halfZoomWindowSizeRaw  | i_srchp(p)=4       |  The smaller search range to zoom-in. In PyCuAmpcor, is determined by zoomWindowSize/(2\*rawDataOversamplingFactor)

**Difference to ROIPAC**

ROIPAC extracts the secondary window centering at the correlation surface peak. If the peak locates near the edge, zeros are padded if the extraction zone exceeds the window range. In PyCuAmpcor, the extraction center may be shifted away from peak to warrant all pixels being in the range of the original window.


### 5.4 Oversampling reference and (extracted) secondary windows

* Oversample both the reference and the (extracted) secondary windows by a factor of 2, which is to avoid aliasing in the complex multiplication of the SAR images. The oversampling is performed with FFT (zero padding), same as in ROIPAC.
* A deramping procedure is in general required for complex signals before oversampling, to shift the band center to 0. The procedure is only designed to remove a linear phase ramp. It doesn't work for TOPSAR, whose ramp goes quadratic. Instead, the amplitudes are taken before oversampling.
* the amplitudes (real) are then taken for each pixel of the complex signals in reference and secondary windows.

**Parameters**

| PyCuAmpcor          | CUDA variable       | ampcor.F equivalent   | Notes                     |
| :---                | :---                | :----                 | :---                      |
| rawDataOversamplingFactor | rawDataOversamplingFactor | i_ovs=2   | the oversampling factor for reference and secondary windows, use 2 for InSAR SLCs. |
| derampMethod        | derampMethod        | 1 or no effect on TOPS | the phase ramp removal method 0=take magnitude (TOPS, default), 1=linear phase ramp, 2=skip deramp. |

**Difference to ROIPAC**

ROIPAC enlarges both windows to a size which is a power of 2; ideal for FFT. PyCuAmpcor uses their original sizes for FFT.

ROIPAC always performs deramping with Method 1, to obtain the ramp by averaging the phase difference between neighboring pixels. For TOPS mode, users need to specify 'mag' as the image *datatype* such that the amplitudes are taken before oversampling. Therefore, deramping has no effect. In PyCuAmpcor, derampMethod=0 is equivalent to *datatype='mag'*, taking amplitudes but skipping deramping. derampMethod=1 always performs deramping, no matter the 'complex' or 'real' image datatypes.

### 5.5 Cross-Correlate the oversampled reference and secondary windows

* cross-correlate the oversampled reference and secondary windows.
* other procedures are needed to obtain the normalized cross-correlation surface, such as calculating and subtracting the mean values.
* the resulting correlation surface is of size (2\*halfZoomWindowSizeRaw\*rawDataOversamplingFactor+1, 2\*halfZoomWindowSizeRaw\*rawDataOversamplingFactor+1). We cut the last row and column to make it an even sequence, or the size 2\*halfZoomWindowSizeRaw\*rawDataOversamplingFactor=zoomWindowSize.

**Parameters**

| PyCuAmpcor          | CUDA variable       | ampcor.F equivalent   | Notes                     |
| :---                | :---                | :----                 | :---                      |
| corrSurfaceZoomInWindow | zoomWindowSize  | i_cw   | The size of correlation surface of the (anti-aliasing) oversampled reference/secondary windows, also used to set halfZoomWindowSizeRaw. Set it to 16 to be consistent with ROIPAC. |

**Difference to ROIPAC**

In ROIPAC, an extra resizing step is performed on the correlation surface, from (2\*halfZoomWindowSizeRaw\*rawDataOversamplingFactor+1, 2\*halfZoomWindowSizeRaw\*rawDataOversamplingFactor+1) to (i_cw, i_cw), centered at the peak (in ROIPAC, the peak seeking is incorporated in the correlation module while is separate in PyCuAmpcor). i_cw is a user configurable variable; it could be smaller or bigger than 2\*i_srchp\*i_ovs+1=17 (fixed), leading to extraction or enlargement by padding 0s. This procedure is not performed in PyCuAmpcor, as it makes little difference in the next oversampling procedure.

### 5.6 Oversample the correlation surface and find the peak position

* oversample the (real) correlation surface by a factor oversamplingFactor, or the resulting surface is of size (zoomWindowSize\*oversamplingFactor, zoomWindowSize\*oversamplingFactor) Two oversampling methods are offered, oversamplingMethod=0 (FFT, default), =1(sinc).
* find the peak position in the oversampled correlation surface, OffsetZoomIn, in range zoomWindowSize\*oversamplingFactor.
* calculate the final offset, from OffsetInit (which is the starting position of secondary window extraction in 2.4),

   offset = (OffsetInit-halfSearchRange)+OffsetZoomIn/(oversamplingFactor\*rawDataOversamplingFactor)

Note that this offset does not include the pre-defined gross offset. Users need to add them together if necessary.


**Parameters**

| PyCuAmpcor          | CUDA variable       | ampcor.F equivalent   | Notes                     |
| :---                | :---                | :----                 | :---                      |
| corrSurfaceOverSamplingFactor | oversamplingFactor  | i_covs   | The oversampling factor for the correlation surface |
| corrSurfaceOverSamplingMethod | oversamplingMethod | i_sinc_fourier=i_sinc | The oversampling method 0=FFT, 1=sinc. |

**Difference to ROIPAC**

ROIPAC by default uses the sinc interpolator (the FFT method is included but one needs to change the FORTRAN code to switch). For since interpolator, there is no difference in implementations. For FFT, ROIPAC always enlarges the window to a size in power of 2.

### 5.7 Coherent cross-correlation (algorithm=2, experimental)

With *algorithm*=2 (*pycuampcor.Algorithm.COHERENCE*, or *"coherence"*), the complex windows are cross-correlated instead of their amplitudes, as for the coregistration of interferometric pairs. The correlation surface is the squared coherence at each shift,

|Σ conj(t − mean t) s|² / (Σ|t − mean t|² · (Σ|s|² − |Σ s|²/n)),

the sums over the n pixels of the reference window t and of the secondary window s under it; it is computed with FFTs, and normalized with sum area tables of the secondary windows. The rest of the two-pass workflow is the same: the peak to the pixel, the windows around it oversampled, correlated again, and the surface oversampled around the peak. Before the oversampling, both windows are deramped with the phase ramp of the reference window, which keeps their relative (interferometric) phase. The peak value output is the squared coherence (for speckle, the correlation of the amplitudes is about the squared coherence too).

* It needs complex images, the two-pass workflow, and *derampMethod* 1 (complex; or 2, none, for images without a Doppler centroid); other settings raise an error.
* It is experimental; the amplitude correlation remains the default. So far:
  * On synthetic images, where the decorrelation is additive noise, it is more precise than the amplitude correlation at moderate coherence (an RMS error of 0.016 vs 0.024 pixel at a coherence of about 0.5), and still works at about 0.2, where the amplitude correlation fails.
  * On a real pair (NISAR, L-band, 12 days, Jakobshavn Isbræ; the roughness of the offsets between neighboring windows), it was about 15% more precise at a coherence of 0.3–0.5, as precise above 0.5, and worse below 0.3, where the phase within the windows varies.
  * Where the coherence is lost (e.g., fast ice, long time spans, C-band in the melt season), it fails: use the amplitudes.
* With *dense_offsets*: *--alg 2 --drmp 1*.
