# PyCuAmpcor

PyCuAmpcor (the Python package `pycuampcor`) measures dense offsets between two images by
amplitude cross-correlation: the AMPCOR method of ROI_PAC and ISCE, also known as offset, pixel,
speckle or feature tracking. It runs on NVIDIA GPUs with CUDA and on CPUs with OpenMP, with the same
interface and the same results. Typical uses are glacier and landslide velocities, earthquake
displacements, and the coregistration of SAR images.

A GPU version of PyCuAmpcor (version 1) is included in [ISCE2](https://github.com/isce-framework/isce2)
(*contrib/PyCuAmpcor*) and in [isce3](https://github.com/isce-framework/isce3). This repository is the
standalone package, version 2.

## Features

* **GPU and CPU:** `PyCuAmpcor` (CUDA, cuFFT) and `PyCPUAmpcor` (OpenMP, FFTW), with the same
  parameters and results (to floating-point round-off).
* **Sub-pixel offsets:** the windows are oversampled (antialiasing) and the correlation surface is
  oversampled around its peak (FFT or sinc). A two-pass workflow (the default) first finds the peak to
  the pixel over the full search range; a one-pass workflow oversamples the whole search range.
* **Images:** raw binary files (memory mapped), datasets in HDF5 files (e.g., NISAR RSLCs, decoded in
  parallel without a copy), and rasters read with GDAL (GeoTIFF, VRT and more, of any pixel type).
* **Gross offsets:** a constant shift, or one per window (e.g., from a velocity model), to keep the
  search range small.
* **Layers:** several window sizes processed together, reading the images once.
* **Command-line tools:** `dense_offsets` for images in radar geometry (with the outputs of
  `cuDenseOffsets.py` of ISCE2, read by MintPy), `geo_offsets` for georeferenced images (east and north
  displacements as a GeoTIFF), and `plot_offsets` for a first look at the results.
* **Experimental:** the coherent cross-correlation of complex windows.

## Installation

From conda-forge, for Linux (x86_64, aarch64) and macOS (x86_64, arm64):

```bash
conda install -c conda-forge pycuampcor
```

Conda installs the `cuda` variant (CPU and GPU) if it finds an NVIDIA driver, and the `cpu` variant
otherwise. To choose one, use `pycuampcor=*=cpu*` or `pycuampcor=*=cuda*`.

From the source, with pip or CMake (a C++17 compiler, CMake ≥ 3.18, pybind11 and FFTW; optionally
CUDA ≥ 11, OpenMP, HDF5 and GDAL). [environment.yml](environment.yml) is a conda environment with all
of them (but CUDA) and the packages of the tests and examples:

```bash
conda env create -f environment.yml && conda activate pycuampcor
pip install . --no-deps --no-build-isolation                    # CUDA if nvcc is found
pip install . -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=native    # for the GPUs of this machine
pip install . -Ccmake.define.PYCUAMPCOR_CUDA=OFF                # CPU only
```

See [docs/development.md](docs/development.md) for the build options, the tests and the code
organization.

## Quick start

### Python

```python
import pycuampcor

# two complex64 images of 2000 x 1000 pixels, in raw binary files;
# windows of 64 x 64 pixels every 64 pixels, searched +/- 20 pixels
ampcor = pycuampcor.PyCuAmpcor(            # or pycuampcor.PyCPUAmpcor
    referenceImageName="reference.slc", referenceImageHeight=2000, referenceImageWidth=1000,
    secondaryImageName="secondary.slc", secondaryImageHeight=2000, secondaryImageWidth=1000,
    windowSizeHeight=64, windowSizeWidth=64,
    halfSearchRangeDown=20, halfSearchRangeAcross=20,
    skipSampleDown=64, skipSampleAcross=64,
    referenceStartPixelDownStatic=20, referenceStartPixelAcrossStatic=20,
    numberWindowDown=29, numberWindowAcross=14,
    offsetImageName="offset.bip", snrImageName="snr.bip",
    grossOffsetDown=0, grossOffsetAcross=0)
ampcor.runAmpcor()
```

The offsets are written to `offset.bip`: float32 pairs (down, across) in pixels, one per window, 29 x
14. [examples/simple_offsets.py](examples/simple_offsets.py) is a complete example on synthetic images
with a known shift, and [docs/parameters.md](docs/parameters.md) describes all the parameters.

### Command line

```bash
# SAR images in radar geometry (raw files with ISCE .xml or GDAL .vrt headers, HDF5 datasets, GeoTIFF)
python -m pycuampcor.dense_offsets -r reference.slc -s secondary.slc --op offsets/offset \
    --ww 64 --wh 64 --sw 20 --sh 20 --kw 32 --kh 32

# georeferenced images, e.g., optical images: east and north displacements in a GeoTIFF
python -m pycuampcor.geo_offsets -r before.tif -s after.tif --op offsets/displacement

# a first look at the results
python -m pycuampcor.plot_offsets offsets/offset.bip --min-snr 5
```

The tools run on the GPU, or on the CPU with `--cpu` (or if the package is built without CUDA). See
[docs/tools.md](docs/tools.md).

## Examples

| Example | What it shows |
|---|---|
| [examples/simple_offsets.py](examples/simple_offsets.py) | The Python API on synthetic images with a known shift |
| [examples/optical](examples/optical) | GeoTIFF and optical images: Sentinel-2 offsets of the Ridgecrest 2019 and Kahramanmaraş 2023 earthquakes, and the fault slip |
| [examples/glacier](examples/glacier) | Glacier velocities from SAR images, without and with a velocity model for the gross offsets: NISAR with isce3, and Sentinel-1 with ISCE2 topsStack |

## Documentation

* [docs/tools.md](docs/tools.md): the command-line tools `dense_offsets`, `geo_offsets` and
  `plot_offsets`.
* [docs/parameters.md](docs/parameters.md): the Python classes, their parameters and methods, the
  image readers, the window grid and the gross offsets.
* [docs/algorithm.md](docs/algorithm.md): the method, the two workflows, each step and its
  differences from ampcor of ROI_PAC, and the coherent cross-correlation.
* [docs/development.md](docs/development.md): building, testing, and the organization of the code.
* [CHANGELOG.md](CHANGELOG.md): the changes of each release.

## With ISCE2, isce3 and MintPy

* **ISCE2** includes PyCuAmpcor version 1 (*contrib/PyCuAmpcor*, GPU only), used by *topsApp* and
  *alos2App*. This package can be installed in the same environment. Its `dense_offsets` tool replaces
  ISCE2's *cuDenseOffsets.py*: it reads the ISCE *.xml* headers if ISCE2 is installed, and writes the
  same outputs.
* **isce3** includes PyCuAmpcor version 1 for the `offsets_product` and `dense_offsets` steps of the
  NISAR workflows. The [glacier example](examples/glacier/nisar_isce3) uses isce3 for the geometry and
  the coregistration, and this package for the offsets.
* **MintPy** reads the outputs of `dense_offsets` (*.bip* with ISCE *.xml* or ENVI *.hdr* headers) for
  offset time series.

## Citation

If you use PyCuAmpcor in your work, please cite it:

> Zhu, L., Zhong, M., & Simons, M. (2022). *PyCuAmpcor - Amplitude Cross-Correlation with GPU*
> (Version 1.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.6614209

```bibtex
@software{zhu_pycuampcor_2022,
  author    = {Zhu, Lijun and Zhong, Minyan and Simons, Mark},
  title     = {{PyCuAmpcor - Amplitude Cross-Correlation with GPU}},
  year      = {2022},
  publisher = {Zenodo},
  version   = {1.0},
  doi       = {10.5281/zenodo.6614209},
  url       = {https://doi.org/10.5281/zenodo.6614209}
}
```

## License

PyCuAmpcor is licensed under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE) for the copyright
notice.
