# Development: building, testing, and the code

## Building

The build needs a C++17 compiler, CMake ≥ 3.18, pybind11 and FFTW (single precision, or double with
`PYCUAMPCOR_DOUBLE`), and optionally:

* CUDA ≥ 11, for the GPU implementation;
* OpenMP, for the parallel CPU implementation;
* HDF5 (≥ 1.14 for the fast reading of compressed datasets) with zlib, to read HDF5 datasets;
* GDAL, to read rasters such as GeoTIFF.

### With pip

```bash
pip install .
```

The CMake options are passed with `-C`:

```bash
pip install . -Ccmake.define.PYCUAMPCOR_CUDA=OFF                # CPU only
pip install . -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=native    # CUDA for the GPUs of this machine
```

### With CMake

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release
cmake --build build -j && cmake --install build
```

| Option | Default | Notes |
|:---|:---|:---|
| `PYCUAMPCOR_CUDA` | AUTO | build the CUDA implementation: AUTO (if a CUDA compiler is found), ON or OFF |
| `PYCUAMPCOR_HDF5` | AUTO | read HDF5 datasets: AUTO (if HDF5 and zlib are found), ON or OFF |
| `PYCUAMPCOR_GDAL` | AUTO | read rasters with GDAL: AUTO (if GDAL is found), ON or OFF |
| `PYCUAMPCOR_DOUBLE` | OFF | double precision for the computations and the outputs |
| `PYCUAMPCOR_BUILD_TESTS` | OFF | build the C++ unit tests (needs GoogleTest) |
| `PYCUAMPCOR_INSTALL_DIR` | `site-packages/pycuampcor` | where to install the Python package |
| `CMAKE_CUDA_ARCHITECTURES` | | the GPU architectures, e.g., `native` (the GPUs of this machine), `70` (V100), `80` (A100), `120` (RTX PRO 6000 Blackwell) |

If nvcc is not in the `PATH`, give it with `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` (or
`-Ccmake.define.CMAKE_CUDA_COMPILER=...` with pip).

### Debug builds

With `-DCMAKE_BUILD_TYPE=Debug`, the processors write the intermediate arrays of each chunk (the
windows, the correlation surfaces, the peaks) to files in the working directory, e.g., to inspect one
window (run with one window, or one window per chunk). It is much slower.

### Conda packages

The conda-forge packages are built from tagged releases by the
[pycuampcor-feedstock](https://github.com/conda-forge/pycuampcor-feedstock), in two variants: `cpu`
(`PyCPUAmpcor` only) and `cuda` (`PyCPUAmpcor` and `PyCuAmpcor`).

## Testing

The tests are in

* *tests/cxx*: C++ unit tests (GoogleTest) of the kernels and workers (copying, statistics, offsets,
  FFT oversamplers and correlators, normalization, deramping, sinc interpolation, parameters, image
  readers), each checked against a simple reference. The sources are compiled for each implementation
  (`test_cpu`, `test_cuda`); the CUDA tests are skipped without a GPU.
* *tests/python*: the tests of the Python package, e.g.,
  * *test_synthetic.py*: accuracy and robustness with synthetic images of known sub-pixel shifts, for
    both workflows and both oversampling methods, and the CPU/GPU consistency;
  * *test_regression.py*: regression tests with the *ovs128-rho0.8* data (from isce3), against the
    isce3 outputs and against reference outputs of this package (*tests/data/ampcor/ovs128-rho0.8/golden*);
  * *test_api.py*: the parameters, the gross offsets, real images, and the errors;
  * *test_layers.py*: several layers run together with `runAmpcorLayers`, identical to separate runs;
  * *test_hdf5.py*, *test_gdal.py*: images read from HDF5 datasets and with GDAL, identical to the same
    images in raw files;
  * *test_coherence.py*: the coherent cross-correlation;
  * *test_dense_offsets.py*, *test_geo_offsets.py*, *test_plot_offsets.py*: the command-line tools.

```bash
# C++ unit tests (both implementations if CUDA is found)
cmake -S . -B build -DPYCUAMPCOR_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake --build build -j && ctest --test-dir build

# Python tests, after the installation (the GPU tests are skipped without a GPU)
pytest tests/python
```

The GitHub Actions CI (*.github/workflows/tests.yml*) tests the CPU implementation only, as the
GitHub-hosted runners have no GPU. Test the CUDA implementation locally, with the CUDA toolkit found by
CMake (e.g., in */usr/local/cuda*).

If a change of the results is intended, regenerate the reference outputs with
`python tests/python/make_golden.py`, and commit them.

The package has type stubs (`_cpu.pyi`, `_cuda.pyi`) for type checkers and IDEs, generated from the
bindings. If the bindings change (parameters, methods or docstrings), regenerate them with
`python tests/python/make_stubs.py` (with pycuampcor built with CUDA), and commit them.
*test_stubs.py* checks that they are up to date, and
`python -m mypy.stubtest pycuampcor._cpu pycuampcor._cuda --allowlist tests/python/stubtest_allowlist.txt`
checks them against the modules.

## The code

* *cxx/pycuampcor/common*: the shared code, in namespace `pycuampcor`, compiled once (library
  `pycuampcor_common`): the parameters and their options, logging, the image readers (raw files, HDF5
  datasets and GDAL rasters), and the layers processed together (chunk footprints); and the processors
  of the two-pass and one-pass workflows, written once as templates on the array type (*Processor\*.h*,
  with the definitions in *Processor\*Impl.h*, and the types that go with the arrays in `array_traits`),
  instantiated by each package (*processors.cpp*). The workers specialized by the packages are class
  templates on the method, the data type and the device: `OverSampler<method, T, device>` and
  `Correlator<algorithm, T, device>`.
* *cxx/pycuampcor/cpu*: the CPU package, in namespace `pycuampcor::cpu` (library `pycuampcor_cpu`):
  arrays and kernels, FFTs with FFTW, and the controller and chunk loader, processing chunks in parallel
  with OpenMP threads.
* *cxx/pycuampcor/cuda*: the CUDA package, in namespace `pycuampcor::cuda` (library `pycuampcor_cuda`):
  arrays and kernels, FFTs with cuFFT, and the controller and chunk loader, processing chunks on CUDA
  streams.
* *python/extensions*: the pybind11 bindings, compiled to `pycuampcor._cpu` and `pycuampcor._cuda`.
* *python/packages/pycuampcor*: the Python package, with the command-line tools.
* *tests*: the tests and their data.
* *examples*: the examples.

The arrays are batches of 2D arrays: `ArraysView<T>` (*common/ArraysView.h*) is a non-owning view
(a pointer and the count, height and width, row major) passed to the kernels by value;
`cpu::Arrays<T>` and `cuda::cuArrays<T>` own the memory (move only) and convert to views.

The headers are organized as in [pyre](https://github.com/pyre/pyre): in each of *common/*, *cpu/* and
*cuda/*, *externals.h* (the external headers), *forward.h* (the forward declarations), *api.h* (the
public type aliases) and *public.h* (everything, in dependency order); those of *cpu/* and *cuda/*
include those of *common/*. The clients (the Python bindings, the C++ tests) include *public.h*, with
the include directories of the library `pycuampcor_cpu` or `pycuampcor_cuda` (the package's directory,
then *common/*). Each header includes *forward.h* and the headers of the types it needs complete, and
compiles on its own; each source includes its own header and the headers it uses, so that changing a
header recompiles only the sources that use it.
