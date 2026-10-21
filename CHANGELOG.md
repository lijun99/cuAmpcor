# Changelog

## Unreleased

### License

- pycuampcor is now licensed under the Apache License 2.0 (as ISCE2 and ISCE3), instead of MIT;
  the copyright notice is in `NOTICE`.

### New

- The parameters can be given to the constructor as keyword arguments (the property names, e.g.,
  `PyCuAmpcor(windowSizeHeight=64, referenceImageName="ref.slc", ...)`), with the gross offsets
  `grossOffsetDown` / `grossOffsetAcross` (constant or varying); the constructor sets the object
  up to run (`setupParams` and the gross offsets).
- File name parameters accept path-like objects; a value of a wrong type raises a `TypeError`
  naming the parameter.
- Options are integer enums (`pycuampcor.Algorithm`, `Workflow`, `DerampMethod`, `DerampAxis`,
  `OversamplingMethod`, `DataType`, in `pycuampcor.options`): `algorithm`, `workflow`,
  `derampMethod`, `derampAxis`, `corrSurfaceOverSamplingMethod`, `referenceImageDataType` and
  `secondaryImageDataType` are set by the member, the integer value as before, or the name
  (e.g., `workflow="one_pass"`, `derampAxis="range"`), and read as the member, equal to the
  integer value. An invalid option raises a `ValueError` when set (instead of in `setupParams`).
- `useMmap` and `mergeGrossOffset` are booleans (set by a bool or an integer).
- Docstrings for the class, its methods and all the parameters (with their default values), and
  argument names for `setConstantGrossOffset` / `setVaryingGrossOffset`.
- `pycuampcor.set_log_level` / `get_log_level`: the level of the messages written (`quiet`,
  `warning`, `info` (default) or `debug`), for both backends; all the messages go through it
  (warnings to stderr, the others to stdout). The one-pass zoom-in window size is now a debug
  message.
- Type stubs (`_cpu.pyi`, `_cuda.pyi`, with `py.typed`) for type checkers and IDEs: the types of
  the parameters (as set and as read) and of the methods, the keyword constructor with all the
  parameters, and the docstrings.

### Changes

- `setupParams` keeps the parameters as set: `numberWindowDownInChunk` / `numberWindowAcrossInChunk`
  read 0 (automatic) after the setup; the numbers of windows in a chunk used are the new read-only
  `windowsDownInChunk` / `windowsAcrossInChunk`.
- `PyCuAmpcor.device_list()` returns the devices (dicts with `id`, `name`, `sm_count`,
  `compute_capability` and `total_memory`) instead of printing them.
- The CUDA utilities adapted from the NVIDIA CUDA samples are replaced by our own: CUDA and cuFFT
  errors are reported with their names and the failed call; in Debug builds, the device is
  synchronized after each kernel, so that execution errors are reported at the failing kernel.

### Fixes

- `setupParams` can be called repeatedly: the one-pass workflow no longer adds its zoom-in margin
  to `halfSearchRangeDown` / `halfSearchRangeAcross`.
- A repeated `setupParams` (e.g., after changing `deviceID`, the numbers of windows or the search
  range) chooses the automatic numbers of windows in a chunk again, and no longer keeps the
  zoom-in window (`corrSurfaceZoomInWindow`) limited by a previous, smaller search range.
- A CUDA error is cleared once reported, so it is no longer reported again by a later kernel
  (e.g., after an invalid `deviceID`).

## 2.2.0 (2026-10-02)

### New

- `runAmpcorLayers`: several layers (different windows and search ranges on the same grid,
  e.g., the isce3 offsets_product layers) are processed together, loading each image chunk once.
- Images can be read directly from 2D datasets in HDF5 files, named `HDF5:<file>:<dataset>`;
  chunks compressed with gzip and shuffle (as in the NISAR products) are decoded in parallel
  into a shared cache, other layouts are read with the HDF5 library. `pycuampcor.has_hdf5`
  tells whether HDF5 support is built (CMake option `PYCUAMPCOR_HDF5`: AUTO, ON or OFF);
  `PYCUAMPCOR_HDF5_THREADS` caps the decoding threads (default 8).
- `referenceImageReader` / `secondaryImageReader`: choose the image reader (`auto`, the default,
  by the name; `raw`; `hdf5`).
- Automatic windows per chunk: `numberWindowDownInChunk` / `numberWindowAcrossInChunk` = 0, now
  the default, picks (SM/4) x 8 windows on the GPU and 1 x 1 on the CPU.
- `PyCuAmpcor.device_list` is a static method.

### Performance

- GPU: buffers are allocated once per worker instead of for each chunk (freeing them for every
  chunk synchronized the device and kept the CUDA streams from overlapping); chunk loads are
  staged through page-locked memory and copied asynchronously (pageable memory as a fallback).
- CPU: vectorized time-domain correlation and peak search; a pruned inverse FFT for the FFT
  oversampling of correlation surfaces (about 4x faster with FFT oversampling at x64).

### Changes in results

- FFT oversampling follows the convention of the isce3 internal code for the Nyquist frequency
  of even-length windows (the transform directions of isce3): two-pass results are bitwise
  identical to isce3 develop with its bugs fixed. Compared with 2.1.0, the sub-pixel peak moves
  by one oversampling step in some windows; the regression goldens are regenerated.
- One-pass workflow: the SNR is estimated as in the two-pass workflow (statistics window at the
  raw spacing within the search range, count of valid pixels); one-pass SNR values change.
- The pruned FFT oversampler on the CPU changes the correlation surface at the rounding level,
  which moves a few near-tie peaks by one oversampling step.

### Fixes

- Shared-memory races in the CUDA block reductions: GPU results are now deterministic.
- The mmap window of raw images is enlarged to chunks that need more than `mmapSize`
  (the automatic batch with a large skip failed).
- `getSMCount` uses the given device.
- Portability: `posix_memalign` (macOS), `<cstdlib>` includes, OpenMP runtime in the recipes.

### Packaging

- Project home moved to https://github.com/earthdef/cuAmpcor.
- conda recipes build with HDF5 and ninja, and test with h5py; CI on linux-64, linux-aarch64,
  osx-64 and osx-arm64.
