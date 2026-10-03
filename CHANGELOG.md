# Changelog

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
