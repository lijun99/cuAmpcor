# Changelog

## 2.3.0 (2026-10-21)

### License

- pycuampcor is now licensed under the Apache License 2.0 (as ISCE2 and ISCE3), instead of MIT;
  the copyright notice is in `NOTICE`.

### New

- `environment.yml`: a conda environment to build from the source, test, and run the tools and
  examples.
- Documentation: a shorter README (features, installation, quick start, examples, citation), with
  the details in *docs/*: the command-line tools (`tools.md`), the Python classes and their
  parameters (`parameters.md`), the algorithm and its differences from ROI_PAC (`algorithm.md`), and
  the build, tests and code organization (`development.md`); a citation (Zenodo,
  doi:10.5281/zenodo.6614209) in the README and in `CITATION.cff`.
- The coherent cross-correlation (experimental), `algorithm="coherence"` (`Algorithm.COHERENCE`,
  2): the complex windows are correlated instead of their amplitudes, as for the coregistration of
  interferometric pairs, with the squared coherence as the correlation surface (and the peak value
  output); one class template specialization per device, `Correlator<Algorithm::Coherence,
  complex, device>` (FFTs, sum area tables of the secondary windows); in the two-pass workflow,
  both windows are deramped with the phase ramp of the reference window. Complex images, the
  two-pass workflow and `derampMethod` complex (or none) only. On a real NISAR pair, about 15%
  more precise than the amplitudes at a coherence of 0.3-0.5, as precise above, worse below
  (docs/algorithm.md).
- `python -m pycuampcor.plot_offsets`: a first look at the outputs of `dense_offsets` (with the SNR
  file found next to the offsets) and `geo_offsets` (on map coordinates): the two components and the
  SNR, the vectors (`--vectors`) or 3D surfaces (`--3d`), with robust color scales, SNR masking,
  crops, and saved to a file without a display. It replaces `examples/plotOffsets.py`.
- `examples/glacier`: glacier velocities with `dense_offsets`, without and with a velocity model
  for the gross offsets (README): NISAR RSLCs of Jakobshavn Isbræ with isce3 (geometry and
  coregistration; `nisar_isce3/`: `jakobshavn.sh`, `insar_runconfig.yaml`, `nisar_download.py`),
  Sentinel-1 SLCs with ISCE2 topsStack (on the GPU with the gpuresamp branch; `s1_isce2/`:
  `jakobshavn.sh`, `s1_download.py` with the precise orbits, `make_dem.py` for a GLO-30 DEM), and
  `velocity_model.py` for the gross offsets of a velocity model (e.g., ITS_LIVE) and the velocities
  of the offsets.
- `examples/optical`: GeoTIFF and optical images with `geo_offsets` (README), with scripts to search
  and download Sentinel-2 scenes (`s2_download.py`) and measure the slip along a fault, and the
  Ridgecrest 2019 and Kahramanmaraş 2023 earthquakes as examples.
- `python -m pycuampcor.geo_offsets`: the displacement field between two georeferenced images
  (e.g., optical images of an earthquake or a glacier), read with GDAL: the shift of their grids
  from the georeferencing as the gross offset, the windows over the area common to the images (or
  `--bbox`), and a GeoTIFF of the east/north displacements, SNR, peak value and covariance on the
  grid of the window centers, with the windows of no-data pixels masked.
- Images can be rasters read with GDAL (GeoTIFF, VRT, ...), if built with GDAL (`PYCUAMPCOR_GDAL`,
  AUTO by default; `pycuampcor.has_gdal`): the first band, converted to the pixel type of the image
  (e.g., UInt16 to float32, CInt16 to complex64). The `auto` reader chooses GDAL for names ending
  with `.tif`, `.tiff` or `.vrt`; or set `referenceImageReader`/`secondaryImageReader` to `gdal`.
  `pycuampcor.dense_offsets` reads them too (by the extension, or `--reader gdal`), and HDF5
  datasets (`HDF5:<file>:<dataset>`, e.g., NISAR RSLCs, or `--reader hdf5`), their size and type
  read with h5py.
- `python -m pycuampcor.dense_offsets`, the dense offsets tool (formerly `cuDenseOffsets.py` of
  ISCE2, used with MintPy) in the package: ISCE2 is optional (the image
  size and type from the ISCE .xml, GDAL, or the options `--length`, `--width`, `--data-type`;
  the output headers ISCE .xml, or ENVI .hdr and .vrt), `--cpu` for the CPU implementation, and
  the numbers of windows in a chunk automatic by default.
- `examples/simple_offsets.py`: a self-contained example (synthetic images of a known shift); the
  examples depending on ISCE2 or a GDAL reader (`GeoTiffSample.py`, `glacierSample.py`,
  `varyGrossOffsetSample.py`, `grossOffsets.py`) are removed.
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

- The C++ code is organized as the shared code (namespace `pycuampcor`: parameters, options,
  logging, image readers, layers), compiled once, and the CPU and CUDA packages
  (`pycuampcor::cpu`, `pycuampcor::cuda`), each with its arrays, kernels, controller and chunk
  loader; the processors of the workflows are written once, as templates on the array type, and
  instantiated by each package. The results are unchanged.
- The C++ files and classes drop the `cuAmpcor` prefix (e.g., `Parameter`, `Processor`,
  `Controller`, `ChunkLoader`); the files of the CUDA package keep its `cu` prefix
  (`cuController.h`, `cuChunkLoader.h`). The python API is unchanged.
- The oversamplers are one class template, `OverSampler<method, T, device>` (FFT or sinc, the
  element type, CPU or CUDA), specialized by the packages for the complex windows (FFT) and the
  real correlation surfaces (FFT, sinc); the processors take the device from their arrays.
- The cross-correlators are one class template too, `Correlator<algorithm, T, device>`
  (frequency or time domain, the element type, CPU or CUDA), with an interface for the algorithm
  chosen at run time (`WindowCorrelator`); other correlators (e.g., the coherence of complex
  windows) can be added as specializations.
- Compiled as standard C++17 (`-std=c++17`, without GNU extensions); the code also builds as
  C++20.
- `setupParams` keeps the parameters as set: `numberWindowDownInChunk` / `numberWindowAcrossInChunk`
  read 0 (automatic) after the setup; the numbers of windows in a chunk used are the new read-only
  `windowsDownInChunk` / `windowsAcrossInChunk`.
- `PyCuAmpcor.device_list()` returns the devices (dicts with `id`, `name`, `sm_count`,
  `compute_capability` and `total_memory`) instead of printing them.
- The CUDA utilities adapted from the NVIDIA CUDA samples are replaced by our own: CUDA and cuFFT
  errors are reported with their names and the failed call; in Debug builds, the device is
  synchronized after each kernel, so that execution errors are reported at the failing kernel.
- The SCons build files (`SConscript`, `src/SConscript`), out of date, are removed; pycuampcor
  builds with CMake (ISCE2 keeps its own build files in `contrib/pycuampcor`).

### Fixes

- `dense_offsets --out-geom`: the geometry is taken at the centers of the windows; it was half a
  window step after them (GDAL's resampling of blocks starting at the window centers, as in
  `cuDenseOffsets.py` of ISCE2).
- Debug builds: the CPU package dumps the intermediate arrays (as the CUDA package), and
  `checkPixelInImageRange` warns about the windows outside the images (both were compiled out:
  the debug flag wasn't included there).
- The time-domain correlation on the GPU (`algorithm=1`) computed only as many rows of the
  correlation surfaces as they have columns: with `halfSearchRangeDown` > `halfSearchRangeAcross`,
  the lower rows were missed (wrong offsets for peaks there); with a wider surface, the extra
  thread blocks read beyond the windows. The CPU backend was not affected.
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
