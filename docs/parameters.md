# The Python classes and their parameters

The package `pycuampcor` has two classes with the same parameters and methods:

* `pycuampcor.PyCuAmpcor`: the CUDA implementation (if built with CUDA, `pycuampcor.has_cuda`);
* `pycuampcor.PyCPUAmpcor`: the CPU implementation (OpenMP), always available.

Each class has docstrings for all its parameters (with their defaults) and methods, e.g.,
`help(pycuampcor.PyCPUAmpcor)`, and type stubs for IDEs and type checkers.

## Using the classes

Set the parameters as properties, set up the processing, set the gross offsets, and run:

```python
import pycuampcor

ampcor = pycuampcor.PyCuAmpcor()        # or pycuampcor.PyCPUAmpcor()
ampcor.referenceImageName = "20151120.slc.full"
ampcor.referenceImageHeight = 2000
# ... the other parameters (see below)
ampcor.setupParams()
ampcor.setConstantGrossOffset(0, 0)     # or setVaryingGrossOffset(down, across), one per window
ampcor.runAmpcor()
```

Or give the parameters to the constructor as keyword arguments, with the gross offsets as
`grossOffsetDown` and `grossOffsetAcross` (numbers for a constant gross offset, or sequences of
numberWindowDown x numberWindowAcross numbers; 0 by default). The object is then set up and ready to
run:

```python
ampcor = pycuampcor.PyCuAmpcor(referenceImageName="20151120.slc.full", windowSizeWidth=64, ...,
                               grossOffsetDown=0, grossOffsetAcross=0)
ampcor.runAmpcor()
```

Unknown parameter names, and values of a wrong type, raise a `TypeError`; invalid values raise a
`ValueError`. File names may be strings or path-like objects (e.g., `pathlib.Path`). The options
(e.g., `workflow`, `derampMethod`) are integer enums of `pycuampcor.options`, set by the member, the
integer value, or the name (e.g., `"one_pass"`).

[examples/simple_offsets.py](../examples/simple_offsets.py) is a complete example on synthetic
images, and [pycuampcor/dense_offsets.py](../python/packages/pycuampcor/dense_offsets.py) a complete
tool.

## Parameters

The defaults are those of a new object.

### Images

| Parameter | Default | Notes |
|:---|:---|:---|
| referenceImageName | `"reference.slc"` | the reference (template) image: a raw binary file, `HDF5:<file>:<dataset>`, or a GDAL raster (see [Image readers](#image-readers)) |
| referenceImageReader | `"auto"` | how to read it: `auto` (by the name), `raw`, `hdf5` or `gdal` |
| referenceImageHeight, referenceImageWidth | 1000, 1000 | its size (lines, columns) |
| referenceImageDataType | `complex64` | its pixel type (`DataType`): `complex64` (2) or `float32` (1) |
| secondaryImageName | `"secondary.slc"` | the secondary (search) image, as the reference |
| secondaryImageReader | `"auto"` | how to read it |
| secondaryImageHeight, secondaryImageWidth | 1000, 1000 | its size |
| secondaryImageDataType | `complex64` | its pixel type |

### Outputs

The outputs are raw binary files, one value (or pair, triple) per window, numberWindowDown x
numberWindowAcross, row major; float32 (float64 if built with `PYCUAMPCOR_DOUBLE`,
`isDoublePrecision()`).

| Parameter | Default | Notes |
|:---|:---|:---|
| offsetImageName | `"DenseOffset.bip"` | the offsets: (down, across) pairs, in pixels |
| grossOffsetImageName | `"GrossOffset.bip"` | the gross offsets: (down, across) pairs |
| snrImageName | `"snr.bip"` | the signal-to-noise ratio of the correlation peak |
| covImageName | `"cov.bip"` | the covariance of the offsets: (down variance, across variance, covariance) triples |
| peakValueImageName | `"peakValue.bip"` | the normalized correlation peak value (the squared coherence with the coherence algorithm); also `corrImageName`, the name used by isce3 |
| mergeGrossOffset | `False` | whether the offsets include the gross offsets |

### Windows and search

| Parameter | Default | Notes |
|:---|:---|:---|
| windowSizeHeight, windowSizeWidth | 64, 64 | the size of the reference (template) windows |
| halfSearchRangeDown, halfSearchRangeAcross | 20, 20 | half of the search range: the secondary windows are larger by twice this range |
| skipSampleDown, skipSampleAcross | 64, 64 | the distance between neighboring windows |
| numberWindowDown, numberWindowAcross | 1, 1 | the number of windows |
| referenceStartPixelDownStatic, referenceStartPixelAcrossStatic | 0, 0 | the first pixel of the first reference window |

### Correlation

| Parameter | Default | Notes |
|:---|:---|:---|
| workflow | `two_pass` | `two_pass` (0) or `one_pass` (1); see [the algorithm](algorithm.md) |
| algorithm | `frequency` | the cross-correlation of the amplitudes, `frequency` (0, FFT) or `time` (1); or `coherence` (2, experimental), of the complex windows |
| rawDataOversamplingFactor | 2 | the antialiasing oversampling of the windows (2 for SLCs) |
| derampMethod | `magnitude` | before the oversampling: `magnitude` (0, take the amplitudes, e.g., for TOPS), `complex` (1, remove a linear phase ramp) or `none` (2) |
| derampAxis | `both` | the axes of the ramp: `down` (0), `across` (1) or `both` (2) |
| corrSurfaceZoomInWindow | 16 | the size of the correlation surface (at the oversampled spacing) around the peak to oversample; 16 as in ROI_PAC |
| corrSurfaceOverSamplingFactor | 16 | the oversampling factor of the correlation surface |
| corrSurfaceOverSamplingMethod | `fft` | `fft` (0) or `sinc` (1) |
| corrStatWindowSize | 21 | the size of the correlation surface around the peak for the SNR and the covariance |

The sub-pixel step of the offsets is 1 / (corrSurfaceOverSamplingFactor x rawDataOversamplingFactor)
pixel: 1/32 pixel with the defaults.

### Processing

| Parameter | Default | Notes |
|:---|:---|:---|
| deviceID | 0 | the CUDA device (GPU only); `CUDA_VISIBLE_DEVICES` also selects the devices |
| nStreams | 1 | the number of CUDA streams (GPU only), each processing a chunk, overlapping the computations with the data transfers; 2 is usually enough, more use more memory |
| nThreads | 0 | the number of threads (CPU only), each processing a chunk; 0 for the OpenMP default (e.g., `OMP_NUM_THREADS`) |
| numberWindowDownInChunk, numberWindowAcrossInChunk | 0, 0 | the number of windows in a chunk (processed together); 0 for automatic |
| mmapSize | 1 | the size in GB of the memory map of a raw image, or of the cache of decoded chunks of an HDF5 image; enlarged if a chunk needs more; at most about 1/4 of the memory |
| useMmap | `True` | unused (raw images are always memory mapped); kept for compatibility |

The windows are processed in chunks. With the automatic size (0), `setupParams()` chooses, limited by
the numbers of windows: on the GPU, twice the number of streaming multiprocessors (SMs), as (SM/4) x 8
windows (down x across), which was found to be the fastest, within a few percent, on V100 and RTX PRO
6000 Blackwell GPUs (the number of windows matters more than the shape of the chunk); on the CPU, one
window per chunk, each thread processing chunks one after another. After `setupParams()`, the sizes
used are `windowsDownInChunk` and `windowsAcrossInChunk`.

## Image readers

The reader of each image is chosen by referenceImageReader / secondaryImageReader: `auto` (the
default) reads the names starting with `HDF5:` as HDF5 datasets, the names ending with `.tif`, `.tiff`
or `.vrt` with GDAL, and the other names as raw binary files; `raw`, `hdf5` and `gdal` choose the
reader explicitly. In all cases the image size and pixel type are given by the parameters.

* **Raw binary files** (e.g., ISCE or ENVI images, or the data file of a VRT raw raster): one band of
  complex64 or float32 pixels, row major, read with a memory map of mmapSize GB.
* **HDF5 datasets** (if built with HDF5, `pycuampcor.has_hdf5`): 2D datasets in HDF5 files, named as in
  GDAL, `HDF5:<file>:<dataset>` (or `HDF5:"<file>":<dataset>` if the file name contains `:`), e.g.,
  `HDF5:rslc.h5:/science/LSAR/RSLC/swaths/frequencyA/HH`; with the `hdf5` reader, the `HDF5:` prefix is
  optional. The dataset must be float32, or complex64 (a compound of two float32, as written by h5py
  and in the NISAR products). Chunked datasets, uncompressed or compressed with deflate (gzip) with or
  without shuffle (as the NISAR products), are read without the HDF5 library: the chunks are located
  once, read and decoded in parallel, and kept in a cache of mmapSize GB shared by the workers
  (`PYCUAMPCOR_HDF5_THREADS` caps the decoding threads, 8 by default). Other datasets (contiguous,
  other filters, files with a user block) are read with the HDF5 library. No intermediate copy of the
  image is made.
* **GDAL rasters** (if built with GDAL, `pycuampcor.has_gdal`): GeoTIFF, VRT, or any raster of GDAL.
  The first band is read, and converted by GDAL to the pixel type of the image: a real band of any type
  (e.g., Byte, UInt16, Int16, Float64) to float32 (referenceImageDataType / secondaryImageDataType =
  `float32`), a complex band (e.g., CInt16, CFloat32) to complex64. The reads of an image are
  serialized among the workers; raw files and HDF5 datasets remain the faster readers for large
  images. Recent GDAL versions read the raw files of VRTs (VRTRawRasterBand) only in the
  directory of the VRT (or below) by default: give the file relative to the VRT (`relativeToVRT="1"`, as
  ISCE2 and `dense_offsets` write them), or set the GDAL option `GDAL_VRT_RAWRASTERBAND_ALLOWED_SOURCE`.

## The window grid and the gross offsets

The program takes the number of windows (numberWindowDown, numberWindowAcross), the distance between
them (skipSampleDown, skipSampleAcross), and the first pixel of the first reference window
(referenceStartPixelDownStatic, referenceStartPixelAcrossStatic). The reference window (i, j) starts
at

    (referenceStartPixelDownStatic + i skipSampleDown, referenceStartPixelAcrossStatic + j skipSampleAcross),

and its secondary (search) window starts halfSearchRange pixels before, shifted by its gross offset.
The tools `dense_offsets` and `geo_offsets` compute the grid; with the classes, compute it, e.g., down:

```python
# the first window after the margin and the search range
ampcor.referenceStartPixelDownStatic = margin + halfSearchRangeDown
# as many windows as fit, with the search range and the margin at the end
ampcor.numberWindowDown = (referenceImageHeight - 2*margin - 2*halfSearchRangeDown - windowSizeHeight) // skipSampleDown
```

and the same across. With a gross offset, make the margin at least the gross offset.

After `setupParams()`, set the gross offsets (in pixels):

* a constant gross offset, for all windows: `setConstantGrossOffset(grossOffsetDown, grossOffsetAcross)`;
* a gross offset per window, e.g., from a velocity model (see [examples/glacier](../examples/glacier)):
  `setVaryingGrossOffset(down, across)`, two sequences of numberWindowDown x numberWindowAcross
  integers, in the order of the outputs (row major).

The windows may extend beyond the images, which are padded with zeros; the results of these windows
may be wrong. `checkPixelInImageRange()` warns about them (in debug builds only).

## Methods

| Method | Notes |
|:---|:---|
| `setupParams()` | set up the processing from the parameters, after they are set |
| `setConstantGrossOffset(down, across)` | a constant gross offset, after setupParams |
| `setVaryingGrossOffset(down, across)` | a gross offset per window, after setupParams |
| `runAmpcor()` | run, and write the outputs |
| `runAmpcorLayers(layers)` (static) | run several objects (layers, e.g., different window sizes and search ranges) with the same images and numbers of windows, loading each chunk of the images once; each layer is set up as for runAmpcor, and the device and the number of streams or threads are those of the first layer |
| `checkPixelInImageRange()` | warn about windows outside the images (debug builds only) |
| `isDoublePrecision()` | whether the computations and outputs are in double precision |
| `PyCuAmpcor.device_list()` (static) | the CUDA devices, as dicts with `id`, `name`, `sm_count`, `compute_capability`, `total_memory` |

The messages (the progress to stdout, the warnings to stderr) are chosen with
`pycuampcor.set_log_level`, e.g., `pycuampcor.set_log_level("warning")` for the warnings only
(levels `quiet`, `warning`, `info` (the default) and `debug`).
