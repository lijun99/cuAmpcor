# The algorithm

This page describes how PyCuAmpcor estimates the offsets, step by step, and how each step differs
from ampcor (*ampcor.F*) of ROI_PAC, which PyCuAmpcor follows.

## The method

Ampcor (amplitude cross-correlation) estimates the shift (offset) between two images with the
feature tracking method, also called speckle, pixel or offset tracking. For each location:

* a rectangular window $R(x, y)$ is taken from the reference image, as the template;
* windows of the same size, $S(x+u, y+v)$, are taken from the secondary (search) image, shifted by
  $(u, v)$ within a search range;
* the normalized cross-correlation of the template with each shifted window gives the correlation
  surface $c(u, v)$;
* the location $(u_m, v_m)$ of the maximum of $c(u, v)$ is the estimate of the offset.

The offsets are in pixels, with sub-pixel values from the oversampling described below. A detailed
formulation of the normalized cross-correlation, with the frequency-domain approach, is given by
J. P. Lewis, [Fast Normalized Cross-Correlation](http://scribblethink.org/Work/nvisionInterface/nip.html).

SAR images are band-limited, so the correlation surface can be interpolated to sub-pixel
precision. To avoid aliasing in the cross-correlation and to improve the accuracy, the windows are
first oversampled (by 2 or more). This antialiasing oversampling makes the windows larger, and the
computation more costly. Two workflows balance efficiency and accuracy:

1. **Two-pass** (the default, as in version 1). A first pass estimates the offset to the pixel,
   $(r_x^0, r_y^0)$, over the full search range, without the antialiasing oversampling. A second pass
   correlates the oversampled windows over a much smaller search range centered at $(r_x^0, r_y^0)$.
2. **One-pass** (added in version 2). The windows are oversampled over the full search range before
   a single cross-correlation; no first pass is needed.

The two-pass workflow is faster and adequate in most cases. When the correlation surfaces are noisy
(high-frequency noise), the one-pass workflow can be more accurate.

PyCuAmpcor processes many windows together (a chunk): on the GPU, a chunk is processed by the CUDA
kernels of a stream, several streams overlapping the computations with the data transfers; on the
CPU, each OpenMP thread processes a chunk. To make this parallel processing efficient, some steps
differ slightly from ampcor.F, as described below.

### Directions

The following names are equivalent in PyCuAmpcor:

* row, height, down, azimuth, along the track: the first (outer) index of the 2D arrays;
* column, width, across, range, along the line of sight: the second (inner, contiguous) index.

Note that ampcor.F and GDAL in general use *y* for the rows and *x* for the columns.

## The steps

The steps below apply to one pair of reference and secondary windows (of the two-pass workflow),
repeated for all windows. The tables give the PyCuAmpcor parameters, the variables of the C++/CUDA
code, and the ampcor.F equivalents.

### 1. Read a window from the reference and secondary images

* Load a window of windowSizeHeight x windowSizeWidth pixels from the reference image, at its
  starting pixel.
* Load a larger window of (windowSizeHeight + 2 halfSearchRangeDown) x (windowSizeWidth + 2
  halfSearchRangeAcross) pixels from the secondary image, starting (halfSearchRangeDown,
  halfSearchRangeAcross) pixels before the reference window, shifted by the gross offset if any. It
  holds all the windows of the reference size shifted within the search range.

| PyCuAmpcor | C++/CUDA | ampcor.F | Notes |
|:---|:---|:---|:---|
| windowSizeHeight | windowSizeHeightRaw | i_wsyi | reference window height |
| windowSizeWidth | windowSizeWidthRaw | i_wsxi | reference window width |
| halfSearchRangeDown | halfSearchRangeDownRaw | i_srchy | half of the search range down |
| halfSearchRangeAcross | halfSearchRangeAcrossRaw | i_srchx | half of the search range across |

**Difference from ROI_PAC:** none of significance. Windows extending beyond the images are padded
with zeros.

### 2. Cross-correlate, and estimate the offset to the pixel

* Take the amplitudes of the windows (of complex or real images), and subtract the mean of the
  reference window.
* Compute the normalized correlation surface, of (2 halfSearchRangeDown + 1) x (2
  halfSearchRangeAcross + 1) pixels, in the frequency domain (with FFTs) or in the time domain.
* Find the peak of the surface.
* Around the peak, take a part of the surface (corrStatWindowSize x corrStatWindowSize) for the
  statistics: the signal-to-noise ratio (SNR) and the covariance of the offset.

This first estimate uses the full search range. The next steps zoom in around the peak, with the
oversampled windows and a smaller search range.

| PyCuAmpcor | C++/CUDA | ampcor.F | Notes |
|:---|:---|:---|:---|
| algorithm | algorithm | N/A | 0 = frequency domain (default), 1 = time domain; 2 = coherence (see below) |
| workflow | workflow | N/A | 0 = two-pass (default), 1 = one-pass |
| corrStatWindowSize | corrStatWindowSize | 21 | size of the surface around the peak for the statistics |

**Differences from ROI_PAC:**

* ROI_PAC only computes the correlation in the time domain. The frequency domain is faster, and is
  the default in PyCuAmpcor.
* ROI_PAC continues from here only for the windows with a good match (high SNR or coherence). To keep
  the processing parallel, PyCuAmpcor continues with all windows; the filtering of the results (e.g.,
  by the SNR) is left to the users.

### 3. Extract a smaller secondary window around the peak

* From the secondary window, extract a smaller window of (windowSizeHeight + 2
  halfZoomWindowSizeRaw) x (windowSizeWidth + 2 halfZoomWindowSizeRaw) pixels, centered at the
  peak. If the peak, e.g., down, is at offsetInit (in [0, 2 halfSearchRangeDown]), the extraction
  starts at offsetInit + halfSearchRangeDown − halfZoomWindowSizeRaw.

| PyCuAmpcor | C++/CUDA | ampcor.F | Notes |
|:---|:---|:---|:---|
| N/A | halfZoomWindowSizeRaw | i_srchp(p) = 4 | the smaller search range: corrSurfaceZoomInWindow / (2 rawDataOversamplingFactor) |

**Difference from ROI_PAC:** ROI_PAC centers the extracted window at the peak, padding zeros where it
extends beyond the secondary window (a peak near its edge). PyCuAmpcor shifts the extraction away
from the peak instead, so that all its pixels are in the secondary window.

### 4. Oversample the reference and the extracted secondary windows

* Remove the phase ramp of complex windows (deramping), to move the center of their spectrum to
  zero; it is needed before oversampling a complex signal. Only a linear phase ramp is removed, which
  does not work for TOPS images, whose ramp is quadratic: for them, the amplitudes are taken instead.
* Oversample both windows by rawDataOversamplingFactor (2 for SLCs), with FFTs (zero padding), to
  avoid aliasing in the cross-correlation.
* Take the amplitudes of the oversampled windows.

| PyCuAmpcor | C++/CUDA | ampcor.F | Notes |
|:---|:---|:---|:---|
| rawDataOversamplingFactor | rawDataOversamplingFactor | i_ovs = 2 | antialiasing oversampling factor, 2 for SLCs |
| derampMethod | derampMethod | 1, or no effect for TOPS | 0 = take the amplitudes (default, e.g., TOPS), 1 = remove a linear phase ramp, 2 = no deramping |
| derampAxis | derampAxis | N/A | the axes of the ramp: 0 = down, 1 = across, 2 = both (default) |

**Differences from ROI_PAC:**

* ROI_PAC enlarges both windows to sizes that are powers of 2, for the FFTs; PyCuAmpcor uses their
  sizes.
* ROI_PAC always deramps, with the average phase difference between neighboring pixels (method 1).
  For TOPS images, the image *datatype* must be set to `'mag'`, so that the amplitudes are taken
  before the oversampling, and the deramping has no effect. In PyCuAmpcor, derampMethod = 0 is the
  equivalent of `'mag'` (the amplitudes, no deramping); derampMethod = 1 always deramps.

### 5. Cross-correlate the oversampled windows

* Cross-correlate the oversampled reference and secondary windows, and normalize, as in step 2.
* The correlation surface is (2 halfZoomWindowSizeRaw rawDataOversamplingFactor + 1) pixels in each
  direction; its last row and column are dropped, so that its size, corrSurfaceZoomInWindow, is even.

| PyCuAmpcor | C++/CUDA | ampcor.F | Notes |
|:---|:---|:---|:---|
| corrSurfaceZoomInWindow | zoomWindowSize | i_cw | size of the correlation surface of the oversampled windows; also sets halfZoomWindowSizeRaw; 16 as in ROI_PAC |

**Difference from ROI_PAC:** ROI_PAC resizes the correlation surface from (2 i_srchp i_ovs + 1 = 17)
to i_cw pixels, centered at the peak (the peak search is part of its correlation step), extracting or
padding zeros. PyCuAmpcor skips this resizing, which makes little difference to the next step.

### 6. Oversample the correlation surface, and find the peak

* Oversample the correlation surface by corrSurfaceOverSamplingFactor (with FFTs, the default, or a
  sinc interpolation), to (corrSurfaceZoomInWindow corrSurfaceOverSamplingFactor) pixels in each
  direction.
* Find its peak, offsetZoomIn.
* The offset is

  offset = (offsetInit − halfSearchRange) + offsetZoomIn / (corrSurfaceOverSamplingFactor rawDataOversamplingFactor),

  with offsetInit the start of the extraction of step 3.

The offset does not include the gross offset, unless mergeGrossOffset is set; the gross offsets are
written to their own output (grossOffsetImageName).

| PyCuAmpcor | C++/CUDA | ampcor.F | Notes |
|:---|:---|:---|:---|
| corrSurfaceOverSamplingFactor | oversamplingFactor | i_covs | oversampling factor of the correlation surface |
| corrSurfaceOverSamplingMethod | oversamplingMethod | i_sinc_fourier = i_sinc | 0 = FFT (default), 1 = sinc |

**Difference from ROI_PAC:** ROI_PAC uses the sinc interpolation by default (the FFT method is in the
code, but switching to it needs a change of the Fortran code). The sinc interpolations are the same.
For the FFT, ROI_PAC enlarges the surface to a size that is a power of 2.

The sub-pixel step of the offsets is 1 / (corrSurfaceOverSamplingFactor rawDataOversamplingFactor)
pixel, e.g., 1/32 pixel with the defaults (16 and 2).

## The one-pass workflow

In the one-pass workflow, the reference window and the whole secondary window are deramped and
oversampled first, and cross-correlated once over the full (oversampled) search range. The
correlation surface around the peak is then oversampled as in step 6. The first pass (steps 2 and 3)
is not needed, but the oversampled windows are larger, and the computation more costly.

## The coherent cross-correlation (experimental)

With algorithm = 2 (`pycuampcor.Algorithm.COHERENCE`, or `"coherence"`), the complex windows are
cross-correlated instead of their amplitudes, as for the coregistration of interferometric pairs.
The correlation surface is the squared coherence at each shift,

$$\frac{\left|\sum \overline{(t - \bar t)}\, s\right|^2}{\sum |t - \bar t|^2 \left(\sum |s|^2 - \left|\sum s\right|^2 / n\right)},$$

the sums over the n pixels of the reference window t and of the secondary window s under it. It is
computed with FFTs, and normalized with sum area tables of the secondary windows. The rest of the
two-pass workflow is the same. Before the oversampling, both windows are deramped with the phase ramp
of the reference window, which keeps their relative (interferometric) phase. The squared coherence
is band limited, as the complex correlation, so that it can be oversampled with FFTs; for speckle,
the correlation of the amplitudes is also about the squared coherence. The peak value output is the
squared coherence.

* It needs complex images, the two-pass workflow, and derampMethod 1 (or 2, for images without a
  Doppler centroid); other settings raise an error. With `dense_offsets`: `--alg 2 --drmp 1`.
* It is experimental; the amplitude correlation remains the default. So far:
  * On synthetic images, where the decorrelation is additive noise, it is more precise than the
    amplitude correlation at moderate coherence (an RMS error of 0.016 vs 0.024 pixel at a coherence of
    about 0.5), and still works at about 0.2, where the amplitude correlation fails.
  * On real NISAR pairs (L-band, 12 days), the offsets were about 10–30% smoother at a coherence of
    0.3–0.7, as smooth at 0.7–0.9, and less smooth below 0.3 and above 0.9. With the coregistration
    settings of the NISAR workflow (a correlation surface oversampled by 16), the two were the same
    where the coherence is above about 0.5.
  * Where the coherence is lost (e.g., fast ice, long time spans, C-band in the melt season), it
    fails: use the amplitudes.
