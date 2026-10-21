/**
 * @file  Parameter.h
 * @brief Parameters of ampcor
 *
 * Author: Lijun Zhu @ Seismo Lab, Caltech
 */

#pragma once

#include "forward.h"
#include "Options.h"

namespace pycuampcor {

/// a pair of integers along the two axes of the images (x: down, y: across), e.g., a size or a
/// position; the backends convert it to their own int2 (CUDA's, or the CPU backend's)
struct Index2 {
    int x = 0;  ///< down
    int y = 0;  ///< across
};

/// Parameters of ampcor
///
/// @note
/// The dimension/direction names used are:
/// The inner-most dimension: x, row, height, down, azimuth, along the track.
/// The outer-most dimension: y, column, width, across, range, along the sight.
/// C/C++/Python use row-major indexing: a[i][j] -> a[i*WIDTH+j]
/// FORTRAN/BLAS/CUBLAS use column-major indexing: a[i][j]->a[i+j*LENGTH]
///
/// @note
/// Usage:
/// 1. create an instance (with the default inputs) and set the inputs, e.g., numberWindowDown = 100;
/// 2. setupParameters(...): compute the sizes of the windows, correlation surfaces and chunks
///    (with the backend's default number of windows in a chunk; see setupParameters(param) of the
///    cpu and cuda controllers);
/// 3. setStartPixels(...): set the starting pixels and the gross offsets of the windows, and the
///    areas of the chunks to load;
/// 4. optionally, checkPixelInImageRange(): warn about windows outside the images (debug builds).
/// setupParameters() does not change the inputs: it may be called again after changing them,
/// followed by setStartPixels().
class Parameter {
public:
    // ==== inputs (not changed by setupParameters) ====

    // processing
    Algorithm algorithm = Algorithm::Frequency;            ///< cross-correlation domain
    int deviceID = 0;           ///< CUDA device (GPU backend)
    int nStreams = 1;           ///< number of CUDA streams, each processing a chunk (GPU backend)
    int nThreads = 0;           ///< number of threads, each processing a chunk (CPU backend; 0 for the OpenMP default)
    DerampMethod derampMethod = DerampMethod::Magnitude;  ///< phase ramp removal before the antialiasing oversampling
    DerampAxis derampAxis = DerampAxis::Both;              ///< axis of the phase ramp
    Workflow workflow = Workflow::TwoPass;                 ///< cross-correlation workflow

    // reference image
    std::string referenceImageName = "reference.slc";  ///< raw binary file, or HDF5:<file>:<dataset>
    std::string referenceImageReader = "auto";         ///< reader: auto (by the name), raw or hdf5
    DataType referenceImageDataType = DataType::Complex64;  ///< pixel type
    int referenceImageHeight = 1000;                   ///< height (lines)
    int referenceImageWidth = 1000;                    ///< width (columns)

    // secondary image
    std::string secondaryImageName = "secondary.slc";  ///< raw binary file, or HDF5:<file>:<dataset>
    std::string secondaryImageReader = "auto";         ///< reader: auto (by the name), raw or hdf5
    DataType secondaryImageDataType = DataType::Complex64;  ///< pixel type
    int secondaryImageHeight = 1000;                   ///< height (lines)
    int secondaryImageWidth = 1000;                    ///< width (columns)

    // windows (in pixels of the images)
    int numberWindowDown = 1;           ///< number of windows (down)
    int numberWindowAcross = 1;         ///< number of windows (across)
    int windowSizeHeightRaw = 64;       ///< reference (template) window height
    int windowSizeWidthRaw = 64;        ///< reference (template) window width
    int halfSearchRangeDownRaw = 20;    ///< half of the search range (down)
    int halfSearchRangeAcrossRaw = 20;  ///< half of the search range (across)
    int skipSampleDownRaw = 64;         ///< distance between neighboring windows (down)
    int skipSampleAcrossRaw = 64;       ///< distance between neighboring windows (across)
    int referenceStartPixelDown0 = 0;   ///< first line of the first reference window
    int referenceStartPixelAcross0 = 0; ///< first column of the first reference window
    int numberWindowDownInChunk = 0;    ///< number of windows in a chunk (down); 0 = automatic
    int numberWindowAcrossInChunk = 0;  ///< number of windows in a chunk (across); 0 = automatic

    // oversampling and correlation surface
    int rawDataOversamplingFactor = 2;  ///< antialiasing oversampling factor of the windows
    int zoomWindowSize = 16;            ///< size of the correlation surface (antialiasing-oversampled) around the peak to oversample
    int corrSurfaceOverSamplingFactor = 16;  ///< oversampling factor of the correlation surface
    OversamplingMethod corrSurfaceOverSamplingMethod = OversamplingMethod::FFT;  ///< oversampling method of the correlation surface
    int corrStatWindowSize = 21;        ///< size of the correlation surface around the peak for the SNR and covariance (21 as in ROI_PAC)

    // outputs
    std::string offsetImageName = "DenseOffset.bip";       ///< dense offsets
    std::string grossOffsetImageName = "GrossOffset.bip";  ///< gross offsets
    std::string snrImageName = "snr.bip";                  ///< signal-to-noise ratio
    std::string covImageName = "cov.bip";                  ///< covariance of the offsets
    std::string peakValueImageName = "peakValue.bip";      ///< normalized correlation peak value
    bool mergeGrossOffset = false;      ///< whether to add the gross offsets to the dense offsets

    // image i/o
    bool useMmap = true;                ///< unused (raw images are always read with a memory map)
    int mmapSizeInGB = 1;               ///< size of the memory map (raw images) or of the cache of decoded chunks (HDF5 images)

    // ==== computed by setupParameters ====

    int numberWindows = 0;              ///< numberWindowDown*numberWindowAcross
    int windowsDownInChunk = 0;         ///< number of windows in a chunk used (down)
    int windowsAcrossInChunk = 0;       ///< number of windows in a chunk used (across)
    int numberChunkDown = 0;            ///< number of chunks (down)
    int numberChunkAcross = 0;          ///< number of chunks (across)
    int numberChunks = 0;               ///< total number of chunks

    // window sizes in pixels of the images
    int windowSizeHeightRawEnlarged = 0;  ///< reference window height loaded (enlarged to the search window in the one-pass workflow)
    int windowSizeWidthRawEnlarged = 0;   ///< reference window width loaded (enlarged to the search window in the one-pass workflow)
    int searchWindowSizeHeightRaw = 0;    ///< search window height (including the zoom-in margin in the one-pass workflow)
    int searchWindowSizeWidthRaw = 0;     ///< search window width (including the zoom-in margin in the one-pass workflow)

    // window sizes after the antialiasing oversampling
    int windowSizeHeight = 0;           ///< reference window height
    int windowSizeWidth = 0;            ///< reference window width
    int windowSizeHeightEnlarged = 0;   ///< reference window height enlarged (one-pass workflow)
    int windowSizeWidthEnlarged = 0;    ///< reference window width enlarged (one-pass workflow)
    int searchWindowSizeHeight = 0;     ///< search window height
    int searchWindowSizeWidth = 0;      ///< search window width

    // correlation surface
    int corrZoomWindowSize = 0;         ///< zoomWindowSize used (limited by the search range in the two-pass workflow)
    int halfZoomWindowSizeRaw = 0;      ///< corrZoomWindowSize/(2*rawDataOversamplingFactor)
    Index2 corrWindowSize{};              ///< correlation surface size (one-pass workflow)
    Index2 corrZoomInSize{};              ///< zoom-in correlation surface size: corrZoomWindowSize+1 (one-pass workflow)
    Index2 corrZoomInOversampledSize{};   ///< corrZoomInSize*corrSurfaceOverSamplingFactor (one-pass workflow)
    Index2 corrZoomInOversampledSearchStart{};  ///< peak search start in the oversampled correlation surface (one-pass workflow)
    Index2 corrZoomInOversampledSearchRange{};  ///< peak search range in the oversampled correlation surface (one-pass workflow)
    int corrRawZoomInHeight = 0;        ///< correlation surface height (at the raw spacing) for the SNR
    int corrRawZoomInWidth = 0;         ///< correlation surface width (at the raw spacing) for the SNR

    // first pass of the two-pass workflow
    int searchWindowSizeHeightRawZoomIn = 0;  ///< search window height of the second pass
    int searchWindowSizeWidthRawZoomIn = 0;   ///< search window width of the second pass

    // loading offsets, depending on the workflow
    int referenceLoadingOffsetDown = 0, referenceLoadingOffsetAcross = 0;  ///< reference (e.g., matching the secondary)
    int secondaryLoadingOffsetDown = 0, secondaryLoadingOffsetAcross = 0;  ///< secondary (e.g., the search range)

    // ==== computed by setStartPixels ====

    std::vector<int> referenceStartPixelDown;    ///< reference starting pixels for each window (down)
    std::vector<int> referenceStartPixelAcross;  ///< reference starting pixels for each window (across)
    std::vector<int> secondaryStartPixelDown;    ///< secondary starting pixels for each window (down)
    std::vector<int> secondaryStartPixelAcross;  ///< secondary starting pixels for each window (across)
    std::vector<int> grossOffsetDown;            ///< gross offsets between reference and secondary windows (down)
    std::vector<int> grossOffsetAcross;          ///< gross offsets between reference and secondary windows (across)

    std::vector<int> referenceChunkStartPixelDown;    ///< reference starting pixels for each chunk (down)
    std::vector<int> referenceChunkStartPixelAcross;  ///< reference starting pixels for each chunk (across)
    std::vector<int> secondaryChunkStartPixelDown;    ///< secondary starting pixels for each chunk (down)
    std::vector<int> secondaryChunkStartPixelAcross;  ///< secondary starting pixels for each chunk (across)
    std::vector<int> referenceChunkHeight;   ///< reference chunk height
    std::vector<int> referenceChunkWidth;    ///< reference chunk width
    std::vector<int> secondaryChunkHeight;   ///< secondary chunk height
    std::vector<int> secondaryChunkWidth;    ///< secondary chunk width
    int maxReferenceChunkHeight = 0, maxReferenceChunkWidth = 0;  ///< max reference chunk size
    int maxSecondaryChunkHeight = 0, maxSecondaryChunkWidth = 0;  ///< max secondary chunk size
    bool chunkStartPixelsSet = false;        ///< whether the window/chunk starting pixels are set

    // ==== methods ====

    /// compute the sizes of the windows, correlation surfaces and chunks from the inputs
    /// @param defaultWindowsDownInChunk the number of windows down in a chunk if
    ///     numberWindowDownInChunk is 0 (automatic), e.g., the backend's default
    /// @param defaultWindowsAcrossInChunk the same across
    void setupParameters(int defaultWindowsDownInChunk, int defaultWindowsAcrossInChunk);

    /// set the starting pixels of the windows and their gross offsets (after setupParameters),
    /// and the areas of the chunks to load: varying reference start pixels and gross offsets,
    /// one per window
    void setStartPixels(const int*, const int*, const int*, const int*);
    /// set the starting pixels of the windows and their gross offsets (after setupParameters),
    /// and the areas of the chunks to load: the first reference start pixel (the windows spaced
    /// by skipSampleDownRaw/skipSampleAcrossRaw) and varying gross offsets, one per window
    void setStartPixels(int, int, const int*, const int*);
    /// set the starting pixels of the windows and their gross offsets (after setupParameters),
    /// and the areas of the chunks to load: the first reference start pixel (the windows spaced
    /// by skipSampleDownRaw/skipSampleAcrossRaw) and a constant gross offset
    void setStartPixels(int, int, int, int);

    /// warn about windows outside the images (debug builds only)
    void checkPixelInImageRange() const;
    /// throw if setupParameters is not called
    void checkArraysAllocated() const;
    /// throw if setupParameters or setStartPixels is not called
    void checkReadyToRun() const;

private:
    /// the sizes of the windows and correlation surfaces for the two-pass workflow
    void _setupParameters_TwoPass();
    /// the sizes of the windows and correlation surfaces for the one-pass workflow
    void _setupParameters_OnePass();
    /// (re)size the arrays of the windows and chunks
    void allocateArrays();
    /// set the starting pixels and sizes of the chunks, from the starting pixels of the windows
    void setChunkStartPixels();
};

} // namespace
//end of file
