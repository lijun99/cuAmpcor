/**
 * @file  cuAmpcorProcessor.h
 * @brief The processor of a chunk (a batch of windows): the base of the workflows
 *
 * A processor runs a workflow (two-pass or one-pass) on the windows of a chunk, loaded by the
 * chunk loader, and writes the offsets and their statistics to the output images. The processors
 * are written once, as templates on the array type (e.g., cpu::Arrays or cuda::cuArrays; see
 * cuAmpcorArrayTraits.h), and instantiated by each package (processors.cpp of the package).
 *
 * The base class holds what the workflows share: the parameters, the output images, the stream,
 * the buffers of the per-window results (peaks, offsets, SNR, covariance), and the steps they
 * have in common (copying the windows from the chunk, correlating, oversampling the correlation
 * surface, writing the results).
 */

#pragma once

#include "forward.h"
#include "cuAmpcorArrayTraits.h"

namespace pycuampcor {

/// the processor of a chunk: the base of the workflows (cuAmpcorProcessorTwoPass, OnePass)
template <template <typename> class arrayT>
class cuAmpcorProcessor {
public:
    /// the types that go with the arrays
    using traits = array_traits<arrayT>;
    /// the arrays: batches of 2D arrays of T, e.g., cuda::cuArrays<float>
    template <typename T>
    using array_type = arrayT<T>;
    /// the element types
    using real_type = typename traits::real_type;
    using complex_type = typename traits::complex_type;
    using real2_type = typename traits::real2_type;
    using real3_type = typename traits::real3_type;
    using int2_type = typename traits::int2_type;
    /// the stream on which the kernels are queued
    using stream_type = typename traits::stream_type;
    /// the chunks of the images loaded by the chunk loader
    using chunk_type = typename traits::chunk_type;
    using loaded_chunk_type = typename traits::loaded_chunk_type;
    /// the workers
    using freq_correlator_type = typename traits::freq_correlator_type;
    using oversampler_c2c_type = typename traits::oversampler_c2c_type;
    using normalizer_type = typename traits::normalizer_type;
    using surface_oversampler_type = typename traits::surface_oversampler_type;

    /// a processor writing to the output images
    /// @param param the parameters (set up)
    /// @param offsetImage, snrImage, covImage, peakValueImage the output images, covering whole chunks
    /// @param stream the stream on which the kernels are queued
    cuAmpcorProcessor(const cuAmpcorParameter *param,
        array_type<real2_type> *offsetImage, array_type<real_type> *snrImage,
        array_type<real3_type> *covImage, array_type<real_type> *peakValueImage,
        stream_type stream);
    /// release the buffers and workers (defined with the processors, where the workers are complete)
    virtual ~cuAmpcorProcessor();

    /// the processor of a workflow (two-pass or one-pass); the arguments are those of the constructor
    static std::unique_ptr<cuAmpcorProcessor> create(Workflow workflow,
        const cuAmpcorParameter *param,
        array_type<real2_type> *offsetImage, array_type<real_type> *snrImage,
        array_type<real3_type> *covImage, array_type<real_type> *peakValueImage,
        stream_type stream);

    /// process the chunk (idxDown, idxAcross), loaded by a chunk loader, and write its results to
    /// the output images
    virtual void run(int idxDown, int idxAcross, const chunk_type &chunk) = 0;

protected:
    const cuAmpcorParameter *param;          ///< the parameters
    array_type<real2_type> *offsetImage;     ///< the output dense offsets
    array_type<real_type> *snrImage;         ///< the output SNR
    array_type<real3_type> *covImage;        ///< the output covariance
    array_type<real_type> *peakValueImage;   ///< the output correlation peak
    stream_type stream;                      ///< the stream on which the kernels are queued

    // the current chunk
    int idxChunkDown = 0;      ///< the index of the chunk, down
    int idxChunkAcross = 0;    ///< the index of the chunk, across
    int nWindowsDown = 0;      ///< the number of windows in the chunk, down (fewer in the last chunk)
    int nWindowsAcross = 0;    ///< the number of windows in the chunk, across

    // the starting pixels of the windows relative to the loaded chunk
    std::unique_ptr<array_type<int>> chunkOffsetDown;    ///< down
    std::unique_ptr<array_type<int>> chunkOffsetAcross;  ///< across

    // the per-window results, shared by the workflows
    std::unique_ptr<array_type<int2_type>> offsetInit;      ///< the peak of the first correlation surface
    std::unique_ptr<array_type<int2_type>> offsetZoomIn;    ///< the peak of the oversampled correlation surface
    std::unique_ptr<array_type<real2_type>> offsetFinal;    ///< the offsets
    std::unique_ptr<array_type<real_type>> r_maxval;        ///< the value of the first peak
    std::unique_ptr<array_type<real_type>> corrMaxValue;    ///< the value of the oversampled peak
    std::unique_ptr<array_type<real3_type>> r_covValue;     ///< the covariance of the offsets
    std::unique_ptr<array_type<real_type>> r_snrValue;      ///< the SNR
    // the SNR estimation: the correlation surface around the peak (at the raw pixel spacing),
    // its valid pixels, their sum and count
    std::unique_ptr<array_type<real_type>> r_corrBatchRawZoomIn;    ///< the surface around the peak
    std::unique_ptr<array_type<int>> i_corrBatchZoomInValid;        ///< its valid pixels
    std::unique_ptr<array_type<real_type>> r_corrBatchSum;          ///< their sum
    std::unique_ptr<array_type<int>> i_corrBatchValidCount;         ///< their count

    /// the oversampler of the correlation surfaces around the peaks (FFT or sinc)
    std::unique_ptr<surface_oversampler_type> surfaceOverSampler;

    /// set the index of the current chunk, and its number of windows
    void setIndex(int idxDown, int idxAcross);

    /// the starting pixels of the windows of the current chunk, relative to the loaded chunk
    /// @param[out] relative the starting pixels relative to the loaded chunk (windows in a chunk)
    /// @param absolute the starting pixels of all windows in the image
    /// @param chunkStart the starting pixel of the loaded chunk in the image
    void getRelativeOffset(int *relative, const std::vector<int> &absolute, int chunkStart);

    /// copy the windows of the current chunk from a loaded chunk to a batch; windows outside the
    /// image are padded with zeros, complex images are copied as amplitudes if derampMethod is
    /// magnitude, and real images are copied as complex
    /// @param chunk the loaded chunk (of the reference or secondary image)
    /// @param startDown, startAcross the starting pixels of all windows in the image
    /// @param[out] batch the windows
    void copyToBatch(const loaded_chunk_type &chunk,
        const std::vector<int> &startDown, const std::vector<int> &startAcross,
        array_type<complex_type> *batch);

    /// cross-correlate the reference windows with the secondary windows (in the frequency or time
    /// domain), and normalize the correlation surfaces
    /// @param reference, secondary the windows (amplitudes, with the mean subtracted)
    /// @param[out] surface the normalized correlation surfaces
    /// @param freqCorrelator the correlator in the frequency domain (if algorithm is frequency)
    /// @param normalizer the normalizer of the correlation surfaces
    /// @param name the name of the surfaces in the debug dumps
    void correlate(array_type<real_type> *reference, array_type<real_type> *secondary,
        array_type<real_type> *surface, freq_correlator_type *freqCorrelator,
        normalizer_type *normalizer, const std::string &name);

    /// create the oversampler of the correlation surfaces of height x width around the peak,
    /// by corrSurfaceOverSamplingFactor (FFT or sinc, by corrSurfaceOverSamplingMethod)
    void createSurfaceOversampler(int height, int width);

    /// oversample the correlation surfaces around the peaks
    /// @param surface the correlation surfaces around the peaks
    /// @param[out] oversampled the oversampled correlation surfaces
    /// @param shift the shift of each peak from the center of its surface (the sinc oversampler
    ///     computes only around the peak), or nullptr if the peaks are at the centers
    void oversampleSurface(array_type<real_type> *surface, array_type<real_type> *oversampled,
        array_type<int2_type> *shift);

    /// write the results of the current chunk (offsets, SNR, covariance, peak) to the output images
    void writeResults();

    /// write an intermediate array to a file named name (debug builds only)
    template <typename T>
    void dump(array_type<T> *array, const std::string &name);
};

} // namespace
//end of file
