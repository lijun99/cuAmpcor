/**
 * @file  ProcessorTwoPass.h
 * @brief The two-pass workflow: a first pass at the raw pixel spacing, then a second pass with
 *     oversampled windows around the first peak
 *
 * First pass: the amplitudes of the windows are correlated at the raw pixel spacing, giving the
 * peak to the pixel (and the SNR and covariance). Second pass: the windows (the secondary ones
 * around the first peak) are deramped and oversampled (antialiasing), their amplitudes correlated,
 * and the correlation surface oversampled again around the peak, giving the sub-pixel offset.
 *
 * The definitions are in ProcessorTwoPassImpl.h, included only by the packages'
 * processors.cpp, which instantiate the processors for their arrays (compiled once per package).
 */

#pragma once

#include "Processor.h"
#include "OverSampler.h"

namespace pycuampcor {

/// the two-pass workflow
template <template <typename> class arrayT>
class ProcessorTwoPass : public Processor<arrayT> {
public:
    /// the base class: the processor of a chunk
    using base_type = Processor<arrayT>;
    /// the arrays: batches of 2D arrays of T (as in the base class)
    template <typename T>
    using array_type = typename base_type::template array_type<T>;
    using typename base_type::real_type;
    using typename base_type::complex_type;
    using typename base_type::real2_type;
    using typename base_type::real3_type;
    using typename base_type::int2_type;
    using typename base_type::stream_type;
    using typename base_type::chunk_type;
    using typename base_type::correlator_type;
    using typename base_type::normalizer_type;
    /// the windows: complex (copied from complex images, or real images as complex)
    using window_type = complex_type;
    /// the oversampler of the windows: FFT, on the device of the arrays
    using window_oversampler_type = OverSampler<OversamplingMethod::FFT, window_type, base_type::device>;

    /// a processor writing to the output images (see Processor)
    ProcessorTwoPass(const Parameter &param,
        array_type<real2_type> &offsetImage, array_type<real_type> &snrImage,
        array_type<real3_type> &covImage, array_type<real_type> &peakValueImage,
        stream_type stream);
    /// release the buffers and workers
    ~ProcessorTwoPass() override;

    /// process the chunk (idxDown, idxAcross), and write its results to the output images
    void run(int idxDown, int idxAcross, const chunk_type &chunk) override;

private:
    // the members of the base class, used as our own
    using base_type::param;
    using base_type::stream;
    using base_type::offsetInit;
    using base_type::offsetZoomIn;
    using base_type::offsetFinal;
    using base_type::r_maxval;
    using base_type::corrMaxValue;
    using base_type::r_covValue;
    using base_type::r_snrValue;
    using base_type::r_corrBatchRawZoomIn;
    using base_type::i_corrBatchZoomInValid;
    using base_type::r_corrBatchSum;
    using base_type::i_corrBatchValidCount;
    using base_type::setIndex;
    using base_type::copyToBatch;
    using base_type::correlate;
    using base_type::createSurfaceOversampler;
    using base_type::oversampleSurface;
    using base_type::writeResults;
    using base_type::dump;

    // first pass: the windows at the raw pixel spacing, and their amplitudes
    array_type<window_type> c_referenceBatchRaw;                    ///< the reference windows
    array_type<window_type> c_secondaryBatchRaw;                    ///< the secondary windows
    array_type<real_type> r_referenceBatchRaw;                      ///< their amplitudes
    array_type<real_type> r_secondaryBatchRaw;                      ///< their amplitudes
    array_type<real_type> r_corrBatchRaw;                           ///< the correlation surfaces
    std::unique_ptr<correlator_type> correlatorRaw;                 ///< the cross-correlator of the windows
    std::unique_ptr<normalizer_type> corrNormalizerRaw;             ///< the normalizer

    // second pass: the oversampled windows (the secondary ones around the first peak)
    array_type<window_type> c_secondaryBatchZoomIn;                         ///< the secondary windows around the peak
    array_type<int2_type> maxLocShift;                                      ///< the shift of the peak from the center of the extracted windows
    array_type<window_type> c_referenceBatchOverSampled;                    ///< the oversampled reference windows
    array_type<window_type> c_secondaryBatchOverSampled;                    ///< the oversampled secondary windows
    array_type<real_type> r_referenceBatchOverSampled;                      ///< their amplitudes
    array_type<real_type> r_secondaryBatchOverSampled;                      ///< their amplitudes
    window_oversampler_type referenceBatchOverSampler;                      ///< the oversampler of the reference windows
    window_oversampler_type secondaryBatchOverSampler;                      ///< the oversampler of the secondary windows
    array_type<real_type> r_corrBatchZoomIn;                                ///< the correlation surfaces
    array_type<real_type> r_corrBatchZoomInAdjust;                          ///< the same, of even size
    array_type<real_type> r_corrBatchZoomInOverSampled;                     ///< the oversampled surfaces
    std::unique_ptr<correlator_type> correlatorOverSampled;                 ///< the cross-correlator of the windows
    std::unique_ptr<normalizer_type> corrNormalizerOverSampled;             ///< the normalizer
};

} // namespace
//end of file
