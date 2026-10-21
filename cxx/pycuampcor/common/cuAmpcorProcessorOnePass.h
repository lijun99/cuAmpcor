/**
 * @file  cuAmpcorProcessorOnePass.h
 * @brief The one-pass workflow: the oversampled windows correlated once
 *
 * The windows (the reference ones enlarged to the secondary size) are deramped and oversampled
 * (antialiasing), and their amplitudes correlated over the whole search range, with an extra
 * margin around it; the correlation surface is then oversampled around the peak, giving the
 * sub-pixel offset.
 *
 * The definitions are in cuAmpcorProcessorOnePassImpl.h, included only by the packages'
 * processors.cpp, which instantiate the processors for their arrays (compiled once per package).
 */

#pragma once

#include "cuAmpcorProcessor.h"

namespace pycuampcor {

/// the one-pass workflow
template <template <typename> class arrayT>
class cuAmpcorProcessorOnePass : public cuAmpcorProcessor<arrayT> {
public:
    /// the base class: the processor of a chunk
    using base_type = cuAmpcorProcessor<arrayT>;
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
    using typename base_type::freq_correlator_type;
    using typename base_type::oversampler_c2c_type;
    using typename base_type::normalizer_type;
    /// the windows: complex (copied from complex images, or real images as complex)
    using window_type = complex_type;

    /// a processor writing to the output images (see cuAmpcorProcessor)
    cuAmpcorProcessorOnePass(const cuAmpcorParameter &param,
        array_type<real2_type> &offsetImage, array_type<real_type> &snrImage,
        array_type<real3_type> &covImage, array_type<real_type> &peakValueImage,
        stream_type stream);
    /// release the buffers and workers
    ~cuAmpcorProcessorOnePass() override;

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

    // the windows, and their oversampled amplitudes
    array_type<window_type> c_referenceBatchRaw;                            ///< the reference windows (enlarged)
    array_type<window_type> c_secondaryBatchRaw;                            ///< the secondary windows
    array_type<window_type> c_referenceBatchOverSampled;                    ///< the oversampled reference windows
    array_type<window_type> c_secondaryBatchOverSampled;                    ///< the oversampled secondary windows
    array_type<real_type> r_referenceBatchOverSampled;                      ///< the amplitudes of the reference windows (not enlarged)
    array_type<real_type> r_secondaryBatchOverSampled;                      ///< the amplitudes of the secondary windows
    oversampler_c2c_type referenceBatchOverSampler;                         ///< the oversampler of the reference windows
    oversampler_c2c_type secondaryBatchOverSampler;                         ///< the oversampler of the secondary windows

    // the correlation surfaces
    array_type<real_type> r_corrBatch;                                      ///< the correlation surfaces
    array_type<real_type> r_corrBatchZoomIn;                                ///< the surfaces around the peak
    array_type<real_type> r_corrBatchZoomInOverSampled;                     ///< the oversampled surfaces
    std::unique_ptr<freq_correlator_type> cuCorrFreqDomain_OverSampled;     ///< the correlator (frequency domain)
    std::unique_ptr<normalizer_type> corrNormalizerOverSampled;             ///< the normalizer
};

} // namespace
//end of file
