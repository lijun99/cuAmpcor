/*
 * @file  cuAmpcorChunk.h
 * @brief Ampcor processor for a batch of windows
 *
 *
 */

#ifndef __CUAMPCORPROCESSOR_H
#define __CUAMPCORPROCESSOR_H

#include "backend.h"
#include "SlcImage.h"
#include "data_types.h"
#include "cuArrays.h"
#include "cuAmpcorParameter.h"
#include "cuOverSampler.h"
#include "cuSincOverSampler.h"
#include "cuCorrFrequency.h"
#include "cuCorrNormalizer.h"
#include "cuAmpcorChunkLoader.h"
#include <memory>
#include <vector>

namespace pycuampcor::PYCUAMPCOR_BACKEND {


/**
 * cuAmpcor batched processor (virtual class)
 */
class cuAmpcorProcessor{
// shared variables
protected:
    int idxChunkDown;     ///< index of the chunk in total batches, down
    int idxChunkAcross;   ///< index of the chunk in total batches, across
    int idxChunk;         ///<
    int nWindowsDown;     ///< number of windows in one chunk, down
    int nWindowsAcross;   ///< number of windows in one chunk, across

    int devId;            ///< GPU device ID to use


    cuAmpcorParameter *param;   ///< reference to the (global) parameters
    cuArrays<real2_type> *offsetImage; ///< output offsets image
    cuArrays<real_type> *snrImage;     ///< snr image
    cuArrays<real3_type> *covImage;    ///< cov image
    cuArrays<real_type> *peakValueImage;     ///< peak value image

    stream_t stream;  ///< stream to use (CUDA stream or dummy for CPU)

    // starting pixels of windows relative to the loaded chunk
    std::unique_ptr<cuArrays<int>> ChunkOffsetDown, ChunkOffsetAcross;

public:
    // default constructor and destructor
    cuAmpcorProcessor(cuAmpcorParameter *param_,
        cuArrays<real2_type> *offsetImage_, cuArrays<real_type> *snrImage_,
        cuArrays<real3_type> *covImage_, cuArrays<real_type> *peakValueImage_,
        stream_t stream_);
    virtual ~cuAmpcorProcessor() = default;

    // Factory method (virtual constructor)
    static std::unique_ptr<cuAmpcorProcessor> create(int workflow,
        cuAmpcorParameter *param_,
        cuArrays<real2_type> *offsetImage_, cuArrays<real_type> *snrImage_,
        cuArrays<real3_type> *covImage_, cuArrays<real_type> *peakValueImage_,
        stream_t stream_);

    // workflow specific methods
    // process the chunk (idxDown, idxAcross), loaded by a chunk loader
    virtual void run(int idxDown, int idxAcross, const cuAmpcorChunk &chunk) = 0;

protected:
    // shared methods
    void setIndex(int idxDown_, int idxAcross_);
    void getRelativeOffset(int *rStartPixel, const std::vector<int> &oStartPixel, int diff);
    // copy the windows starting at {startDown, startAcross} (in the image) from a loaded chunk to a batch
    // (complex images are copied as amplitudes if derampMethod == 0)
    void copyToBatch(const cuAmpcorLoadedChunk &chunk,
        const std::vector<int> &startDown, const std::vector<int> &startAcross,
        cuArrays<complex_type> *batch);

};

} // namespace

#endif //__CUAMPCORPROCESSOR_H
