/**
 * @file  cuFFTOverSampler.h
 * @brief The FFT oversamplers on the CUDA device: OverSampler<OversamplingMethod::FFT, T,
 *     Device::CUDA> for complex arrays (windows) and real arrays (correlation surfaces)
 *
 * The arrays are oversampled by zero padding their spectra (cufft plans saved for the batches).
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"
#include "OverSampler.h"
#include "cuArrays.h"
#include "cufftPlan.h"

namespace pycuampcor {

/// the FFT oversampler of complex arrays (windows) on the CUDA device, by zero padding their spectra
template <>
class OverSampler<OversamplingMethod::FFT, cuda::complex_type, Device::CUDA>
{
public:
    using complex_type = cuda::complex_type;   ///< the complex type

    // disable the default constructor
    OverSampler() = delete;
    /// constructor, for nImages images of inNX x inNY oversampled to outNX x outNY
    OverSampler(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream_);
    /// execute the oversampling (imagesOut of outNX x outNY)
    void execute(ArraysView<const complex_type> imagesIn, ArraysView<complex_type> imagesOut);
    /// destructor
    ~OverSampler();

private:
    cuda::CufftPlan forwardPlan;              ///< the forward fft handle
    cuda::CufftPlan backwardPlan;             ///< the backward fft handle
    cudaStream_t stream;                      ///< the cuda stream
    cuda::cuArrays<complex_type> workIn;      ///< the work array to hold the forward fft data
    cuda::cuArrays<complex_type> workOut;     ///< the work array to hold the padded data
};

/// the FFT oversampler of real arrays (correlation surfaces) on the CUDA device, by zero padding
/// their spectra
template <>
class OverSampler<OversamplingMethod::FFT, cuda::real_type, Device::CUDA>
{
public:
    using real_type = cuda::real_type;         ///< the real type
    using complex_type = cuda::complex_type;   ///< the complex type (of the spectra)

    OverSampler() = delete;
    /// constructor, for nImages images of inNX x inNY oversampled to outNX x outNY
    OverSampler(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream_);
    /// execute the oversampling (imagesOut takes the top left part of the oversampled images)
    void execute(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut);
    /// destructor
    ~OverSampler();

private:
    cuda::CufftPlan forwardPlan;                  ///< the forward fft (in place) of all images
    cuda::CufftPlan backwardPlan;                 ///< the inverse fft (in place) of the padded spectra
    cudaStream_t stream;                          ///< the cuda stream
    cuda::cuArrays<complex_type> workSizeIn;      ///< the input images (as complex), then their spectra
    cuda::cuArrays<complex_type> workSizeOut;     ///< the padded spectra, then the oversampled images
};

} // namespace

namespace pycuampcor::cuda {

/// an oversampler of count correlation surfaces of height x width around their peaks, by factor:
/// FFT or sinc, by method; with rawFactor, the oversampling of the windows (the sinc oversampler
/// shifts the peaks by factor*rawFactor)
std::unique_ptr<SurfaceOverSampler<cuArrays>> newSurfaceOverSampler(OversamplingMethod method,
    int height, int width, int factor, int rawFactor, int count, cudaStream_t stream);

} // namespace
// end of file
