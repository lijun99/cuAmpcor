/**
 * @file  cufftPlan.h
 * @brief A cuFFT plan, destroyed with its owner
 */

#pragma once

#include "cudaError.h"
#include <cufft.h>
#include <cstdio>
#include <utility>

namespace pycuampcor::cuda {

/// a cuFFT plan, destroyed with its owner; move-only
class CufftPlan {
public:
    CufftPlan() = default;
    CufftPlan(CufftPlan &&other) noexcept
        : handle_(other.handle_), valid_(std::exchange(other.valid_, false)) {}
    CufftPlan &operator=(CufftPlan &&other) noexcept
    {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            valid_ = std::exchange(other.valid_, false);
        }
        return *this;
    }
    CufftPlan(const CufftPlan &) = delete;
    CufftPlan &operator=(const CufftPlan &) = delete;
    ~CufftPlan() { reset(); }

    /// create a batched plan (as cufftPlanMany), replacing the current one
    void planMany(int rank, int *n, int *inembed, int istride, int idist,
                  int *onembed, int ostride, int odist, cufftType type, int batch)
    {
        reset();
        cufftHandle handle;
        checkCudaErrors(cufftPlanMany(&handle, rank, n, inembed, istride, idist,
                                      onembed, ostride, odist, type, batch));
        handle_ = handle;
        valid_ = true;
    }

    operator cufftHandle() const { return handle_; }

private:
    void reset() noexcept
    {
        // not throwing (called by the destructor)
        if (valid_ && cufftDestroy(handle_) != CUFFT_SUCCESS)
            std::fprintf(stderr, "Warning: failed to destroy a cuFFT plan\n");
        valid_ = false;
    }
    cufftHandle handle_ = 0;
    bool valid_ = false;
};

} // namespace
//end of file
