/**
 * @file  cufftPlan.h
 * @brief A cuFFT plan, destroyed with its owner
 */

#pragma once

#include "externals.h"
#include "cudaError.h"
#include "logging.h"

namespace pycuampcor::cuda {

/// a cuFFT plan, destroyed with its owner; move-only
class CufftPlan {
public:
    CufftPlan() = default;
    /// the plan of other, which is left empty
    CufftPlan(CufftPlan &&other) noexcept
        : handle_(other.handle_), valid_(std::exchange(other.valid_, false)) {}
    /// destroy the plan, and take the plan of other, which is left empty
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
    /// destroy the plan
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

    /// the plan, for the cuFFT execute functions (valid once planned)
    operator cufftHandle() const { return handle_; }

private:
    /// destroy the plan, if any (a failure is logged, not thrown)
    void reset() noexcept
    {
        // not throwing (called by the destructor)
        if (valid_ && cufftDestroy(handle_) != CUFFT_SUCCESS)
            logWarning("failed to destroy a cuFFT plan");
        valid_ = false;
    }
    cufftHandle handle_ = 0;   ///< the plan (valid if valid_)
    bool valid_ = false;       ///< whether handle_ is a plan (to destroy)
};

} // namespace
//end of file
