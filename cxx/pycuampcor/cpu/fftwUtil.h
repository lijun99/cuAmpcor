/*
 * @file  fftwUtil.h
 * @brief Helpers to use fftw (single or double precision) for the CPU backend
 *
 * fftw planning routines are not thread-safe; plans are created/destroyed under a mutex.
 * Plans are executed with the new-array interface on arrays allocated with Arrays,
 *   which have the same alignment as those used for planning.
 */

#pragma once

#include "externals.h"

namespace pycuampcor::cpu {

#ifdef CUAMPCOR_DOUBLE
    using fftw_plan_type = fftw_plan;
    using fftw_complex_type = fftw_complex;
    #define PYCUAMPCOR_FFTW(name) fftw_ ## name
#else
    using fftw_plan_type = fftwf_plan;
    using fftw_complex_type = fftwf_complex;
    #define PYCUAMPCOR_FFTW(name) fftwf_ ## name
#endif

// fftw planner flag: FFTW_ESTIMATE gives deterministic plans (FFTW_MEASURE may vary between runs)
constexpr unsigned fftwFlags = FFTW_ESTIMATE;

/// the mutex to guard the fftw planner
std::mutex & fftwPlannerMutex();

/// an fftw plan, destroyed (under the planner mutex) with its owner; move-only
class FftwPlan {
public:
    FftwPlan() = default;
    explicit FftwPlan(fftw_plan_type plan) : plan_(plan) {}
    FftwPlan(FftwPlan &&other) noexcept : plan_(std::exchange(other.plan_, nullptr)) {}
    FftwPlan &operator=(FftwPlan &&other) noexcept
    {
        if (this != &other) {
            reset();
            plan_ = std::exchange(other.plan_, nullptr);
        }
        return *this;
    }
    /// take a plan (created under the planner mutex; the replaced plan, if any, is destroyed)
    FftwPlan &operator=(fftw_plan_type plan)
    {
        reset();
        plan_ = plan;
        return *this;
    }
    FftwPlan(const FftwPlan &) = delete;
    FftwPlan &operator=(const FftwPlan &) = delete;
    ~FftwPlan() { reset(); }

    operator fftw_plan_type() const { return plan_; }

private:
    void reset()
    {
        // lock only to destroy a plan (the planner mutex may be held while plans are created)
        if (plan_) {
            std::lock_guard<std::mutex> lock(fftwPlannerMutex());
            PYCUAMPCOR_FFTW(destroy_plan)(plan_);
            plan_ = nullptr;
        }
    }
    fftw_plan_type plan_ = nullptr;
};

/// cast complex arrays to fftw complex type
inline fftw_complex_type * fftwCast(complex_type *data)
{
    return reinterpret_cast<fftw_complex_type *>(data);
}

} // namespace
// end of file
