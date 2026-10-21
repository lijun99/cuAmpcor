/**
 * @file test_util.h
 * @brief Helpers for the backend-agnostic unit tests
 */

#pragma once

#include <gtest/gtest.h>

#include "public.h"

#include <cmath>
#include <complex>
#include <memory>
#include <random>
#include <vector>

#ifdef PYCUAMPCOR_BACKEND_CUDA
#include <cuda_runtime.h>
#endif

namespace pycuampcor::PYCUAMPCOR_BACKEND::test {

/// whether the backend (e.g., a cuda device) is available
inline bool backendAvailable()
{
#ifdef PYCUAMPCOR_BACKEND_CUDA
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
    return true;
#endif
}

/// test fixture with a stream and helpers to move data between host and backend
class BackendTest : public ::testing::Test {
protected:
    stream_t stream{};
    bool hasStream = false;

    /// skip the test if the device is not available; create the stream
    void SetUp() override
    {
        if (!backendAvailable())
            GTEST_SKIP() << "the backend (device) is not available";
#ifdef PYCUAMPCOR_BACKEND_CUDA
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
#endif
        hasStream = true;
    }

    /// destroy the stream
    void TearDown() override
    {
#ifdef PYCUAMPCOR_BACKEND_CUDA
        if (hasStream)
            cudaStreamDestroy(stream);
#endif
    }

    /// wait for the work queued on the device (nothing to wait for on the CPU)
    static void synchronize()
    {
#ifdef PYCUAMPCOR_BACKEND_CUDA
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
#endif
    }

    /// allocate a batch of images in both backend and host memory
    template<typename T>
    std::unique_ptr<array_type<T>> make(int height, int width, int countH = 1, int countW = 1)
    {
        auto array = std::make_unique<array_type<T>>(height, width, countH, countW);
        array->allocate();
        array->allocateHost();
        return array;
    }

    /// copy values to the backend memory
    template<typename T>
    void upload(array_type<T> &array, const std::vector<T> &values)
    {
        ASSERT_EQ(values.size(), array.getSize());
        std::copy(values.begin(), values.end(), array.hostData);
        array.copyToDevice(stream);
        synchronize();
    }

    /// allocate and fill a batch of images
    template<typename T>
    std::unique_ptr<array_type<T>> makeFrom(const std::vector<T> &values, int height, int width,
        int countH = 1, int countW = 1)
    {
        auto array = make<T>(height, width, countH, countW);
        upload(*array, values);
        return array;
    }

    /// copy values from the backend memory
    template<typename T>
    std::vector<T> download(array_type<T> &array)
    {
        synchronize();
        array.copyToHost(stream);
        synchronize();
        return std::vector<T>(array.hostData, array.hostData + array.getSize());
    }
};

/// random values in [-1, 1)
inline std::vector<real_type> randomReal(size_t n, unsigned seed = 1)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<real_type> v(n);
    for (auto &x : v) x = dist(gen);
    return v;
}

/// random complex values with real/imaginary parts in [-1, 1)
template<typename T = complex_type>
std::vector<T> randomComplex(size_t n, unsigned seed = 1)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<T> v(n);
    for (auto &x : v) { x.x = dist(gen); x.y = dist(gen); }
    return v;
}

/// tolerance for values computed from the (single precision) image data
constexpr double imageTol = 1e-6;

/// tolerance for floating point comparisons in the internal precision
#ifdef CUAMPCOR_DOUBLE
constexpr double tol = 1e-10;
#else
constexpr double tol = 1e-5;
#endif

} // namespace
