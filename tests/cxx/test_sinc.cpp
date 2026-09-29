/**
 * @file test_sinc.cpp
 * @brief Tests for the sinc interpolator of correlation surfaces
 */

#include "test_util.h"
#include "cuSincOverSampler.h"

namespace pycuampcor::PYCUAMPCOR_BACKEND::test {

class SincTest : public BackendTest {
protected:
    static constexpr int n = 32;       // input size
    static constexpr int covs = 8;     // oversampling factor
    static constexpr int window = 2;   // only +/- window*covs around the center is interpolated
    static constexpr double pi = 3.14159265358979323846;

    // a slowly varying (band-limited) function
    static double f(double x, double y)
    {
        return std::cos(2*pi*0.04*x + 0.3)*std::cos(2*pi*0.05*y) + 0.2;
    }

    std::vector<real_type> input(int count)
    {
        std::vector<real_type> v(n*n*count);
        for (int k = 0; k < count; k++)
            for (int x = 0; x < n; x++)
                for (int y = 0; y < n; y++)
                    v[(k*n + x)*n + y] = f(x, y);
        return v;
    }

    // check the interpolated values within the window centered at (center + shift*factor)
    void check(const std::vector<real_type> &out, int k, int2 shift, int factor)
    {
        const int N = n*covs;
        const int cx = N/2 + shift.x*factor, cy = N/2 + shift.y*factor;
        for (int x = 0; x < N; x++)
            for (int y = 0; y < N; y++) {
                double v = out[(k*N + x)*N + y];
                bool inside = std::abs(x - cx) <= window*covs && std::abs(y - cy) <= window*covs;
                if (inside)
                    EXPECT_NEAR(v, f(double(x)/covs, double(y)/covs), 2e-2) << x << " " << y;
                else
                    EXPECT_EQ(v, 0) << x << " " << y;
            }
    }
};

TEST_F(SincTest, ConstantCenter)
{
    auto in = makeFrom(input(1), n, n);
    auto out = make<real_type>(n*covs, n*covs);
    cuSincOverSamplerR2R sinc(covs, stream);
    for (int2 shift : {make_int2(0, 0), make_int2(1, -2)}) {
        const int factor = covs;
        sinc.execute(in.get(), out.get(), shift, factor);
        check(download(*out), 0, shift, factor);
    }
}

TEST_F(SincTest, VaryingCenter)
{
    const int count = 3;
    auto in = makeFrom(input(count), n, n, 1, count);
    auto out = make<real_type>(n*covs, n*covs, 1, count);
    std::vector<int2> shiftV = {make_int2(0, 0), make_int2(-1, 1), make_int2(2, 0)};
    auto shifts = makeFrom(shiftV, 1, count);
    cuSincOverSamplerR2R sinc(covs, stream);
    const int factor = covs/2;
    sinc.execute(in.get(), out.get(), shifts.get(), factor);
    auto r = download(*out);
    for (int k = 0; k < count; k++)
        check(r, k, shiftV[k], factor);
}

} // namespace
