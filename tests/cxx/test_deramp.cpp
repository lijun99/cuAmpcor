/**
 * @file test_deramp.cpp
 * @brief Tests for removing linear phase ramps from complex images
 */

#include "test_util.h"

namespace pycuampcor::PYCUAMPCOR_BACKEND::test {

using DerampTest = BackendTest;

// random amplitudes with a linear phase ramp exp(i(ax*x + ay*y))
static std::vector<complex_type> rampImage(int h, int w, double ax, double ay, std::vector<double> &amp)
{
    std::mt19937 gen(9);
    std::uniform_real_distribution<double> dist(0.5, 1.5);
    std::vector<complex_type> v(h*w);
    amp.resize(h*w);
    for (int x = 0; x < h; x++)
        for (int y = 0; y < w; y++) {
            amp[x*w + y] = dist(gen);
            double phase = ax*x + ay*y;
            v[x*w + y] = make_complex_type(amp[x*w + y]*std::cos(phase), amp[x*w + y]*std::sin(phase));
        }
    return v;
}

TEST_F(DerampTest, BothAxes)
{
    const int h = 16, w = 24;
    std::vector<double> amp;
    auto v = rampImage(h, w, 0.7, -1.3, amp);
    auto images = makeFrom(v, h, w);
    cuDeramp(1, images.get(), 2, stream);
    auto r = download(*images);
    for (int i = 0; i < h*w; i++) {
        EXPECT_NEAR(r[i].x, amp[i], 1e-4);
        EXPECT_NEAR(r[i].y, 0, 1e-4);
    }
}

TEST_F(DerampTest, SingleAxis)
{
    const int h = 10, w = 12;
    const double ax = 0.4, ay = 2.1;
    std::vector<double> amp;
    auto v = rampImage(h, w, ax, ay, amp);
    for (int axis : {0, 1}) {
        auto images = makeFrom(v, h, w);
        cuDeramp(1, images.get(), axis, stream);
        auto r = download(*images);
        for (int x = 0; x < h; x++)
            for (int y = 0; y < w; y++) {
                // the ramp along the other axis remains
                double phase = axis == 0 ? ay*y : ax*x;
                int i = x*w + y;
                EXPECT_NEAR(r[i].x, amp[i]*std::cos(phase), 1e-4);
                EXPECT_NEAR(r[i].y, amp[i]*std::sin(phase), 1e-4);
            }
    }
}

TEST_F(DerampTest, OtherMethodsSkip)
{
    const int h = 4, w = 5;
    auto v = randomComplex(h*w);
    auto images = makeFrom(v, h, w);
    for (int method : {0, 2}) {
        cuDeramp(method, images.get(), 2, stream);
        auto r = download(*images);
        for (int i = 0; i < h*w; i++) {
            EXPECT_EQ(r[i].x, v[i].x);
            EXPECT_EQ(r[i].y, v[i].y);
        }
    }
}

} // namespace
