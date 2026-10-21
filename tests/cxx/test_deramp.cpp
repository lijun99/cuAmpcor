/**
 * @file  test_deramp.cpp
 * @brief Tests for removing linear phase ramps from complex images
 */

#include "test_util.h"

namespace pycuampcor::PYCUAMPCOR_BACKEND::test {

using DerampTest = BackendTest;

/**
 * Random amplitudes with a linear phase ramp exp(i(ax*x + ay*y))
 */
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
    cuDeramp(DerampMethod::Complex, images, DerampAxis::Both, stream);
    auto r = download(images);
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
    for (auto axis : {DerampAxis::Down, DerampAxis::Across}) {
        auto images = makeFrom(v, h, w);
        cuDeramp(DerampMethod::Complex, images, axis, stream);
        auto r = download(images);
        for (int x = 0; x < h; x++)
            for (int y = 0; y < w; y++) {
                // the ramp along the other axis remains
                double phase = axis == DerampAxis::Down ? ay*y : ax*x;
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
    for (auto method : {DerampMethod::Magnitude, DerampMethod::None}) {
        cuDeramp(method, images, DerampAxis::Both, stream);
        auto r = download(images);
        for (int i = 0; i < h*w; i++) {
            EXPECT_EQ(r[i].x, v[i].x);
            EXPECT_EQ(r[i].y, v[i].y);
        }
    }
}

/**
 * Deramp by the reference: the ramp of the reference window is removed from it and from its
 * secondary window (of another size and ramp), which keeps their relative phase
 */
TEST_F(DerampTest, ByReference)
{
    const int h = 16, w = 24, H = 20, W = 30;
    std::vector<double> amp, amp2;
    auto rv = rampImage(h, w, 0.7, -1.3, amp);
    auto sv = rampImage(H, W, 0.9, -1.1, amp2);
    auto reference = makeFrom(rv, h, w);
    auto secondary = makeFrom(sv, H, W);
    cuDerampByReference(DerampMethod::Complex, reference, secondary, DerampAxis::Both, stream);
    auto r = download(reference);
    auto s = download(secondary);
    for (int i = 0; i < h*w; i++) {
        EXPECT_NEAR(r[i].x, amp[i], 1e-4);
        EXPECT_NEAR(r[i].y, 0, 1e-4);
    }
    // the secondary keeps the difference of the ramps
    for (int x = 0; x < H; x++)
        for (int y = 0; y < W; y++) {
            const double phase = 0.2*x + 0.2*y;
            EXPECT_NEAR(s[x*W + y].x, amp2[x*W + y]*std::cos(phase), 1e-4);
            EXPECT_NEAR(s[x*W + y].y, amp2[x*W + y]*std::sin(phase), 1e-4);
        }
    // the other methods leave the windows as they are
    auto r2 = makeFrom(rv, h, w);
    auto s2 = makeFrom(sv, H, W);
    cuDerampByReference(DerampMethod::None, r2, s2, DerampAxis::Both, stream);
    auto rr = download(r2);
    EXPECT_EQ(rr[5].x, rv[5].x);
}

} // namespace
