/**
 * @file test_stats.cpp
 * @brief Tests for mean subtraction, sums, and the SNR/variance estimates
 */

#include "test_util.h"

namespace pycuampcor::PYCUAMPCOR_BACKEND::test {

using StatsTest = BackendTest;

TEST_F(StatsTest, SubtractMean)
{
    const int h = 7, w = 9, n = 3;
    auto v = randomReal(h*w*n);
    for (auto &x : v) x += 3;  // non-zero mean
    auto a = makeFrom(v, h, w, 1, n);
    cuArraysSubtractMean(a.get(), stream);
    auto r = download(*a);
    for (int k = 0; k < n; k++) {
        double mean = 0;
        for (int i = 0; i < h*w; i++) mean += v[k*h*w + i];
        mean /= h*w;
        for (int i = 0; i < h*w; i++)
            EXPECT_NEAR(r[k*h*w + i], v[k*h*w + i] - mean, 10*tol);
    }
}

TEST_F(StatsTest, SumSquareAndSumCorr)
{
    const int h = 5, w = 6, n = 2;
    auto v = randomReal(h*w*n);
    std::vector<int> valid(h*w*n);
    for (size_t i = 0; i < valid.size(); i++) valid[i] = (i % 3 != 0);
    auto a = makeFrom(v, h, w, 1, n);
    auto f = makeFrom(valid, h, w, 1, n);
    auto sum = make<real_type>(1, n);
    auto sum2 = make<real_type>(1, n);
    auto count = make<int>(1, n);
    cuArraysSumSquare(a.get(), sum.get(), stream);
    cuArraysSumCorr(a.get(), f.get(), sum2.get(), count.get(), stream);
    auto s = download(*sum), s2 = download(*sum2);
    auto c = download(*count);
    for (int k = 0; k < n; k++) {
        double ref = 0;
        int refCount = 0;
        for (int i = 0; i < h*w; i++) {
            ref += v[k*h*w+i]*v[k*h*w+i];
            refCount += valid[k*h*w+i];
        }
        EXPECT_NEAR(s[k], ref, 10*tol*ref);
        EXPECT_NEAR(s2[k], ref, 10*tol*ref);
        EXPECT_EQ(c[k], refCount);
    }
}

TEST_F(StatsTest, EstimateSnr)
{
    std::vector<real_type> sumV = {10, 20}, peakV = {0.9, 0.5};
    std::vector<int> countV = {21, 41};
    auto sum = makeFrom(sumV, 1, 2);
    auto peak = makeFrom(peakV, 1, 2);
    auto count = makeFrom(countV, 1, 2);
    auto snr = make<real_type>(1, 2);

    // with the count of valid pixels
    cuEstimateSnr(sum.get(), count.get(), peak.get(), snr.get(), stream);
    auto r = download(*snr);
    for (int k = 0; k < 2; k++) {
        double p2 = double(peakV[k])*peakV[k];
        EXPECT_NEAR(r[k], p2/((sumV[k] - p2)/(countV[k] - 1)), 10*tol*r[k]);
    }

    // with a fixed size
    const int size = 100;
    cuEstimateSnr(sum.get(), peak.get(), snr.get(), size, stream);
    r = download(*snr);
    for (int k = 0; k < 2; k++) {
        double p2 = double(peakV[k])*peakV[k];
        EXPECT_NEAR(r[k], p2/((sumV[k] - p2)/(size - 1)), 10*tol*r[k]);
    }
}

// reference covariance from the curvature of the correlation surface at the peak
static std::vector<double> refCovariance(const std::vector<real_type> &c, int NY, int2 loc,
    double peak, int templateSize, int d)
{
    auto at = [&](int x, int y) { return double(c[x*NY + y]); };
    int px = loc.x, py = loc.y;
    double dxx = -(at(px+d, py) + at(px-d, py) - 2*at(px, py)) * templateSize;
    double dyy = -(at(px, py+d) + at(px, py-d) - 2*at(px, py)) * templateSize;
    double dxy = (at(px+d, py+d) + at(px-d, py-d) - at(px+d, py-d) - at(px-d, py+d)) * 0.25 * templateSize;
    double n2 = std::max(1.0 - peak, 0.0);
    double n4 = n2*n2*0.5*templateSize;
    n2 *= 2;
    double u = dxy*dxy - dxx*dyy;
    double u2 = u*u;
    return {(-n2*u*dyy + n4*(dyy*dyy + dxy*dxy))/u2,
            (-n2*u*dxx + n4*(dxx*dxx + dxy*dxy))/u2,
            (n2*u*dxy - n4*(dxx + dyy)*dxy)/u2};
}

TEST_F(StatsTest, EstimateVariance)
{
    const int NX = 11, NY = 13, n = 2, templateSize = 64;
    // quadratic correlation surfaces peaked at (5, 6) and at the margin (0, 3)
    std::vector<int2> locV = {make_int2(5, 6), make_int2(0, 3)};
    std::vector<real_type> c(NX*NY*n), peakV(n);
    for (int k = 0; k < n; k++) {
        for (int x = 0; x < NX; x++)
            for (int y = 0; y < NY; y++) {
                double dx = x - locV[k].x, dy = y - locV[k].y;
                c[(k*NX + x)*NY + y] = 0.9 - 0.02*dx*dx - 0.03*dy*dy - 0.01*dx*dy;
            }
        peakV[k] = 0.9;
    }
    auto corr = makeFrom(c, NX, NY, 1, n);
    auto loc = makeFrom(locV, 1, n);
    auto peak = makeFrom(peakV, 1, n);
    auto cov = make<real3_type>(1, n);
    for (int d : {1, 2}) {
        cuEstimateVariance(corr.get(), loc.get(), peak.get(), templateSize, d, cov.get(), stream);
        auto r = download(*cov);
        std::vector<real_type> c0(c.begin(), c.begin() + NX*NY);
        auto ref = refCovariance(c0, NY, locV[0], peakV[0], templateSize, d);
        EXPECT_NEAR(r[0].x, ref[0], 1e-3*std::abs(ref[0]));
        EXPECT_NEAR(r[0].y, ref[1], 1e-3*std::abs(ref[1]));
        EXPECT_NEAR(r[0].z, ref[2], 1e-3*std::abs(ref[2]));
        EXPECT_GT(r[0].x, 0);
        EXPECT_GT(r[0].y, 0);
        // the peak at the margin is flagged
        EXPECT_EQ(r[1].x, 99);
        EXPECT_EQ(r[1].y, 99);
        EXPECT_EQ(r[1].z, 0);
    }
}

} // namespace
