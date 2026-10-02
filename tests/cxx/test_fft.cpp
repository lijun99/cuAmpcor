/**
 * @file test_fft.cpp
 * @brief Tests for the FFT oversamplers, correlators and the correlation surface normalization
 */

#include "test_util.h"
#include "cuOverSampler.h"
#include "cuCorrFrequency.h"
#include "cuCorrNormalizer.h"

namespace pycuampcor::PYCUAMPCOR_BACKEND::test {

class FFTTest : public BackendTest {
protected:
    void checkNormalizer(int h, int w, int H, int W, int2 lag);
};

constexpr double pi = 3.14159265358979323846;

// complex exponential exp(i 2pi (kx x/nx + ky y/ny)) sampled on a nx x ny grid, with a scale factor
static std::vector<complex_type> exponential(int nx, int ny, int kx, int ky, double scale = 1)
{
    std::vector<complex_type> v(nx*ny);
    for (int x = 0; x < nx; x++)
        for (int y = 0; y < ny; y++) {
            double phase = 2*pi*(double(kx)*x/nx + double(ky)*y/ny);
            v[x*ny + y] = make_complex_type(scale*std::cos(phase), scale*std::sin(phase));
        }
    return v;
}

// a band-limited signal is exactly interpolated by the FFT oversampling
TEST_F(FFTTest, OverSamplerC2C)
{
    // even and odd sizes, with positive and negative frequencies
    const int nx = 16, ny = 15, ratio = 2, n = 2;
    const int kx[n] = {3, -5}, ky[n] = {-2, 6};
    std::vector<complex_type> v;
    for (int k = 0; k < n; k++) {
        auto e = exponential(nx, ny, kx[k], ky[k], k+1);
        v.insert(v.end(), e.begin(), e.end());
    }
    auto in = makeFrom(v, nx, ny, 1, n);
    auto out = make<complex_type>(nx*ratio, ny*ratio, 1, n);
    cuOverSamplerC2C oversampler(nx, ny, nx*ratio, ny*ratio, n, stream);
    oversampler.execute(in.get(), out.get());
    auto r = download(*out);
    // the input is unchanged (out-of-place)
    auto v2 = download(*in);
    EXPECT_EQ(v2[5].x, v[5].x);
    for (int k = 0; k < n; k++) {
        auto ref = exponential(nx*ratio, ny*ratio, kx[k], ky[k], k+1);
        for (size_t i = 0; i < ref.size(); i++) {
            EXPECT_NEAR(r[k*ref.size() + i].x, ref[i].x, 100*tol);
            EXPECT_NEAR(r[k*ref.size() + i].y, ref[i].y, 100*tol);
        }
    }
}

// the Nyquist frequency of even lengths is sided with the positive frequencies (as the isce3 v1 pycuampcor):
// (-1)^x is interpolated as exp(+i pi x), for both even dimensions, and the odd dimension is exact
TEST_F(FFTTest, OverSamplerNyquistPositive)
{
    const int nx = 8, ny = 7, ratio = 2;
    // Nyquist along x (kx = nx/2); a regular frequency along the odd y
    auto v = exponential(nx, ny, nx/2, 2);
    auto in = makeFrom(v, nx, ny, 1, 1);
    auto out = make<complex_type>(nx*ratio, ny*ratio, 1, 1);
    cuOverSamplerC2C oversampler(nx, ny, nx*ratio, ny*ratio, 1, stream);
    oversampler.execute(in.get(), out.get());
    auto r = download(*out);
    auto ref = exponential(nx*ratio, ny*ratio, nx/2, 2);
    for (size_t i = 0; i < ref.size(); i++) {
        EXPECT_NEAR(r[i].x, ref[i].x, 100*tol);
        EXPECT_NEAR(r[i].y, ref[i].y, 100*tol);
    }
}

TEST_F(FFTTest, OverSamplerR2R)
{
    const int nx = 12, ny = 10, ratio = 4;
    std::vector<real_type> v(nx*ny);
    auto f = [](double x, double y) { return std::cos(2*pi*(2*x + 3*y)) + 0.5*std::sin(2*pi*x); };
    for (int x = 0; x < nx; x++)
        for (int y = 0; y < ny; y++)
            v[x*ny + y] = f(double(x)/nx, double(y)/ny);
    auto in = makeFrom(v, nx, ny);
    auto out = make<real_type>(nx*ratio, ny*ratio);
    cuOverSamplerR2R oversampler(nx, ny, nx*ratio, ny*ratio, 1, stream);
    oversampler.execute(in.get(), out.get());
    auto r = download(*out);
    for (int x = 0; x < nx*ratio; x++)
        for (int y = 0; y < ny*ratio; y++)
            EXPECT_NEAR(r[x*ny*ratio + y], f(double(x)/(nx*ratio), double(y)/(ny*ratio)), 100*tol);
}

// direct (time domain) cross-correlation for reference
static std::vector<double> naiveCorrelation(const std::vector<real_type> &t, int h, int w,
    const std::vector<real_type> &s, int H, int W, int n)
{
    const int ch = H - h + 1, cw = W - w + 1;
    std::vector<double> c(ch*cw*n, 0.0);
    for (int k = 0; k < n; k++)
        for (int i = 0; i < ch; i++)
            for (int j = 0; j < cw; j++) {
                double sum = 0;
                for (int m = 0; m < h; m++)
                    for (int l = 0; l < w; l++)
                        sum += double(t[(k*h + m)*w + l]) * s[(k*H + i + m)*W + j + l];
                c[(k*ch + i)*cw + j] = sum;
            }
    return c;
}

TEST_F(FFTTest, Correlators)
{
    const int h = 8, w = 12, H = 20, W = 30, n = 3;
    const int ch = H - h + 1, cw = W - w + 1;
    auto tv = randomReal(h*w*n, 3);
    auto sv = randomReal(H*W*n, 4);
    auto ref = naiveCorrelation(tv, h, w, sv, H, W, n);
    auto t = makeFrom(tv, h, w, 1, n);
    auto s = makeFrom(sv, H, W, 1, n);
    auto c = make<real_type>(ch, cw, 1, n);

    cuFreqCorrelator correlator(H, W, n, stream);
    correlator.execute(t.get(), s.get(), c.get());
    auto rf = download(*c);
    cuCorrTimeDomain(t.get(), s.get(), c.get(), stream);
    auto rt = download(*c);
    for (size_t i = 0; i < ref.size(); i++) {
        EXPECT_NEAR(rf[i], ref[i], 1e3*tol) << i;
        EXPECT_NEAR(rt[i], ref[i], 1e2*tol) << i;
    }
}

// normalized cross-correlation: the template is a (mean-subtracted) part of the search image
void FFTTest::checkNormalizer(int h, int w, int H, int W, int2 lag)
{
    const int ch = H - h + 1, cw = W - w + 1, n = 1;
    auto sv = randomReal(H*W, 5);
    for (auto &x : sv) x += 0.5;  // non-zero mean
    std::vector<real_type> tv(h*w);
    double mean = 0;
    for (int m = 0; m < h; m++)
        for (int l = 0; l < w; l++) {
            tv[m*w + l] = sv[(lag.x + m)*W + lag.y + l];
            mean += tv[m*w + l];
        }
    mean /= h*w;
    for (auto &x : tv) x -= mean;

    auto corr = naiveCorrelation(tv, h, w, sv, H, W, n);
    std::vector<real_type> cv(corr.begin(), corr.end());
    auto t = makeFrom(tv, h, w);
    auto s = makeFrom(sv, H, W);
    auto c = makeFrom(cv, ch, cw);
    std::unique_ptr<cuNormalizeProcessor> normalizer(newCuNormalizer(H, W, n));
    normalizer->execute(c.get(), t.get(), s.get(), stream);
    auto r = download(*c);

    double t2 = 0;
    for (auto x : tv) t2 += double(x)*x;
    for (int i = 0; i < ch; i++)
        for (int j = 0; j < cw; j++) {
            double s1 = 0, s2 = 0;
            for (int m = 0; m < h; m++)
                for (int l = 0; l < w; l++) {
                    double v = sv[(i + m)*W + j + l];
                    s1 += v; s2 += v*v;
                }
            double ref = corr[i*cw + j]/std::sqrt(t2*(s2 - s1*s1/(h*w)));
            EXPECT_NEAR(r[i*cw + j], ref, 1e-3) << i << " " << j;
        }
    // the template matches at the lag
    EXPECT_NEAR(r[lag.x*cw + lag.y], 1.0, 1e-3);
}

TEST_F(FFTTest, Normalizer)
{
    // small search windows (for GPU, fixed-size normalizer)
    checkNormalizer(6, 10, 14, 40, make_int2(3, 17));
}

TEST_F(FFTTest, NormalizerWide)
{
    // search windows wider than 1024 (for GPU, sum area table normalizer)
    checkNormalizer(3, 1090, 7, 1100, make_int2(2, 6));
}

} // namespace
