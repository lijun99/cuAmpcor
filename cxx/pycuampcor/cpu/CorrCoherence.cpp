/**
 * @file  CorrCoherence.cpp
 * @brief The coherent cross-correlator of the complex windows on the CPU
 */

#include "CorrCoherence.h"
#include "float2.h"
#include "ArraysCopy.h"
#include "Arrays.h"

namespace pycuampcor::cpu {

// the mutex of the fftw planner (CorrFrequency.cpp)
std::mutex & fftwPlannerMutex();

} // namespace

namespace pycuampcor {

// the definitions use the CPU package: its types, arrays and fftw plans
using namespace cpu;

/// the coherent cross-correlator on the CPU
using CoherenceCorrelator = Correlator<Algorithm::Coherence, cpu::complex_type, Device::CPU>;

/**
 * Constructor of the coherent cross-correlator: the fftw plans and the work arrays
 * @param imageNX,imageNY the size of the secondary windows
 * @param nImages the number of windows
 * @param stream_ (not used)
 */
CoherenceCorrelator::Correlator(int imageNX, int imageNY, int nImages, stream_t stream_)
    : stream(stream_)
{
    const int imageSize = imageNX*imageNY;
    int n[2] = {imageNX, imageNY};
    workT = array_type<complex_type>(imageNX, imageNY, nImages);
    workS = array_type<complex_type>(imageNX, imageNY, nImages);
    sat = array_type<double>(imageNX, imageNY);
    satImag = array_type<double>(imageNX, imageNY);
    sat2 = array_type<double>(imageNX, imageNY);

    std::lock_guard<std::mutex> lock(fftwPlannerMutex());
    forwardPlan = PYCUAMPCOR_FFTW(plan_many_dft)(2, n, nImages,
        fftwCast(workT.data_handle()), nullptr, 1, imageSize,
        fftwCast(workT.data_handle()), nullptr, 1, imageSize,
        FFTW_FORWARD, fftwFlags);
    backwardPlan = PYCUAMPCOR_FFTW(plan_many_dft)(2, n, nImages,
        fftwCast(workT.data_handle()), nullptr, 1, imageSize,
        fftwCast(workT.data_handle()), nullptr, 1, imageSize,
        FFTW_BACKWARD, fftwFlags);
}

/**
 * Release the fftw plans and the work arrays
 */
CoherenceCorrelator::~Correlator() = default;

/**
 * Compute the squared coherence surfaces
 * @param[in] templates the reference windows
 * @param[in] images the secondary windows
 * @param[out] results the squared coherence surfaces (secondary - reference + 1 each way)
 */
void CoherenceCorrelator::execute(ArraysView<const complex_type> templates,
    ArraysView<const complex_type> images, ArraysView<real_type> results)
{
    const int h = templates.height(), w = templates.width(), n = h*w;
    const int H = images.height(), W = images.width();
    const int corNX = results.height(), corNY = results.width();
    const int count = images.count();
    const ArraysView<complex_type> padded = workT;
    const ArraysView<double> satReal = sat, satIm = satImag, satNorm = sat2;

    // the reference windows with their means subtracted, zero padded to the size of the secondary
    // windows, and the sums of their squared magnitudes
    std::vector<double> refSum2(count);
    workT.setZero(stream);
    for (int k = 0; k < count; k++) {
        const ArraysView<const complex_type> t = templates.array(k);
        double mx = 0, my = 0;
        for (int i = 0; i < n; i++) { mx += t[i].x; my += t[i].y; }
        mx /= n; my /= n;
        double sum2 = 0;
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++) {
                const double x = t(i, j).x - mx, y = t(i, j).y - my;
                padded(k, i, j) = make_complex_type(x, y);
                sum2 += x*x + y*y;
            }
        refSum2[k] = sum2;
    }
    // the spectra
    std::copy(images.data_handle(), images.data_handle() + images.size(), workS.data_handle());
    PYCUAMPCOR_FFTW(execute_dft)(forwardPlan, fftwCast(workT.data_handle()), fftwCast(workT.data_handle()));
    PYCUAMPCOR_FFTW(execute_dft)(forwardPlan, fftwCast(workS.data_handle()), fftwCast(workS.data_handle()));
    // conj(T) S, back to the correlations sum conj(t) s at each shift (fftw doesn't normalize)
    const real_type coef = real_type(1)/(H*W);
    complex_type *ft = workT.data_handle();
    const complex_type *fs = workS.data_handle();
    for (size_t i = 0; i < workT.size(); i++) {
        const complex_type a = ft[i], b = fs[i];
        ft[i] = make_complex_type(coef*(a.x*b.x + a.y*b.y), coef*(a.x*b.y - a.y*b.x));
    }
    PYCUAMPCOR_FFTW(execute_dft)(backwardPlan, fftwCast(workT.data_handle()), fftwCast(workT.data_handle()));

    for (int k = 0; k < count; k++) {
        // the sum area tables of the secondary window: its values (real and imaginary parts) and
        // their squared magnitudes
        const ArraysView<const complex_type> s = images.array(k);
        for (int i = 0; i < H; i++) {
            double sx = 0, sy = 0, s2 = 0;
            for (int j = 0; j < W; j++) {
                const double x = s(i, j).x, y = s(i, j).y;
                sx += x; sy += y; s2 += x*x + y*y;
                satReal(i, j) = (i > 0 ? satReal(i-1, j) : 0.0) + sx;
                satIm(i, j) = (i > 0 ? satIm(i-1, j) : 0.0) + sy;
                satNorm(i, j) = (i > 0 ? satNorm(i-1, j) : 0.0) + s2;
            }
        }
        // the sum over the template at the shift (tx, ty)
        auto area = [&](ArraysView<const double> a, int tx, int ty) {
            const double bottomright = a(tx+h-1, ty+w-1);
            const double topleft = (tx > 0 && ty > 0) ? a(tx-1, ty-1) : 0.0;
            const double topright = (tx > 0) ? a(tx-1, ty+w-1) : 0.0;
            const double bottomleft = (ty > 0) ? a(tx+h-1, ty-1) : 0.0;
            return bottomright + topleft - topright - bottomleft;
        };
        // the squared coherence at each shift
        for (int tx = 0; tx < corNX; tx++)
            for (int ty = 0; ty < corNY; ty++) {
                const double sx = area(satReal, tx, ty), sy = area(satIm, tx, ty);
                const double norm2 = (area(satNorm, tx, ty) - (sx*sx + sy*sy)/n)*refSum2[k];
                const complex_type c = padded(k, tx, ty);
                results(k, tx, ty) = (double(c.x)*c.x + double(c.y)*c.y)/(norm2 + EPSILON);
            }
    }
}

} // namespace
// end of file
