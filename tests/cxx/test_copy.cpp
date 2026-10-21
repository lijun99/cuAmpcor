/**
 * @file test_copy.cpp
 * @brief Tests for the copy/extract/insert/padding/conversion kernels
 */

#include "test_util.h"

namespace pycuampcor::PYCUAMPCOR_BACKEND::test {

using CopyTest = BackendTest;

// a chunk of image with values (i*width+j, -(i*width+j))
static std::vector<image_complex_type> chunkValues(int height, int width)
{
    std::vector<image_complex_type> v(height*width);
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++) {
            v[i*width+j].x = i*width + j;
            v[i*width+j].y = -(i*width + j);
        }
    return v;
}

// copy windows from a chunk with varying offsets, filling zeros outside the chunk
TEST_F(CopyTest, CopyToBatchWithOffset)
{
    const int H = 6, W = 7, h = 3, w = 4;
    auto chunkV = chunkValues(H, W);
    auto chunk = makeFrom(chunkV, H, W);
    // two windows: inside, and partially outside (bottom-right)
    std::vector<int> offH = {1, 4}, offW = {2, 5};
    auto offsetH = makeFrom(offH, 1, 2);
    auto offsetW = makeFrom(offW, 1, 2);

    auto batch = make<complex_type>(h, w, 1, 2);
    cuArraysCopyToBatchWithOffset(chunk.get(), H, W, batch.get(), offsetH->devData, offsetW->devData, stream);
    auto out = download(*batch);

    auto batchAbs = make<complex_type>(h, w, 1, 2);
    cuArraysCopyToBatchAbsWithOffset(chunk.get(), H, W, batchAbs.get(), offsetH->devData, offsetW->devData, stream);
    auto outAbs = download(*batchAbs);

    for (int k = 0; k < 2; k++)
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++) {
                int idx = (k*h + i)*w + j;
                int ci = offH[k] + i, cj = offW[k] + j;
                bool inside = ci < H && cj < W;
                double re = inside ? chunkV[ci*W+cj].x : 0;
                double im = inside ? chunkV[ci*W+cj].y : 0;
                EXPECT_EQ(out[idx].x, re) << k << " " << i << " " << j;
                EXPECT_EQ(out[idx].y, im) << k << " " << i << " " << j;
                EXPECT_NEAR(outAbs[idx].x, std::hypot(re, im), imageTol*(1+std::hypot(re, im)));
                EXPECT_EQ(outAbs[idx].y, 0);
            }
}

// real images are converted to complex
TEST_F(CopyTest, CopyToBatchWithOffsetR2C)
{
    const int H = 5, W = 5, h = 2, w = 3;
    std::vector<image_real_type> chunkV(H*W);
    for (int i = 0; i < H*W; i++) chunkV[i] = i;
    auto chunk = makeFrom(chunkV, H, W);
    std::vector<int> offH = {-1}, offW = {3};
    auto offsetH = makeFrom(offH, 1, 1);
    auto offsetW = makeFrom(offW, 1, 1);
    auto batch = make<complex_type>(h, w);
    cuArraysCopyToBatchWithOffsetR2C(chunk.get(), H, W, batch.get(), offsetH->devData, offsetW->devData, stream);
    auto out = download(*batch);
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            int ci = offH[0] + i, cj = offW[0] + j;
            bool inside = ci >= 0 && ci < H && cj < W;
            EXPECT_EQ(out[i*w+j].x, inside ? chunkV[ci*W+cj] : 0);
            EXPECT_EQ(out[i*w+j].y, 0);
        }
}

// extract with a fixed offset: real->real, complex->real (real part), complex->complex
TEST_F(CopyTest, CopyExtractFixedOffset)
{
    const int H = 6, W = 5, h = 3, w = 2, n = 2;
    const int2 offset = make_int2(2, 1);
    auto rv = randomReal(H*W*n);
    auto cv = randomComplex(H*W*n);
    auto rin = makeFrom(rv, H, W, 1, n);
    auto cin = makeFrom(cv, H, W, 1, n);
    auto rout = make<real_type>(h, w, 1, n);
    auto crout = make<real_type>(h, w, 1, n);
    auto cout = make<complex_type>(h, w, 1, n);
    cuArraysCopyExtract(rin.get(), rout.get(), offset, stream);
    cuArraysCopyExtract(cin.get(), crout.get(), offset, stream);
    cuArraysCopyExtract(cin.get(), cout.get(), offset, stream);
    auto r = download(*rout), cr = download(*crout);
    auto c = download(*cout);
    for (int k = 0; k < n; k++)
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++) {
                int o = (k*h + i)*w + j;
                int s = (k*H + i + offset.x)*W + j + offset.y;
                EXPECT_EQ(r[o], rv[s]);
                EXPECT_EQ(cr[o], cv[s].x);
                EXPECT_EQ(c[o].x, cv[s].x);
                EXPECT_EQ(c[o].y, cv[s].y);
            }
}

// extract with a varying offset for each image
TEST_F(CopyTest, CopyExtractVaryingOffset)
{
    const int H = 7, W = 6, h = 3, w = 3, n = 2;
    auto rv = randomReal(H*W*n);
    auto rin = makeFrom(rv, H, W, 1, n);
    std::vector<int2> offV = {make_int2(0, 3), make_int2(4, 1)};
    auto offsets = makeFrom(offV, 1, n);
    auto rout = make<real_type>(h, w, 1, n);
    cuArraysCopyExtract(rin.get(), rout.get(), offsets.get(), stream);
    auto r = download(*rout);
    for (int k = 0; k < n; k++)
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                EXPECT_EQ(r[(k*h + i)*w + j], rv[(k*H + i + offV[k].x)*W + j + offV[k].y]);
}

// extract with amplitudes
TEST_F(CopyTest, CopyExtractAbs)
{
    const int H = 4, W = 5, h = 2, w = 3;
    auto cv = randomComplex(H*W);
    auto cin = makeFrom(cv, H, W);
    auto rout = make<real_type>(h, w);
    cuArraysCopyExtractAbs(cin.get(), rout.get(), make_int2(1, 2), stream);
    auto r = download(*rout);
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            auto v = cv[(i+1)*W + j+2];
            EXPECT_NEAR(r[i*w+j], std::hypot(v.x, v.y), tol);
        }
}

// extract windows centered at the max locations, with validity flags
TEST_F(CopyTest, CopyExtractCorr)
{
    const int H = 9, W = 9, h = 5, w = 5, n = 2;
    auto rv = randomReal(H*W*n);
    auto rin = makeFrom(rv, H, W, 1, n);
    // centered inside, and near the top-left corner
    std::vector<int2> locV = {make_int2(4, 5), make_int2(1, 0)};
    auto maxloc = makeFrom(locV, 1, n);
    auto out = make<real_type>(h, w, 1, n);
    auto valid = make<int>(h, w, 1, n);
    cuArraysCopyExtractCorr(rin.get(), out.get(), valid.get(), maxloc.get(), stream);
    auto r = download(*out);
    auto f = download(*valid);
    for (int k = 0; k < n; k++)
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++) {
                int o = (k*h + i)*w + j;
                int si = i + locV[k].x - h/2, sj = j + locV[k].y - w/2;
                bool inside = si >= 0 && sj >= 0 && si < H && sj < W;
                EXPECT_EQ(f[o], inside ? 1 : 0);
                EXPECT_EQ(r[o], inside ? rv[(k*H + si)*W + sj] : 0);
            }

    // without validity flags (the window is assumed to be inside)
    std::vector<int2> inV = {make_int2(4, 5), make_int2(2, 6)};
    auto inside = makeFrom(inV, 1, n);
    cuArraysCopyExtractCorr(rin.get(), out.get(), inside.get(), stream);
    r = download(*out);
    for (int k = 0; k < n; k++)
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                EXPECT_EQ(r[(k*h + i)*w + j], rv[(k*H + i + inV[k].x - h/2)*W + j + inV[k].y - w/2]);
}

// extract windows with a stride (e.g., a raw pixel spacing on an oversampled surface),
// valid within a region of the input
TEST_F(CopyTest, CopyExtractCorrStrided)
{
    const int H = 21, W = 25, h = 5, w = 7, n = 2, stride = 2;
    const int2 start = make_int2(3, 2), range = make_int2(15, 20);
    auto rv = randomReal(H*W*n);
    auto rin = makeFrom(rv, H, W, 1, n);
    // centered inside, and near the edge of the valid region
    std::vector<int2> locV = {make_int2(10, 12), make_int2(4, 19)};
    auto maxloc = makeFrom(locV, 1, n);
    auto out = make<real_type>(h, w, 1, n);
    auto valid = make<int>(h, w, 1, n);
    cuArraysCopyExtractCorr(rin.get(), out.get(), valid.get(), maxloc.get(), stride, start, range, stream);
    auto r = download(*out);
    auto f = download(*valid);
    int nInvalid = 0;
    for (int k = 0; k < n; k++)
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++) {
                int o = (k*h + i)*w + j;
                int si = locV[k].x + (i - h/2)*stride, sj = locV[k].y + (j - w/2)*stride;
                bool inside = si >= start.x && sj >= start.y && si < start.x + range.x && sj < start.y + range.y;
                nInvalid += !inside;
                EXPECT_EQ(f[o], inside ? 1 : 0);
                EXPECT_EQ(r[o], inside ? rv[(k*H + si)*W + sj] : 0);
            }
    // the second window crosses the edges of the valid region
    EXPECT_GT(nInvalid, 0);
}

// insert a small image into a larger one
TEST_F(CopyTest, CopyInsert)
{
    const int H = 5, W = 6, h = 2, w = 3;
    auto big = make<real_type>(H, W);
    big->setZero(stream);
    auto sv = randomReal(h*w);
    auto small = makeFrom(sv, h, w);
    cuArraysCopyInsert(small.get(), big.get(), 2, 3, stream);
    auto r = download(*big);
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) {
            bool inside = i >= 2 && i < 2+h && j >= 3 && j < 3+w;
            EXPECT_EQ(r[i*W+j], inside ? sv[(i-2)*w + j-3] : 0);
        }
}

// copy to a larger size with zero padding, real->complex
TEST_F(CopyTest, CopyPadded)
{
    const int h = 2, w = 3, H = 4, W = 5, n = 2;
    auto sv = randomReal(h*w*n);
    auto small = makeFrom(sv, h, w, 1, n);
    auto big = make<complex_type>(H, W, 1, n);
    cuArraysCopyPadded(small.get(), big.get(), stream);
    auto r = download(*big);
    for (int k = 0; k < n; k++)
        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++) {
                bool inside = i < h && j < w;
                EXPECT_EQ(r[(k*H + i)*W + j].x, inside ? sv[(k*h + i)*w + j] : 0);
                EXPECT_EQ(r[(k*H + i)*W + j].y, 0);
            }
}

// element-wise conversions
TEST_F(CopyTest, Conversions)
{
    const int n = 17;
    auto cv = randomComplex(n);
    auto rv = randomReal(n);
    auto c = makeFrom(cv, 1, n);
    auto r = makeFrom(rv, 1, n);
    auto rout = make<real_type>(1, n);
    auto cout = make<complex_type>(1, n);

    cuArraysAbs(c.get(), rout.get(), stream);
    auto a = download(*rout);
    cuArraysC2R(c.get(), rout.get(), stream);
    auto re = download(*rout);
    cuArraysR2C(r.get(), cout.get(), stream);
    auto rc = download(*cout);
    cuArraysSetConstant(r.get(), 2.5, stream);
    auto k = download(*r);
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(a[i], std::hypot(cv[i].x, cv[i].y), tol);
        EXPECT_EQ(re[i], cv[i].x);
        EXPECT_EQ(rc[i].x, rv[i]);
        EXPECT_EQ(rc[i].y, 0);
        EXPECT_EQ(k[i], real_type(2.5));
    }
}

} // namespace
