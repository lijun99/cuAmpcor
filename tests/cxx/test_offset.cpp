/**
 * @file test_offset.cpp
 * @brief Tests for max location and offset determination
 */

#include "test_util.h"

namespace pycuampcor::PYCUAMPCOR_BACKEND::test {

using OffsetTest = BackendTest;

TEST_F(OffsetTest, Maxloc2D)
{
    const int H = 9, W = 11, n = 3;
    auto v = randomReal(H*W*n);
    // global maxima of each image; image 2 also has a larger value outside the search range below
    std::vector<int2> peaks = {make_int2(0, 0), make_int2(8, 10), make_int2(4, 5)};
    for (int k = 0; k < n; k++)
        v[(k*H + peaks[k].x)*W + peaks[k].y] = 2 + k;
    v[(2*H + 0)*W + 1] = 10;
    auto images = makeFrom(v, H, W, 1, n);
    auto loc = make<int2>(1, n);
    auto val = make<real_type>(1, n);

    // whole image
    cuArraysMaxloc2D(images.get(), loc.get(), val.get(), stream);
    auto l = download(*loc);
    auto m = download(*val);
    EXPECT_EQ(l[0].x, 0); EXPECT_EQ(l[0].y, 0); EXPECT_EQ(m[0], 2);
    EXPECT_EQ(l[1].x, 8); EXPECT_EQ(l[1].y, 10); EXPECT_EQ(m[1], 3);
    EXPECT_EQ(l[2].x, 0); EXPECT_EQ(l[2].y, 1); EXPECT_EQ(m[2], 10);

    // in a range, excluding the first row
    cuArraysMaxloc2D(images.get(), make_int2(1, 0), make_int2(H-1, W), loc.get(), val.get(), stream);
    l = download(*loc);
    m = download(*val);
    EXPECT_EQ(l[2].x, 4); EXPECT_EQ(l[2].y, 5); EXPECT_EQ(m[2], 4);
    EXPECT_EQ(l[1].x, 8); EXPECT_EQ(l[1].y, 10);
    // image 0: its max is excluded; check against a reference search
    int2 ref = make_int2(1, 0);
    double best = -1e30;
    for (int i = 1; i < H; i++)
        for (int j = 0; j < W; j++)
            if (v[i*W+j] > best) { best = v[i*W+j]; ref = make_int2(i, j); }
    EXPECT_EQ(l[0].x, ref.x); EXPECT_EQ(l[0].y, ref.y); EXPECT_EQ(m[0], real_type(best));
}

TEST_F(OffsetTest, SubPixelOffset)
{
    std::vector<int2> initV = {make_int2(10, 12), make_int2(3, 25)};
    std::vector<int2> zoomV = {make_int2(260, 250), make_int2(128, 300)};
    auto init = makeFrom(initV, 1, 2);
    auto zoom = makeFrom(zoomV, 1, 2);
    auto final_ = make<real2_type>(1, 2);
    const int2 initOrigin = make_int2(10, 10), zoomOrigin = make_int2(256, 256);
    const int initFactor = 2, zoomFactor = 128;
    cuSubPixelOffset(init.get(), zoom.get(), final_.get(), initOrigin, initFactor, zoomOrigin, zoomFactor, stream);
    auto f = download(*final_);
    for (int k = 0; k < 2; k++) {
        double x = double(initV[k].x - initOrigin.x)/initFactor + double(zoomV[k].x - zoomOrigin.x)/zoomFactor;
        double y = double(initV[k].y - initOrigin.y)/initFactor + double(zoomV[k].y - zoomOrigin.y)/zoomFactor;
        EXPECT_NEAR(f[k].x, x, tol);
        EXPECT_NEAR(f[k].y, y, tol);
    }
}

TEST_F(OffsetTest, SubPixelOffset2Pass)
{
    std::vector<int2> initV = {make_int2(18, 22), make_int2(0, 40)};
    std::vector<int2> zoomV = {make_int2(64, 100), make_int2(3, 255)};
    auto init = makeFrom(initV, 1, 2);
    auto zoom = makeFrom(zoomV, 1, 2);
    auto final_ = make<real2_type>(1, 2);
    const int zoomOS = 64, rawOS = 2, halfX = 20, halfY = 22;
    cuSubPixelOffset2Pass(init.get(), zoom.get(), final_.get(), zoomOS, rawOS, halfX, halfY, stream);
    auto f = download(*final_);
    for (int k = 0; k < 2; k++) {
        EXPECT_NEAR(f[k].x, double(zoomV[k].x)/(zoomOS*rawOS) + initV[k].x - halfX, tol*100);
        EXPECT_NEAR(f[k].y, double(zoomV[k].y)/(zoomOS*rawOS) + initV[k].y - halfY, tol*100);
    }
}

// the zoom-in extraction start and the shift of the max location from the extraction center
TEST_F(OffsetTest, DetermineSecondaryExtractOffset)
{
    const int oldRange = 12, newRange = 2;  // max locations in [0, 2*oldRange]
    const int rbound = 2*(oldRange - newRange);
    // inside, left edge, right edge (x); and the same pattern shifted for y
    std::vector<int2> locV = {make_int2(10, 1), make_int2(0, 24), make_int2(24, 12)};
    auto loc = makeFrom(locV, 1, 3);
    auto shift = make<int2>(1, 3);
    cuDetermineSecondaryExtractOffset(loc.get(), shift.get(), oldRange, oldRange, newRange, newRange, stream);
    auto start = download(*loc);
    auto s = download(*shift);
    auto expect = [&](int maxloc, int st, int sh) {
        int refStart = maxloc - newRange, refShift = 0;
        if (refStart < 0) { refShift = refStart; refStart = 0; }
        else if (refStart > rbound) { refShift = refStart - rbound; refStart = rbound; }
        EXPECT_EQ(st, refStart) << "maxloc " << maxloc;
        EXPECT_EQ(sh, refShift) << "maxloc " << maxloc;
        // the max location is at (newRange + shift) in the extracted window
        EXPECT_EQ(maxloc - st, newRange + sh) << "maxloc " << maxloc;
    };
    for (int k = 0; k < 3; k++) {
        expect(locV[k].x, start[k].x, s[k].x);
        expect(locV[k].y, start[k].y, s[k].y);
    }
    // explicit values for the edges
    EXPECT_EQ(s[1].x, -2);  // left edge: the max is on the left of the center
    EXPECT_EQ(s[1].y, 2);   // right edge: on the right
}

} // namespace
