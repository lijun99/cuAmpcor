/**
 * @file test_arrays.cpp
 * @brief Tests for cuArrays: allocation, copying between host and backend memory
 */

#include "test_util.h"

namespace pycuampcor::PYCUAMPCOR_BACKEND::test {

using ArraysTest = BackendTest;

TEST_F(ArraysTest, Sizes)
{
    cuArrays<real_type> a(3, 4, 2, 5);
    EXPECT_EQ(a.size, 12);
    EXPECT_EQ(a.count, 10);
    EXPECT_EQ(a.getSize(), 120u);
    EXPECT_EQ((size_t)a.getByteSize(), 120*sizeof(real_type));
    EXPECT_FALSE(a.is_allocated);
}

TEST_F(ArraysTest, RoundTripAndSetZero)
{
    std::vector<real_type> v(3*4*2);
    for (size_t i = 0; i < v.size(); i++) v[i] = i;
    auto a = makeFrom(v, 3, 4, 1, 2);
    // clear host data to make sure the values come back from the backend
    std::fill(a->hostData, a->hostData + a->getSize(), real_type(-1));
    EXPECT_EQ(download(*a), v);

    a->setZero(stream);
    for (auto x : download(*a)) EXPECT_EQ(x, 0);
}

TEST_F(ArraysTest, ComplexRoundTrip)
{
    auto v = randomComplex(5*7);
    auto a = makeFrom(v, 5, 7);
    auto w = download(*a);
    for (size_t i = 0; i < v.size(); i++) {
        EXPECT_EQ(w[i].x, v[i].x);
        EXPECT_EQ(w[i].y, v[i].y);
    }
}

} // namespace
