/**
 * @file test_arrays.cpp
 * @brief Tests for cuArrays: allocation, copying between host and backend memory
 */

#include "test_util.h"

namespace pycuampcor::PYCUAMPCOR_BACKEND::test {

using ArraysTest = BackendTest;

TEST_F(ArraysTest, Sizes)
{
    array_type<real_type> a(3, 4, 2, 5);
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

TEST(ArraysViewTest, ExtentsAndIndexing)
{
    // 2 arrays of 3 x 4, the element (k, i, j) holds 100k + 10i + j
    std::vector<int> v(2*3*4);
    for (int k = 0; k < 2; k++)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++) v[(k*3 + i)*4 + j] = 100*k + 10*i + j;
    ArraysView<int> a(v.data(), 3, 4, 2);
    EXPECT_EQ(a.rank(), 3);
    EXPECT_EQ(a.extent(0), 2);
    EXPECT_EQ(a.extent(1), 3);
    EXPECT_EQ(a.extent(2), 4);
    EXPECT_EQ(a.count(), 2);
    EXPECT_EQ(a.height(), 3);
    EXPECT_EQ(a.width(), 4);
    EXPECT_EQ(a.arraySize(), 12);
    EXPECT_EQ(a.size(), 24u);
    EXPECT_FALSE(a.empty());
    EXPECT_EQ(a.data_handle(), v.data());
    EXPECT_EQ(a(1, 2, 3), 123);
    EXPECT_EQ(a(2, 1), 21);
    EXPECT_EQ(a[13], 101);

    // the second array
    auto b = a.array(1);
    EXPECT_EQ(b.count(), 1);
    EXPECT_EQ(b.height(), 3);
    EXPECT_EQ(b.width(), 4);
    EXPECT_EQ(b(2, 3), 123);
    b(0, 0) = -1;
    EXPECT_EQ(v[12], -1);

    // read-only view of the same data
    ArraysView<const int> c = a;
    EXPECT_EQ(c.data_handle(), v.data());
    EXPECT_EQ(c(1, 0, 0), -1);
    static_assert(std::is_same_v<decltype(c(0, 0, 0)), const int &>);
    static_assert(!std::is_convertible_v<ArraysView<const int>, ArraysView<int>>);

    // an empty view
    EXPECT_TRUE(ArraysView<int>().empty());
}

TEST_F(ArraysTest, View)
{
    array_type<real_type> a(3, 4, 2, 5);
    a.allocate();
    ArraysView<real_type> v = a;
    EXPECT_EQ(v.data_handle(), a.devData);
    EXPECT_EQ(v.count(), 10);
    EXPECT_EQ(v.height(), 3);
    EXPECT_EQ(v.width(), 4);
    const auto &c = a;
    ArraysView<const real_type> w = c;
    EXPECT_EQ(w.data_handle(), a.devData);
    EXPECT_EQ(w.size(), a.getSize());
}

} // namespace
