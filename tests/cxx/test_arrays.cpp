/**
 * @file  test_arrays.cpp
 * @brief Tests for the arrays (cuArrays, Arrays) and their views: allocation, moves, copies
 *     between host and backend memory, indexing
 */

#include "test_util.h"

namespace pycuampcor::PYCUAMPCOR_BACKEND::test {

using ArraysTest = BackendTest;

TEST_F(ArraysTest, Sizes)
{
    array_type<real_type> a(3, 4, 10);
    EXPECT_EQ(a.height(), 3);
    EXPECT_EQ(a.width(), 4);
    EXPECT_EQ(a.count(), 10);
    EXPECT_EQ(a.arraySize(), 12);
    EXPECT_EQ(a.size(), 120u);
    EXPECT_FALSE(a.empty());
    EXPECT_NE(a.data_handle(), nullptr);
    EXPECT_TRUE(array_type<real_type>().empty());
}

TEST_F(ArraysTest, Move)
{
    array_type<real_type> a(3, 4, 2);
    real_type *data = a.data_handle();
    // move construction: b takes the memory, a is left empty
    array_type<real_type> b(std::move(a));
    EXPECT_EQ(b.data_handle(), data);
    EXPECT_EQ(b.count(), 2);
    EXPECT_TRUE(a.empty());
    EXPECT_EQ(a.data_handle(), nullptr);
    // move assignment: c frees its memory and takes that of b
    array_type<real_type> c(5, 5);
    c = std::move(b);
    EXPECT_EQ(c.data_handle(), data);
    EXPECT_EQ(c.height(), 3);
    EXPECT_TRUE(b.empty());
    static_assert(!std::is_copy_constructible_v<array_type<real_type>>);
    static_assert(!std::is_copy_assignable_v<array_type<real_type>>);
}

TEST_F(ArraysTest, RoundTripAndSetZero)
{
    std::vector<real_type> v(3*4*2);
    for (size_t i = 0; i < v.size(); i++) v[i] = i;
    auto a = makeFrom(v, 3, 4, 2);
    EXPECT_EQ(download(a), v);

    a.setZero(stream);
    for (auto x : download(a)) EXPECT_EQ(x, 0);
}

TEST_F(ArraysTest, ComplexRoundTrip)
{
    auto v = randomComplex(5*7);
    auto a = makeFrom(v, 5, 7);
    auto w = download(a);
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
    array_type<real_type> a(3, 4, 10);
    ArraysView<real_type> v = a;
    EXPECT_EQ(v.data_handle(), a.data_handle());
    EXPECT_EQ(v.count(), 10);
    EXPECT_EQ(v.height(), 3);
    EXPECT_EQ(v.width(), 4);
    const auto &c = a;
    ArraysView<const real_type> w = c;
    EXPECT_EQ(w.data_handle(), a.data_handle());
    EXPECT_EQ(w.size(), a.size());
    // read-only arrays give read-only views only
    static_assert(!std::is_convertible_v<const array_type<real_type> &, ArraysView<real_type>>);
}

} // namespace
