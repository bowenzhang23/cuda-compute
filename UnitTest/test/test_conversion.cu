#include "Matrix.cuh"
#include "Vector.cuh"
#include "gtest/gtest.h"

#include <algorithm>
#include <memory>
#include <vector>

TEST(Conversion, M2V_Construct)
{
    constexpr std::size_t row = 201;
    constexpr std::size_t col = 251;
    constexpr std::size_t len = row * col;
    Matrix<float>         mf(row, col);
    Vector<float>         vf(std::move(mf));
    EXPECT_EQ(vf.Nlen(), len);
}

TEST(Conversion, M2V_Into)
{
    constexpr std::size_t row = 201;
    constexpr std::size_t col = 251;
    constexpr std::size_t len = row * col;
    Matrix<float>         mf(row, col);
    Vector<float>         vf = mf.Into();
    EXPECT_EQ(vf.Nlen(), len);
}

TEST(Conversion, M2V_Row)
{
    constexpr std::size_t row = 201;
    constexpr std::size_t col = 251;
    constexpr std::size_t len = row * col;
    std::vector<int>      x(len);
    std::generate(x.begin(), x.end(),
                  [i = 0]() mutable { return (i++) / col; });
    Matrix<int> m(x.data(), row, col);
    Vector<int> v = m.Row(100);
    auto        p = v.ToCPU();
    EXPECT_TRUE(
        std::all_of(p.begin(), p.end(), [](int i) { return i == 100; }));
}

TEST(Conversion, M2V_Col)
{
    constexpr std::size_t row = 201;
    constexpr std::size_t col = 251;
    constexpr std::size_t len = row * col;
    std::vector<int>      x(len);
    std::generate(x.begin(), x.end(),
                  [i = 0]() mutable { return (i++) % col; });
    Matrix<int> m(x.data(), row, col);
    Vector<int> v = m.Col(100);
    auto        p = v.ToCPU();
    EXPECT_TRUE(
        std::all_of(p.begin(), p.end(), [](int i) { return i == 100; }));
}

TEST(Conversion, V2M_Construct)
{
    constexpr std::size_t row = 201;
    constexpr std::size_t col = 251;
    constexpr std::size_t len = row * col;
    Vector<float>         vf(len);
    Matrix<float>         mf(std::move(vf), col);
    EXPECT_EQ(mf.Nrow(), row);
    EXPECT_EQ(mf.Ncol(), col);
}

TEST(Conversion, V2M_Into)
{
    constexpr std::size_t row = 201;
    constexpr std::size_t col = 251;
    constexpr std::size_t len = row * col;
    Vector<float>         vf(len);
    Matrix<float>         mf = vf.Into(col);
    EXPECT_EQ(mf.Nrow(), row);
    EXPECT_EQ(mf.Ncol(), col);
}