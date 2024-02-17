#include "Matrix.cuh"
#include "gtest/gtest.h"

#include <algorithm>
#include <memory>
#include <vector>

TEST(Matrix, Construct)
{
    constexpr std::size_t row = 201;
    constexpr std::size_t col = 251;
    constexpr std::size_t len = row * col;
    Matrix<float>         mf(row, col);
    EXPECT_EQ(mf.Shape().at(0), row);
    EXPECT_EQ(mf.Shape().at(1), col);

    std::vector<int> x(len);
    std::generate(x.begin(), x.end(), []() { return rand() % 10; });
    Matrix<int> mi(x.data(), row, col);
    mi      = mi;
    auto px = mi.ToCPU();
    EXPECT_EQ(mi.Nrow(), row);
    EXPECT_EQ(mi.Ncol(), col);
    EXPECT_EQ(px, x);
}

TEST(Matrix, Linear)
{
    constexpr std::size_t row = 201;
    constexpr std::size_t col = 251;
    constexpr std::size_t len = row * col;
    std::vector<int>      x(len);
    std::vector<int>      y(len);
    std::generate(x.begin(), x.end(), [i = 0]() mutable { return +(i++); });
    std::generate(y.begin(), y.end(), [i = 0]() mutable { return -(i++); });
    Matrix<int> mx(x.data(), row, col);
    Matrix<int> my(y.data(), row, col);
    Matrix<int> mz = Linear(3, mx, 2, my, 1);
    auto        pz = mz.ToCPU();
    EXPECT_TRUE(std::all_of(pz.begin(), pz.end(),
                            [i = 0](int j) mutable { return j == 1 + (i++); }));
}
TEST(Matrix, Power)
{
    constexpr std::size_t row = 201;
    constexpr std::size_t col = 251;
    constexpr std::size_t len = row * col;
    std::vector<int>      x(len);
    std::vector<int>      y(len);
    std::generate(x.begin(), x.end(), []() { return -2; });
    std::generate(y.begin(), y.end(), []() { return -1; });
    Matrix<int> mx(x.data(), row, col);
    Matrix<int> my(y.data(), row, col);
    Matrix<int> mz = Power(3, mx, 2, my, 1);
    auto        pz = mz.ToCPU();
    EXPECT_TRUE(
        std::all_of(pz.begin(), pz.end(), [](int j) { return j == -12; }));
}

TEST(Vector, Binary)
{
    constexpr std::size_t row = 201;
    constexpr std::size_t col = 251;
    constexpr std::size_t len = row * col;
    std::vector<int>      x(len);
    std::vector<int>      y(len);
    std::generate(x.begin(), x.end(), [i = 0]() mutable { return 1 - (i++) % 2; });
    std::generate(y.begin(), y.end(), [i = 0]() mutable { return (i++) % 2; });
    Matrix<int> mx(x.data(), row, col);
    Matrix<int> my(y.data(), row, col);
    Matrix<int> mz = Binary<int, int>(mx, my, BinaryOp::GT);
    auto        pz = mz.ToCPU();
    EXPECT_EQ(pz, x);
}

TEST(Matrix, Scale)
{
    constexpr std::size_t row = 201;
    constexpr std::size_t col = 251;
    constexpr std::size_t len = row * col;
    std::vector<int>      x(len);
    std::generate(x.begin(), x.end(), [i = 0]() mutable { return i++; });
    Matrix<int> mx(x.data(), row, col);
    Matrix<int> mz = -2 * mx * -1;
    auto        pz = mz.ToCPU();
    EXPECT_TRUE(std::all_of(pz.begin(), pz.end(),
                            [i = 0](int j) mutable { return j == 2 * (i++); }));
}

TEST(Matrix, ScaleSelf)
{
    constexpr std::size_t row = 200;
    constexpr std::size_t col = 250;
    constexpr std::size_t len = row * col;
    std::vector<int>      x(len);
    std::generate(x.begin(), x.end(), [i = 0]() mutable { return i++; });
    Matrix<int> mx(x.data(), row, col);
    mx      = -2 * mx * -1;
    auto px = mx.ToCPU();
    EXPECT_TRUE(std::all_of(px.begin(), px.end(),
                            [i = 0](int j) mutable { return j == 2 * (i++); }));
}

TEST(Matrix, Transpose)
{
    constexpr std::size_t row = 201;
    constexpr std::size_t col = 251;
    constexpr std::size_t len = row * col;
    std::vector<int>      x(len);
    std::generate(x.begin(), x.end(), [i = 0]() mutable { return i++; });
    Matrix<int> mx(x.data(), row, col);
    Matrix<int> mx_t = mx.Transpose();
    auto        px   = mx.ToCPU();
    EXPECT_EQ(mx.Nrow(), mx_t.Ncol());
    EXPECT_EQ(mx.Ncol(), mx_t.Nrow());
    bool all_equal = true;
    for (std::size_t i = 0; i < col; ++i) {
        for (std::size_t j = 0; j < row; ++j) {
            all_equal |= (px[i * row + j] == x[j * col + i]);
        }
    }
    EXPECT_TRUE(all_equal);
}

TEST(Matrix, TransposeSelf)
{
    constexpr std::size_t row = 201;
    constexpr std::size_t col = 251;
    constexpr std::size_t len = row * col;
    std::vector<int>      x(len);
    std::generate(x.begin(), x.end(), [i = 0]() mutable { return i++; });
    Matrix<int> mx(x.data(), row, col);
    mx      = mx.Transpose();
    auto px = mx.ToCPU();
    EXPECT_EQ(row, mx.Ncol());
    EXPECT_EQ(col, mx.Nrow());
    bool all_equal = true;
    for (std::size_t i = 0; i < col; ++i) {
        for (std::size_t j = 0; j < row; ++j) {
            all_equal |= (px[i * row + j] == x[j * col + i]);
        }
    }
    EXPECT_TRUE(all_equal);
}

TEST(Matrix, Reshape)
{
    constexpr std::size_t row = 10;
    constexpr std::size_t col = 10;
    constexpr std::size_t len = row * col;
    std::vector<int>      x(len);
    std::generate(x.begin(), x.end(), [i = 0]() mutable { return i++; });
    Matrix<int> mx(x.data(), row, col);
    mx.Reshape_(1);
    EXPECT_EQ(mx.Nrow(), 1);
    EXPECT_EQ(mx.Ncol(), 100);
    EXPECT_EQ(mx.ToCPU(), x);
}

TEST(Matrix, MatMulSmall)
{
    constexpr std::size_t m = 641;
    constexpr std::size_t k = 301;
    constexpr std::size_t n = 541;
    std::vector<int>      a(m * k);
    std::vector<int>      b(k * n);
    std::vector<int>      c(m * n);
    std::generate(a.begin(), a.end(), [i = +5]() mutable { return (i++) % 7; });
    std::generate(b.begin(), b.end(), [i = -5]() mutable { return (i++) % 9; });
    Matrix<int> ma(a.data(), m, k);
    Matrix<int> mb(b.data(), k, n);
    Matrix<int> mc = MatMulSmall(ma, mb);
    EXPECT_EQ(mc.Nrow(), ma.Nrow());
    EXPECT_EQ(mc.Ncol(), mb.Ncol());
    // TODO cpu matmul
    for (std::size_t mi = 0; mi < m; ++mi) {
        for (std::size_t ni = 0; ni < n; ++ni) {
            c[mi * n + ni] = 0;
            for (std::size_t ki = 0; ki < k; ++ki) {
                c[mi * n + ni] += a[mi * k + ki] * b[ki * n + ni];
            }
        }
    }
    auto pc = mc.ToCPU();
    EXPECT_EQ(pc.size(), m * n);
    EXPECT_EQ(pc, c);
}

TEST(Matrix, MatMulLarge)
{
    constexpr std::size_t m = 641;
    constexpr std::size_t k = 301;
    constexpr std::size_t n = 541;
    std::vector<int>      a(m * k);
    std::vector<int>      b(k * n);
    std::vector<int>      c(m * n);
    std::generate(a.begin(), a.end(), [i = +5]() mutable { return (i++) % 7; });
    std::generate(b.begin(), b.end(), [i = -5]() mutable { return (i++) % 9; });
    Matrix<int> ma(a.data(), m, k);
    Matrix<int> mb(b.data(), k, n);
    Matrix<int> mc = MatMulLarge(ma, mb);
    EXPECT_EQ(mc.Nrow(), ma.Nrow());
    EXPECT_EQ(mc.Ncol(), mb.Ncol());
    // TODO cpu matmul
    for (std::size_t mi = 0; mi < m; ++mi) {
        for (std::size_t ni = 0; ni < n; ++ni) {
            c[mi * n + ni] = 0;
            for (std::size_t ki = 0; ki < k; ++ki) {
                c[mi * n + ni] += a[mi * k + ki] * b[ki * n + ni];
            }
        }
    }
    auto pc = mc.ToCPU();
    EXPECT_EQ(pc.size(), m * n);
    EXPECT_EQ(pc, c);
}

TEST(Matrix, MatMul1)
{
    constexpr std::size_t m = 65;
    constexpr std::size_t k = 33;
    constexpr std::size_t n = 121;
    std::vector<int>      a(m * k);
    std::vector<int>      b(k * n);
    std::vector<int>      c(m * n);
    std::generate(a.begin(), a.end(), [i = +5]() mutable { return (i++) % 7; });
    std::generate(b.begin(), b.end(), [i = -5]() mutable { return (i++) % 9; });
    Matrix<int> ma(a.data(), m, k);
    Matrix<int> mb(b.data(), k, n);
    Matrix<int> mc = MatMul(ma, mb);
    EXPECT_EQ(mc.Nrow(), ma.Nrow());
    EXPECT_EQ(mc.Ncol(), mb.Ncol());
    // TODO cpu matmul
    for (std::size_t mi = 0; mi < m; ++mi) {
        for (std::size_t ni = 0; ni < n; ++ni) {
            c[mi * n + ni] = 0;
            for (std::size_t ki = 0; ki < k; ++ki) {
                c[mi * n + ni] += a[mi * k + ki] * b[ki * n + ni];
            }
        }
    }
    auto pc = mc.ToCPU();
    EXPECT_EQ(pc.size(), m * n);
    EXPECT_EQ(pc, c);
}

TEST(Matrix, MatMul2)
{
    constexpr std::size_t m = 677;
    constexpr std::size_t k = 299;
    constexpr std::size_t n = 987;
    std::vector<int>      a(m * k);
    std::vector<int>      b(k * n);
    std::vector<int>      c(m * n);
    std::generate(a.begin(), a.end(), [i = +5]() mutable { return (i++) % 7; });
    std::generate(b.begin(), b.end(), [i = -5]() mutable { return (i++) % 9; });
    Matrix<int> ma(a.data(), m, k);
    Matrix<int> mb(b.data(), k, n);
    Matrix<int> mc = MatMul(ma, mb);
    EXPECT_EQ(mc.Nrow(), ma.Nrow());
    EXPECT_EQ(mc.Ncol(), mb.Ncol());
    // TODO cpu matmul
    for (std::size_t mi = 0; mi < m; ++mi) {
        for (std::size_t ni = 0; ni < n; ++ni) {
            c[mi * n + ni] = 0;
            for (std::size_t ki = 0; ki < k; ++ki) {
                c[mi * n + ni] += a[mi * k + ki] * b[ki * n + ni];
            }
        }
    }
    auto pc = mc.ToCPU();
    EXPECT_EQ(pc.size(), m * n);
    EXPECT_EQ(pc, c);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}