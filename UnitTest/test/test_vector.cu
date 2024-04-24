#include "Stream.cuh"
#include "Vector.cuh"
#include "gtest/gtest.h"

#include <algorithm>
#include <memory>
#include <vector>

TEST(Vector, Construct)
{
    constexpr std::size_t len = 4001;
    Vector<float>         vf(len);
    EXPECT_EQ(vf.Shape().at(0), len);

    std::vector<int> x(len);
    std::generate(x.begin(), x.end(), []() { return rand() % 10; });
    Vector<int> vi(x.data(), x.size());
    vi      = vi;
    auto px = vi.ToCPU();
    EXPECT_EQ(vi.Nlen(), len);
    EXPECT_EQ(px, x);
}

TEST(Vector, ConstructSTLVector)
{
    constexpr std::size_t len = 4001;
    std::vector<float>    x(len);
    std::generate(x.begin(), x.end(), [i = 0]() mutable { return i++; });
    Vector<float> vf(x, len);
    auto          pf = vf.ToCPU();
    EXPECT_EQ(pf, x);
}

TEST(Vector, CopyConstruct)
{
    constexpr std::size_t len = 4001;
    std::vector<float>    x(len);
    std::generate(x.begin(), x.end(), [i = 0]() mutable { return i++; });
    Vector<float> vf(x, len);
    Vector<float> vf_copy(vf);
    EXPECT_EQ(vf_copy.ToCPU(), vf.ToCPU());
    EXPECT_EQ(vf_copy.Shape().at(0), vf.Shape().at(0));
}

TEST(Vector, MoveConstruct)
{
    constexpr std::size_t len = 4001;
    std::vector<float>    x(len);
    std::generate(x.begin(), x.end(), [i = 0]() mutable { return i++; });
    Vector<float> vf(x, len);
    Vector<float> vf_copy(std::move(vf));
    EXPECT_EQ(vf_copy.ToCPU(), x);
    EXPECT_EQ(vf_copy.Shape().at(0), len);
}

TEST(Vector, SumAndMean)
{
    constexpr std::size_t len = 4001;
    std::vector<int>      x(len);
    std::generate(x.begin(), x.end(), []() { return 42; });
    Vector<int> vx(x, len);
    EXPECT_EQ(vx.Sum(), 4001 * 42);
    EXPECT_EQ(vx.Mean(), 42);
}

TEST(Vector, MaxAndMin)
{
    constexpr std::size_t len = 4001;
    std::vector<int>      x(len);
    std::generate(x.begin(), x.end(), []() { return 0; });
    x[1001] = 1001;
    x[2002] = -2002;
    Vector<int>     vx(x, len);
    ValueIndex<int> max_vi = vx.Max();
    ValueIndex<int> min_vi = vx.Min();
    EXPECT_EQ(max_vi.val, 1001);
    EXPECT_EQ(max_vi.idx, 1001);
    EXPECT_EQ(min_vi.val, -2002);
    EXPECT_EQ(min_vi.idx, 2002);
}

TEST(Vector, Reversed)
{
    constexpr std::size_t len = 255 + 4096;
    std::vector<int>      x(len);
    std::generate(x.begin(), x.end(), [i = 0]() mutable { return i++; });
    Vector<int> vx(x, len);
    std::reverse(x.begin(), x.end());
    EXPECT_EQ(vx.Reversed().ToCPU(), x);
}

TEST(Vector, SortPowerOfTwo)
{
    constexpr std::size_t len = 4096;
    std::vector<int>      x(len);
    std::generate(x.begin(), x.end(), [i = len]() mutable { return i--; });
    Vector<int> vx(x, len);
    vx.Sort_();
    std::sort(x.begin(), x.end());
    EXPECT_EQ(vx.ToCPU(), x);
}

TEST(Vector, SortAscending)
{
    constexpr std::size_t len = 255 + 4096;
    std::vector<int>      x(len);
    std::generate(x.begin(), x.end(), [i = len]() mutable { return i--; });
    Vector<int> vx(x, len);
    vx.Sort_();
    std::sort(x.begin(), x.end());
    EXPECT_EQ(vx.ToCPU(), x);
}

TEST(Vector, SortDescending)
{
    constexpr std::size_t len = 255 + 4096;
    std::vector<int>      x(len);
    std::generate(x.begin(), x.end(), [i = 0]() mutable { return i++; });
    Vector<int> vx(x, len);
    vx.Sort_(false);
    std::sort(x.begin(), x.end(), std::greater<int>());
    EXPECT_EQ(vx.ToCPU(), x);
}

TEST(Vector, Linear)
{
    constexpr std::size_t len = 4001;
    std::vector<int>      x(len);
    std::vector<int>      y(len);
    std::generate(x.begin(), x.end(), [i = 0]() mutable { return +(i++); });
    std::generate(y.begin(), y.end(), [i = 0]() mutable { return -(i++); });
    Vector<int> vx(x.data(), len);
    Vector<int> vy(y.data(), len);
    Vector<int> vz = Linear(3, vx, 2, vy, 1);
    auto        pz = vz.ToCPU();
    EXPECT_TRUE(std::all_of(pz.begin(), pz.end(),
                            [i = 0](int j) mutable { return j == 1 + (i++); }));
}

TEST(Vector, LinearStream)
{
    auto stream = std::unique_ptr<Stream>(Stream::CreateNonBlocking());
    constexpr std::size_t len = 4001;
    std::vector<int>      x(len);
    std::vector<int>      y(len);
    std::generate(x.begin(), x.end(), [i = 0]() mutable { return +(i++); });
    std::generate(y.begin(), y.end(), [i = 0]() mutable { return -(i++); });
    Vector<int> vx(x.data(), len, stream->Cuda());
    Vector<int> vy(y.data(), len, stream->Cuda());
    Vector<int> vz = Linear(3, vx, 2, vy, 1);
    auto        pz = vz.ToCPU();
    EXPECT_EQ(vx.S(), vz.S());
    EXPECT_TRUE(std::all_of(pz.begin(), pz.end(),
                            [i = 0](int j) mutable { return j == 1 + (i++); }));
}

TEST(Vector, Power)
{
    constexpr std::size_t len = 4001;
    std::vector<int>      x(len);
    std::vector<int>      y(len);
    std::generate(x.begin(), x.end(), []() { return -2; });
    std::generate(y.begin(), y.end(), []() { return -1; });
    Vector<int> vx(x.data(), len);
    Vector<int> vy(y.data(), len);
    Vector<int> vz = Power(3, vx, 2, vy, 1);
    auto        pz = vz.ToCPU();
    EXPECT_TRUE(
        std::all_of(pz.begin(), pz.end(), [](int j) { return j == -12; }));
}

TEST(Vector, Binary)
{
    constexpr std::size_t len = 4001;
    std::vector<int>      x(len);
    std::vector<int>      y(len);
    std::generate(x.begin(), x.end(),
                  [i = 0]() mutable { return 1 - (i++) % 2; });
    std::generate(y.begin(), y.end(), [i = 0]() mutable { return (i++) % 2; });
    Vector<int> vx(x.data(), len);
    Vector<int> vy(y.data(), len);
    Vector<int> vz = Binary<int, int>(vx, vy, BinaryOp::GT);
    auto        pz = vz.ToCPU();
    EXPECT_EQ(pz, x);
}

TEST(Vector, Scale)
{
    constexpr std::size_t len = 4001;
    std::vector<int>      x(len);
    std::generate(x.begin(), x.end(), [i = 0]() mutable { return i++; });
    Vector<int> vx(x.data(), len);
    Vector<int> vz = -2 * vx * -1;
    auto        pz = vz.ToCPU();
    EXPECT_TRUE(std::all_of(pz.begin(), pz.end(),
                            [i = 0](int j) mutable { return j == 2 * (i++); }));
}

TEST(Vector, ScaleSelf)
{
    constexpr std::size_t len = 4001;
    std::vector<int>      x(len);
    std::generate(x.begin(), x.end(), [i = 0]() mutable { return i++; });
    Vector<int> vx(x.data(), len);
    vx      = -2 * vx * -1;
    auto px = vx.ToCPU();
    EXPECT_TRUE(std::all_of(px.begin(), px.end(),
                            [i = 0](int j) mutable { return j == 2 * (i++); }));
}

TEST(Vector, Inner)
{
    constexpr std::size_t len = 255 + 4096;
    std::vector<int>      x(len);
    std::vector<int>      y(len);
    std::generate(x.begin(), x.end(), []() { return 1; });
    std::generate(y.begin(), y.end(), []() { return 1; });
    Vector<int> vx(x.data(), len);
    Vector<int> vy(y.data(), len);
    int         result = Inner(vx, vy);
    EXPECT_EQ(result, len);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}