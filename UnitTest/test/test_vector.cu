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

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}