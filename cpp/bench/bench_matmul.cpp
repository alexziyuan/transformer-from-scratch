// cpp/bench/bench_matmul.cpp
#include "ops.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>

using Clock = std::chrono::high_resolution_clock;

double benchmark_matmul(int M, int K, int N, int iterations = 10)
{
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N);

    // Warmup
    ops::matmul(A.data(), B.data(), C.data(), M, K, N);

    auto start = Clock::now();
    for (int i = 0; i < iterations; i++)
    {
        ops::matmul(A.data(), B.data(), C.data(), M, K, N);
    }
    auto end = Clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    return ms / iterations;
}

int main()
{
    std::cout << "Matmul benchmark (GPT-2 small relevant sizes)\n";
    std::cout << std::string(55, '-') << "\n";
    std::cout << "Shape                   Avg ms    GFLOPS\n";
    std::cout << std::string(55, '-') << "\n";

    // Representative GPT-2 small matmul sizes
    struct Shape
    {
        int M, K, N;
        const char *label;
    };
    std::vector<Shape> shapes = {
        {1, 768, 2304, "QKV proj (T=1)"},
        {8, 768, 2304, "QKV proj (T=8)"},
        {64, 768, 2304, "QKV proj (T=64)"},
        {64, 768, 768, "out proj (T=64)"},
        {64, 768, 3072, "FFN expand (T=64)"},
        {64, 3072, 768, "FFN contract (T=64)"},
    };

    for (auto &s : shapes)
    {
        double ms = benchmark_matmul(s.M, s.K, s.N);
        double flops = 2.0 * s.M * s.K * s.N; // multiply-add = 2 ops
        double gflops = (flops / 1e9) / (ms / 1e3);

        std::printf("%-24s  %6.2f ms  %6.2f\n", s.label, ms, gflops);
    }

    std::cout << std::string(55, '-') << "\n";
    std::cout << "Build with -DUSE_OPENMP to see parallel speedup.\n";
    return 0;
}