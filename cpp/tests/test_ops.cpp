// cpp/tests/test_ops.cpp
#include "ops.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <numeric>

static bool close(float a, float b, float atol = 1e-4f)
{
    return std::abs(a - b) <= atol;
}

static void assert_close_vec(
    const std::vector<float> &got,
    const std::vector<float> &ref,
    const std::string &name,
    float atol = 1e-4f)
{
    assert(got.size() == ref.size());
    float max_diff = 0.0f;
    for (size_t i = 0; i < got.size(); i++)
    {
        max_diff = std::max(max_diff, std::abs(got[i] - ref[i]));
    }
    if (max_diff > atol)
    {
        std::cerr << "FAIL [" << name << "] max_diff=" << max_diff << "\n";
        std::exit(1);
    }
    std::cout << "  PASS [" << name << "] max_diff=" << max_diff << "\n";
}

void test_softmax()
{
    std::cout << "softmax\n";

    // Row 1: [1, 2, 3] Row 2: [1, 1, 1]
    std::vector<float> x = {1.0f, 2.0f, 3.0f,
                            1.0f, 1.0f, 1.0f};
    ops::softmax(x.data(), 2, 3);

    // Row 1 reference (computed from NumPy)
    // softmax([1,2,3]) = [0.0900, 0.2447, 0.6652]
    assert(close(x[0], 0.0900f, 1e-3f));
    assert(close(x[1], 0.2447f, 1e-3f));
    assert(close(x[2], 0.6652f, 1e-3f));
    std::cout << "  PASS [known values]\n";

    // Row 2: uniform input -> uniform output
    assert(close(x[3], 1.0f / 3.0f, 1e-5f));
    assert(close(x[4], 1.0f / 3.0f, 1e-5f));
    assert(close(x[5], 1.0f / 3.0f, 1e-5f));
    std::cout << "  PASS [uniform input]\n";

    // Rows sum to 1
    float sum1 = x[0] + x[1] + x[2];
    float sum2 = x[3] + x[4] + x[5];
    assert(close(sum1, 1.0f, 1e-5f));
    assert(close(sum2, 1.0f, 1e-5f));
    std::cout << "  PASS [rows sum to 1]\n";
}

void test_gelu()
{
    std::cout << "gelu\n";

    std::vector<float> x = {0.0f, 1.0f, -1.0f, 2.0f};
    ops::gelu(x.data(), 4);

    // gelu(0) = 0
    assert(close(x[0], 0.0f, 1e-6f));
    std::cout << "  PASS [gelu(0)]\n";

    // gelu(1) ≈ 0.8413
    assert(close(x[1], 0.8413f, 1e-3f));
    std::cout << "  PASS [gelu(1)≈0.8413]\n";

    // gelu(-1) ≈ -0.1587
    assert(close(x[2], -0.1587f, 1e-3f));
    std::cout << "  PASS [gelu(-1)≈-0.1587]\n";
}

void test_layer_norm()
{
    std::cout << "layer_norm\n";
    int T = 2, D = 4;

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f,
                            2.0f, 2.0f, 2.0f, 2.0f};
    std::vector<float> gamma(D, 1.0f);
    std::vector<float> beta(D, 0.0f);
    std::vector<float> out(T * D);

    ops::layer_norm(x.data(), out.data(), gamma.data(), beta.data(), T, D);

    // Row 2 is constant — after layernorm should be all zeros
    assert(close(out[4], 0.0f, 1e-5f));
    assert(close(out[5], 0.0f, 1e-5f));
    assert(close(out[6], 0.0f, 1e-5f));
    assert(close(out[7], 0.0f, 1e-5f));
    std::cout << "  PASS [constant row -> zeros]\n";

    // Row 1 should have mean≈0, std≈1
    float mean = (out[0] + out[1] + out[2] + out[3]) / 4.0f;
    assert(close(mean, 0.0f, 1e-5f));
    std::cout << "  PASS [mean≈0]\n";
}

void test_causal_mask()
{
    std::cout << "causal_mask\n";
    int T = 4;
    std::vector<float> mask(T * T);
    ops::causal_mask(mask.data(), T);

    // Diagonal and below: 0.0
    for (int i = 0; i < T; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            assert(close(mask[i * T + j], 0.0f, 1e-6f));
        }
    }
    std::cout << "  PASS [lower triangle = 0]\n";

    // Above diagonal: large negative
    for (int i = 0; i < T; i++)
    {
        for (int j = i + 1; j < T; j++)
        {
            assert(mask[i * T + j] < -1e9f);
        }
    }
    std::cout << "  PASS [upper triangle = -inf]\n";
}

void test_matmul()
{
    std::cout << "matmul\n";

    // 2x3 @ 3x2 = 2x2
    std::vector<float> A = {1, 2, 3, 4, 5, 6};
    std::vector<float> B = {7, 8, 9, 10, 11, 12};
    std::vector<float> C(4);

    ops::matmul(A.data(), B.data(), C.data(), 2, 3, 2);

    // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    assert(close(C[0], 58.0f));
    assert(close(C[1], 64.0f));
    assert(close(C[2], 139.0f));
    assert(close(C[3], 154.0f));
    std::cout << "  PASS [2x3 @ 3x2 = 2x2]\n";
}

int main()
{
    std::cout << std::string(50, '=') << "\n";
    std::cout << "ops.cpp unit tests\n";
    std::cout << std::string(50, '=') << "\n";

    test_matmul();
    test_softmax();
    test_gelu();
    test_layer_norm();
    test_causal_mask();

    std::cout << std::string(50, '=') << "\n";
    std::cout << "All tests passed.\n";
    return 0;
};