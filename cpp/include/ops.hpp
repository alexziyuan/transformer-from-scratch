// cpp include/ops.hpp
#pragma once
#include <vector>
#include <cstddef>

// ─────────────────────────────────────────────
//
// ─────────────────────────────────────────────

namespace ops
{

    // Matrix multiply: C = A @ B
    // A: [M, K], B: [K, N], C: [M, N]
    void matmul(
        const float *A, const float *B, float *C,
        int M, int K, int N);

    // Softmax in-place over last axis
    // x : [M, N] - modified in place
    void softmax(float *x, int M, int N);

    // GELU activation in-place (tanh approximation)
    // x: any shape, n_elem total elements
    void gelu(float *x, int n_elem);

    // Layer normalization
    // x:   [T, D] input
    // out: [T, D] output (separate buffer)
    // gamma, beta: [D]
    void layer_norm(
        const float *x, float *out,
        const float *gamma, const float *beta,
        int T, int D, float eps = 1e-5f);

    // Build causal mask into pre-allocated buffer
    // mask: [T, T] - filled with 0.0 on/below diagonal, -inf above
    void causal_mask(float *mask, int T);

} // namespace ops