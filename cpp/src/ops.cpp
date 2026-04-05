// cpp/src/ops.cpp
#include "ops.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ops
{
    // ─────────────────────────────────────────────
    // Matrix multiply
    // ─────────────────────────────────────────────

    void matmul(
        const float *A, const float *B, float *C,
        int M, int K, int N)
    {
        // Naive triple loop -
        // extension of this phase: replace inner loop with BLAS sgemm call.
        // The OpenMP pragma parallelizes the outer loop across CPU cores.

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < M; i++)
        {
            for (
                int j = 0;
                j < N; j++)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; k++)
                {
                    // A[i, k] = A[i*K + k]
                    // B[k, j] = B[k*N + j]
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    // ─────────────────────────────────────────────
    // Softmax (in-place, row-wise)
    // ─────────────────────────────────────────────

    void softmax(float *x, int M, int N)
    {
        // For each row: subtract max, exp, normalize.
        // Identical logic to Python ops.py softmax().
        for (int i = 0; i < M; i++)
        {
            float *row = x + i * N;

            // Find row max for numerical stability
            float row_max = row[0];
            for (int j = 1; j < N; j++)
            {
                if (row[j] > row_max)
                    row_max = row[j];
            }

            // Subtract max and exponentiate
            float sum = 0.0f;
            for (int j = 0; j < N; j++)
            {
                row[j] = std::exp(row[j] - row_max);
                sum += row[j];
            }

            // Normalize
            for (int j = 0; j < N; j++)
            {
                row[j] /= sum;
            }
        }
    }

    // ─────────────────────────────────────────────
    //  GELU (tanh approximation, in-place)
    // ─────────────────────────────────────────────

    void gelu(float *x, int n_elem)
    {
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // Must match Python ops.py gelu() exactly - GPT-2 weights depend on this form.
        constexpr float SQRT_2_OVER_PI = 0.7978845608f; // sqrt(2/pi)
        constexpr float COEFF = 0.044715f;

        for (int i = 0; i < n_elem; i++)
        {
            float v = x[i];
            float inner = SQRT_2_OVER_PI * (v + COEFF * v * v * v);
            x[i] = 0.5f * v * (1.0f + std::tanh(inner));
        }
    }
    // ─────────────────────────────────────────────
    //  Layer Normalization
    // ─────────────────────────────────────────────

    void layer_norm(
        const float *x, float *out,
        const float *gamma, const float *beta,
        int T, int D, float eps)
    {
        // Normalize each token (row) independently over the D dimension.
        // Matches Python ops.py layer_norm() exactly.
        for (int t = 0; t < T; t++)
        {
            const float *row = x + t * D;
            float *dst = out + t * D;

            // Compute mean
            float mean = 0.0f;
            for (int d = 0; d < D; d++)
                mean += row[d];
            mean /= D;

            // Compute variance
            float var = 0.0f;
            for (int d = 0; d < D; d++)
            {
                float diff = row[d] - mean;
                var += diff * diff;
            }
            var /= D;

            // Normalize and apply affine parameters
            float inv_std = 1.0f / std::sqrt(var + eps);
            for (int d = 0; d < D; d++)
            {
                dst[d] = gamma[d] * ((row[d] - mean) * inv_std) + beta[d];
            }
        }
    }

    // ─────────────────────────────────────────────
    //  Causal Mask
    // ─────────────────────────────────────────────

    void causal_mask(float *mask, int T)
    {
        constexpr float NEG_INF = -1e10f; // large negative, not true -inf
                                          // avoids NaN in exp(-inf) on some hardware.
        for (int i = 0; i < T; i++)
        {
            for (int j = 0; j < T; j++)
            {
                // Upper triangle (j > i): future tokens, mask out
                mask[i * T + j] = (j > i) ? NEG_INF : 0.0f;
            }
        }
    }

} // namespace ops
