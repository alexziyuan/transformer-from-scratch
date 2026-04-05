// cpp/src/attention.cpp
#include "attention.hpp"
#include "ops.hpp"
#include <cmath>
#include <cstring>

namespace attention
{
    void scaled_dot_product(
        const float *Q, const float *K, const float *V,
        const float *mask,
        float *out, float *weights,
        int T, int d_k, int d_v,
        float *score_buf)
    {
        float scale = 1.0 / std::sqrt((float)d_k);

        // scores = Q @ K^T / sqrt(d_k)
        // K^T: [d_k, T] - we treat K as [T, d_k] and compute Q @ K^T
        // by iterating manually since ops::matmul expects row-major non-transposed.
        for (int i = 0; i < T; i++)
        {
            for (int j = 0; j < T; j++)
            {
                float dot = 0.0f;
                for (int k = 0; k < d_k; k++)
                {
                    dot += Q[i * d_k + k] * K[j * d_k + k]; // K transposed: K[j, k]
                }
                score_buf[i * T + j] = dot * scale;
            }
        }

        // Add causal mask
        if (mask != nullptr)
        {
            for (int i = 0; i < T * T; i++)
            {
                score_buf[i] += mask[i];
            }
        }

        // Softmax over scores
        std::memcpy(weights, score_buf, T * T * sizeof(float));
        ops::softmax(weights, T, T);

        // out = weights @ V
        ops::matmul(weights, V, out, T, T, d_v);
    }

    void multi_head(
        const float *x,
        const float *c_attn_w, const float *c_attn_b,
        const float *c_proj_w, const float *c_proj_b,
        const float *mask,
        float *out, float *attn_weights,
        int T, int d_model, int n_head,
        float *qkv_buf,
        float *head_buf, float *score_buf)
    {
        int d_head = d_model / n_head;
        int d_model3 = 3 * d_model;

        // ── Fused QKV projection ──────────────────────────────
        // qkv_buf = x @ c_attn_w + c_attn_b  [T, 3*d_model]
        ops::matmul(x, c_attn_w, qkv_buf, T, d_model, d_model3);
        for (int i = 0; i < T; i++)
        {
            for (int j = 0; j < d_model3; j++)
            {
                qkv_buf[i * d_model3 + j] += c_attn_b[j];
            }
        }

        // Q = qkv_buf[:, 0:d_model]
        // K = qkv_buf[:, d_model:2*d_model]
        // V = qkv_buf[:, 2*d_model:3*d_model]
        const float *Q_full = qkv_buf;
        const float *K_full = qkv_buf + d_model;
        const float *V_full = qkv_buf + 2 * d_model;

        // ── Per-head attention ────────────────────────────────
        // For each head h:
        //   Q_h = Q_full[:, h*d_head : (h+1)*d_head]  — non-contiguous slice
        //   We copy into contiguous scratch buffers for matmul.

        // Allocate small per-head buffers on the stack (d_head <= 64 for GPT-2)
        // For larger models use heap allocation.
        float Q_h[1024 * 64]; // [T, d_head] - max T=1024, d_head=64
        float K_h[1024 * 64];
        float V_h[1024 * 64];
        float out_h[1024 * 64];
        float w_h[1024 * 1024]; // [T, T] attention weights

        for (int h = 0; h < n_head; h++)
        {
            int offset = h * d_head;

            // Extract head slice - copy strided slice into contiguous buffer
            for (int t = 0; t < T; t++)
            {
                for (int d = 0; d < d_head; d++)
                {
                    Q_h[t * d_head + d] = Q_full[t * d_model3 + offset + d];
                    K_h[t * d_head + d] = K_full[t * d_model3 + offset + d];
                    V_h[t * d_head + d] = V_full[t * d_model3 + offset + d];
                }
            }

            // Run attention for this head
            scaled_dot_product(
                Q_h, K_h, V_h, mask,
                out_h, w_h,
                T, d_head, d_head,
                score_buf);

            // Store attention weights: attn_weights[h, :, :] = w_h
            std::memcpy(
                attn_weights + h * T * T,
                w_h,
                T * T * sizeof(float));

            // Write head output into head_buf at the correct column slice
            for (int t = 0; t < T; t++)
            {
                for (int d = 0; d < d_head; d++)
                {
                    head_buf[t * d_model + offset + d] = out_h[t * d_head + d];
                }
            }
        }

        // ── Output projection ────────────────────────────────
        // out = head_buf @ c_proj_w + c_proj_b
        ops::matmul(head_buf, c_proj_w, out, T, d_model, d_model);
        for (int i = 0; i < T; i++)
        {
            for (int j = 0; j < d_model; j++)
            {
                out[i * d_model + j] += c_proj_b[j];
            }
        }
    }
} // namespace attention