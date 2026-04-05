// cpp/include/attention.hpp
#pragma once

namespace attention
{

    // Scaled dot-product attention/
    // Q, K: [T, d_k]  V: [T, d_v]  mask: [T, T] (causal)
    // out:     [T, d_v] (separate buffer)
    // weights: [T, T] - written out for visualization
    void scaled_dot_product(
        const float *Q, const float *K, const float *V,
        const float *mask,
        float *out, float *weights,
        int T, int d_k, int d_v,
        float *score_buf // scratch buffer [T, T]
    );

    // Multi-head self-attention (GPT-2 fused QKV style).
    // x:        [T, d_model]
    // c_attn_w: [d_model, 3*d_model]   c_attn_b: [3*d_model]
    // c_proj_w: [d_model, d_model]     c_proj_b: [d_model]
    // mask:     [T, T]
    // out:      [T, d_model]
    // attn_out: [n_head, T, T]  — per-head attention weights
    //
    // Scratch buffers (pre-allocated by caller):
    //   qkv_buf:   [T, 3*d_model]
    //   head_buf:  [T, d_model]    (concat of head outputs)
    //   score_buf: [T, T]          (reused per head)
    void multi_head(
        const float *x,
        const float *c_attn_w, const float *c_attn_b,
        const float *c_proj_w, const float *c_proj_b,
        const float *mask,
        float *out, float *attn_weights,
        int T, int d_model, int n_head,
        float *qkv_buf, float *head_buf, float *score_buf);

} // namespace attention