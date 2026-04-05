// cpp/src/transformer.cpp
#include "transformer.hpp"
#include "ops.hpp"
#include "attention.hpp"
#include <cstring>
#include <stdexcept>
#include <numeric>

GPT2Model::GPT2Model(const std::string &weights_dir)
{
    weights_ = load_weights(weights_dir);
}

void GPT2Model::forward_block(
    int layer_idx,
    int T,
    InferenceBuffers &buf,
    std::vector<float> &layer_attn_weights)
{
    const auto &cfg = weights_.config;
    const auto &l = weights_.layers[layer_idx];
    int D = cfg.d_model;

    // ── Pre-norm attention ────────────────────────────────
    ops::layer_norm(
        buf.x.data(), buf.x_norm.data(),
        l.ln1_w.data(), l.ln1_b.data(),
        T, D);

    layer_attn_weights.resize(cfg.n_head * T * T);

    attention::multi_head(
        buf.x_norm.data(),
        l.c_attn_w.data(), l.c_attn_b.data(),
        l.c_proj_w.data(), l.c_proj_b.data(),
        buf.mask.data(),
        buf.attn_out.data(),
        layer_attn_weights.data(),
        T, D, cfg.n_head,
        buf.qkv_buf.data(),
        buf.head_buf.data(),
        buf.score_buf.data());

    // Residual connection: x = x + attn_out
    for (int i = 0; i < T * D; i++)
    {
        buf.x[i] += buf.attn_out[i];
    }

    // ── Pre-norm FFN ──────────────────────────────────────
    ops::layer_norm(
        buf.x.data(), buf.x_norm.data(),
        l.ln2_w.data(), l.ln2_b.data(),
        T, D);

    // FFN: expand -> gelu -> contract
    int D4 = 4 * D;
    std::vector<float> h(T * D4);

    // h = x_norm @ fc_w + fc_b
    ops::matmul(buf.x_norm.data(), l.fc_w.data(), h.data(), T, D, D4);
    for (int i = 0; i < T; i++)
    {
        for (int j = 0; j < D4; j++)
        {
            h[i * D4 + j] += l.fc_b[j];
        }
    }
    ops::gelu(h.data(), T * D4);

    // ff_out = h @ proj_w + proj_b
    ops::matmul(h.data(), l.proj_w.data(), buf.ff_out.data(), T, D4, D);
    for (int i = 0; i < T; i++)
    {
        for (int j = 0; j < D; j++)
        {
            buf.ff_out[i * D + j] += l.proj_b[j];
        }
    }

    // Residual connection: x= x + ff_out
    for (int i = 0; i < T * D; i++)
    {
        buf.x[i] += buf.ff_out[i];
    }
}

ForwardResult GPT2Model::forward(const std::vector<int> &token_ids)
{
    const auto &cfg = weights_.config;
    int T = (int)token_ids.size();
    int D = cfg.d_model;

    if (T > cfg.n_ctx)
    {
        throw std::runtime_error("Sequence length exceeds model context size");
    }

    // ── Allocate buffers ──────────────────────────────────
    InferenceBuffers buf;
    buf.x.resize(T * D);
    buf.x_norm.resize(T * D);
    buf.attn_out.resize(T * D);
    buf.ff_out.resize(T * D);
    buf.qkv_buf.resize(T * 3 * D);
    buf.head_buf.resize(T * D);
    buf.score_buf.resize(T * T);
    buf.mask.resize(T * T);

    // Token + positional embeddings.
    ops::causal_mask(buf.mask.data(), T);

    for (int t = 0; t < T; t++)
    {
        int tok = token_ids[t];
        for (int d = 0; d < D; d++)
        {
            buf.x[t * D + d] =
                weights_.wte[tok * D + d] +
                weights_.wpe[t * D + d];
        }
    }

    // ── Transformer blocks ────────────────────────────────
    ForwardResult result;
    result.attn_weights.resize(cfg.n_layer);

    for (int i = 0; i < cfg.n_layer; i++)
    {
        forward_block(i, T, buf, result.attn_weights[i]);
    }

    // ── Final layer norm ──────────────────────────────────
    std::vector<float> x_final(T * D);
    ops::layer_norm(
        buf.x.data(), x_final.data(),
        weights_.ln_f_w.data(), weights_.ln_f_b.data(),
        T, D);

    // ── LM head (weight-tied with wte) ────────────────────
    // logits = x_final @ wte^T
    result.logits.resize(T * cfg.n_vocab);
    for (int t = 0; t < T; t++)
    {
        for (int v = 0; v < cfg.n_vocab; v++)
        {
            float dot = 0.0f;
            for (int d = 0; d < D; d++)
            {
                dot += x_final[t * D + d] * weights_.wte[v * D + d];
            }
            result.logits[t * cfg.n_vocab + v] = dot;
        }
    }

    return result;
}