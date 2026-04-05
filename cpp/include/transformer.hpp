// cpp/include/transformer.hpp
#pragma once
#include "loader.hpp"
#include <vector>

struct InferenceBuffers
{
    // Pre-allocated activation buffers reused across all layers.
    // Allocated once at model init based on max sequence length.
    std::vector<float> x;         // [T, d_model]
    std::vector<float> x_norm;    // [T, d_model]
    std::vector<float> attn_out;  // [T, d_model]
    std::vector<float> ff_out;    // [T, d_model]
    std::vector<float> qkv_buf;   // [T, 3*d_model]
    std::vector<float> head_buf;  // [T, d_model]
    std::vector<float> score_buf; // [T, T]
    std::vector<float> mask;      // [T, T]
};

struct ForwardResult
{
    std::vector<float> logits; // [T, n_vocab]
    // attn_weights[layer][head * T * T]
    std::vector<std::vector<float>> attn_weights; // [n_layer, n_head*T*T]
};

class GPT2Model
{
public:
    explicit GPT2Model(const std::string &weights_dir);

    // Run forward pass. Returns logits + all attention weights.
    ForwardResult forward(const std::vector<int> &token_ids);

    const GPT2Config &config() const { return weights_.config; }

private:
    GPT2Weights weights_;

    void forward_block(
        int layer_idx,
        int T,
        InferenceBuffers &buf,
        std::vector<float> &layer_attn_weights);
};