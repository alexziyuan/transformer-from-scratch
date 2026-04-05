// cpp/include/loader.hpp
#pragma once
#include <string>
#include <vector>
#include <unordered_map>

struct GPT2Config
{
    int n_vocab;
    int d_model;
    int n_ctx;
    int n_layer;
    int n_head;
};

struct LayerWeights
{
    std::vector<float> ln1_w, ln1_b;
    std::vector<float> c_attn_w, c_attn_b;
    std::vector<float> c_proj_w, c_proj_b;
    std::vector<float> ln2_w, ln2_b;
    std::vector<float> fc_w, fc_b;
    std::vector<float> proj_w, proj_b;
};

struct GPT2Weights
{
    GPT2Config config;
    std::vector<float> wte; // [n_vocab, d_model]
    std::vector<float> wpe; // [n_ctx,   d_model]
    std::vector<LayerWeights> layers;
    std::vector<float> ln_f_w, ln_f_b;
};

// Load all weights from bin_dir (produced by weights/dump_weights.py)
GPT2Weights load_weights(const std::string &bindir);
