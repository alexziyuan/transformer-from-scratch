// cpp/src/loader.cpp
#include "loader.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Replace __ back to . to get HuggingFace key name
static std::string safe_to_key(const std::string &safe_name)
{
    std::string key = safe_name;
    size_t pos = 0;
    while ((pos = key.find("__", pos)) != std::string::npos)
    {
        key.replace(pos, 2, ".");
        pos += 1;
    }
    return key;
}

static std::vector<float> load_bin(
    const std::string &bin_dir,
    const std::string &hf_key)
{
    // Convert HuggingFace key to filename
    std::string safe_name = hf_key;
    size_t pos = 0;
    while ((pos = safe_name.find(".", pos)) != std::string::npos)
    {
        safe_name.replace(pos, 1, "__");
        pos += 2;
    }

    std::string path = bin_dir + "/" + safe_name + ".bin";
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
    {
        throw std::runtime_error("Cannot open weight file: " + path);
    }

    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<float> buf(size / sizeof(float));
    if (!f.read(reinterpret_cast<char *>(buf.data()), size))
    {
        throw std::runtime_error("Failed to read: " + path);
    }

    return buf;
}

GPT2Weights load_weights(const std::string &bin_dir)
{
    // Read manifest for config
    std::string manifest_path = bin_dir + "/manifest.json";
    std::ifstream mf(manifest_path);
    if (!mf.is_open())
    {
        throw std::runtime_error(
            "manifest.json not found in " + bin_dir +
            ". Run weights/dump_weights.py first.");
    }

    json manifest;
    mf >> manifest;

    // Infer config from shapes
    GPT2Weights w;
    auto &cfg = w.config;

    auto wte_shape = manifest["wte.weight"].get<std::vector<int>>();
    auto wpe_shape = manifest["wpe.weight"].get<std::vector<int>>();

    cfg.n_vocab = wte_shape[0];
    cfg.d_model = wte_shape[1];
    cfg.n_ctx = wpe_shape[0];

    // Count layers
    cfg.n_layer = 0;
    while (manifest.contains("h." + std::to_string(cfg.n_layer) + ".ln_1.weight"))
    {
        cfg.n_layer++;
    }

    // n_head from d_model
    std::unordered_map<int, int> head_map = {
        {768, 12}, {1024, 16}, {1280, 20}, {1600, 25}};
    cfg.n_head = head_map.count(cfg.d_model) ? head_map[cfg.d_model] : 12;

    std::cerr << "Config: n_vocab=" << cfg.n_vocab
              << " d_model=" << cfg.d_model
              << " n_layer=" << cfg.n_layer
              << " n_head=" << cfg.n_head << "\n";

    // Load embeddings
    w.wte = load_bin(bin_dir, "wte.weight");
    w.wpe = load_bin(bin_dir, "wpe.weight");

    // Load transformer blocks
    w.layers.resize(cfg.n_layer);
    for (int i = 0; i < cfg.n_layer; i++)
    {
        std::string p = "h." + std::to_string(i);
        auto &l = w.layers[i];

        l.ln1_w = load_bin(bin_dir, p + ".ln_1.weight");
        l.ln1_b = load_bin(bin_dir, p + ".ln_1.bias");
        l.c_attn_w = load_bin(bin_dir, p + ".attn.c_attn.weight");
        l.c_attn_b = load_bin(bin_dir, p + ".attn.c_attn.bias");
        l.c_proj_w = load_bin(bin_dir, p + ".attn.c_proj.weight");
        l.c_proj_b = load_bin(bin_dir, p + ".attn.c_proj.bias");
        l.ln2_w = load_bin(bin_dir, p + ".ln_2.weight");
        l.ln2_b = load_bin(bin_dir, p + ".ln_2.bias");
        l.fc_w = load_bin(bin_dir, p + ".mlp.c_fc.weight");
        l.fc_b = load_bin(bin_dir, p + ".mlp.c_fc.bias");
        l.proj_w = load_bin(bin_dir, p + ".mlp.c_proj.weight");
        l.proj_b = load_bin(bin_dir, p + ".mlp.c_proj.bias");
    }

    // Final layer norm
    w.ln_f_w = load_bin(bin_dir, "ln_f.weight");
    w.ln_f_b = load_bin(bin_dir, "ln_f.bias");

    std::cerr << "Loaded " << cfg.n_layer << " transformer blocks.\n";
    return w;
}