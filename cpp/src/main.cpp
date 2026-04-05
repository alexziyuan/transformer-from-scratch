// cpp/src/main.cpp
#include "transformer.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ── Minimal BPE tokenizer (GPT-2) ────────────────────────
// We use a simple whitespace tokenize as a placeholder for now.
// Replace with a full BPE implementation or call Python's tiktoken
// via subprocess for correctness.
// The JSON output mode is called from Python which handles tokenization.
std::vector<int> simple_tokenize(const std::string & /*text*/)
{
    // Placeholder: return a fixed test sequence
    // Real tokenization is handled by Python caller in server mode
    return {15496, 11, 995}; // "Hello world"
}

// ── Greedy decode ─────────────────────────────────────────
int argmax(const float *logits, int n_vocab)
{
    int best = 0;
    for (int i = 1; i < n_vocab; i++)
    {
        if (logits[i] > logits[best])
            if (logits[i] > logits[best])
                best = i;
    }
    return best;
}

// ── JSON output mode (called by Flask server) ─────────────
void output_json(
    const ForwardResult &result,
    const std::vector<int> &token_ids,
    const GPT2Config &cfg,
    int top_k = 20)
{
    int T = (int)token_ids.size();

    // Last-position logits for next-token prediction
    const float *last_logits = result.logits.data() + (T - 1) * cfg.n_vocab;

    // Top-k tokens and raw logits
    std::vector<int> indices(cfg.n_vocab);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(), indices.begin() + top_k, indices.end(),
        [&](int a, int b)
        { return last_logits[a] > last_logits[b]; });

    json out;
    out["token_ids"] = token_ids;

    // Top-k next token logits
    json topk = json::array();
    for (int i = 0; i < top_k; i++)
    {
        topk.push_back({{"token_id", indices[i]}, {"logit", last_logits[indices[i]]}});
    }
    out["top_k_logits"] = topk;

    // Attention weights: [n_layer][n_head][T][T]
    json attn = json::array();
    for (int l = 0; l < cfg.n_layer; l++)
    {
        json layer = json::array();
        for (int h = 0; h < cfg.n_head; h++)
        {
            json head = json::array();
            for (int i = 0; i < T; i++)
            {
                json row = json::array();
                for (int j = 0; j < T; j++)
                {
                    row.push_back(result.attn_weights[l][h * T * T + i * T + j]);
                }
                head.push_back(row);
            }
            layer.push_back(head);
        }
        attn.push_back(layer);
    }
    out["attn_weights"] = attn;

    std::cout << out.dump() << std::endl;
}

int main(int argc, char *argv[])
{
    std::string weights_dir = "weights/bin";
    std::string mode = "--json";
    std::string prompt = "";

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--weights_dir" && i + 1 < argc)
            weights_dir = argv[++i];
        if (arg == "--prompt" && i + 1 < argc)
            prompt = argv[++i];
        if (arg == "--json")
            mode = "--json";
        if (arg == "--generate")
            mode = "--generate";
    }

    GPT2Model model(weights_dir);
    const auto &cfg = model.config();

    // Read token IDs from stdin if no prompt given
    // (Python caller encodes text and passes IDs via stdin)
    std::vector<int> token_ids;

    if (prompt.empty())
    {
        // Read space-separated token IDs from stdin
        int id;
        while (std::cin >> id)
            token_ids.push_back(id);
    }
    else
    {
        token_ids = simple_tokenize(prompt);
    }

    if (token_ids.empty())
    {
        std::cerr << "No token IDs provided.\n";
        return 1;
    }

    auto result = model.forward(token_ids);

    if (mode == "--json")
    {
        output_json(result, token_ids, cfg);
    }
    else
    {
        // Generate mode: print top-5 next tokens
        int T = (int)token_ids.size();
        const float *last = result.logits.data() + (T - 1) * cfg.n_vocab;
        std::cout << "Top next tokens (by logit):\n";
        std::vector<int> idx(cfg.n_vocab);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(
            idx.begin(), idx.begin() + 5, idx.end(),
            [&](int a, int b)
            { return last[a] > last[b]; });
        for (int i = 0; i < 5; i++)
        {
            std::cout << "  token " << idx[i] << " logit=" << last[idx[i]] << "\n";
        }
    }

    return 0;
}