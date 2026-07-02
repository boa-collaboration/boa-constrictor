#include "gru_backbone.h"
#include <torch/torch.h>
#include <memory>

namespace boa {

struct GRUBackbone::Impl {
    torch::nn::Embedding embedding{nullptr};
    torch::nn::GRU gru{nullptr};
    torch::nn::Linear output_proj{nullptr};
    bool is_training = true;

    Impl(const BackboneConfig& config) {
        embedding = torch::nn::Embedding(
            torch::nn::EmbeddingOptions(config.vocab_size, config.d_model)
        );
        gru = torch::nn::GRU(
            torch::nn::GRUOptions(config.d_model, config.d_model)
                .num_layers(config.num_layers)
                .dropout(config.dropout)
                .batch_first(true)
        );
        output_proj = torch::nn::Linear(config.d_model, config.vocab_size);
    }
};

GRUBackbone::GRUBackbone(const BackboneConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

GRUBackbone::~GRUBackbone() = default;

std::vector<float> GRUBackbone::forward(const std::vector<int>& input) {
    // Must copy to int64_t; from_blob with kLong over int* (32-bit) causes corruption
    std::vector<int64_t> buf(input.begin(), input.end());
    auto input_tensor = torch::from_blob(
        buf.data(),
        {1, static_cast<int64_t>(buf.size())},
        torch::TensorOptions().dtype(torch::kLong)
    ).clone();

    pimpl_->embedding->train(pimpl_->is_training);
    auto embedded = pimpl_->embedding->forward(input_tensor);
    auto gru_out = pimpl_->gru->forward(embedded);
    auto logits = pimpl_->output_proj->forward(std::get<0>(gru_out));

    auto flat = logits.flatten().cpu().contiguous();
    std::vector<float> result(flat.data_ptr<float>(), flat.data_ptr<float>() + flat.numel());
    return result;
}

std::vector<float> GRUBackbone::get_probabilities(const std::vector<int>& input) {
    pimpl_->is_training = false;
    auto logits = forward(input);
    // Apply softmax per timestep: each block of vocab_size values is one distribution
    const int vocab_size = 256;
    const int seq_len = static_cast<int>(logits.size()) / vocab_size;
    for (int t = 0; t < seq_len; ++t) {
        float* start = logits.data() + t * vocab_size;
        double sum = 0.0;
        for (int i = 0; i < vocab_size; ++i) sum += std::exp(start[i]);
        for (int i = 0; i < vocab_size; ++i) start[i] = std::exp(start[i]) / sum;
    }
    return logits;
}

void GRUBackbone::train() { pimpl_->is_training = true; }
void GRUBackbone::eval() { pimpl_->is_training = false; }
void GRUBackbone::to_device(const std::string& device) { /* TODO */ }

int GRUBackbone::param_count() const {
    int count = 0;
    count += pimpl_->embedding->parameters()[0].numel();
    for (const auto& p : pimpl_->gru->parameters()) {
        count += p.numel();
    }
    count += pimpl_->output_proj->parameters()[0].numel();
    return count;
}

} // namespace boa
