#include "lstm_backbone.h"
#include <cmath>

namespace boa {

LSTMBackbone::LSTMBackbone(const BackboneConfig& config)
    : embedding_(torch::nn::EmbeddingOptions(config.vocab_size, config.d_model)),
      lstm_(torch::nn::LSTMOptions(config.d_model, config.d_model)
                .num_layers(config.num_layers)
                .dropout(config.dropout)
                .batch_first(true)),
      output_proj_(config.d_model, config.vocab_size),
      is_training_(true) {}

std::vector<float> LSTMBackbone::forward(const std::vector<int>& input) {
    std::vector<int64_t> buf(input.begin(), input.end());
    auto input_tensor = torch::from_blob(
        buf.data(),
        {1, static_cast<int64_t>(buf.size())},
        torch::TensorOptions().dtype(torch::kLong)
    ).clone();

    embedding_->train(is_training_);
    auto embedded = embedding_->forward(input_tensor);
    auto lstm_out = lstm_->forward(embedded);
    auto logits = output_proj_->forward(std::get<0>(lstm_out));

    auto flat = logits.flatten().cpu().contiguous();
    std::vector<float> result(flat.data_ptr<float>(), flat.data_ptr<float>() + flat.numel());
    return result;
}

std::vector<float> LSTMBackbone::get_probabilities(const std::vector<int>& input) {
    auto logits = forward(input);
    double sum = 0;
    for (float v : logits) sum += std::exp(v);
    for (float& v : logits) v = std::exp(v) / sum;
    return logits;
}

void LSTMBackbone::train() { is_training_ = true; }
void LSTMBackbone::eval() { is_training_ = false; }
void LSTMBackbone::to_device(const std::string& device) { /* TODO */ }

int LSTMBackbone::param_count() const {
    int count = 0;
    count += embedding_->parameters()[0].numel();
    for (const auto& p : lstm_->parameters()) count += p.numel();
    count += output_proj_->parameters()[0].numel();
    return count;
}

} // namespace boa
