#include "mingru_backbone.h"
#include <cmath>

namespace boa {

MinGRUCell::MinGRUCell(int d_model)
    : gate_(torch::nn::LinearOptions(d_model, d_model)),
      candidate_(torch::nn::LinearOptions(d_model, d_model)) {}

torch::Tensor MinGRUCell::forward(const torch::Tensor& x, const torch::Tensor& h) {
    auto z = torch::sigmoid(gate_->forward(x));
    auto c = torch::tanh(candidate_->forward(x));
    return (1 - z) * h + z * c;
}

MinGRUBackbone::MinGRUBackbone(const BackboneConfig& config)
    : embedding_(torch::nn::EmbeddingOptions(config.vocab_size, config.d_model)),
      output_proj_(torch::nn::LinearOptions(config.d_model, config.vocab_size)),
      is_training_(true) {
    for (int i = 0; i < config.num_layers; ++i) {
        cells_.push_back(std::make_shared<MinGRUCell>(config.d_model));
    }
}

std::vector<float> MinGRUBackbone::forward(const std::vector<int>& input) {
    std::vector<int64_t> buf(input.begin(), input.end());
    auto input_tensor = torch::from_blob(
        buf.data(),
        {1, static_cast<int64_t>(buf.size())},
        torch::TensorOptions().dtype(torch::kLong)
    ).clone();

    embedding_->train(is_training_);
    auto h = embedding_->forward(input_tensor);

    for (auto& cell : cells_) {
        // Simplified version: pass the whole sequence through the cell
        // In proper MinGRU, this requires a parallel prefix scan.
        h = cell->forward(h, h);
    }

    auto logits = output_proj_->forward(h);
    auto flat = logits.flatten().cpu().contiguous();
    std::vector<float> result(flat.data_ptr<float>(), flat.data_ptr<float>() + flat.numel());
    return result;
}

std::vector<float> MinGRUBackbone::get_probabilities(const std::vector<int>& input) {
    auto logits = forward(input);
    double sum = 0;
    for (float v : logits) sum += std::exp(v);
    for (float& v : logits) v = std::exp(v) / sum;
    return logits;
}

void MinGRUBackbone::train() { is_training_ = true; }
void MinGRUBackbone::eval() { is_training_ = false; }
void MinGRUBackbone::to_device(const std::string& device) { /* TODO */ }

int MinGRUBackbone::param_count() const {
    int count = 0;
    count += embedding_->parameters()[0].numel();
    count += cells_.size() * 2 * embedding_->parameters()[0].size(1) * embedding_->parameters()[0].size(0); // approximate
    count += output_proj_->parameters()[0].numel();
    return count;
}

} // namespace boa
