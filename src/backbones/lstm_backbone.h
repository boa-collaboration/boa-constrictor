#pragma once
#include "backbone_interface.h"
#include <torch/torch.h>

namespace boa {

class LSTMBackbone : public Backbone {
public:
    explicit LSTMBackbone(const BackboneConfig& config = BackboneConfig{});
    ~LSTMBackbone() override = default;

    std::vector<float> forward(const std::vector<int>& input) override;
    std::vector<float> get_probabilities(const std::vector<int>& input) override;
    void train() override;
    void eval() override;
    void to_device(const std::string& device) override;
    int param_count() const override;
    std::string name() const override { return "LSTM"; }

private:
    torch::nn::Embedding embedding_{nullptr};
    torch::nn::LSTM lstm_{nullptr};
    torch::nn::Linear output_proj_{nullptr};
    bool is_training_ = true;
};

} // namespace boa
