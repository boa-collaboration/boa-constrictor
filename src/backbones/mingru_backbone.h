#pragma once
#include "backbone_interface.h"
#include <torch/torch.h>
#include <memory>
#include <vector>

namespace boa {

class MinGRUCell {
public:
    MinGRUCell(int d_model);
    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& h);

private:
    torch::nn::Linear gate_{nullptr};
    torch::nn::Linear candidate_{nullptr};
};

class MinGRUBackbone : public Backbone {
public:
    explicit MinGRUBackbone(const BackboneConfig& config = BackboneConfig{});
    ~MinGRUBackbone() override = default;

    std::vector<float> forward(const std::vector<int>& input) override;
    std::vector<float> get_probabilities(const std::vector<int>& input) override;
    void train() override;
    void eval() override;
    void to_device(const std::string& device) override;
    int param_count() const override;
    std::string name() const override { return "MinGRU"; }

private:
    torch::nn::Embedding embedding_{nullptr};
    std::vector<std::shared_ptr<MinGRUCell>> cells_;
    torch::nn::Linear output_proj_{nullptr};
    bool is_training_ = true;
};

} // namespace boa
