#pragma once
#include "backbone_interface.h"
#include <memory>

namespace boa {

class GRUBackbone : public Backbone {
public:
    explicit GRUBackbone(const BackboneConfig& config = BackboneConfig{});
    ~GRUBackbone() override;

    std::vector<float> forward(const std::vector<int>& input) override;
    std::vector<float> get_probabilities(const std::vector<int>& input) override;
    void train() override;
    void eval() override;
    void to_device(const std::string& device) override;
    int param_count() const override;
    std::string name() const override { return "GRU"; }

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace boa
