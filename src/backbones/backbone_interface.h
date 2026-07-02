#pragma once
#include <vector>
#include <string>

namespace boa {

struct BackboneConfig {
    int d_model = 256;
    int num_layers = 1;
    int vocab_size = 256;
    float dropout = 0.1;
    std::string device = "cpu";   // "cpu", "cuda", "mps"
};

class Backbone {
public:
    virtual ~Backbone() = default;
    virtual std::vector<float> forward(const std::vector<int>& input) = 0;
    virtual std::vector<float> get_probabilities(const std::vector<int>& input) = 0;
    virtual void train() = 0;
    virtual void eval() = 0;
    virtual void to_device(const std::string& device) = 0;
    virtual int param_count() const = 0;
    virtual std::string name() const = 0;
};

} // namespace boa
