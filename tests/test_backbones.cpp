#include <gtest/gtest.h>
#include "gru_backbone.h"
#include "lstm_backbone.h"
#include "mingru_backbone.h"
#include <vector>
#include <iostream>

using namespace boa;

TEST(BackbonesTest, GRUForward) {
    GRUBackbone backbone;
    std::vector<int> input(100, 42);
    auto output = backbone.forward(input);
    EXPECT_EQ(output.size(), 100 * 256);
}

TEST(BackbonesTest, GRUProbabilities) {
    GRUBackbone backbone;
    std::vector<int> input(100, 42);
    auto probs = backbone.get_probabilities(input);
    EXPECT_EQ(probs.size(), 100 * 256);
    float sum = 0;
    for (float p : probs) sum += p;
    EXPECT_NEAR(sum, 100.0, 0.01);
}

TEST(BackbonesTest, LSTMFoward) {
    LSTMBackbone backbone;
    std::vector<int> input(100, 42);
    auto output = backbone.forward(input);
    EXPECT_EQ(output.size(), 100 * 256);
}

TEST(BackbonesTest, MinGRUForward) {
    MinGRUBackbone backbone;
    std::vector<int> input(100, 42);
    auto output = backbone.forward(input);
    EXPECT_EQ(output.size(), 100 * 256);
}

TEST(BackbonesTest, ParamCounts) {
    BackboneConfig config;
    config.d_model = 256;
    config.num_layers = 1;

    GRUBackbone gru(config);
    LSTMBackbone lstm(config);
    MinGRUBackbone mingru(config);

    EXPECT_GT(gru.param_count(), 0);
    EXPECT_GT(lstm.param_count(), 0);
    EXPECT_GT(mingru.param_count(), 0);
    std::cout << "GRU params: " << gru.param_count() << std::endl;
    std::cout << "LSTM params: " << lstm.param_count() << std::endl;
    std::cout << "MinGRU params: " << mingru.param_count() << std::endl;
}
