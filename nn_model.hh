#pragma once

#include <vector>
#include "nn_layer.hh"
#include "cross_entropy.hh"

class NeuralNetwork
{
    private:
        std::vector<NNLayer *>layers;
        CECost ce_cost;

        host_vec Y;
        host_vec dY;
        float lr;

    public:
        NeuralNetwork(float lr=0.01);
        ~NeuralNetwork();

        host_vec forward(host_vec X, Shape X_shape);
        void backprop(host_vec preds, int targets, int mb_size=1);
        void update_weight_bias(float lr);

        void addLayer(NNLayer *layer);
        std::vector<NNLayer *>getLayers() const;
};