#pragma once

#include "nn_layer.hh"

class ReLU : public NNLayer
{
    private:
        host_vec A;
        device_vec A_device;;
        Shape A_shape;

        host_vec Z;
        device_vec Z_device;
        Shape Z_shape;

        host_vec dZ;
        device_vec dZ_device;
        Shape dZ_shape;

    public:
        ReLU(std::string name);
        ~ReLU();

        host_vec& forward(host_vec& Z, Shape& Z_shape);
        host_vec& backprop(host_vec& dA, float lr=0.01, int mb_size=1);
        void update_weights_bias(float learning_rate);
};