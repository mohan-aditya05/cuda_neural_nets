#pragma once
#include "nn_layer.hh"

class LinearLayer : public NNLayer 
{
    private:
        const float weights_init_thresh = 0.01;
        const float dropout_rate=0.0;
        int mb_size=1;

        Shape W_shape;
        host_vec W;
        device_vec W_device;

        host_vec b;
        device_vec b_device;
        Shape b_shape;

        host_vec Z;
        device_vec Z_device;
        Shape Z_shape;

        host_vec A;
        device_vec A_device;
        Shape A_shape;

        host_vec dA;
        device_vec dA_device;
        Shape dA_shape;

        host_vec dropped;
        device_vec dropped_device;
        Shape dropped_shape; // (1, n_neurons)

        host_vec m_weight_deriv;
        device_vec m_weight_deriv_device;
        Shape m_weight_deriv_shape;

        host_vec m_bias_deriv;
        device_vec m_bias_deriv_device;
        Shape m_bias_deriv_shape;

        void initializeBiasWithZeros();
        void initializeWeightsRandomly();

        void computeAndStoreBackpropError(host_vec& dZ);
        void computeAndStoreLayerOutput(host_vec& A);
        void updateWeights(float learning_rate);
        void accumulateWeightDeriv(host_vec& dZ, float learning_rate);
        void updateBias(float learning_rate);
        void accumulateBiasDeriv(host_vec& dZ, float learning_rate);

    public:
        LinearLayer(std::string name, Shape W_shape, float dropout=0);
        ~LinearLayer();

        host_vec& forward(host_vec& A, Shape& A_shape);
        host_vec& backprop(host_vec& dZ, float learning_rate = 0.01, int mb_size=1);
        void update_weights_bias(float learning_rate);

        int getXDim() const;
        int getYDim() const;

        host_vec getWeightsMatrix() const;
        host_vec getBiasVector() const;
};