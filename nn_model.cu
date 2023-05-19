#include "nn_model.hh"

NeuralNetwork::NeuralNetwork(float lr):lr(lr)
{}

NeuralNetwork::~NeuralNetwork()
{
    //TODO
}

void NeuralNetwork::addLayer(NNLayer *layer)
{
    this->layers.push_back(layer);
}

host_vec NeuralNetwork::forward(host_vec X, Shape X_shape)
{
    host_vec Z = X;

    for (auto layer : layers)
    {
        Z = layer->forward(Z, X_shape);
    }

    Y = Z;
    return Y;
}

void NeuralNetwork::backprop(host_vec preds, int target, int mb_size)
{
    if(dY.size()==0)
    {
        host_vec temp_dY(preds.size(), 0);
        dY = temp_dY;
    }
    host_vec error = ce_cost.dCost(preds, target, dY);

    for(auto it = this->layers.rbegin(); it!=this->layers.rend(); it++)
    {
        error = (*it)->backprop(error, lr, mb_size);
    }

    cudaDeviceSynchronize();
}

void NeuralNetwork::update_weight_bias(float lr)
{
    for(auto it = this->layers.begin(); it!= this->layers.end(); it++)
    {
        (*it)->update_weights_bias(lr);
    }

    cudaDeviceSynchronize();
}

std::vector<NNLayer *>NeuralNetwork::getLayers() const
{
    return layers;
}