#pragma once
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "shape.hh"

typedef thrust::host_vector<float> host_vec;
typedef thrust::device_vector<float> device_vec;

class NNLayer
{
    protected:
        std::string name;
    
    public:
        virtual host_vec& forward(host_vec& A, Shape& A_shape) = 0;
        virtual host_vec& backprop(host_vec& dZ, float lr, int mb_size=1) = 0;
        virtual void update_weights_bias(float learning_rate) = 0;

        std::string getName()
        {
            return this->name;
        }
};