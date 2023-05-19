#include "softmax.hh"
#include <iostream>

__global__ void softmaxActivationForward(float* Z, float* A,
                                            int Z_x_dim, int Z_y_dim, float C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < Z_x_dim * Z_y_dim)
    {
        A[idx] = exp(Z[idx] - C);
    } 
}

__global__ void softmaxActivationBackprop(float* Z, float* dA, float* dZ,
										  int Z_x_dim, int Z_y_dim, int N) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=0; i<N; i++)
    {
        if(index == i)
        {
            dZ[index] += (Z[index] * (1- Z[i]))*dA[index];
        }
        else
        {
            dZ[index] += (-Z[i] * Z[index])*dA[index];
        }
    }
	
}

host_vec& Softmax::forward(host_vec& Z, Shape& Z_shape)
{
    this->Z = Z;
    Z_device = Z;
    this->Z_shape = Z_shape;

    float C=0;

    for(int i=0; i<Z.size(); i++)
    {
        C = max(Z[i], C);
    }
    
    if(A.size()==0)
    {
        host_vec temp_A(Z_shape.x * Z_shape.y, 0.0);
        A = temp_A;
        A_device = A;
        A_shape = Z_shape;
    }

    dim3 block_size(256);
    dim3 num_of_blocks((Z_shape.x * Z_shape.y + block_size.x - 1) / block_size.x);

    float *A_device_ptr = thrust::raw_pointer_cast(A_device.data());
    float *Z_device_ptr = thrust::raw_pointer_cast(Z_device.data());
    softmaxActivationForward<<<num_of_blocks, block_size>>>(Z_device_ptr,
                                                            A_device_ptr,
                                                            Z_shape.x, Z_shape.y, C);

    float _sum=0;
    A = A_device;
    
    for(int i=0; i<A.size(); i++)
    {
        _sum += A[i];
    }
    
    for (size_t i = 0; i < A.size(); i++) {
        A[i] = A[i]/_sum;
    }

    A_device = A;
    Z_shape = A_shape;
    return A;
}

host_vec& Softmax::backprop(host_vec& dA, float lr, int mb_size)
{
    if(dZ.size()==0)
    {
        host_vec temp_dZ(Z_shape.x * Z_shape.y, 0);
        dZ = temp_dZ;
        dZ_device = dZ;
        dZ_shape = Z_shape;
    }

    for(int index = 0; index < dZ.size(); index++)
    {
        dZ[index] = 0;
        for(int i=0; i<dZ.size(); i++)
        {
            if(index == i)
            {
                dZ[index] += (Z[index] * (1- Z[i]))*dA[i];
            }
            else
            {
                dZ[index] += (-Z[i] * Z[index])*dA[i];
            }
        }
    }

    return dZ;
}

void Softmax::update_weights_bias(float learning_rate)
{
    /*do nothing*/
}

Softmax::Softmax(std::string name)
{
    this->name = name;
}

Softmax::~Softmax()
{}