#include "relu.hh"

__global__ void reluActivationForward(float *Z, float *A,
                                        int Z_x_dim, int Z_y_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < Z_x_dim * Z_y_dim)
    {
        A[idx] = fmaxf(Z[idx], 0);
    }
}

__global__ void reluActivationBackprop(float* Z, float* dA, float* dZ,
									   int Z_x_dim, int Z_y_dim) 
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < Z_x_dim * Z_y_dim) 
    {
		if (Z[idx] > 0) 
        {
			dZ[idx] = dA[idx];
		}
		else 
        {
			dZ[idx] = 0;
		}
	}
}

ReLU::ReLU(std::string name)
{
    this->name = name;
}

ReLU::~ReLU()
{
    // do nothing
}

host_vec& ReLU::forward(host_vec& Z, Shape& Z_shape)
{
    this->Z = Z;
    Z_device = Z;
    this->Z_shape = Z_shape;

    if(A.size()==0)
    {
        host_vec temp_A(Z_shape.x*Z_shape.y, 0);
        A = temp_A;
        A_device = A;
        A_shape = Z_shape;
    }
    
    dim3 block_size(256);
    dim3 num_of_blocks((Z_shape.x * Z_shape.y + block_size.x - 1)/block_size.x);

    float* Z_device_ptr = thrust::raw_pointer_cast(Z_device.data());
    float* A_device_ptr = thrust::raw_pointer_cast(A_device.data());

    reluActivationForward<<<num_of_blocks, block_size>>>(Z_device_ptr, A_device_ptr,
														 Z_shape.x, Z_shape.y);
    
    A = A_device;
    Z_shape = A_shape;
    return A;
}

host_vec& ReLU::backprop(host_vec& dA, float learning_rate, int mb_size) 
{
    device_vec dA_device = dA;

    if(dZ.size()==0)
    {
        host_vec temp_dZ(Z_shape.x*Z_shape.y, 0);
        dZ = temp_dZ;
        dZ_device = dZ;
        this->dZ_shape = Z_shape;
    }

	dim3 block_size(256);
	dim3 num_of_blocks((Z_shape.y * Z_shape.x + block_size.x - 1) / block_size.x);
	
    float* Z_device_ptr = thrust::raw_pointer_cast(Z_device.data());
    float* dA_device_ptr = thrust::raw_pointer_cast(dA_device.data());
    float* dZ_device_ptr = thrust::raw_pointer_cast(dZ_device.data());

    reluActivationBackprop<<<num_of_blocks, block_size>>>(Z_device_ptr, dA_device_ptr,
													      dZ_device_ptr,
														  Z_shape.x, Z_shape.y);

	dZ = dZ_device;
    return dZ;
}

void ReLU::update_weights_bias(float lr)
{
    /*do nothing*/
}