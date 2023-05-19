#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>

#include "linear_layer.hh"

LinearLayer::LinearLayer(std::string name, Shape W_shape, float dropout): W_shape(W_shape), dropout_rate(1 - dropout)
{
    host_vec temp_W(W_shape.y*W_shape.x, 0);
    W = temp_W;
    W_device = W;

    host_vec temp_b(W_shape.y, 0);
    b = temp_b;
    b_device = b;

    this->name = name;
    initializeWeightsRandomly();

	std::uniform_real_distribution<double> dist(0, 1);
	std::default_random_engine m_eng;

	host_vec temp_dropped(W_shape.y, 0.0);
	dropped = temp_dropped;

	std::generate(dropped.begin(), dropped.end(),
             [&]() { return dist(m_eng) < dropout_rate ? 1/dropout_rate : 0; });
	dropped_device = dropped;
	dropped_shape = Shape(W_shape.y, 1);
}

void LinearLayer::initializeWeightsRandomly()
{
    std::default_random_engine generator;
	std::normal_distribution<float> normal_distribution(0.0, 1.0);

    for(int i=0; i<W_shape.x*W_shape.y; i++)
    {
        W[i] = normal_distribution(generator)/sqrt(W.size());   
    }

    W_device = W;
}

host_vec& LinearLayer::forward(host_vec& A, Shape& in_shape){
    assert(W_shape.x == in_shape.y);

    this->A = A;
    this->A_shape = in_shape;
	A_device = A;
    Shape Z_shape(in_shape.x, W_shape.y);
	this->Z_shape = Z_shape;

    if(Z.size()==0)
    {
        host_vec temp_Z(Z_shape.y*Z_shape.x, 0.0);
        Z = temp_Z;
        Z_device = Z;
    }

    computeAndStoreLayerOutput(A);

	in_shape = Z_shape;

    return Z;
}

__global__ void linearLayerForward(float* W, float* A, float* Z, 
                                    float *b, float *dropped, int W_x_dim, int W_y_dim,
                                    int A_x_dim, int A_y_dim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int Z_x_dim = A_x_dim;
    int Z_y_dim = W_y_dim;

    float Z_value = 0;

    if(row < Z_y_dim && col < Z_x_dim)
    {
        for(int i=0; i<W_x_dim; i++)
        {
            Z_value += W[row * W_x_dim + i] * A[i * A_x_dim + col];
        }
        Z[row * Z_x_dim + col] = Z_value + b[row];
		Z[row * Z_x_dim + col] *= dropped[row];
    }
}

__global__ void linearLayerBackprop(float* W, float* dZ, float *dA, float *dropped,
									int W_x_dim, int W_y_dim,
									int dZ_x_dim, int dZ_y_dim) 
{

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int dA_x_dim = dZ_x_dim;
	int dA_y_dim = W_x_dim;

	float dA_value = 0.0f;

	if (row < dA_y_dim && col < dA_x_dim) {
		if(dropped[row] > 0)
		{
			for (int i = 0; i < W_y_dim; i++) 
			{
				dA_value += dropped[row] * W[i * W_x_dim + row] * dZ[i * dZ_x_dim + col];
			}
			dA[row * dA_x_dim + col] = dA_value;
		}
		
	}
}

__global__ void linearLayerAccumulateWeights(  float* dZ, float* A, float* W, float* dropped,
										   float* m_weight_deriv, int dZ_x_dim, int dZ_y_dim,
										   int A_x_dim, int A_y_dim,
										   float learning_rate, int mb_size) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int W_x_dim = A_y_dim;
	int W_y_dim = dZ_y_dim;

	if (row < W_y_dim && col < W_x_dim) {
		if(dropped[row] > 0)
		{
			for (int i = 0; i < dZ_x_dim; i++) {
				m_weight_deriv[row * W_x_dim + col] += dropped[row] * dZ[row * dZ_x_dim + i] * A[col * A_x_dim + i];
		}
		m_weight_deriv[row * W_x_dim + col] /= mb_size;
		
		}
	}
	
}

__global__ void linearLayerUpdateWeights(  float* W, float* dropped,
										   float* m_weight_deriv, int dZ_x_dim, int dZ_y_dim,
										   int A_x_dim, int A_y_dim,
										   float learning_rate) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int W_x_dim = A_y_dim;
	int W_y_dim = dZ_y_dim;
	
	if (row < W_y_dim && col < W_x_dim) {
		if(dropped[row] > 0)
		{
			W[row * W_x_dim + col] = W[row * W_x_dim + col] - learning_rate * (m_weight_deriv[row * W_x_dim + col] / A_x_dim);
			m_weight_deriv[row * W_x_dim + col] = 0.0;
		}
	}
	
}

__global__ void linearLayerAccumulateBias(  float* dZ, float* b, float* dropped,
										float* m_bias_deriv, int dZ_x_dim, int dZ_y_dim,
										int b_x_dim,
										float learning_rate, int mb_size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < dZ_x_dim * dZ_y_dim) {
		int dZ_x = index % dZ_x_dim;
		int dZ_y = index / dZ_x_dim;
		if(dropped[dZ_y] > 0)
		{
			m_bias_deriv[dZ_y] += dropped[dZ_y] * dZ[dZ_y * dZ_x_dim + dZ_x] / mb_size ;
		}
	}
}

__global__ void linearLayerUpdateBias(  float* b, float* dropped,
										float* m_bias_deriv, int dZ_x_dim, int dZ_y_dim,
										int b_x_dim,
										float learning_rate) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < dZ_x_dim * dZ_y_dim) {
		int dZ_x = index % dZ_x_dim;
		int dZ_y = index / dZ_x_dim;
		if(dropped[dZ_y] > 0)
		{
			atomicAdd(&b[dZ_y], - learning_rate * m_bias_deriv[dZ_y]);
			m_bias_deriv[dZ_y] = 0.0;
		}
	}
}

void LinearLayer::computeAndStoreLayerOutput(host_vec& A)
{
    dim3 block_size(8, 8);
    dim3 num_of_blocks((Z_shape.x + block_size.x - 1)/ block_size.x,
                            (Z_shape.y + block_size.y - 1)/block_size.y);


    float *W_device_ptr = thrust::raw_pointer_cast(W_device.data());
    float *A_device_ptr = thrust::raw_pointer_cast(A_device.data());
    float *Z_device_ptr = thrust::raw_pointer_cast(Z_device.data());
    float *b_device_ptr = thrust::raw_pointer_cast(b_device.data());
	float *dropped_device_ptr = thrust::raw_pointer_cast(dropped_device.data());
    linearLayerForward<<<num_of_blocks, block_size>>>(W_device_ptr, A_device_ptr,
                                                        Z_device_ptr, b_device_ptr, dropped_device_ptr,
                                                        W_shape.x, W_shape.y, A_shape.x, A_shape.y);
	A = A_device;
	Z = Z_device;
}

host_vec& LinearLayer::backprop(host_vec& dZ, float learning_rate, int mb_size) {
	
	this->mb_size = mb_size;
    if(dA.size()==0)
    {
        host_vec temp_dA(A_shape.x*A_shape.y, 0.0);
        dA = temp_dA;
        dA_device = dA;
		dA_shape = Shape(A_shape.x, A_shape.y);
    }

	if(m_weight_deriv.size()==0)
	{
		host_vec temp_w(W_shape.x*W_shape.y, 0.0);
		m_weight_deriv = temp_w;
		m_weight_deriv_device = m_weight_deriv;
		m_weight_deriv_shape = W_shape;
	}

	if(m_bias_deriv.size()==0)
	{
		host_vec temp_b(b_shape.x*b_shape.y, 0.0);
		m_bias_deriv = temp_b;
		m_bias_deriv_device = m_bias_deriv;
		m_bias_deriv_shape = b_shape;
	}

	computeAndStoreBackpropError(dZ);

	accumulateBiasDeriv(dZ, learning_rate);

	accumulateWeightDeriv(dZ, learning_rate);

	dA = dA_device;
	return dA;
}

void LinearLayer::update_weights_bias(float learning_rate)
{
	updateWeights(learning_rate);
	updateBias(learning_rate);
}

void LinearLayer::computeAndStoreBackpropError(host_vec& dZ) {
	device_vec dZ_device = dZ;
    
    dim3 block_size(8, 8);
	dim3 num_of_blocks(	(A_shape.x + block_size.x - 1) / block_size.x,
						(A_shape.y + block_size.y - 1) / block_size.y);
	float *W_device_ptr = thrust::raw_pointer_cast(W_device.data());
    float *dA_device_ptr = thrust::raw_pointer_cast(dA_device.data());
    float *dZ_device_ptr = thrust::raw_pointer_cast(dZ_device.data());
	float *dropped_device_ptr = thrust::raw_pointer_cast(dropped_device.data());

    linearLayerBackprop<<<num_of_blocks, block_size>>>( W_device_ptr,
														dZ_device_ptr,
														dA_device_ptr,
														dropped_device_ptr,
														W_shape.x, W_shape.y,
														Z_shape.x, Z_shape.y);

    dZ = dZ_device;
}


int LinearLayer::getXDim() const {
	return W_shape.x;
}

int LinearLayer::getYDim() const {
	return W_shape.y;
}

host_vec LinearLayer::getWeightsMatrix() const {
	return W;
}

host_vec LinearLayer::getBiasVector() const {
	return b;
}

bool
check_weights(host_vec old, host_vec new_vec, Shape W_shape)
{
	int idx=0;
	for(int i=0; i<W_shape.y; i++)
	{
		for(int j=0; j<W_shape.x; j++)
		{
			if(old[idx]!=new_vec[idx])
				return false;
			idx++;
		}
	}
	return true;
}

void LinearLayer::updateWeights(float learning_rate) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(W_shape.x + block_size.x - 1) / block_size.x,
						(W_shape.y + block_size.y - 1) / block_size.y);
                        
    float *W_device_ptr = thrust::raw_pointer_cast(W_device.data());
	float* dropped_device_ptr = thrust::raw_pointer_cast(dropped_device.data());
	float* m_weight_ptr = thrust::raw_pointer_cast(m_weight_deriv_device.data());

    linearLayerUpdateWeights<<<num_of_blocks, block_size>>>(
															W_device_ptr,
															dropped_device_ptr,
															m_weight_ptr,
															Z_shape.x, Z_shape.y,
															A_shape.x, A_shape.y,
															learning_rate);
	
	W = W_device;
}

void LinearLayer::accumulateWeightDeriv(host_vec& dZ, float learning_rate) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(W_shape.x + block_size.x - 1) / block_size.x,
						(W_shape.y + block_size.y - 1) / block_size.y);
                        
	device_vec dZ_device = dZ;
    float *W_device_ptr = thrust::raw_pointer_cast(W_device.data());
    float *A_device_ptr = thrust::raw_pointer_cast(A_device.data());
    float *dZ_device_ptr = thrust::raw_pointer_cast(dZ_device.data());
	float* dropped_device_ptr = thrust::raw_pointer_cast(dropped_device.data());
	float* m_weight_ptr = thrust::raw_pointer_cast(m_weight_deriv_device.data());

    linearLayerAccumulateWeights<<<num_of_blocks, block_size>>>(dZ_device_ptr,
															A_device_ptr,
															W_device_ptr,
															dropped_device_ptr,
															m_weight_ptr,
															Z_shape.x, Z_shape.y,
															A_shape.x, A_shape.y,
															learning_rate, mb_size);
	
	m_weight_deriv = m_weight_deriv_device;
}



void LinearLayer::updateBias(float learning_rate) 
{
	dim3 block_size(256);
	dim3 num_of_blocks( (Z.size() + block_size.x - 1) / block_size.x);
	
    float *b_device_ptr = thrust::raw_pointer_cast(b_device.data());
    float *dropped_device_ptr = thrust::raw_pointer_cast(dropped_device.data());
	float *m_bias_ptr = thrust::raw_pointer_cast(m_weight_deriv_device.data());
	linearLayerUpdateBias<<<num_of_blocks, block_size>>>(
														 b_device_ptr,
														 dropped_device_ptr,
														 m_bias_ptr,
														 Z_shape.x, Z_shape.y,
														 b_shape.x, learning_rate);
	b = b_device;
}

void LinearLayer::accumulateBiasDeriv(host_vec& dZ, float learning_rate) 
{
	dim3 block_size(256);
	dim3 num_of_blocks( (dZ.size() + block_size.x - 1) / block_size.x);

    device_vec dZ_device = dZ;
	
    float *dZ_device_ptr = thrust::raw_pointer_cast(dZ_device.data());
    float *b_device_ptr = thrust::raw_pointer_cast(b_device.data());
    float *dropped_device_ptr = thrust::raw_pointer_cast(dropped_device.data());
	float *m_bias_ptr = thrust::raw_pointer_cast(m_weight_deriv_device.data());
	linearLayerAccumulateBias<<<num_of_blocks, block_size>>>(dZ_device_ptr,
														 b_device_ptr,
														 dropped_device_ptr,
														 m_bias_ptr,
														 Z_shape.x, Z_shape.y,
														 b_shape.x, learning_rate, mb_size);
	m_weight_deriv = m_weight_deriv_device;
}