#pragma once

#include <string>
#include <thrust/host_vector.h>

class MNISTDataset {
private:
	size_t batch_size;
	size_t number_of_batches;
    std::string fn_images, fn_labels;

	thrust::host_vector<thrust::host_vector<thrust::host_vector<float>>> batches;
	thrust::host_vector<thrust::host_vector<int>> targets;

    void read_mnist_labels();
    void read_mnist_images();

public:

	MNISTDataset(size_t batch_size, const std::string &fn_images, const std::string &fn_labels);

	int getNumOfBatches();
	thrust::host_vector<thrust::host_vector<thrust::host_vector<float>>>& getBatches();
	thrust::host_vector<thrust::host_vector<int>>& getTargets();
};