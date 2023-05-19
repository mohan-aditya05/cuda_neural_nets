#include <iostream>
#include <time.h>
#include <chrono>
#include <fstream>

#include "nn_model.hh"
#include "linear_layer.hh"
#include "relu.hh"
#include "dataset.hh"
#include "softmax.hh"

bool isCorrectPredicted(const host_vec& predictions, const int target) {

	auto pos = std::max_element(predictions.begin(), predictions.end());
	auto idx = std::distance(predictions.begin(), pos);

	if (idx == target) {
		return true;
	}
	return false;
}

int main()
{
    srand( time(NULL) );

	std::ofstream myfile;
	myfile.open("cuda_profiling.csv");
	myfile << "step,time" <<std::endl;

	int batch_size = 100;
	float lr = 0.001;

	MNISTDataset dataset(batch_size, "mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte");
	MNISTDataset test_dataset(batch_size, "mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte");
	CECost ce_cost;

	NeuralNetwork nn(lr);
	nn.addLayer(new LinearLayer("linear_1", Shape(28*28, 1024), 0.4));
	nn.addLayer(new ReLU("relu_1"));
	nn.addLayer(new LinearLayer("linear_2", Shape(1024, 10), 0.0));
	nn.addLayer(new Softmax("softmax_output"));

	// network training
	host_vec Y;
	int total_batches = dataset.getNumOfBatches();
	int total_test_batches = test_dataset.getNumOfBatches();

	//epoch
	for (int epoch = 0; epoch < 20; epoch++) {
		float cost=0.0;

		int64_t epoch_time = 0;
		int step=0;
		for (int batch = 0; batch < total_batches; batch++) {

			//mini-batch
			float batch_cost = 0.0;
			int64_t mini_batch_time = 0;
			
			for(int i=0; i<batch_size; i++)
			{
				//single step
				auto start = std::chrono::high_resolution_clock::now();

				Y = nn.forward(dataset.getBatches()[batch][i], Shape(1,28*28));
				nn.backprop(Y, dataset.getTargets()[batch][i], batch_size);
				batch_cost += ce_cost.cost(Y, dataset.getTargets()[batch][i]);

				auto end = std::chrono::high_resolution_clock::now();
				// std::cout<< "Time Taken for one step: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() <<" microseconds"<<std::endl;;
				auto step_time = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
				mini_batch_time += step_time;
				myfile << step << "," << step_time << std::endl;
				step++;
			}

			epoch_time += mini_batch_time;
			// std::cout<< "Mini-Batch Cost: " << batch_cost/batch_size << std::endl;
			
			nn.update_weight_bias(lr);
			cost += batch_cost/batch_size;
		}

		myfile.close();

		std::cout<< "Time per epoch: " << epoch_time << std::endl;
		std::cout<< "Avg Time Per Step: " << epoch_time/(batch_size*total_batches) << std::endl;

		if (epoch % 1 == 0) {
			std::cout 	<< "Epoch: " << epoch
						<< ", Cost: " << cost / (dataset.getNumOfBatches())
						<< std::endl;

			float correct = 0;
			float total = 0;

			//iterating over test set batches
			for(int j=0; j<total_test_batches; j++)
			{
				auto curr_batch = test_dataset.getBatches()[j];
				auto curr_targets = test_dataset.getTargets()[j];
				for(int i=0; i<batch_size; i++)
				{
					Y = nn.forward(curr_batch[i], Shape(1, 28*28));
					if(isCorrectPredicted(Y, curr_targets[i]))
					{
						correct++;
					}
					total++;
				}
			}
			std::cout<< "Accuracy on test set: " << correct/total << std::endl;
		}
	}

	return 0;
}