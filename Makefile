CC = nvcc
CFLAGS = -g -G

main: main.cu
	$(CC) $(CFLAGS) shape.cu dataset.cu cross_entropy.cu linear_layer.cu relu.cu softmax.cu nn_model.cu main.cu -o main

sequential:sequential.cpp
	g++ -o sequential sequential.cpp

clean:
	-rm main sequential