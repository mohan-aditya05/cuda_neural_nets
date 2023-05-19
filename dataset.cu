#include "dataset.hh"
#include "utils.hh"
#include <fstream>

void MNISTDataset::read_mnist_images() {

    int rv;
    int fd;
    fd = open(fn_images.c_str(), O_RDONLY);
    assert(fd >= 0);

    int magic = read_int(fd);
    assert(magic == 0x803);

    int n_images = read_int(fd);

    int n_rows = read_int(fd);
    assert(n_rows == 28);

    int n_cols = read_int(fd);
    assert(n_cols == 28);

    number_of_batches = n_images/batch_size;

    for(int i=0; i<number_of_batches; i++)
    {
        thrust::host_vector<thrust::host_vector<float>> batch(batch_size);
        batches.push_back(batch);

        thrust::host_vector<int> batch_targets(batch_size, 0);
        targets.push_back(batch_targets);
    }

    int batch_idx=0;
    int idx=0;
    int batch_img_idx=0;
    for (int i = 0; i < n_images; i++) {
        if(i> 0 && i%batch_size==0)
        {
            batch_idx++;
            batch_img_idx = 0;
        }

        unsigned char tmp[28][28];
        rv = read(fd, tmp, 28*28); assert(rv == 28*28);
        
        idx=0;
        thrust::host_vector<float> temp_img(28*28, 0.0);
        for (int r = 0; r < 28; r++) 
        {
            for (int c = 0; c < 28; c++) 
            {
                // Make go from -1 to 1.
                temp_img[idx] = float(tmp[r][c])/127.5 - 1;
                idx++;
            }
        }
        batches[batch_idx][batch_img_idx] = temp_img;
    }

    rv = close(fd); assert(rv == 0);
}

void MNISTDataset::read_mnist_labels() {

    int rv;
    int fd;
    fd = open(fn_labels.c_str(), O_RDONLY);
    assert(fd >= 0);

    int magic = read_int(fd);
    assert(magic == 0x801);

    int n_labels = read_int(fd);
    unsigned char labels[n_labels];

    rv = read(fd, labels, n_labels); assert(rv == n_labels);

    int batch_idx=0;
    int idx=0;
    for (int i = 0; i < n_labels; i++) {
        assert(labels[i] >= 0 && labels[i] <= 9);

        if(i> 0 && i%batch_size==0)
        {
            batch_idx++;
            idx=0;
        }
        
        targets[batch_idx][idx] = static_cast<int>(labels[i]);
    }
    std::cout<< n_labels<<std::endl;
    std::cout<< targets[0].size()<<std::endl;

    rv = close(fd); assert(rv == 0);
}

MNISTDataset::MNISTDataset(size_t batch_size, const std::string &fn_images, const std::string &fn_labels):
	batch_size(batch_size), fn_images(fn_images), fn_labels(fn_labels)
{
    read_mnist_images();
    read_mnist_labels();
}

int MNISTDataset::getNumOfBatches() {
	return number_of_batches;
}

thrust::host_vector<thrust::host_vector<thrust::host_vector<float>>>& MNISTDataset::getBatches() {
	return batches;
}

thrust::host_vector<thrust::host_vector<int>>& MNISTDataset::getTargets() {
	return targets;
}

void output_pgm(const std::string &fn, thrust::host_vector<float>& img) {

    std::ofstream ofs(fn, std::fstream::out|std::fstream::trunc);

    ofs << "P2\n";
    ofs << "28 28\n";
    ofs << "255\n";
    int idx=0;
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            if (j > 0) {
                ofs << " ";
            }
            ofs << 255 - int(std::round(127.5*(img[idx] + 1)));
            idx++;
        }
        ofs << "\n";
    }
}

/* testing */
// int main()
// {
//     MNISTDataset data_obj(1, "mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte");
//     auto batches = data_obj.getBatches();
//     output_pgm("img0.pgm", batches[0]);
//     // output_pgm("img1.pgm", batches[5][4]);

//     return 0;
// }