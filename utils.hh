#pragma once

#include <fcntl.h>
#include <unistd.h>
#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

void
swap(int &i) {
    // Some of the & are superfluous.
    i =
     (0xff&(i >> 24)) |
     (0xff00&(i >> 8)) |
     (0xff0000&(i << 8)) |
     (0xff000000&(i << 24));
}

int
read_int(int fd) {
    int rv;
    int i;
    rv = read(fd, &i, 4); assert(rv == 4);
    swap(i);
    return i;
}

// thrust::device_vector<float>
// flatten(thrust::host_vector<thrust::host_vector<float>> vec, int& C)
// {
//     thrust::host_vector<float> temp_vec;
//     for(int i=0; i<vec.size(); i++)
//     {
//         for(int j=0; j<vec[0].size(); j++)
//         {
//             temp_vec.push_back(vec[i][j]);
//             C = max(C, vec[i][j]);
//         }
//     }

//     thrust::device_vector<float> temp_device_vec = temp_vec;
//     return temp_device_vec;
// }
