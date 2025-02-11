#include <cuda.h>
#include <cuda_runtime_api.h>

#include "test.cuh"


void get_device_count(std::int32_t* device_count){  
    cudaGetDeviceCount(device_count);
}