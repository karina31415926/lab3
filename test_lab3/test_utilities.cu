#include <cuda.h>
#include <cuda_runtime_api.h>

#include "test_utilities.cuh"

#include <cstdint>

bool compare_device_double(void* d_double_ptr1, void* d_double_ptr2){
    double buffer_1 = 0.0;
    double buffer_2 = 0.0;

    cudaMemcpy(&buffer_1, d_double_ptr1, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&buffer_2, d_double_ptr2, sizeof(double), cudaMemcpyDeviceToHost);    

    return buffer_1 == buffer_2;
}

bool compare_device_doublevector(void* d_ptr1, void* d_ptr2, size_t size){
    double* buffer_1 = (double*) malloc(sizeof(double)*size);
    double* buffer_2 = (double*) malloc(sizeof(double)*size);

    cudaMemcpy(buffer_1, d_ptr1, sizeof(double)*size, cudaMemcpyDeviceToHost);
    cudaMemcpy(buffer_2, d_ptr2, sizeof(double)*size, cudaMemcpyDeviceToHost);

    bool ret_val = true;

    for(std::int32_t i = 0; i < size; i++){
        if(buffer_1[i] != buffer_2[i]){
            ret_val = false;
            break;
        }
    }

    free(buffer_1);
    free(buffer_2);

    return ret_val;
}

bool compare_device_double2vector(void* d_ptr1, void* d_ptr2, size_t size){
    double2* buffer_1 = (double2*) malloc(sizeof(double2)*size);
    double2* buffer_2 = (double2*) malloc(sizeof(double2)*size);

    cudaMemcpy(buffer_1, d_ptr1, sizeof(double2)*size, cudaMemcpyDeviceToHost);
    cudaMemcpy(buffer_2, d_ptr2, sizeof(double2)*size, cudaMemcpyDeviceToHost);    

    bool ret_val = true;

    for(std::int32_t i = 0; i < size; i++){
        if((buffer_1[i].x != buffer_2[i].x) || (buffer_2[i].y != buffer_2[i].y)){
            ret_val = false;
            break;
        }
    }

    free(buffer_1);
    free(buffer_2);

    return ret_val;
}