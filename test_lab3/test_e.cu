#include "test.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "structures/universe.h"

#include "input_generator/input_generator.h"

#include "test_e.cuh"

#include "reference_implementations.cuh"

#include "simulation/naive_cuda_simulation.cuh"

void call_calculate_forces_kernel(Universe& universe, void* d_positions, void* d_weights, void* d_forces){
    // launch at least universe.num_bodies threads in total
    // calculate grid dimensions
    std::uint32_t block_dim = 512;
    std::uint32_t grid_dim;
    
    if(universe.num_bodies % block_dim == 0){
        grid_dim = universe.num_bodies / block_dim;
    }
    else{
        grid_dim = (universe.num_bodies - (universe.num_bodies % block_dim) + block_dim) / block_dim;
    }
     

    NaiveCudaSimulation::calculate_forces_kernel_test_adapter(grid_dim, block_dim, universe.num_bodies, (double2*) d_positions, (double*) d_weights, (double2*) d_forces);
    // wait for kernel to finish
    cudaDeviceSynchronize();
}