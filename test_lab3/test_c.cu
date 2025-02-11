#include "test.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "structures/universe.h"

#include "input_generator/input_generator.h"

#include "test_c.cuh"

void reference_copy_data_to_device(Universe& universe, void*  d_weights_reference, void* d_forces_reference, void* d_velocities_reference, void* d_positions_reference){
    for(std::uint32_t i = 0; i < universe.num_bodies; i++){
            cudaMemcpy(&(((double*)d_weights_reference)[i]), &(universe.weights[i]), sizeof(double), cudaMemcpyHostToDevice);    
            double2 dummy_force = make_double2(universe.forces[i][0], universe.forces[i][1]);
            cudaMemcpy(&(((double2*)d_forces_reference)[i]), &dummy_force, sizeof(double2), cudaMemcpyHostToDevice);
            double2 dummy_velocity = make_double2(universe.velocities[i][0], universe.velocities[i][1]);
            cudaMemcpy(&(((double2*)d_velocities_reference)[i]), &dummy_velocity, sizeof(double2), cudaMemcpyHostToDevice);
            double2 dummy_position = make_double2(universe.positions[i][0], universe.positions[i][1]);
            cudaMemcpy(&(((double2*)d_positions_reference)[i]), &dummy_position, sizeof(double2), cudaMemcpyHostToDevice);
        }
}