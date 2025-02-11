#include "test.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "structures/universe.h"

#include "input_generator/input_generator.h"

#include "test_a.cuh"


void fill_and_check_device(Universe& universe, void* d_weights, void* d_forces, void* d_velocities, void* d_positions){
    double* weights = new double[universe.num_bodies];
    double2* forces = new double2[universe.num_bodies];
    double2* velocities = new double2[universe.num_bodies];
    double2* positions = new double2[universe.num_bodies];

    for(std::uint32_t i = 0; i < universe.num_bodies; i++){
        weights[i] = (double) i;
        forces[i].x = (double)i;
        forces[i].y = (double) 2*i;
        velocities[i].x = (double) i;
        velocities[i].y = (double) 3*i;
        positions[i].x = (double) i;
        positions[i].y = (double) 4*i;
    }

    double* weights_buffer = new double[universe.num_bodies];
    double2* forces_buffer = new double2[universe.num_bodies];
    double2* velocities_buffer = new double2[universe.num_bodies];
    double2* positions_buffer = new double2[universe.num_bodies];

    for(std::uint32_t i = 0; i < universe.num_bodies; i++){
        weights_buffer[i] = 0.0;
        forces_buffer[i].x = 0.0;
        forces_buffer[i].y = 0.0;
        velocities_buffer[i].x = 0.0;
        velocities_buffer[i].y = 0.0;
        positions_buffer[i].x = 0.0;
        positions_buffer[i].y = 0.0;
    }

    cudaMemcpy(d_weights, weights, sizeof(double)*universe.num_bodies, cudaMemcpyHostToDevice);
    cudaMemcpy(d_forces, forces, sizeof(double2)*universe.num_bodies, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, velocities, sizeof(double2)*universe.num_bodies, cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, positions, sizeof(double2)*universe.num_bodies, cudaMemcpyHostToDevice);

    cudaMemcpy(weights_buffer, d_weights, sizeof(double)*universe.num_bodies, cudaMemcpyDeviceToHost);
    cudaMemcpy(forces_buffer, d_forces, sizeof(double2)*universe.num_bodies, cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities_buffer, d_velocities, sizeof(double2)*universe.num_bodies, cudaMemcpyDeviceToHost);
    cudaMemcpy(positions_buffer, d_positions, sizeof(double2)*universe.num_bodies, cudaMemcpyDeviceToHost);

    for(std::uint32_t i = 0; i < universe.num_bodies; i++){
        ASSERT_FLOAT_EQ(weights_buffer[i], weights[i]);
        ASSERT_FLOAT_EQ(forces_buffer[i].x, forces[i].x);
        ASSERT_FLOAT_EQ(forces_buffer[i].y, forces[i].y);
        ASSERT_FLOAT_EQ(velocities_buffer[i].x, velocities[i].x);
        ASSERT_FLOAT_EQ(velocities_buffer[i].y, velocities[i].y);
        ASSERT_FLOAT_EQ(positions_buffer[i].x, positions[i].x);
        ASSERT_FLOAT_EQ(positions_buffer[i].y, positions[i].y);
    }

    delete(weights);
    delete(forces);
    delete(velocities);
    delete(positions);

    delete(weights_buffer);
    delete(forces_buffer);
    delete(velocities_buffer);
    delete(positions_buffer);
}

