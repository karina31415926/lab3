#include "test.h"

#include "simulation/naive_cuda_simulation.cuh"
#include "simulation/constants.h"

#include "input_generator/input_generator.h"

#include <exception>
#include <iostream>

#include "test_c.cuh"

#include "utilities.h"

#include "test_utilities.cuh"

class Lab3Test : public LabTest {};

TEST_F(Lab3Test, test_c){
    // check: data converted, moved, and stored according to specification
    auto universe = Universe();
    InputGenerator::create_random_universe(1337, universe);

    auto universe_reference = Universe();
    InputGenerator::create_random_universe(1337, universe_reference);
    for(std::int32_t i = 0; i < universe.num_bodies; i++){
        universe_reference.weights[i] = universe.weights[i];
        universe_reference.forces[i].set(universe.forces[i][0], universe.forces[i][1]);
        universe_reference.positions[i].set(universe.positions[i][0], universe.positions[i][1]);
        universe_reference.velocities[i].set(universe.velocities[i][0], universe.velocities[i][1]);
    }

    void* d_weights;
    void* d_forces;
    void* d_velocities;
    void* d_positions;

    void* d_weights_reference;
    void* d_forces_reference;
    void* d_velocities_reference;
    void* d_positions_reference;

    NaiveCudaSimulation::allocate_device_memory(universe_reference, &d_weights_reference, &d_forces_reference, &d_velocities_reference, &d_positions_reference);

    reference_copy_data_to_device(universe_reference, d_weights_reference, d_forces_reference, d_velocities_reference, d_positions_reference);

    NaiveCudaSimulation::allocate_device_memory(universe, &d_weights, &d_forces, &d_velocities, &d_positions);
    NaiveCudaSimulation::copy_data_to_device(universe, d_weights, d_forces, d_velocities, d_positions);

    ASSERT_TRUE(compare_device_doublevector(d_weights, d_weights_reference, universe.num_bodies));
    ASSERT_TRUE(compare_device_double2vector(d_forces, d_forces_reference, universe.num_bodies));
    ASSERT_TRUE(compare_device_double2vector(d_velocities, d_velocities_reference, universe.num_bodies));
    ASSERT_TRUE(compare_device_double2vector(d_positions, d_positions_reference, universe.num_bodies));

    NaiveCudaSimulation::free_device_memory(&d_weights_reference, &d_forces_reference, &d_velocities_reference, &d_positions_reference);
    NaiveCudaSimulation::free_device_memory(&d_weights, &d_forces, &d_velocities, &d_positions);
}
