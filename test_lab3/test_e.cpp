#include "test.h"

#include "simulation/naive_cuda_simulation.cuh"
#include "simulation/constants.h"

#include "input_generator/input_generator.h"

#include <exception>
#include <iostream>

#include "test_e.cuh"

#include "utilities.h"
#include "utilities/import.hpp"

class Lab3Test : public LabTest {};

TEST_F(Lab3Test, test_e){
    // check forces correctly calculated
    Universe original_uni;
    auto tmp = std::filesystem::path{"../test_input/test_five_ppws24_D75C_universe.txt"};
    load_universe(tmp, original_uni);

    Universe uni;
    load_universe(tmp, uni);

    void* d_weights;
    void* d_forces;
    void* d_velocities;
    void* d_positions;

    NaiveCudaSimulation::allocate_device_memory(uni, &d_weights, &d_forces, &d_velocities, &d_positions);
    NaiveCudaSimulation::copy_data_to_device(uni, d_weights, d_forces, d_velocities, d_positions);

    call_calculate_forces_kernel(uni, d_positions, d_weights, d_forces);

    NaiveCudaSimulation::copy_data_from_device(uni, d_weights, d_forces, d_velocities, d_positions);

    NaiveCudaSimulation::free_device_memory(&d_weights, &d_forces, &d_velocities, &d_positions);

    // load reference
    Universe reference_uni;
    auto tmp2 = std::filesystem::path{"../test_input/test_five_ppws24_D75C_universe_after_calculate_forces.txt"};
    load_universe(tmp2, reference_uni);

    ASSERT_EQ(uni.num_bodies, reference_uni.num_bodies);
    ASSERT_EQ(uni.weights.size(), reference_uni.weights.size());
    ASSERT_EQ(uni.forces.size(), reference_uni.forces.size());
    ASSERT_EQ(uni.velocities.size(), reference_uni.velocities.size());
    ASSERT_EQ(uni.positions.size(), reference_uni.positions.size());

    for(int i = 0; i < uni.num_bodies; i++){
        ASSERT_FLOAT_EQ(round_to(uni.weights[i], 0.000001), round_to(reference_uni.weights[i], 0.000001));
        ASSERT_FLOAT_EQ(round_to(uni.forces[i][0], 0.000001), round_to(reference_uni.forces[i][0], 0.000001));
        ASSERT_FLOAT_EQ(round_to(uni.forces[i][1], 0.000001), round_to(reference_uni.forces[i][1], 0.000001));
        ASSERT_FLOAT_EQ(round_to(uni.velocities[i][0], 0.000001), round_to(reference_uni.velocities[i][0], 0.000001));
        ASSERT_FLOAT_EQ(round_to(uni.velocities[i][1], 0.000001), round_to(reference_uni.velocities[i][1], 0.000001));
        ASSERT_FLOAT_EQ(round_to(uni.positions[i][0], 0.000001), round_to(reference_uni.positions[i][0], 0.000001));
        ASSERT_FLOAT_EQ(round_to(uni.positions[i][1], 0.000001), round_to(reference_uni.positions[i][1], 0.000001));
    }
}
