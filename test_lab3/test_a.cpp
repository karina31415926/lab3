#include "test.h"

#include "simulation/naive_cuda_simulation.cuh"
#include "simulation/constants.h"

#include "input_generator/input_generator.h"

#include <exception>
#include <iostream>

#include "test_a.cuh"

#include "utilities.h"

class Lab3Test : public LabTest {};

TEST_F(Lab3Test, test_a){
    auto universe = Universe();
    InputGenerator::create_random_universe(51342, universe);

    void* d_weights;
    void* d_forces;
    void* d_velocities;
    void* d_positions;

    NaiveCudaSimulation::allocate_device_memory(universe, &d_weights, &d_forces, &d_velocities, &d_positions);
    
    fill_and_check_device(universe, d_weights, d_forces, d_velocities, d_positions);
}


