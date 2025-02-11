#include "test.h"

#include <exception>
#include <iostream>

#include "structures/universe.h"
#include "utilities/import.hpp"
#include "simulation/naive_parallel_simulation.h"
#include "simulation/naive_sequential_simulation.h"

#include "utilities.h"


class Ex2Test : public LabTest {};

TEST_F(Ex2Test, test_two_b){

    // check if velocities are updated
    Universe original_uni;
    auto tmp3 = std::filesystem::path{"../test_input_grading/test_five_ppws24_D75C_universe_after_calculate_forces.txt"};
    load_universe(tmp3, original_uni);

    Universe uni3;
    load_universe(tmp3, uni3);
    NaiveParallelSimulation::calculate_velocities(uni3);

    ASSERT_EQ(original_uni.num_bodies, uni3.num_bodies);
    for(int i = 0; i < original_uni.num_bodies; i++){
        ASSERT_NE(round_to(uni3.velocities[i][0], 0.000001), round_to(original_uni.velocities[i][0], 0.000001));
        ASSERT_NE(round_to(uni3.velocities[i][1], 0.000001), round_to(original_uni.velocities[i][1], 0.000001));
    }

    // check for correct velocities if no forces are applied
    Universe uni2;
    load_universe(tmp3, uni2);
    for(int i = 0; i < uni2.num_bodies; i++){
        uni2.forces[i].set(0, 0);
    }
    NaiveParallelSimulation::calculate_velocities(uni2);
    for(int i = 0; i < uni2.num_bodies; i++){
        ASSERT_FLOAT_EQ(round_to(uni2.velocities[i][0], 0.000001), round_to(original_uni.velocities[i][0], 0.000001));
        ASSERT_FLOAT_EQ(round_to(uni2.velocities[i][1], 0.000001), round_to(original_uni.velocities[i][1], 0.000001));
    }

    // check for correct velocities if forces are applied
    Universe uni;
    auto tmp = std::filesystem::path{"../test_input_grading/test_five_ppws24_D75C_universe_after_calculate_forces.txt"};
    load_universe(tmp, uni);

    NaiveParallelSimulation::calculate_velocities(uni);

    Universe reference_uni;
    auto tmp2 = std::filesystem::path{"../test_input_grading/test_five_ppws24_D75C_universe_after_calculate_velocities.txt"};
    load_universe(tmp2, reference_uni);

    ASSERT_EQ(uni.num_bodies, reference_uni.num_bodies);
    for(int i = 0; i < uni.num_bodies; i++){
        ASSERT_FLOAT_EQ(round_to(uni.velocities[i][0], 0.000001), round_to(reference_uni.velocities[i][0], 0.000001));
        ASSERT_FLOAT_EQ(round_to(uni.velocities[i][1], 0.000001), round_to(reference_uni.velocities[i][1], 0.000001));
    }

    // check for consistency of other data
    ASSERT_EQ(uni.num_bodies, original_uni.num_bodies);
    ASSERT_EQ(uni.weights.size(), original_uni.weights.size());
    ASSERT_EQ(uni.forces.size(), original_uni.forces.size());
    ASSERT_EQ(uni.velocities.size(), original_uni.velocities.size());
    ASSERT_EQ(uni.positions.size(), original_uni.positions.size());

    for(int i = 0; i < uni.num_bodies; i++){
        ASSERT_FLOAT_EQ(round_to(uni.weights[i], 0.000001), round_to(original_uni.weights[i], 0.000001));
        ASSERT_FLOAT_EQ(round_to(uni.forces[i][0], 0.000001), round_to(original_uni.forces[i][0], 0.000001));
        ASSERT_FLOAT_EQ(round_to(uni.forces[i][1], 0.000001), round_to(original_uni.forces[i][1], 0.000001));
        ASSERT_FLOAT_EQ(round_to(uni.positions[i][0], 0.000001), round_to(original_uni.positions[i][0], 0.000001));
        ASSERT_FLOAT_EQ(round_to(uni.positions[i][1], 0.000001), round_to(original_uni.positions[i][1], 0.000001));
    }

}



