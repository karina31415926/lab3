#include "test.h"

#include "simulation/naive_sequential_simulation.h"
#include "simulation/constants.h"

#include "utilities/import.hpp"
#include "utilities/export.hpp"

#include <exception>
#include <iostream>

#include "utilities.h"

class SimulationTest : public LabTest {};




TEST_F(SimulationTest, test_five_a){
    // forces korrekt berechnet
    Universe original_uni;
    auto tmp = std::filesystem::path{"../test_input/test_five_universe.txt"};
    load_universe(tmp, original_uni);

    Universe uni;
    load_universe(tmp, uni);
    NaiveSequentialSimulation::calculate_forces(uni);

    // load reference
    Universe reference_uni;
    auto tmp2 = std::filesystem::path{"../test_input/test_five_universe_after_calculate_forces.txt"};
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


std::int32_t count_files_in_directory(std::filesystem::path directory){
    std::int32_t count = 0;
    for(auto entry : std::filesystem::directory_iterator(directory)){
        if(entry.is_regular_file()){
            count++;
        }
    }
    return count;
}

TEST_F(SimulationTest, test_four_d){
    Universe uni; 
    auto tmp = std::filesystem::path{"../test_input/test_five_universe.txt"};
    load_universe(tmp, uni);
    

    BoundingBox bb(-939948456012620.000000,942861260429425.875000, -914208968239933.750000,936301039007632.625000);

    auto tmp_path = std::filesystem::path{"test_four_d_plot"};
    //  cleanup potentially remaining results in directory
    std::filesystem::remove_all(tmp_path);
    std::filesystem::create_directories(tmp_path);
    Plotter plotter(bb, tmp_path, 400, 400);
    NaiveSequentialSimulation::simulate_epoch(plotter, uni, true, 1);

    // check if epoch counter is incremented 
    ASSERT_EQ(uni.current_simulation_epoch, 1);
    // check if epoch simulation results are correct
    Universe reference_uni;
    auto tmp2 = std::filesystem::path{"../test_input/test_five_universe_after_one_epoch.txt"};
    load_universe(tmp2, reference_uni);

    ASSERT_EQ(uni.num_bodies, reference_uni.num_bodies);
    for(int i = 0; i < uni.num_bodies; i++){
        ASSERT_FLOAT_EQ(round_to(uni.weights[i], 0.000001), round_to(reference_uni.weights[i], 0.000001));
        ASSERT_FLOAT_EQ(round_to(uni.forces[i][0], 0.000001), round_to(reference_uni.forces[i][0], 0.000001));
        ASSERT_FLOAT_EQ(round_to(uni.forces[i][1], 0.000001), round_to(reference_uni.forces[i][1], 0.000001));
        ASSERT_FLOAT_EQ(round_to(uni.velocities[i][0], 0.000001), round_to(reference_uni.velocities[i][0], 0.000001));
        ASSERT_FLOAT_EQ(round_to(uni.velocities[i][1], 0.000001), round_to(reference_uni.velocities[i][1], 0.000001));
        ASSERT_FLOAT_EQ(round_to(uni.positions[i][0], 0.000001), round_to(reference_uni.positions[i][0], 0.000001));
        ASSERT_FLOAT_EQ(round_to(uni.positions[i][1], 0.000001), round_to(reference_uni.positions[i][1], 0.000001));
    }
    // One plot should have been created
    ASSERT_EQ(count_files_in_directory(tmp_path), 1);
    //  cleanup created directory
    std::filesystem::remove_all(tmp_path);

    // check if plotting can be disabled fully
    std::filesystem::create_directories(tmp_path);
    Plotter plotter1(bb, tmp_path, 400, 400);
    Universe uni1;
    load_universe(tmp, uni1);
    NaiveSequentialSimulation::simulate_epoch(plotter, uni1, false, 1);
    NaiveSequentialSimulation::simulate_epoch(plotter, uni1, false, 1);
    NaiveSequentialSimulation::simulate_epoch(plotter, uni1, false, 1);
    NaiveSequentialSimulation::simulate_epoch(plotter, uni1, false, 1);
    NaiveSequentialSimulation::simulate_epoch(plotter, uni1, false, 1);
    ASSERT_EQ(count_files_in_directory(tmp_path), 0);
    std::filesystem::remove_all(tmp_path);


    // check if plotting every n epochs works
    std::filesystem::create_directories(tmp_path);
    Plotter plotter2(bb, tmp_path, 400, 400);
    Universe uni2;
    load_universe(tmp, uni2);
    // important: account for potential differences in implementation: plot before / after increasing epoch counter
    NaiveSequentialSimulation::simulate_epoch(plotter, uni1, true, 3); // plot
    NaiveSequentialSimulation::simulate_epoch(plotter, uni1, true, 3);
    NaiveSequentialSimulation::simulate_epoch(plotter, uni1, true, 3);
    NaiveSequentialSimulation::simulate_epoch(plotter, uni1, true, 3); // plot
    NaiveSequentialSimulation::simulate_epoch(plotter, uni1, true, 3);
    NaiveSequentialSimulation::simulate_epoch(plotter, uni1, true, 3);
    NaiveSequentialSimulation::simulate_epoch(plotter, uni1, true, 3); // plot
    NaiveSequentialSimulation::simulate_epoch(plotter, uni1, true, 3); 
    NaiveSequentialSimulation::simulate_epoch(plotter, uni1, true, 3);
    NaiveSequentialSimulation::simulate_epoch(plotter, uni1, true, 3); // plot
    NaiveSequentialSimulation::simulate_epoch(plotter, uni1, true, 3);
    ASSERT_EQ(count_files_in_directory(tmp_path), 4);
    std::filesystem::remove_all(tmp_path);
}

