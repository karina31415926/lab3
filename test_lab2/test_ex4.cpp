#include "test.h"

#include <exception>
#include <iostream>

#include "structures/universe.h"
#include "utilities/import.hpp"

#include "quadtree/quadtree.h"

#include "simulation/barnes_hut_simulation.h"

class Ex4Test : public LabTest {};

bool legacy_test_solution(){
    Universe uni;
    auto tmp = std::filesystem::path{"../test_input_grading/test_five_ppws24_D75C_universe.txt"};
    load_universe(tmp, uni);

    BoundingBox bb = uni.get_bounding_box();
    Quadtree qt(uni, bb, 0);

    std::vector<QuadtreeNode*> relevant_nodes;
    // select body 0
    std::int32_t body_index = 0;
    auto body_position = uni.positions[body_index];
    double threshold_theta;

    bool test_result = true;

    threshold_theta = 0.1;
    relevant_nodes.clear();
    BarnesHutSimulation::get_relevant_nodes(uni, qt, relevant_nodes, body_position, body_index, threshold_theta);
    test_result = test_result && (relevant_nodes.size() == 97);
    

    threshold_theta = 0.2;
    relevant_nodes.clear();
    BarnesHutSimulation::get_relevant_nodes(uni, qt, relevant_nodes, body_position, body_index, threshold_theta);
    test_result = test_result && (relevant_nodes.size() == 93);

    threshold_theta = 0.3;
    relevant_nodes.clear();
    BarnesHutSimulation::get_relevant_nodes(uni, qt, relevant_nodes, body_position, body_index, threshold_theta);
    test_result = test_result && (relevant_nodes.size() == 81);

    threshold_theta = 0.4;
    relevant_nodes.clear();
    BarnesHutSimulation::get_relevant_nodes(uni, qt, relevant_nodes, body_position, body_index, threshold_theta);
    test_result = test_result && (relevant_nodes.size() == 81);
    return test_result;
}

bool fixed_test_solution(){
    Universe uni;
    auto tmp = std::filesystem::path{"../test_input_grading/test_five_ppws24_D75C_universe.txt"};
    load_universe(tmp, uni);

    BoundingBox bb = uni.get_bounding_box();
    Quadtree qt(uni, bb, 0);

    std::vector<QuadtreeNode*> relevant_nodes;
    // select body 0
    std::int32_t body_index = 0;
    auto body_position = uni.positions[body_index];
    double threshold_theta;

    bool test_result = true;

    threshold_theta = 0.1;
    relevant_nodes.clear();
    BarnesHutSimulation::get_relevant_nodes(uni, qt, relevant_nodes, body_position, body_index, threshold_theta);
    test_result = test_result && (relevant_nodes.size() == 96);
    

    threshold_theta = 0.2;
    relevant_nodes.clear();
    BarnesHutSimulation::get_relevant_nodes(uni, qt, relevant_nodes, body_position, body_index, threshold_theta);
    test_result = test_result && (relevant_nodes.size() == 84);

    threshold_theta = 0.3;
    relevant_nodes.clear();
    BarnesHutSimulation::get_relevant_nodes(uni, qt, relevant_nodes, body_position, body_index, threshold_theta);
    test_result = test_result && (relevant_nodes.size() == 66);

    threshold_theta = 0.4;
    relevant_nodes.clear();
    BarnesHutSimulation::get_relevant_nodes(uni, qt, relevant_nodes, body_position, body_index, threshold_theta);
    test_result = test_result && (relevant_nodes.size() == 60);
    return test_result;
}

TEST_F(Ex4Test, test_four_b){
    ASSERT_TRUE( legacy_test_solution() || fixed_test_solution());
}




