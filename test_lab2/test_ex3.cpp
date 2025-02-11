#include "test.h"

#include <exception>
#include <iostream>

#include "structures/universe.h"
#include "input_generator/input_generator.h"

#include "quadtree/quadtree.h"

class Ex3Test : public LabTest {};

TEST_F(Ex3Test, test_three_c){
    // initialize
    Universe uni;
    InputGenerator::create_random_universe(1000, uni);
    BoundingBox BB = uni.get_bounding_box();
    // construct quadtree sequentially
    Quadtree qt(uni, BB, 0);
    // validate quadtree
    ASSERT_TRUE(qt.root != nullptr);
    std::vector<QuadtreeNode*> queue = {qt.root};
    std::set<QuadtreeNode*> visited;

    while(queue.size() > 0){
        auto current = queue.back();
        queue.pop_back();

        // check for cycles
        ASSERT_FALSE(visited.find(current) != visited.end());
        
        visited.insert(current);

        // max 4 children per node
        ASSERT_TRUE(current->children.size() <= 4);

        bool is_leaf_node = current->children.size() == 0;
        
        if(is_leaf_node){
            // validate leaf nodes
            ASSERT_EQ(current->children.size(), 0);
            ASSERT_FALSE(current->body_identifier == -1);
            ASSERT_TRUE(current->body_identifier < uni.num_bodies);

            std::int32_t contained_count = 0;
            for(std::int32_t body_index = 0; body_index < uni.num_bodies; body_index++){
                auto body_position = uni.positions[body_index];
                if(current->bounding_box.contains(body_position)){
                    contained_count++;
                }
            }
            ASSERT_TRUE(contained_count <= 1);
        }
        else{
            // validate regular nodes
            BoundingBox current_bb = current->bounding_box;

            // check if child BB containd in parent bounding box
            for(auto child: current->children){
                BoundingBox child_bb = child->bounding_box;
                ASSERT_TRUE(current_bb.contains(Vector2d<double>(child_bb.x_min, child_bb.y_min)));
                ASSERT_TRUE(current_bb.contains(Vector2d<double>(child_bb.x_min, child_bb.y_max)));
                ASSERT_TRUE(current_bb.contains(Vector2d<double>(child_bb.x_max, child_bb.y_min)));
                ASSERT_TRUE(current_bb.contains(Vector2d<double>(child_bb.x_max, child_bb.y_max)));
            }
            // check if body_identifier = -1
            ASSERT_EQ(current->body_identifier, -1);
        }

        // add children to queue
        for(auto child: current->children){
            queue.push_back(child);
        }
    }

}

TEST_F(Ex3Test, test_three_g){
    // initialize
    Universe uni;
    uni.forces;
    uni.weights;
    uni.positions;
    uni.velocities;

    // register body 1
    uni.forces.push_back(Vector2d<double>(0.0,0.0));
    uni.velocities.push_back(Vector2d<double>(0.0,0.0));
    uni.positions.push_back(Vector2d<double>(100.0, 100.0));
    uni.weights.push_back(100.0);
    // register body 2
    uni.forces.push_back(Vector2d<double>(0.0,0.0));
    uni.velocities.push_back(Vector2d<double>(0.0,0.0));
    uni.positions.push_back(Vector2d<double>(-100.0, -100.0));
    uni.weights.push_back(100.0);
    // register body 3
    uni.forces.push_back(Vector2d<double>(0.0,0.0));
    uni.velocities.push_back(Vector2d<double>(0.0,0.0));
    uni.positions.push_back(Vector2d<double>(100.0, -100.0));
    uni.weights.push_back(200.0);
    // register body 4
    uni.forces.push_back(Vector2d<double>(0.0,0.0));
    uni.velocities.push_back(Vector2d<double>(0.0,0.0));
    uni.positions.push_back(Vector2d<double>(-100.0, 100.0));
    uni.weights.push_back(100.0);

    uni.num_bodies = 4;

    // construct quadtree sequentially
    BoundingBox BB = uni.get_bounding_box();
    Quadtree qt(uni, BB, 0);
    qt.calculate_cumulative_masses();
    qt.calculate_center_of_mass();

    // check if all center of masses are ready

    ASSERT_TRUE(qt.root != nullptr);
    ASSERT_TRUE(qt.root->center_of_mass_ready);
    ASSERT_TRUE(qt.root->children[0]->center_of_mass_ready);
    ASSERT_TRUE(qt.root->children[1]->center_of_mass_ready);
    ASSERT_TRUE(qt.root->children[2]->center_of_mass_ready);
    ASSERT_TRUE(qt.root->children[3]->center_of_mass_ready);

    // check the calculated center of masses of the children. Account for different ordering in tree construction
    std::vector<Vector2d<double>> valid_positions;
    valid_positions.push_back(Vector2d<double>(100.0, 100.0));
    valid_positions.push_back(Vector2d<double>(-100.0, -100.0));
    valid_positions.push_back(Vector2d<double>(100.0, -100.0));
    valid_positions.push_back(Vector2d<double>(-100.0, 100.0));

    std::int32_t valid_counter = 0;
    
    for(auto child: qt.root->children){
        for(std::int32_t index = 0; index < valid_positions.size(); index++){
            if(valid_positions[index] == child->center_of_mass){
                valid_counter++;
                valid_positions.erase(valid_positions.begin() + index);
                break;
            }
        }
    }   
    ASSERT_EQ(valid_counter, 4);

    // check the calculated center of mass of the root node
    ASSERT_EQ(qt.root->center_of_mass, Vector2d<double>(20.0, -20.0));
}



