#pragma once

#include <vector>
#include "structures/vector2d.h"
#include "structures/bounding_box.h"

class QuadtreeNode{
public:
    QuadtreeNode(BoundingBox arg_bounding_box);
    ~QuadtreeNode();
    double calculate_node_cumulative_mass();
    Vector2d<double> calculate_node_center_of_mass();
    std::vector<QuadtreeNode*> children;    
    Vector2d<double> center_of_mass;
    double cumulative_mass;
    std::int32_t body_identifier = -1;

    bool center_of_mass_ready = false;
    bool cumulative_mass_ready = false;

    BoundingBox bounding_box;
};