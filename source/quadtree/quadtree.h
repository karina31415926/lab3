#pragma once

#include "structures/vector2d.h"
#include "structures/universe.h"
#include "quadtreeNode.h"

class Quadtree{
public: 
    Quadtree(Universe& universe, BoundingBox bounding_box, std::int8_t construct_mode);
    ~Quadtree();

    std::vector<QuadtreeNode*> construct(Universe& universe, BoundingBox BB, std::vector<std::int32_t> body_indices);
    std::vector<QuadtreeNode*> construct_task(Universe& universe, BoundingBox BB, std::vector<std::int32_t> body_indices);
    std::vector<QuadtreeNode*> construct_task_with_cutoff(Universe& universe, BoundingBox& BB, std::vector<std::int32_t>& body_indices);

    void calculate_cumulative_masses();
    void calculate_center_of_mass();
    QuadtreeNode* root = nullptr;

    std::vector<BoundingBox> get_bounding_boxes(QuadtreeNode* qtn);
};