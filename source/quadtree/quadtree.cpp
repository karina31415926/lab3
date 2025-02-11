#include "quadtree.h"

#include "quadtreeNode.h"
#include <set>
#include <algorithm>
#include <stdexcept>
#include <omp.h>

Quadtree::Quadtree(Universe& universe, BoundingBox bounding_box, std::int8_t construct_mode){
    std::vector<std::int32_t> all_body_indices;
    for(std::uint32_t i; i < universe.num_bodies; i++){
        all_body_indices.push_back(i);
    }
    root = new QuadtreeNode(bounding_box);


    switch(construct_mode){
        case 0:
            root->children = construct(universe, bounding_box, all_body_indices);
            break;
        case 1:
            #pragma omp parallel
            {
                #pragma omp single
                #pragma omp task
                root->children = construct_task(universe, bounding_box, all_body_indices);
                #pragma omp taskwait
            }
            break;
        case 2:
            #pragma omp parallel
            {
                #pragma omp single
                #pragma omp task
                root->children = construct_task_with_cutoff(universe, bounding_box, all_body_indices);
                #pragma omp taskwait
            }
            break;

        default:
            root->children = construct(universe, bounding_box, all_body_indices);
            break;
    }
}

Quadtree::~Quadtree(){
    delete root;
}

void Quadtree::calculate_cumulative_masses(){
    root->calculate_node_cumulative_mass();
}

void Quadtree::calculate_center_of_mass(){
    root->calculate_node_center_of_mass();
}


std::vector<QuadtreeNode*> Quadtree::construct(Universe& universe, BoundingBox BB, std::vector<std::int32_t> body_indices){
    // body_indices: indices of the contained bodies 
    // returns list of children created from the current configuration
    std::vector<QuadtreeNode*> children;

    // check if a leaf node has been reached and return if so
    if(body_indices.size() == 1){
        // create vector with single entry
        QuadtreeNode* leaf_node = new QuadtreeNode(BB);

        // register body identifier
        leaf_node->body_identifier = body_indices[0];
        
        // register BB for debug plotting
        leaf_node->bounding_box = BB;

        leaf_node->center_of_mass = universe.positions[body_indices[0]];
        // mark center of mass as valid
        leaf_node->center_of_mass_ready = true;

        leaf_node->cumulative_mass = universe.weights[body_indices[0]];
        // mark cumulative mass as valid
        leaf_node->cumulative_mass_ready = true;
        children.push_back(leaf_node);
        return children;
    }

    // get bounding boxes of quadrants
    BoundingBox quadrant_0 = BB.get_quadrant(0), quadrant_1 = BB.get_quadrant(1), quadrant_2 = BB.get_quadrant(2), quadrant_3 = BB.get_quadrant(3);

    // split bodies into quadrants
    std::vector<std::int32_t> bodies_quadrant_0, bodies_quadrant_1, bodies_quadrant_2, bodies_quadrant_3;
    // split bodies into quadrants
    for (std::int32_t body_idx : body_indices){
        if(quadrant_0.contains(universe.positions[body_idx])){
            bodies_quadrant_0.push_back(body_idx);
        }
        else if(quadrant_1.contains(universe.positions[body_idx])){
            bodies_quadrant_1.push_back(body_idx);
        }
        else if(quadrant_2.contains(universe.positions[body_idx])){
            bodies_quadrant_2.push_back(body_idx);
        }
        else if(quadrant_3.contains(universe.positions[body_idx])){
            bodies_quadrant_3.push_back(body_idx);
        }
    }

    // enter recursion
    if(bodies_quadrant_0.size() > 0){
        // create child node for quadrant 0 and compute its children
        QuadtreeNode* child_0 = new QuadtreeNode(quadrant_0);
        child_0->children = construct(universe, quadrant_0, bodies_quadrant_0);
        children.push_back(child_0);
    }
    if(bodies_quadrant_1.size() > 0){
        // create child node for quadrant 1 and compute its children
        QuadtreeNode* child_1 = new QuadtreeNode(quadrant_1);
        child_1->children = construct(universe, quadrant_1, bodies_quadrant_1);
        children.push_back(child_1);
    }
    if(bodies_quadrant_2.size() > 0){
        // create child node for quadrant 2 and compute its children
        QuadtreeNode* child_2 = new QuadtreeNode(quadrant_2);
        child_2->children = construct(universe, quadrant_2, bodies_quadrant_2);
        children.push_back(child_2);
    }
    if(bodies_quadrant_3.size() > 0){
        // create child node for quadrant 3 and compute its children
        QuadtreeNode* child_3 = new QuadtreeNode(quadrant_3);
        child_3->children = construct(universe, quadrant_3, bodies_quadrant_3);
        children.push_back(child_3);
    }

    return children;
}

std::vector<QuadtreeNode*> Quadtree::construct_task(Universe& universe, BoundingBox BB, std::vector<std::int32_t> body_indices){
    // body_indices: indices of the contained bodies 
    // returns list of children created from the current configuration
    std::vector<QuadtreeNode*> children;

    // check if a leaf node has been reached and return if so
    if(body_indices.size() == 1){
        // create vector with single entry
        QuadtreeNode* leaf_node = new QuadtreeNode(BB);

        // register body identifier
        leaf_node->body_identifier = body_indices[0];
        
        // register BB for debug plotting
        leaf_node->bounding_box = BB;

        leaf_node->center_of_mass = universe.positions[body_indices[0]];
        // mark center of mass as valid
        leaf_node->center_of_mass_ready = true;

        leaf_node->cumulative_mass = universe.weights[body_indices[0]];
        // mark cumulative mass as valid
        leaf_node->cumulative_mass_ready = true;
        children.push_back(leaf_node);
        return children;
    }

    // get bounding boxes of quadrants
    BoundingBox quadrant_0 = BB.get_quadrant(0), quadrant_1 = BB.get_quadrant(1), quadrant_2 = BB.get_quadrant(2), quadrant_3 = BB.get_quadrant(3);

    // split bodies into quadrants
    std::vector<std::int32_t> bodies_quadrant_0, bodies_quadrant_1, bodies_quadrant_2, bodies_quadrant_3;
    // split bodies into quadrants
    for (std::int32_t body_idx : body_indices){
        if(quadrant_0.contains(universe.positions[body_idx])){
            bodies_quadrant_0.push_back(body_idx);
        }
        else if(quadrant_1.contains(universe.positions[body_idx])){
            bodies_quadrant_1.push_back(body_idx);
        }
        else if(quadrant_2.contains(universe.positions[body_idx])){
            bodies_quadrant_2.push_back(body_idx);
        }
        else if(quadrant_3.contains(universe.positions[body_idx])){
            bodies_quadrant_3.push_back(body_idx);
        }
    }

    // enter recursion
    std::vector<QuadtreeNode*> children_q0;
    std::vector<QuadtreeNode*> children_q1;
    std::vector<QuadtreeNode*> children_q2;
    std::vector<QuadtreeNode*> children_q3;
    if(bodies_quadrant_0.size() > 0){
        // create child node for quadrant 0 and compute its children
        #pragma omp task shared(universe, children_q0, quadrant_0, bodies_quadrant_0)
        {
            QuadtreeNode* child_0 = new QuadtreeNode(quadrant_0);
            child_0->children = construct_task_with_cutoff(universe, quadrant_0, bodies_quadrant_0);
            children_q0.push_back(child_0);
        }
    }
    if(bodies_quadrant_1.size() > 0){
        // create child node for quadrant 1 and compute its children
        #pragma omp task shared(universe, children_q1, quadrant_1, bodies_quadrant_1)
        {
            QuadtreeNode* child_1 = new QuadtreeNode(quadrant_1);
            child_1->children = construct_task_with_cutoff(universe, quadrant_1, bodies_quadrant_1);
            children_q1.push_back(child_1);
        }
    }
    if(bodies_quadrant_2.size() > 0){
        // create child node for quadrant 2 and compute its children
        #pragma omp task shared(universe, children_q2, quadrant_2, bodies_quadrant_2)
        {
            QuadtreeNode* child_2 = new QuadtreeNode(quadrant_2);
            child_2->children = construct_task_with_cutoff(universe, quadrant_2, bodies_quadrant_2);
            children_q2.push_back(child_2);
        }
    }
    if(bodies_quadrant_3.size() > 0){
        // create child node for quadrant 3 and compute its children
        #pragma omp task shared(universe, children_q3, quadrant_3, bodies_quadrant_3)
        {
            QuadtreeNode* child_3 = new QuadtreeNode(quadrant_3);
            child_3->children = construct_task_with_cutoff(universe, quadrant_3, bodies_quadrant_3);
            children_q3.push_back(child_3);
        }
    }
    #pragma omp taskwait
    // collect local results
    children.reserve(children_q0.size() + children_q1.size() + children_q2.size() + children_q3.size());
    children.insert(children.end(), children_q0.begin(), children_q0.end());
    children.insert(children.end(), children_q1.begin(), children_q1.end());
    children.insert(children.end(), children_q2.begin(), children_q2.end());
    children.insert(children.end(), children_q3.begin(), children_q3.end());

    return children;
}

std::vector<QuadtreeNode*> Quadtree::construct_task_with_cutoff(Universe& universe, BoundingBox& BB, std::vector<std::int32_t>& body_indices){
    // body_indices: indices of the contained bodies 
    // returns list of children created from the current configuration
    std::vector<QuadtreeNode*> children;

    // check if a leaf node has been reached and return if so
    if(body_indices.size() == 1){
        // create vector with single entry
        QuadtreeNode* leaf_node = new QuadtreeNode(BB);

        // register body identifier
        leaf_node->body_identifier = body_indices[0];
        
        // register BB for debug plotting
        leaf_node->bounding_box = BB;

        leaf_node->center_of_mass = universe.positions[body_indices[0]];
        // mark center of mass as valid
        leaf_node->center_of_mass_ready = true;

        leaf_node->cumulative_mass = universe.weights[body_indices[0]];
        // mark cumulative mass as valid
        leaf_node->cumulative_mass_ready = true;
        children.push_back(leaf_node);
        return children;
    }

    // get bounding boxes of quadrants
    BoundingBox quadrant_0 = BB.get_quadrant(0), quadrant_1 = BB.get_quadrant(1), quadrant_2 = BB.get_quadrant(2), quadrant_3 = BB.get_quadrant(3);

    // split bodies into quadrants
    std::vector<std::int32_t> bodies_quadrant_0, bodies_quadrant_1, bodies_quadrant_2, bodies_quadrant_3;
    // split bodies into quadrants
    for (std::int32_t body_idx : body_indices){
        if(quadrant_0.contains(universe.positions[body_idx])){
            bodies_quadrant_0.push_back(body_idx);
        }
        else if(quadrant_1.contains(universe.positions[body_idx])){
            bodies_quadrant_1.push_back(body_idx);
        }
        else if(quadrant_2.contains(universe.positions[body_idx])){
            bodies_quadrant_2.push_back(body_idx);
        }
        else if(quadrant_3.contains(universe.positions[body_idx])){
            bodies_quadrant_3.push_back(body_idx);
        }
    }

    // enter recursion
    std::vector<QuadtreeNode*> children_q0;
    std::vector<QuadtreeNode*> children_q1;
    std::vector<QuadtreeNode*> children_q2;
    std::vector<QuadtreeNode*> children_q3;
    
    if(bodies_quadrant_0.size() > 0){
        // create child node for quadrant 0 and compute its children
        #pragma omp task shared(universe, children_q0, quadrant_0, bodies_quadrant_0) final(bodies_quadrant_0.size() < 10000)
        {
            QuadtreeNode* child_0 = new QuadtreeNode(quadrant_0);
            child_0->children = construct_task_with_cutoff(universe, quadrant_0, bodies_quadrant_0);
            children_q0.push_back(child_0);
        }
    }
    if(bodies_quadrant_1.size() > 0){
        // create child node for quadrant 1 and compute its children
        #pragma omp task shared(universe, children_q1, quadrant_1, bodies_quadrant_1) final(bodies_quadrant_1.size() < 10000)
        {
            QuadtreeNode* child_1 = new QuadtreeNode(quadrant_1);
            child_1->children = construct_task_with_cutoff(universe, quadrant_1, bodies_quadrant_1);
            children_q1.push_back(child_1);
        }
    }
    
    if(bodies_quadrant_2.size() > 0){
        // create child node for quadrant 2 and compute its children
        #pragma omp task shared(universe, children_q2, quadrant_2, bodies_quadrant_2) final(bodies_quadrant_2.size() < 10000)
        {
            QuadtreeNode* child_2 = new QuadtreeNode(quadrant_2);
            child_2->children = construct_task_with_cutoff(universe, quadrant_2, bodies_quadrant_2);
            children_q2.push_back(child_2);
        }
    }
    if(bodies_quadrant_3.size() > 0){
        // create child node for quadrant 3 and compute its children
        #pragma omp task shared(universe, children_q3, quadrant_3, bodies_quadrant_3) final(bodies_quadrant_3.size() < 10000)
        {
            QuadtreeNode* child_3 = new QuadtreeNode(quadrant_3);
            child_3->children = construct_task_with_cutoff(universe, quadrant_3, bodies_quadrant_3);
            children_q3.push_back(child_3);
        }
    }
    #pragma omp taskwait
    // collect local results
    children.reserve(children_q0.size() + children_q1.size() + children_q2.size() + children_q3.size());
    children.insert(children.end(), children_q0.begin(), children_q0.end());
    children.insert(children.end(), children_q1.begin(), children_q1.end());
    children.insert(children.end(), children_q2.begin(), children_q2.end());
    children.insert(children.end(), children_q3.begin(), children_q3.end());

    return children;
}

std::vector<BoundingBox> Quadtree::get_bounding_boxes(QuadtreeNode* qtn){
    // traverse quadtree and collect bounding boxes
    std::vector<BoundingBox> result;
    // collect bounding boxes from children
    for(auto child: qtn->children){
        for(auto bb: get_bounding_boxes(child)){
            result.push_back(bb);
        }
    }
    result.push_back(qtn->bounding_box);
    return result;
}








