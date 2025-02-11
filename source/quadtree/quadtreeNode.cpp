#include "quadtreeNode.h"

#include <iostream>


double QuadtreeNode::calculate_node_cumulative_mass(){
    // calculates, stores, and returns the cumulative mass of the current node by recursively summing the cumulative masses of it's children
    
    // check if a leaf node is reached
    if(cumulative_mass_ready){
        // leaf node reached
        return cumulative_mass; 
    }

    // enter recursion and sum cumulative masses
    double tmp_cum_mass = 0;
    for(QuadtreeNode* child : children){
        tmp_cum_mass += child->calculate_node_cumulative_mass();
    }
    cumulative_mass = tmp_cum_mass;
    // mark cumulative mass as valid
    cumulative_mass_ready = true;
    return tmp_cum_mass;
}

QuadtreeNode::QuadtreeNode(BoundingBox arg_bounding_box){
    bounding_box = arg_bounding_box;
}

QuadtreeNode::~QuadtreeNode(){
    // delete children before deleting the node
    for(QuadtreeNode* child : children){
        delete child;
    }
}

Vector2d<double> QuadtreeNode::calculate_node_center_of_mass(){
    // calculates, stores, and returns the center of mass of the current node by recursively calculating the centers of masses of it's children

    // check if a leaf node is reached
    if(center_of_mass_ready){
        // leaf node reached
        return center_of_mass; 
    }

    // enter recursion and calculate the summed weighted positions
    Vector2d<double> summed_positions;
    for(QuadtreeNode* child : children){
        if(!child->cumulative_mass_ready){
            child->calculate_node_cumulative_mass();
        }
        summed_positions = summed_positions + child->calculate_node_center_of_mass()*child->cumulative_mass;
    }

    // center of mass: summed weighted positions / cumulative mass
    if(!cumulative_mass_ready){
        calculate_node_cumulative_mass();
    }
    center_of_mass = summed_positions / cumulative_mass;
    // mark center of mass as valid
    center_of_mass_ready = true;

    return center_of_mass;

}