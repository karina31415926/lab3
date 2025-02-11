#include "plotting/plotter.h"


void Plotter::add_quadtree_to_bitmap(Quadtree& quadtree){
    // fill bitmap with quadtree bounding boxes
    std::vector<BoundingBox> bounding_boxes = quadtree.get_bounding_boxes(quadtree.root);
    std::set<std::tuple<std::uint32_t, std::uint32_t>> bounding_box_pixels = get_bounding_box_pixels(bounding_boxes);
    for(auto pixel_pair : bounding_box_pixels){
        Pixel<std::uint8_t> green_pixel = Pixel<std::uint8_t>(0, 255, 0); 
        image.set_pixel(std::get<1>(pixel_pair), std::get<0>(pixel_pair), green_pixel);
    }

}

void Plotter::add_quadtreenode_to_bitmap(QuadtreeNode* qtn, std::uint8_t red, std::uint8_t green, std::uint8_t blue){
    // fill bitmap with quadtree bounding boxes
    std::vector<BoundingBox> bounding_boxes;
    bounding_boxes.push_back(qtn->bounding_box);
    std::set<std::tuple<std::uint32_t, std::uint32_t>> bounding_box_pixels = get_bounding_box_pixels(bounding_boxes);
    for(auto pixel_pair : bounding_box_pixels){
        Pixel<std::uint8_t> green_pixel = Pixel<std::uint8_t>(red, green, blue); 
        image.set_pixel(std::get<1>(pixel_pair), std::get<0>(pixel_pair), green_pixel);
    }
}
