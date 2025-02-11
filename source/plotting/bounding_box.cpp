
#include "plotting/plotter.h"

std::set<std::tuple<std::uint32_t, std::uint32_t>> Plotter::get_bounding_box_pixels(std::vector<BoundingBox>& bounding_boxes){
    std::set<std::tuple<std::uint32_t, std::uint32_t>> result_set;

    for(BoundingBox bb: bounding_boxes){

        // restrict plot to plot_bounding_box
        double restricted_bb_x_min = bb.x_min < plot_bounding_box.x_min ? plot_bounding_box.x_min : bb.x_min;
        restricted_bb_x_min = restricted_bb_x_min > plot_bounding_box.x_max ? plot_bounding_box.x_max : restricted_bb_x_min;

        double restricted_bb_x_max = bb.x_max > plot_bounding_box.x_max ? plot_bounding_box.x_max : bb.x_max;
        restricted_bb_x_max = restricted_bb_x_max < plot_bounding_box.x_min ? plot_bounding_box.x_min : restricted_bb_x_max;

        double restricted_bb_y_min = bb.y_min < plot_bounding_box.y_min ? plot_bounding_box.y_min : bb.y_min;
        restricted_bb_y_min = restricted_bb_y_min > plot_bounding_box.y_max ? plot_bounding_box.y_max : restricted_bb_y_min;

        double restricted_bb_y_max = bb.y_max > plot_bounding_box.y_max ? plot_bounding_box.y_max : bb.y_max;
        restricted_bb_y_max = restricted_bb_y_max < plot_bounding_box.y_min ? plot_bounding_box.y_min : restricted_bb_y_max;


        // convert position to pixel by normalizing the position to [0-plot_bounding_box_x/y_max]
        std::int32_t pixel_coord_x_min = (((double)(restricted_bb_x_min - plot_bounding_box.x_min)) / (plot_bounding_box.x_max - plot_bounding_box.x_min)) * plot_width;
        std::int32_t pixel_coord_x_max = (((double)(restricted_bb_x_max - plot_bounding_box.x_min)) / (plot_bounding_box.x_max - plot_bounding_box.x_min)) * plot_width;
        std::int32_t pixel_coord_y_min = (((double)(restricted_bb_y_min - plot_bounding_box.y_min)) / (plot_bounding_box.y_max - plot_bounding_box.y_min)) * plot_height;    
        std::int32_t pixel_coord_y_max = (((double)(restricted_bb_y_max - plot_bounding_box.y_min)) / (plot_bounding_box.y_max - plot_bounding_box.y_min)) * plot_height;    

        // prevent issues with plotting due to rounding
        if(pixel_coord_x_max >= plot_width){
            pixel_coord_x_max = plot_width - 2;
        }
        if(pixel_coord_y_max >= plot_height){
            pixel_coord_y_max = plot_height - 2;
        }
        if(pixel_coord_x_min >= plot_width){
            pixel_coord_x_min = plot_width - 2;
        }
        if(pixel_coord_y_min >= plot_height){
            pixel_coord_y_min = plot_height - 2;
        }
        
        for(int i = pixel_coord_x_min; i <= pixel_coord_x_max; i++){
            // upper boundary
            result_set.insert(std::make_pair(i, pixel_coord_y_max));
            // lower boundary
            result_set.insert(std::make_pair(i, pixel_coord_y_min));
        }
        for(int i = pixel_coord_y_min; i <= pixel_coord_y_max; i++){
            // left boundary
            result_set.insert(std::make_pair(pixel_coord_x_min, i));
            // right boundary
            result_set.insert(std::make_pair(pixel_coord_x_max, i));
        }
        
    }

    return result_set;
}