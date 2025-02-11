#include "plotting/plotter.h"

void Plotter::add_bodies_to_image(Universe& universe){
    // fill bitmap

    for(auto position : universe.positions){
        double position_x = position[0];
        double position_y = position[1];

        if(!plot_bounding_box.contains(position)){
            // body not within the plotted box
            continue;
        }

        // plot pixel
        mark_position(position, 255, 255, 255);
    }
}


void Plotter::add_active_pixels_to_image(std::vector<std::uint8_t>& pixels){
    // merges the passed pixels into the current image
    std::uint32_t width = plot_width;
    std::uint32_t height = plot_height;

    #pragma omp parallel for shared(pixels) firstprivate(width, height) collapse(2)
    for(std::uint32_t i = 0; i < width; i++){
        for(std::uint32_t j = 0; j < height; j++){
            if(pixels[i + j*width] != 0){
                #pragma omp critical
                {
                    mark_pixel(i, j, 255, 255, 255);
                }
            }
        }
    }
}


void Plotter::add_compressed_pixels_to_image(std::vector<std::uint8_t>& compressed_pixels){
    // merges the compressed pixels into the current image
    std::vector<std::uint8_t> inflated_pixels;

    std::uint8_t bitmask_bit_0 = 1;
    std::uint8_t bitmask_bit_1 = 2;
    std::uint8_t bitmask_bit_2 = 4;
    std::uint8_t bitmask_bit_3 = 8;
    std::uint8_t bitmask_bit_4 = 16;
    std::uint8_t bitmask_bit_5 = 32;
    std::uint8_t bitmask_bit_6 = 64;
    std::uint8_t bitmask_bit_7 = 128;

    // inflate pixels
    for(std::uint8_t compressed_pixel : compressed_pixels){
        // unpack
        std::vector<bool> px_values;
        
        px_values.push_back(compressed_pixel & bitmask_bit_0);
        px_values.push_back(compressed_pixel & bitmask_bit_1);
        px_values.push_back(compressed_pixel & bitmask_bit_2);
        px_values.push_back(compressed_pixel & bitmask_bit_3);
        px_values.push_back(compressed_pixel & bitmask_bit_4);
        px_values.push_back(compressed_pixel & bitmask_bit_5);
        px_values.push_back(compressed_pixel & bitmask_bit_6);
        px_values.push_back(compressed_pixel & bitmask_bit_7);
        
        // store
        for(bool px_val : px_values){
            if(px_val){
                inflated_pixels.push_back(255);
            }
            else{
                inflated_pixels.push_back(0);
            }    
        }
    }
    add_active_pixels_to_image(inflated_pixels);
}

