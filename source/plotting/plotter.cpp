#include "plotting/plotter.h"
#include "io/image_parser.h"

#include <exception>

void Plotter::highlight_position(Vector2d<double> position, std::uint8_t red, std::uint8_t green, std::uint8_t blue){
    // check if position inside plot_bounding_box
    if(!plot_bounding_box.contains(position)){
        // do nothing
        return;
    }

    // convert center position to pixel 
    std::uint32_t pixel_coord_x = ((position[0] - plot_bounding_box.x_min) / (plot_bounding_box.x_max - plot_bounding_box.x_min)) * (plot_width-1);
    std::uint32_t pixel_coord_y = ((position[1] - plot_bounding_box.y_min) / (plot_bounding_box.y_max - plot_bounding_box.y_min)) * (plot_height-1);

    // draw red vertical line
    for(int idx_y = 0; idx_y < plot_height; idx_y++){
        // create a red pixel
        mark_pixel(pixel_coord_x, idx_y, red, green, blue);
    }

    // draw red horizontal line
    for(int idx_x = 0; idx_x < plot_width; idx_x++){
        // create a red pixel
        mark_pixel(idx_x, pixel_coord_y, red, green, blue );
    }
}

void Plotter::write_and_clear(){
    // create plot serial number string
    std::string serial_number_string = std::to_string(image_serial_number);
    while(serial_number_string.length() < 9){
        serial_number_string = "0" + serial_number_string;
    }

    std::string file_name = filename_prefix + "_" + serial_number_string + ".bmp";
    ImageParser::write_bitmap(output_folder_path / file_name, image);
    clear_image();
    image_serial_number += 1;
}

void Plotter::mark_position(Vector2d<double> position, std::uint8_t red, std::uint8_t green, std::uint8_t blue){
    // check if position inside plot_bounding_box
    if(!plot_bounding_box.contains(position)){
        // do nothing
        return;
    }

    // convert position to pixel 
    std::uint32_t pixel_coord_x = ((position[0] - plot_bounding_box.x_min) / (plot_bounding_box.x_max - plot_bounding_box.x_min)) * (plot_width-1);
    std::uint32_t pixel_coord_y = ((position[1] - plot_bounding_box.y_min) / (plot_bounding_box.y_max - plot_bounding_box.y_min)) * (plot_height-1);

    mark_pixel(pixel_coord_x, pixel_coord_y, red, green, blue);
}

void Plotter::mark_pixel(std::uint32_t x, std::uint32_t y, std::uint8_t red, std::uint8_t green, std::uint8_t blue){
    // check for bounds 
    if((x >= plot_width) || (y >= plot_height)){
        throw std::invalid_argument("pixel out of bounds!");
    }
    // set the pixel to the specified color
    Pixel<std::uint8_t> pixel = Pixel<std::uint8_t>(red, green, blue);
    image.set_pixel(y, x, pixel);
}

BitmapImage::BitmapPixel Plotter::get_pixel(std::uint32_t x, std::uint32_t y){
    return image.get_pixel(y, x);
}