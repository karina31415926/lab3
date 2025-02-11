#include "test.h"

#include "structures/universe.h"
#include "input_generator/input_generator.h"
#include "plotting/plotter.h"

#include <exception>
#include <iostream>
#include <filesystem>

class PlottingTest : public LabTest {};

TEST_F(PlottingTest, test_three_b){
    // test mark pixel
    BoundingBox bb(-10, 10, -10, 10);
    auto tmp_path = std::filesystem::path{"test_three_b_plot"};

    // test mark position
    Plotter plotter(bb, tmp_path, 800, 800);
    plotter.mark_position(Vector2d<double>(-10, -10), 255, 255, 255);
    plotter.mark_position(Vector2d<double>(-10, 10), 255, 255, 255);
    plotter.mark_position(Vector2d<double>(10, -10), 255, 255, 255);
    plotter.mark_position(Vector2d<double>(0, 0), 255, 255, 255);

    for(int x = 0; x < 800; x++){
        for(int y = 0; y < 800; y++){
            auto px = plotter.get_pixel(x, y);
            if((x == 0 && y == 0) || (x == 0 && y == 799) || (x == 799 && y == 0) || (x == 399 && y == 399) ){
                ASSERT_EQ(px.get_blue_channel(), 255);
                ASSERT_EQ(px.get_green_channel(), 255);
                ASSERT_EQ(px.get_red_channel(), 255);
            }
            else{
                ASSERT_EQ(px.get_blue_channel(), 0);
                ASSERT_EQ(px.get_green_channel(), 0);
                ASSERT_EQ(px.get_red_channel(), 0);
            }
        }
    }
}
