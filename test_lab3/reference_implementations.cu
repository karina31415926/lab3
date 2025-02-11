#include <cuda.h>
#include <cuda_runtime_api.h>

#include "reference_implementations.cuh"

#include "simulation/naive_cuda_simulation.cuh"

void reference_inflate_compressed_pixels(std::vector<std::uint8_t>& compressed_pixels, std::vector<std::uint8_t>& inflated_pixels){
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
}
