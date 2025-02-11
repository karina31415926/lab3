#include "test.h"

#include "simulation/naive_cuda_simulation.cuh"
#include "simulation/constants.h"

#include "input_generator/input_generator.h"

#include <exception>
#include <iostream>

#include "test_m.cuh"

#include "utilities.h"

#include "reference_implementations.cuh"

class Lab3Test : public LabTest {};

TEST_F(Lab3Test, test_m){

    std::vector<std::uint8_t> raw_pixels;
    std::vector<std::uint8_t> compressed_pixels;
    std::vector<std::uint8_t> inflated_pixels;

    // check: compression of 0000 correct
    raw_pixels.clear();
    compressed_pixels.clear();
    inflated_pixels.clear();
    for(std::int32_t i = 0; i < 4096; i++){
        raw_pixels.push_back(0);
    }
    compressed_pixels.resize(raw_pixels.size()/8);

    NaiveCudaSimulation::compress_pixels(raw_pixels, compressed_pixels);

    reference_inflate_compressed_pixels(compressed_pixels, inflated_pixels);
    ASSERT_EQ(raw_pixels.size(), inflated_pixels.size());
    for(std::int32_t i = 0; i < raw_pixels.size(); i++){
        ASSERT_EQ(raw_pixels[i] == 0, inflated_pixels[i] == 0);
    }

    // check: compression of 1111 correct
    raw_pixels.clear();
    compressed_pixels.clear();
    inflated_pixels.clear();
    for(std::int32_t i = 0; i < 2048; i++){
        raw_pixels.push_back(255);
    }
    compressed_pixels.resize(raw_pixels.size()/8);

    NaiveCudaSimulation::compress_pixels(raw_pixels, compressed_pixels);

    reference_inflate_compressed_pixels(compressed_pixels, inflated_pixels);
    ASSERT_EQ(raw_pixels.size(), inflated_pixels.size());
    for(std::int32_t i = 0; i < raw_pixels.size(); i++){
        ASSERT_EQ(raw_pixels[i] == 0, inflated_pixels[i] == 0);
    }

    // check different non-0 values
    raw_pixels.clear();
    compressed_pixels.clear();
    inflated_pixels.clear();
    for(std::uint32_t i = 0; i < 2048; i++){
        raw_pixels.push_back(i%256);
    }
    compressed_pixels.resize(raw_pixels.size()/8);
    
    NaiveCudaSimulation::compress_pixels(raw_pixels, compressed_pixels);

    reference_inflate_compressed_pixels(compressed_pixels, inflated_pixels);
    ASSERT_EQ(raw_pixels.size(), inflated_pixels.size());
    for(std::int32_t i = 0; i < raw_pixels.size(); i++){
        ASSERT_EQ(raw_pixels[i] == 0, inflated_pixels[i] == 0);
    }

    // check individual positions
    raw_pixels.clear();
    compressed_pixels.clear();
    inflated_pixels.clear();

    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);

    raw_pixels.push_back(1);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);

    raw_pixels.push_back(0);
    raw_pixels.push_back(1);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);

    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(1);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);

    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(1);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);

    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(1);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);

    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(1);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);

    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(1);
    raw_pixels.push_back(0);

    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(0);
    raw_pixels.push_back(1);
        
    std::vector<std::uint8_t> original_raw_pixels;
    for(auto val : raw_pixels){
        original_raw_pixels.push_back(val);
    }
    ASSERT_EQ(raw_pixels.size(), original_raw_pixels.size());  // if this fails, this is NOT an error of the student code

    compressed_pixels.resize(raw_pixels.size()/8);

    NaiveCudaSimulation::compress_pixels(raw_pixels, compressed_pixels);

    reference_inflate_compressed_pixels(compressed_pixels, inflated_pixels);
    ASSERT_EQ(raw_pixels.size(), inflated_pixels.size());
    for(std::int32_t i = 0; i < raw_pixels.size(); i++){
        ASSERT_EQ(raw_pixels[i] == 0, inflated_pixels[i] == 0);
    }

    // check: original data not modified
    ASSERT_EQ(raw_pixels.size(), original_raw_pixels.size());
    for(std::uint32_t i = 0; i < raw_pixels.size(); i++){
        ASSERT_EQ(raw_pixels[i] == 0, original_raw_pixels[i] == 0);
    }
}

