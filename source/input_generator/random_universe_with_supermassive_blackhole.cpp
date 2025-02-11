#include "input_generator.h"
#include <ctime>
#include <cstdint>
#include <random>
#include <iostream>

void InputGenerator::create_random_universe_with_supermassive_blackholes(std::uint32_t bodies, Universe& universe, std::uint32_t black_holes){
    std::srand(std::time(nullptr));  // seed random number generator with the current time

    //std::cout << "Creating universe randomly with " << bodies << " bodies." << std::endl;

    universe.num_bodies = bodies;

    if(bodies == 0){
        return;
    }

    // reserve space in the vectors 
    universe.weights.resize(bodies);
    universe.velocities.resize(bodies);
    universe.positions.resize(bodies);
    universe.forces.resize(bodies);

    for(int i = 0; i < bodies; i++){
        // generate random weights roughly between the mass of the black hole in the milky way and merkur

        double mantissa = (double)rand() / RAND_MAX;
        int exponent = (std::rand() % 13) + 20;

        double random_weight = mantissa * std::pow(10, exponent);

        universe.weights[i] = random_weight;  // kg

        // set all forces to 0 initially
        Vector2d<double> initial_force;
        initial_force.set(0,0);
        universe.forces[i] = initial_force;

        // generate random velocities similar to that of the earth
        Vector2d<double> initial_velocity;
        std::uint64_t max_velocity = 30000; 

        double rand_velocity_x = ((double)rand() / RAND_MAX) * max_velocity;
        double rand_velocity_y = ((double)rand() / RAND_MAX) * max_velocity;
        if(((double)rand() / RAND_MAX) > 0.5){
            rand_velocity_x = rand_velocity_x * (-1);  // flip direction
        }
        if(((double)rand() / RAND_MAX) > 0.5){
            rand_velocity_y = rand_velocity_y * (-1);  // flip direction
        }

        initial_velocity.set(rand_velocity_x, rand_velocity_y);
        universe.velocities[i] = initial_velocity;

        // generate random positions. Use a square of size 0.1ly
        double max_universe_radius = 9.46*1e14;  // m
        Vector2d<double> initial_position;
        double rand_position_x = ((double)rand() / RAND_MAX) * max_universe_radius;
        double rand_position_y = ((double)rand() / RAND_MAX) * max_universe_radius;
        if(((double)rand() / RAND_MAX) > 0.5){
            rand_position_x = rand_position_x * (-1);  // flip direction
        }
        if(((double)rand() / RAND_MAX) > 0.5){
            rand_position_y = rand_position_y * (-1);  // flip direction
        }

        initial_position.set(rand_position_x, rand_position_y);
        universe.positions[i] = initial_position;
    }

    // create supermassive black holes
    for(std::uint32_t i = 0; i < black_holes; i++){
        // set weight of body i to the weight of Sagittarius A*
        universe.weights[i] = 8.54*std::pow(10, 36);
        // increase movement speed for more interesting simulations
        universe.velocities[i] = universe.velocities[i] * 40;
    }
    
}