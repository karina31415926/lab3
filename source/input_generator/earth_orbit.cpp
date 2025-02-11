#include "input_generator.h"
#include <ctime>
#include <cstdint>
#include <random>
#include <iostream>

void InputGenerator::create_earth_orbit(Universe& universe){
std::srand(std::time(nullptr));  // seed random number generator with the current time

    std::cout << "Creating earth orbit" << std::endl;
    std::uint32_t bodies = 2;

    universe.num_bodies = bodies;

    // reserve space in the vectors 
    universe.weights.resize(bodies);
    universe.velocities.resize(bodies);
    universe.positions.resize(bodies);
    universe.forces.resize(bodies);

    // define sun
    universe.weights[0] = 1.989 * 1e30;  // kg

    Vector2d<double> sun_initial_force;
    sun_initial_force.set(0,0);
    universe.forces[0] = sun_initial_force;

    Vector2d<double> sun_initial_velocity;
    sun_initial_velocity.set(0,0);
    universe.velocities[0] = sun_initial_velocity;

    Vector2d<double> sun_initial_position;
    sun_initial_position.set(0,0);
    universe.positions[0] = sun_initial_position;

    // define earth
    universe.weights[1] = 5.972 * 1e24;  // kg

    Vector2d<double> earth_initial_force;
    earth_initial_force.set(0,0);
    universe.forces[1] = earth_initial_force;

    Vector2d<double> earth_initial_velocity;
    earth_initial_velocity.set(29784.8 ,0);  // m/s
    universe.velocities[1] = earth_initial_velocity;

    Vector2d<double> earth_initial_position;
    earth_initial_position.set(0, 1.496*1e11);  // m
    universe.positions[1] = earth_initial_position;

}