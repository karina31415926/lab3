#pragma once

#include <cmath>

static const double gravitational_constant = 6.67430*1e-11; // (m^3)/(kg*s^2)

[[nodiscard]] static double gravitational_force(double mass_1,  double mass_2, double distance){
    return gravitational_constant * ((mass_1 * mass_2)/(pow(distance, 2)));
}   

