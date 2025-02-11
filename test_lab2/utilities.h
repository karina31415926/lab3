#pragma once

#include <cmath>

inline double round_to(double value, double precision = 1.0)
{
    return std::round(value / precision) * precision;
}