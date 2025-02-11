#pragma once

#include "structures/vector2d.h"

template <typename T>
[[nodiscard]] static Vector2d<T> calculate_acceleration(Vector2d<T> applied_force, double mass){
    // calculate acceleration 
    // a = F / m
    return applied_force / mass;
}

template <typename T> 
[[nodiscard]] static Vector2d<T> calculate_velocity(Vector2d<T> base_velocity, Vector2d<T> acceleration, double time_in_seconds){
    // v = v0 + a * t
    return base_velocity + (acceleration * time_in_seconds);
}