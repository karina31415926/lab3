#pragma once
#include <vector>
#include "structures/vector2d.h"
#include "structures/universe.h"
#include <cstdint>

class InputGenerator{
public:
    static void create_random_universe(std::uint32_t bodies, Universe& universe);
    static void create_earth_orbit(Universe& universe);
    static void create_random_universe_with_supermassive_blackholes(std::uint32_t bodies, Universe& universe, std::uint32_t black_holes);
    static void create_two_body_collision(Universe& universe);
};