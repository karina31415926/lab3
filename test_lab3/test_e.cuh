#include "structures/universe.h"

void call_calculate_forces_kernel(Universe& universe, void* d_positions, void* d_weights, void* d_forces);