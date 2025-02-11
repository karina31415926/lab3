#pragma once

#ifndef __global__
// Falls der NVCC die Datei inkludiert, hat dieser __global__ bereits definiert und ignoriert das hier.
// Falls der Host-Compiler die Datei inkludiert, ist dieser von __global__ verwirrt, also sagen wir ihm, das ist eigentlich leer.
#define __global__
#endif

#ifndef __device__
// Falls der NVCC die Datei inkludiert, hat dieser __device__ bereits definiert und ignoriert das hier.
// Falls der Host-Compiler die Datei inkludiert, ist dieser von __device__ verwirrt, also sagen wir ihm, das ist eigentlich leer.
#define __device__
#endif

#ifndef __host__
// Falls der NVCC die Datei inkludiert, hat dieser __host__ bereits definiert und ignoriert das hier.
// Falls der Host-Compiler die Datei inkludiert, ist dieser von __host__ verwirrt, also sagen wir ihm, das ist eigentlich leer.
#define __host__
#endif