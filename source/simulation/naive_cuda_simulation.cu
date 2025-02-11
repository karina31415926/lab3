#include "naive_cuda_simulation.cuh"
#include "physics/gravitation.h"
#include "physics/mechanics.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_wrappers.cuh"

// Define gravitational constant as device constant
__device__ const double d_gravitational_constant = 6.67430 * 1e-11;
__device__  const double d_epoch_in_seconds = 2.628e+6;

void NaiveCudaSimulation::allocate_device_memory(Universe& universe, void** d_weights, void** d_forces, void** d_velocities, void** d_positions){
    std::size_t num_bodies = universe.num_bodies;

    // Speicher für die Gewichte allokieren
    parprog_cudaMalloc(d_weights, num_bodies * sizeof(double));

    // Speicher für die Kräfte, Geschwindigkeiten und Positionen allokieren (double2 Typ)
    parprog_cudaMalloc(d_forces, num_bodies * sizeof(double2));
    parprog_cudaMalloc(d_velocities, num_bodies * sizeof(double2));
    parprog_cudaMalloc(d_positions, num_bodies * sizeof(double2));

}

void NaiveCudaSimulation::free_device_memory(void** d_weights, void** d_forces, void** d_velocities, void** d_positions) {
    parprog_cudaFree(*d_weights);
    parprog_cudaFree(*d_forces);
    parprog_cudaFree(*d_velocities);
    parprog_cudaFree(*d_positions);

    *d_weights = nullptr;
    *d_forces = nullptr;
    *d_velocities = nullptr;
    *d_positions = nullptr;
}

void NaiveCudaSimulation::copy_data_to_device(Universe& universe, void* d_weights, void* d_forces, void* d_velocities, void* d_positions){
    std::size_t num_bodies = universe.num_bodies;

    // Kopiere Gewichte
    parprog_cudaMemcpy(d_weights, universe.weights.data(), num_bodies * sizeof(double), cudaMemcpyHostToDevice);

    // Konvertiere Vector2d<double> zu double2 und kopiere Kräfte, Geschwindigkeiten und Positionen
    std::vector<double2> forces(num_bodies);
    std::vector<double2> velocities(num_bodies);
    std::vector<double2> positions(num_bodies);

    for (std::size_t i = 0; i < num_bodies; ++i) {
        forces[i] = { universe.forces[i][0], universe.forces[i][1] };
        velocities[i] = { universe.velocities[i][0], universe.velocities[i][1] };
        positions[i] = { universe.positions[i][0], universe.positions[i][1] };
    }

    parprog_cudaMemcpy(d_forces, forces.data(), num_bodies * sizeof(double2), cudaMemcpyHostToDevice);
    parprog_cudaMemcpy(d_velocities, velocities.data(), num_bodies * sizeof(double2), cudaMemcpyHostToDevice);
    parprog_cudaMemcpy(d_positions, positions.data(), num_bodies * sizeof(double2), cudaMemcpyHostToDevice);
}

void NaiveCudaSimulation::copy_data_from_device(Universe& universe, void* d_weights, void* d_forces, void* d_velocities, void* d_positions){
    std::size_t num_bodies = universe.num_bodies;

    // Kopiere Gewichte
    parprog_cudaMemcpy(universe.weights.data(), d_weights, num_bodies * sizeof(double), cudaMemcpyDeviceToHost);

    // Puffer für die Daten von der GPU
    std::vector<double2> forces(num_bodies);
    std::vector<double2> velocities(num_bodies);
    std::vector<double2> positions(num_bodies);

    parprog_cudaMemcpy(forces.data(), d_forces, num_bodies * sizeof(double2), cudaMemcpyDeviceToHost);
    parprog_cudaMemcpy(velocities.data(), d_velocities, num_bodies * sizeof(double2), cudaMemcpyDeviceToHost);
    parprog_cudaMemcpy(positions.data(), d_positions, num_bodies * sizeof(double2), cudaMemcpyDeviceToHost);

    // Konvertiere double2 zu Vector2d<double>
    for (std::size_t i = 0; i < num_bodies; ++i) {
        universe.forces[i] = { forces[i].x, forces[i].y };
        universe.velocities[i] = { velocities[i].x, velocities[i].y };
        universe.positions[i] = { positions[i].x, positions[i].y };
    }
}

__global__
void calculate_forces_kernel(std::uint32_t num_bodies, double2* d_positions, double* d_weights, double2* d_forces){

    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bodies) return;

    double2 force = { 0.0, 0.0 };
    double2 pos_i = d_positions[i];
    double weight_i = d_weights[i];

    // Erste Summe: j von 0 bis i-1
    for (std::uint32_t j = 0; j < i; ++j) {
        double2 pos_j = d_positions[j];
        double weight_j = d_weights[j];

        double dx = pos_j.x - pos_i.x;
        double dy = pos_j.y - pos_i.y;
        double dist_sq = dx * dx + dy * dy + 1e-9;  // Vermeidung von Division durch Null
        double dist = sqrt(dist_sq);

        double force_magnitude = (d_gravitational_constant * weight_i * weight_j) / dist_sq;
        force.x += force_magnitude * (dx / dist);
        force.y += force_magnitude * (dy / dist);
    }

    // Zweite Summe: j von i+1 bis num_bodies-1
    for (std::uint32_t j = i + 1; j < num_bodies; ++j) {
        double2 pos_j = d_positions[j];
        double weight_j = d_weights[j];

        double dx = pos_j.x - pos_i.x;
        double dy = pos_j.y - pos_i.y;
        double dist_sq = dx * dx + dy * dy + 1e-9;
        double dist = sqrt(dist_sq);

        double force_magnitude = (d_gravitational_constant * weight_i * weight_j) / dist_sq;
        force.x += force_magnitude * (dx / dist);
        force.y += force_magnitude * (dy / dist);
    }

    d_forces[i] = force;
}

void NaiveCudaSimulation::calculate_forces(Universe& universe, void* d_positions, void* d_weights, void* d_forces){

    std::size_t num_bodies = universe.num_bodies;
    int threads_per_block = 256;
    int num_blocks = (num_bodies + threads_per_block - 1) / threads_per_block;

    calculate_forces_kernel << <num_blocks, threads_per_block >> > (num_bodies, static_cast<double2*>(d_positions), static_cast<double*>(d_weights), static_cast<double2*>(d_forces));
    cudaDeviceSynchronize();
}

__global__
void calculate_velocities_kernel(std::uint32_t num_bodies, double2* d_forces, double* d_weights, double2* d_velocities){
    // Thread index
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don't go out of bounds
    if (i >= num_bodies) return;

    // Load the force, weight, and current velocity for the ith body
    double2 force = d_forces[i];
    double weight = d_weights[i];
    double2 velocity = d_velocities[i];

    // Compute acceleration: a = F / m
    double2 acceleration = { force.x / weight, force.y / weight };

    // Update the velocity: v = v0 + a * t
    velocity.x += acceleration.x * d_epoch_in_seconds;
    velocity.y += acceleration.y * d_epoch_in_seconds;

    // Store the updated velocity back to the device memory
    d_velocities[i] = velocity;
}

void NaiveCudaSimulation::calculate_velocities(Universe& universe, void* d_forces, void* d_weights, void* d_velocities){
    std::size_t num_bodies = universe.num_bodies;

    // Define number of threads per block and number of blocks
    int threads_per_block = 256;
    int num_blocks = (num_bodies + threads_per_block - 1) / threads_per_block;

    // Launch the kernel to compute velocities
    calculate_velocities_kernel << <num_blocks, threads_per_block >> > (num_bodies, static_cast<double2*>(d_forces), static_cast<double*>(d_weights), static_cast<double2*>(d_velocities));

    // Synchronize device to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();
}

__global__
void calculate_positions_kernel(std::uint32_t num_bodies, double2* d_velocities, double2* d_positions){
    // Thread index
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don't go out of bounds
    if (i >= num_bodies) return;

    // Load the velocity and initial position for the ith body
    double2 velocity = d_velocities[i];
    double2 position = d_positions[i];

    // Calculate the new position: p = p0 + v * t
    position.x += velocity.x * d_epoch_in_seconds;
    position.y += velocity.y * d_epoch_in_seconds;

    // Store the updated position back to the device memory
    d_positions[i] = position;
}

void NaiveCudaSimulation::calculate_positions(Universe& universe, void* d_velocities, void* d_positions){
    std::size_t num_bodies = universe.num_bodies;

    // Define number of threads per block and number of blocks
    int threads_per_block = 256;
    int num_blocks = (num_bodies + threads_per_block - 1) / threads_per_block;

    // Launch the kernel to compute positions
    calculate_positions_kernel << <num_blocks, threads_per_block >> > (num_bodies, static_cast<double2*>(d_velocities), static_cast<double2*>(d_positions));

    // Synchronize device to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();
}

void NaiveCudaSimulation::simulate_epochs(Plotter& plotter, Universe& universe, std::uint32_t num_epochs, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs){
    std::size_t num_bodies = universe.num_bodies;

    // Allocate device memory for the data structures
    void* d_weights = nullptr, * d_forces = nullptr, * d_velocities = nullptr, * d_positions = nullptr;
    allocate_device_memory(universe, &d_weights, &d_forces, &d_velocities, &d_positions);

    // Copy the initial data from host to device
    copy_data_to_device(universe, d_weights, d_forces, d_velocities, d_positions);

    // Simulate the specified number of epochs
    for (std::uint32_t epoch = 0; epoch < num_epochs; ++epoch) {
        // Call simulate_epoch to handle a single epoch simulation
        simulate_epoch(plotter, universe, create_intermediate_plots, plot_intermediate_epochs, d_weights, d_forces, d_velocities, d_positions);
    }

    // Copy the final state back from device to host
    copy_data_from_device(universe, d_weights, d_forces, d_velocities, d_positions);

    // Free the allocated device memory
    free_device_memory(&d_weights, &d_forces, &d_velocities, &d_positions);
}


__global__
void get_pixels_kernel(std::uint32_t num_bodies, double2* d_positions, std::uint8_t* d_pixels, std::uint32_t plot_width, std::uint32_t plot_height, double plot_bounding_box_x_min, double plot_bounding_box_x_max, double plot_bounding_box_y_min, double plot_bounding_box_y_max){
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bodies) return;

    // Get the position of the body
    double2 pos = d_positions[i];

    // Ensure the body is within the bounding box
    if (pos.x < plot_bounding_box_x_min || pos.x > plot_bounding_box_x_max ||
        pos.y < plot_bounding_box_y_min || pos.y > plot_bounding_box_y_max) {
        return;
    }

    // Convert position to pixel coordinates
    std::uint32_t pixel_x = (std::uint32_t)((pos.x - plot_bounding_box_x_min) /
        (plot_bounding_box_x_max - plot_bounding_box_x_min) * (plot_width - 1));
    std::uint32_t pixel_y = (std::uint32_t)((pos.y - plot_bounding_box_y_min) /
        (plot_bounding_box_y_max - plot_bounding_box_y_min) * (plot_height - 1));

    // Compute the pixel index in row-major order
    std::uint32_t pixel_index = pixel_y * plot_width + pixel_x;

    // Mark the pixel as active
    d_pixels[pixel_index] = 255;
}

std::vector<std::uint8_t> NaiveCudaSimulation::get_pixels(std::uint32_t plot_width, std::uint32_t plot_height, BoundingBox plot_bounding_box, void* d_positions, std::uint32_t num_bodies){

    // Allocate memory for pixels on the GPU
    std::uint8_t* d_pixels = nullptr;
    parprog_cudaMalloc((void**)&d_pixels, plot_width * plot_height * sizeof(std::uint8_t));

    // Launch the CUDA kernel
    int threads_per_block = 256;
    int num_blocks = (num_bodies + threads_per_block - 1) / threads_per_block;

    // Call the CUDA kernel
    get_pixels_kernel << <num_blocks, threads_per_block >> > (
        num_bodies,
        static_cast<double2*>(d_positions),
        d_pixels,
        plot_width,
        plot_height,
        plot_bounding_box.x_min,
        plot_bounding_box.x_max,
        plot_bounding_box.y_min,
        plot_bounding_box.y_max
        );

    // Synchronize after kernel execution
    cudaDeviceSynchronize();

    // Check for errors after kernel launch
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    // Copy the pixel data from device to host
    std::vector<std::uint8_t> pixels(plot_width * plot_height);
    parprog_cudaMemcpy(pixels.data(), d_pixels, plot_width * plot_height * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);

    // Free device memory
    parprog_cudaFree(d_pixels);

    // Return the pixels
    return pixels;
}

__global__
void compress_pixels_kernel(std::uint32_t num_raw_pixels, std::uint8_t* d_raw_pixels, std::uint8_t* d_compressed_pixels){
    std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx * 8 >= num_raw_pixels) return; // Ensure no out-of-bounds access

    std::uint8_t compressed_pixel = 0;

    for (int i = 0; i < 8; ++i) {
        std::uint32_t pixel_idx = idx * 8 + i;
        if (d_raw_pixels[pixel_idx] != 0) {
            compressed_pixel |= (1 << i); // Corrected bit ordering
        }
    }

    d_compressed_pixels[idx] = compressed_pixel;
}

void NaiveCudaSimulation::compress_pixels(std::vector<std::uint8_t>& raw_pixels, std::vector<std::uint8_t>& compressed_pixels){
    std::size_t num_pixels = raw_pixels.size();

    // Allocate memory for compressed pixels on the GPU
    std::uint8_t* d_raw_pixels = nullptr;
    std::uint8_t* d_compressed_pixels = nullptr;

    // Allocate memory for raw pixels and compressed pixels on the device
    parprog_cudaMalloc((void**)&d_raw_pixels, num_pixels * sizeof(std::uint8_t));
    parprog_cudaMalloc((void**)&d_compressed_pixels, num_pixels / 8 * sizeof(std::uint8_t));

    // Copy raw pixels data from host to device
    parprog_cudaMemcpy(d_raw_pixels, raw_pixels.data(), num_pixels * sizeof(std::uint8_t), cudaMemcpyHostToDevice);

    // Define CUDA kernel launch parameters
    int threads_per_block = 256;
    int num_blocks = (num_pixels / 8 + threads_per_block - 1) / threads_per_block;

    // Launch the CUDA kernel to compress the pixels
    compress_pixels_kernel << <num_blocks, threads_per_block >> > (num_pixels, d_raw_pixels, d_compressed_pixels);
    cudaDeviceSynchronize();  // Ensure the kernel finishes execution

    // Check for errors after kernel launch
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    // Copy the compressed pixels from device to host
    compressed_pixels.resize(num_pixels / 8);
    parprog_cudaMemcpy(compressed_pixels.data(), d_compressed_pixels, num_pixels / 8 * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);

    // Free device memory
    parprog_cudaFree(d_raw_pixels);
    parprog_cudaFree(d_compressed_pixels);
}

void NaiveCudaSimulation::simulate_epoch(Plotter& plotter, Universe& universe, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs, void* d_weights, void* d_forces, void* d_velocities, void* d_positions){
    calculate_forces(universe, d_positions, d_weights, d_forces);
    calculate_velocities(universe, d_forces, d_weights, d_velocities);
    calculate_positions(universe, d_velocities, d_positions);

    universe.current_simulation_epoch++;
    if(create_intermediate_plots){
        if(universe.current_simulation_epoch % plot_intermediate_epochs == 0){
            std::vector<std::uint8_t> pixels = get_pixels(plotter.get_plot_width(), plotter.get_plot_height(), plotter.get_plot_bounding_box(), d_positions, universe.num_bodies);
            plotter.add_active_pixels_to_image(pixels);

            // This is a dummy to use compression in plotting, although not beneficial performance-wise
            // ----
            // std::vector<std::uint8_t> compressed_pixels;
            // compressed_pixels.resize(pixels.size()/8);
            // compress_pixels(pixels, compressed_pixels);
            // plotter.add_compressed_pixels_to_image(compressed_pixels);
            // ----

            plotter.write_and_clear();
        }
    }
}

void NaiveCudaSimulation::calculate_forces_kernel_test_adapter(std::uint32_t grid_dim, std::uint32_t block_dim, std::uint32_t num_bodies, void* d_positions, void* d_weights, void* d_forces){
    // adapter function used by automatic tests. DO NOT MODIFY.
    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    calculate_forces_kernel<<<gridDim, blockDim>>>(num_bodies, (double2*) d_positions, (double*) d_weights, (double2*) d_forces);
}

void NaiveCudaSimulation::calculate_velocities_kernel_test_adapter(std::uint32_t grid_dim, std::uint32_t block_dim, std::uint32_t num_bodies, void* d_forces, void* d_weights, void* d_velocities){
    // adapter function used by automatic tests. DO NOT MODIFY.
    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    calculate_velocities_kernel<<<gridDim, blockDim>>>(num_bodies, (double2*) d_forces, (double*) d_weights, (double2*) d_velocities);
}

void NaiveCudaSimulation::calculate_positions_kernel_test_adapter(std::uint32_t grid_dim, std::uint32_t block_dim, std::uint32_t num_bodies, void* d_velocities, void* d_positions){
    // adapter function used by automatic tests. DO NOT MODIFY.
    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    calculate_positions_kernel<<<gridDim, blockDim>>>(num_bodies, (double2*) d_velocities, (double2*) d_positions);
}
