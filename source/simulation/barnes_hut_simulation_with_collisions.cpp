#include "simulation/barnes_hut_simulation_with_collisions.h"
#include "simulation/barnes_hut_simulation.h"
#include "simulation/naive_parallel_simulation.h"
#include <omp.h>

void BarnesHutSimulationWithCollisions::simulate_epochs(Plotter& plotter, Universe& universe, std::uint32_t num_epochs, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs){
    for(int i = 0; i < num_epochs; i++){
        simulate_epoch(plotter, universe, create_intermediate_plots, plot_intermediate_epochs);
    }
}

void BarnesHutSimulationWithCollisions::simulate_epoch(Plotter& plotter, Universe& universe, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs){
    // calculate bounding box
    BoundingBox bounding_box = universe.parallel_cpu_get_bounding_box();

    // create quadtree
    Quadtree quadtree = Quadtree(universe, bounding_box, 2);
    // calculate cumulated masses of quadtree nodes
    quadtree.calculate_cumulative_masses();
    // calculate centers of masses of quadtree nodes
    quadtree.calculate_center_of_mass();

    BarnesHutSimulation::calculate_forces(universe, quadtree);
    NaiveParallelSimulation::calculate_velocities(universe);
    NaiveParallelSimulation::calculate_positions(universe);

    // find collisions 
    find_collisions_parallel(universe);

    universe.current_simulation_epoch++;
    if(create_intermediate_plots){
        if(universe.current_simulation_epoch % plot_intermediate_epochs == 0){
            plotter.add_bodies_to_image(universe);
            plotter.write_and_clear();
        }
    }
}

void BarnesHutSimulationWithCollisions::find_collisions(Universe& universe){
    // collision, if distance between two bodies < 100,000,000km  (~distance earth / sun)

    // NOTE: this is a naive implementation. More sophisticated solutions are conceivable!
    // NOTE: Results may become non-deterministic and may depend on the implementation!

    // step 1: determine collisions
    std::vector<std::tuple<std::uint32_t, std::uint32_t>> collisions;
    for(std::uint32_t body_index_0 = 0; body_index_0 < universe.num_bodies; body_index_0++){
        auto position_0 = universe.positions[body_index_0];
        for(std::uint32_t body_index_1 = 0; body_index_1 < universe.num_bodies; body_index_1++){
            if(body_index_0 == body_index_1){
                continue;
            }
            auto position_1 = universe.positions[body_index_1];
            // calculate the distance between both bodies
            Vector2d<double> distance_vector = position_0 - position_1;
            double distance = sqrt(pow(distance_vector[0], 2) + pow(distance_vector[1], 2));
            // register collision if necessary 
            if (distance < 100000000000.0){  // 100,000,000km
                collisions.push_back(std::make_pair(body_index_0, body_index_1));
            } 
        }   
    }

    // step 2: perform collisions ordered by weight, if a single body partakes in multiple collisions (prioritize descending by weight) (sorting)
    // first, all collisions of the heaviest body are caried out and so on
    while(collisions.size() > 0){
        // find heaviest body
        std::uint32_t heaviest_body_idx;
        double heaviest_body_mass;
        for(std::uint32_t i = 0; i < collisions.size(); i++){
            if(i == 0){
                // initialize
                heaviest_body_idx = std::get<0>(collisions[i]);
                heaviest_body_mass = universe.weights[std::get<0>(collisions[i])];
            }
            if(universe.weights[std::get<0>(collisions[i])] > heaviest_body_mass){
                heaviest_body_mass = universe.weights[std::get<0>(collisions[i])];
                heaviest_body_idx = std::get<0>(collisions[i]);
            }
            if(universe.weights[std::get<1>(collisions[i])] > heaviest_body_mass){
                heaviest_body_mass = universe.weights[std::get<1>(collisions[i])];
                heaviest_body_idx = std::get<1>(collisions[i]);
            }
        }

        // find collision partners
        std::set<std::uint32_t> collision_partners;
        for(auto collision : collisions){
            if(std::get<0>(collision) == heaviest_body_idx){
                collision_partners.insert(std::get<1>(collision));
            }
            if(std::get<1>(collision) == heaviest_body_idx){
                collision_partners.insert(std::get<0>(collision));
            }
        }
        
        // perform collisions
        for(std::uint32_t collision_partner : collision_partners){
            auto m1 = universe.weights[heaviest_body_idx];
            auto m2 = universe.weights[collision_partner];
            auto v1 = universe.velocities[heaviest_body_idx];
            auto v2 = universe.velocities[collision_partner];

            // (Impulserhaltung: m1*v1 + m2*v2 = m1'*v1' + m2'*v2' mit m2'=0 für m1 >=m 2)
            //  update movement vectors 
            //    v1' = (m1*v1 + m2*v2)/(m1+m2)
            universe.velocities[heaviest_body_idx] = ((v1*m1) + (v2*m2))/(m1+m2);

            //  update weights
            //    m1' = m1 + m2
            universe.weights[heaviest_body_idx] = m1 + m2;
        }

        // cleanup collisions list
        for(auto collision_partner : collision_partners){
            if(collisions.size() == 0){
                break;
            }

            std::int32_t i = collisions.size()-1;
            while(i >= 0){
                if(collision_partner == std::get<0>(collisions[i])){
                    collisions.erase(collisions.begin() + i);
                    if(collisions.size() == 0){
                        break;
                    }
                    i--;
                    continue;
                }
                if(collision_partner == std::get<1>(collisions[i])){
                    collisions.erase(collisions.begin() + i);
                    if(collisions.size() == 0){
                        break;
                    }
                    i--;
                    continue;
                }
                i--;
            }
        }

        // convert collision_partners set to vector
        std::vector<std::uint32_t> collision_partners_vector(collision_partners.begin(), collision_partners.end());

        // delete collision partners entries in universe in descending order
        while(collision_partners_vector.size() > 0){
            // find the biggest body index
            std::uint32_t highest_body_idx;
            std::uint32_t selected_index; 
            for(std::uint32_t i = 0; i < collision_partners_vector.size(); i++){
                if(i == 0){
                    // initialize
                    highest_body_idx = collision_partners_vector[i];
                    selected_index = 0;
                }
                if(collision_partners_vector[i] > highest_body_idx){
                    highest_body_idx = collision_partners_vector[i];
                    selected_index = i;
                }
            }
            // delete this body
            universe.forces.erase(universe.forces.begin() + highest_body_idx);
            universe.positions.erase(universe.positions.begin() + highest_body_idx);
            universe.velocities.erase(universe.velocities.begin() + highest_body_idx);
            universe.weights.erase(universe.weights.begin() + highest_body_idx);
            universe.num_bodies--;
            // delete selected collision partner from collision_partners_vector
            collision_partners_vector.erase(collision_partners_vector.begin() + selected_index);
        }
    }
}

void BarnesHutSimulationWithCollisions::find_collisions_parallel(Universe& universe){
    // collision, if distance between two bodies < 100,000,000km  (~distance earth / sun)

    // NOTE: this is a naive implementation. More sophisticated solutions are conceivable!
    // NOTE: Results may become non-deterministic and may depend on the implementation!

    // step 1: determine collisions
    std::vector<std::tuple<std::uint32_t, std::uint32_t>> collisions;
    #pragma omp parallel shared(collisions)
    {
        std::vector<std::tuple<std::uint32_t, std::uint32_t>> partial_collisions;
        std::uint32_t chunk_size = universe.num_bodies / omp_get_num_threads();
        std::uint32_t chunk_start = chunk_size * omp_get_thread_num();
        std::uint32_t chunk_end = chunk_start + chunk_size;
        // let last thread handle the remaining bodies
        if(omp_get_thread_num() == (omp_get_num_threads() - 1)){
            chunk_end = universe.num_bodies;
        }

        for(std::uint32_t body_index_0 = chunk_start; body_index_0 < chunk_end; body_index_0++){
            auto position_0 = universe.positions[body_index_0];
            for(std::uint32_t body_index_1 = 0; body_index_1 < universe.num_bodies; body_index_1++){
                if(body_index_0 == body_index_1){
                    continue;
                }
                auto position_1 = universe.positions[body_index_1];
                // calculate the distance between both bodies
                Vector2d<double> distance_vector = position_0 - position_1;
                double distance = sqrt(pow(distance_vector[0], 2) + pow(distance_vector[1], 2));
                // register collision if necessary 
                if (distance < 100000000000.0){  // 100,000,000km
                    partial_collisions.push_back(std::make_pair(body_index_0, body_index_1));
                } 
            }   
        }

        // combine partial collisions
        #pragma omp critical
        {
            collisions.insert(collisions.end(), partial_collisions.begin(), partial_collisions.end());
        }
    }

    // step 2: perform collisions ordered by weight, if a single body partakes in multiple collisions (prioritize descending by weight) (sorting)
    // first, all collisions of the heaviest body are caried out and so on
    while(collisions.size() > 0){
        // find heaviest body
        std::uint32_t heaviest_body_idx;
        double heaviest_body_mass;
        for(std::uint32_t i = 0; i < collisions.size(); i++){
            if(i == 0){
                // initialize
                heaviest_body_idx = std::get<0>(collisions[i]);
                heaviest_body_mass = universe.weights[std::get<0>(collisions[i])];
            }
            if(universe.weights[std::get<0>(collisions[i])] > heaviest_body_mass){
                heaviest_body_mass = universe.weights[std::get<0>(collisions[i])];
                heaviest_body_idx = std::get<0>(collisions[i]);
            }
            if(universe.weights[std::get<1>(collisions[i])] > heaviest_body_mass){
                heaviest_body_mass = universe.weights[std::get<1>(collisions[i])];
                heaviest_body_idx = std::get<1>(collisions[i]);
            }
        }

        // find collision partners
        std::set<std::uint32_t> collision_partners;
        for(auto collision : collisions){
            if(std::get<0>(collision) == heaviest_body_idx){
                collision_partners.insert(std::get<1>(collision));
            }
            if(std::get<1>(collision) == heaviest_body_idx){
                collision_partners.insert(std::get<0>(collision));
            }
        }
        
        // perform collisions
        for(std::uint32_t collision_partner : collision_partners){
            auto m1 = universe.weights[heaviest_body_idx];
            auto m2 = universe.weights[collision_partner];
            auto v1 = universe.velocities[heaviest_body_idx];
            auto v2 = universe.velocities[collision_partner];

            // (Impulserhaltung: m1*v1 + m2*v2 = m1'*v1' + m2'*v2' mit m2'=0 für m1 >=m 2)
            //  update movement vectors 
            //    v1' = (m1*v1 + m2*v2)/(m1+m2)
            universe.velocities[heaviest_body_idx] = ((v1*m1) + (v2*m2))/(m1+m2);

            //  update weights
            //    m1' = m1 + m2
            universe.weights[heaviest_body_idx] = m1 + m2;
        }

        // cleanup collisions list
        for(auto collision_partner : collision_partners){
            if(collisions.size() == 0){
                break;
            }

            std::int32_t i = collisions.size()-1;
            while(i >= 0){
                if(collision_partner == std::get<0>(collisions[i])){
                    collisions.erase(collisions.begin() + i);
                    if(collisions.size() == 0){
                        break;
                    }
                    i--;
                    continue;
                }
                if(collision_partner == std::get<1>(collisions[i])){
                    collisions.erase(collisions.begin() + i);
                    if(collisions.size() == 0){
                        break;
                    }
                    i--;
                    continue;
                }
                i--;
            }
        }

        // convert collision_partners set to vector
        std::vector<std::uint32_t> collision_partners_vector(collision_partners.begin(), collision_partners.end());

        // delete collision partners entries in universe in descending order
        while(collision_partners_vector.size() > 0){
            // find the biggest body index
            std::uint32_t highest_body_idx;
            std::uint32_t selected_index; 
            for(std::uint32_t i = 0; i < collision_partners_vector.size(); i++){
                if(i == 0){
                    // initialize
                    highest_body_idx = collision_partners_vector[i];
                    selected_index = 0;
                }
                if(collision_partners_vector[i] > highest_body_idx){
                    highest_body_idx = collision_partners_vector[i];
                    selected_index = i;
                }
            }
            // delete this body
            universe.forces.erase(universe.forces.begin() + highest_body_idx);
            universe.positions.erase(universe.positions.begin() + highest_body_idx);
            universe.velocities.erase(universe.velocities.begin() + highest_body_idx);
            universe.weights.erase(universe.weights.begin() + highest_body_idx);
            universe.num_bodies--;
            // delete selected collision partner from collision_partners_vector
            collision_partners_vector.erase(collision_partners_vector.begin() + selected_index);
        }
    }
}