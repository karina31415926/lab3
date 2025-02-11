#include "simulation/barnes_hut_simulation.h"
#include "simulation/naive_parallel_simulation.h"
#include "physics/gravitation.h"
#include "physics/mechanics.h"

#include <cmath>

void BarnesHutSimulation::simulate_epochs(Plotter& plotter, Universe& universe, std::uint32_t num_epochs, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs){
    for(int i = 0; i < num_epochs; i++){
        simulate_epoch(plotter, universe, create_intermediate_plots, plot_intermediate_epochs);
    }
}

void BarnesHutSimulation::simulate_epoch(Plotter& plotter, Universe& universe, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs){
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

    universe.current_simulation_epoch++;
    if(create_intermediate_plots){
        if(universe.current_simulation_epoch % plot_intermediate_epochs == 0){

            plotter.add_bodies_to_image(universe);
            // optional: add quadtree to image
            // plotter.add_quadtree_to_bitmap(quadtree);
            
            plotter.write_and_clear();
        }
    }
}

void BarnesHutSimulation::get_relevant_nodes(Universe& universe, Quadtree& quadtree, std::vector<QuadtreeNode*>& relevant_nodes, Vector2d<double>& body_position, std::int32_t body_index, double threshold_theta){
    // initialize
    std::vector<QuadtreeNode*> queue;
    queue.push_back(quadtree.root);
    // iterate over queue
    while(queue.size() > 0){
        QuadtreeNode* current = queue.back();
        queue.pop_back();

        // resolve further, if the body is contained in the bounding box and the node does not represent the body
        if(current->bounding_box.contains(body_position) && current->body_identifier != body_index){
            for(QuadtreeNode* child: current->children){
                queue.push_back(child);
            }
            continue;
        }

        // stop, if the body is encountered
        if(current->body_identifier == body_index){
            continue;
        }

        // calculate ratio s/d
        // s: region width
        // d: distance body <-> center of mass
        double s = current->bounding_box.get_diagonal();
        Vector2d<double> distance_vector = current->center_of_mass - body_position;
        double d = sqrt(pow(distance_vector[0], 2) + pow(distance_vector[1], 2));

        // check if s/d > theta
        // Theta = 0.35; --> rough simulation;  // theta = 0 --> full calculation
        
        if((s/d) < threshold_theta){
            // current is sufficiently far away from body to not required further resolution
            relevant_nodes.push_back(current);
            continue;
        }

        // current needs to be resolved further or contributes directly as a body
        if(current->body_identifier == -1){
            // not a body
            for(QuadtreeNode* child : current->children){
                queue.push_back(child);
            }
        }
        else{
            // current is a body
            relevant_nodes.push_back(current);
        }
    }

}

void BarnesHutSimulation::calculate_forces(Universe& universe, Quadtree& quadtree){
    #pragma omp parallel for shared(universe)
    for(std::int32_t body_idx = 0; body_idx < universe.num_bodies; body_idx++){
        // prepare force vector
        Vector2d<double> applied_force_vector;
        // get body positions
        Vector2d<double> body_position = universe.positions[body_idx];
        // get body mass
        double body_mass = universe.weights[body_idx];

        // get QuadtreeNode
        //quadtree.find_body(body_idx, body_position);

        // Identify relevant quadtree nodes for force calculation by traversing downwards and adding nodes which do not contain the body position to the queue.
        // The children of each node in the queue are added to the queue, if the ratio s/d is bigger than the configurable accuracy threshold theta for at least one child.
        // If this is not the case, the node itself is added to the list of relevant nodes.
        std::vector<QuadtreeNode*> relevant_nodes;        
        get_relevant_nodes(universe, quadtree, relevant_nodes, body_position, body_idx, 0.2);

        // calculate forces (by design implemented analogous to prior force calculation)
        for(QuadtreeNode* qtn : relevant_nodes){
            // get distant body positions
            Vector2d<double> distant_body_position = qtn->center_of_mass;

            // calculate vector between bodies to get the direction of the gravitational force
            Vector2d<double> direction_vector = distant_body_position - body_position;

            // calculate the distance between the bodies
            double distance = sqrt(pow(direction_vector[0], 2) + pow(direction_vector[1], 2));

            // calculate gravitational force between the bodies
            double force = gravitational_force(body_mass, qtn->cumulative_mass, distance);

            // create the force vector
            Vector2d<double> force_vector = direction_vector * (force / distance);

            // sum forces applied to body
            applied_force_vector = applied_force_vector + force_vector;
        }

        // store applied force 
        universe.forces[body_idx] = applied_force_vector;
    }
}