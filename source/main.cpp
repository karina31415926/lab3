
#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>

#include <cstdint>
#include <iostream>
#include "io/image_parser.h"
#include "structures/universe.h"
#include "simulation/naive_sequential_simulation.h"
#include "simulation/naive_parallel_simulation.h"
#include "simulation/barnes_hut_simulation.h"
#include "simulation/barnes_hut_simulation_with_collisions.h"
#include "simulation/naive_cuda_simulation.cuh"
#include "utilities/export.hpp"
#include "utilities/import.hpp"
#include "input_generator/input_generator.h"
#include "plotting/plotter.h"
#include <exception>

int main(int argc, char** argv) {
	auto lab_cli_app = CLI::App{ "" };

	auto number_epochs = std::uint32_t{ 100 };
	auto output_image_width = std::uint32_t{800};
	auto output_image_height = std::uint32_t{800};
	//auto output_path = std::filesystem::path{"INVALID"};
	auto output_path = std::string{"INVALID"};
	auto save_universe_path = std::filesystem::path{"./universe.txt"};
	auto load_universe_path = std::filesystem::path{"./NOTHING_TO_LOAD"};
	bool output_intermediate_states = bool{true};
	bool save_initial_universe = bool{true};
	auto num_bodies = std::uint32_t{100};
	auto plot_intermediate_epochs = std::uint32_t{5};
	auto plot_bounding_box_scale = std::uint32_t{5};
	auto universe_generator = std::uint32_t{ 0 };
	auto simulation_mode = std::uint32_t{0};

	lab_cli_app.add_option("--output-image-width", output_image_width, "Width of the image. Must be a multiple of 8. default: 800px");
	lab_cli_app.add_option("--output-image-height", output_image_height, "Height of the image. Must be a multiple of 8. default: 800px");
	lab_cli_app.add_option("--num-epochs", number_epochs, "default: 100");
	lab_cli_app.add_option("--output-intermediate-states", output_intermediate_states, "Toggle the output of Bitmaps during the simulation. Default: true");
	lab_cli_app.add_option("--num-bodies", num_bodies, "Number of randomly generated bodies, if not input is specified. Default: 100");
	lab_cli_app.add_option("--plot-intermediate-epochs", plot_intermediate_epochs, "Control the amount of plotted states. Value of 1 creates a plot for every epoch, a value of 5 plots every 5th intermediate epoch etc. Default: 5");
	lab_cli_app.add_option("--save-universe-path", save_universe_path, "Path to store the current universe for reproducibility. Default: ./universe.txt");
	lab_cli_app.add_option("--plot-bounding-box-scale", plot_bounding_box_scale, "Scale of the plotted bounding box compared to the initial bounding box of the system. Default: 5");
	lab_cli_app.add_option("--universe-generator", universe_generator, "Select universe generator. Options: 0 -> Random universe. 1 -> Earth Orbit. 2 -> Random universe with at least one supermassive black hole. 3 -> Random universe with at least two supermassive black holes. Please feel free to add new generators. 4 -> Create two colliding bodies. Default: 0");
	auto load_universe_option = lab_cli_app.add_option("--load-universe-path", load_universe_path, "Path to the universe file to be loaded.");
	lab_cli_app.add_option("--simulation-mode", simulation_mode, "Select simulation mode. Options: 0 -> Naive sequential. 1 -> Naive parallel. 2 -> Barnes-Hut. 3 -> Barnes-Hut with collisions. 4 -> Naive CUDA. Default: 0");
	lab_cli_app.add_option("--save-initial-universe", save_initial_universe, "Toggle saving the initial universe to --save-universe-path. Default: true");

	auto output_option = lab_cli_app.add_option("--output", output_path, "Required argument. Set the path to the output directory. MUST contain 'scratch'.");

	CLI11_PARSE(lab_cli_app, argc, argv);

	// check required arguments
	std::cout << "output: " << output_path << std::endl;
	if(output_path == "INVALID"){
		throw std::invalid_argument("Please specify an output path via --output!");
	}

	// validity checks
	std::string output_name_requirement{"scratch"};
	if(output_path.find(output_name_requirement) == std::string::npos){
		// output path does not contain "scratch"
		throw std::invalid_argument("The output path must contain 'scratch'. When using the Lichtenberg cluster, position your output folder under /work/scratch/kurse/kurs00084/<TU-ID>");
	}
	output_option->check(CLI::ExistingDirectory);
	
	if(output_image_height % 8 != 0){
		throw std::invalid_argument("Output image height must be a multiple of 8!");
	}
	if(output_image_width % 8 != 0){
		throw std::invalid_argument("Output image width must be a multiple of 8!");
	}


	// check if a universe shall be loaded or created
	auto universe = Universe();
	if(std::filesystem::exists(load_universe_path)){
		// load existing universe
		load_universe(load_universe_path, universe);
	}	
	else{
		switch(universe_generator){
			case 0:
				// Create random universe
				InputGenerator::create_random_universe(num_bodies, universe);
				break;
			case 1:
				// create earth orbit
				InputGenerator::create_earth_orbit(universe);
				break;
			case 2:
				// Create random universe with at least one supermassive black hole
				InputGenerator::create_random_universe_with_supermassive_blackholes(num_bodies, universe, 1);
				break;
			case 3:
				// Create random universe with at least two supermassive black hole
				InputGenerator::create_random_universe_with_supermassive_blackholes(num_bodies, universe, 2);
				break;
			case 4:
				// Create two colliding bodies
				InputGenerator::create_two_body_collision(universe);
				break;
			default:
				throw std::invalid_argument("Invalid Argument for --universe-generator");
		}		
	}

	// create output_path if not already existing
	if(!std::filesystem::is_directory(output_path)){
		std::filesystem::create_directory(output_path);
	}

	// calculate plot bounding box
	BoundingBox plot_bounding_box = universe.get_bounding_box();
	plot_bounding_box.plotting_sanity_check();
	plot_bounding_box = plot_bounding_box.get_scaled(plot_bounding_box_scale);

	// initialize plotter
	Plotter plotter(plot_bounding_box, output_path, output_image_width, output_image_height);
	plotter.set_filename_prefix("simulation_result");

	// plot initial state of the universe
	plotter.add_bodies_to_image(universe);
	plotter.write_and_clear();

	// save experiment before starting the simulation for reproducibility
	if(save_initial_universe){
		save_universe(save_universe_path, universe);
	}

	// simulate universe
	switch(simulation_mode){
		case 0:
			NaiveSequentialSimulation::simulate_epochs(plotter, universe, number_epochs, output_intermediate_states, plot_intermediate_epochs);
			break;
		case 1:
			NaiveParallelSimulation::simulate_epochs(plotter, universe, number_epochs, output_intermediate_states, plot_intermediate_epochs);
			break;
		case 2:
			BarnesHutSimulation::simulate_epochs(plotter, universe, number_epochs, output_intermediate_states, plot_intermediate_epochs);
			break;
		case 3:
			BarnesHutSimulationWithCollisions::simulate_epochs(plotter, universe, number_epochs, output_intermediate_states, plot_intermediate_epochs);
			break;
		case 4:
			NaiveCudaSimulation::simulate_epochs(plotter, universe, number_epochs, output_intermediate_states, plot_intermediate_epochs);
			break;
		default:
			throw std::invalid_argument("unknown simulation mode: " + std::to_string(simulation_mode));
	}

	// plot simulation result
	plotter.add_bodies_to_image(universe);
	plotter.write_and_clear();

	return 0;
}
