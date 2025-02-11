#include "benchmark.h"


#include <cstdint>
#include <vector>
#include <iostream>

#include "simulation/naive_sequential_simulation.h"


#include "simulation/naive_parallel_simulation.h"
#include "simulation/barnes_hut_simulation.h"
#include "simulation/barnes_hut_simulation_with_collisions.h"

#include "input_generator/input_generator.h"


static void benchmark_get_bounding_box_sequential(benchmark::State& state){
	const auto number_bodies = state.range(0);

	for (auto _ : state) {
		state.PauseTiming();
		// initialize universe
		Universe uni;
		InputGenerator::create_random_universe(number_bodies, uni);
		// calculate bounding box
		state.ResumeTiming();
		uni.get_bounding_box();
	}

}

static void benchmark_get_bounding_box_parallel(benchmark::State& state){
	const auto number_bodies = state.range(0);

	for (auto _ : state) {
		state.PauseTiming();
		// initialize universe
		Universe uni;
		InputGenerator::create_random_universe(number_bodies, uni);
		// calculate bounding box
		state.ResumeTiming();
		uni.parallel_cpu_get_bounding_box();
	}

}

static void benchmark_naive_sequential(benchmark::State& state) {
	const auto number_bodies = state.range(0);
	const auto number_epochs = state.range(1);

	for (auto _ : state) {
		state.PauseTiming();
		// initialize universe
		Universe uni;
		InputGenerator::create_random_universe(number_bodies, uni);
		// create dummy plotter
		BoundingBox bb(-5, 5, -5, 5);
		auto tmp_path = std::filesystem::path{"dummy_plot"};
		Plotter plotter(bb, tmp_path, 400, 400);

		state.ResumeTiming();
		NaiveSequentialSimulation::simulate_epochs(plotter, uni, number_epochs, false, 1);
		//benchmark::DoNotOptimize(autStat.print());
	}	
}


static void benchmark_naive_parallel(benchmark::State& state) {
	const auto number_bodies = state.range(0);
	const auto number_epochs = state.range(1);

	for (auto _ : state) {
		state.PauseTiming();
		// initialize universe
		Universe uni;
		InputGenerator::create_random_universe(number_bodies, uni);
		// create dummy plotter
		BoundingBox bb(-5, 5, -5, 5);
		auto tmp_path = std::filesystem::path{"dummy_plot"};
		Plotter plotter(bb, tmp_path, 400, 400);

		state.ResumeTiming();
		NaiveParallelSimulation::simulate_epochs(plotter, uni, number_epochs, false, 1);
		//benchmark::DoNotOptimize(autStat.print());
	}	
}


static void benchmark_barnes_hut(benchmark::State& state) {
	const auto number_bodies = state.range(0);
	const auto number_epochs = state.range(1);

	for (auto _ : state) {
		state.PauseTiming();
		// initialize universe
		Universe uni;
		InputGenerator::create_random_universe(number_bodies, uni);
		// create dummy plotter
		BoundingBox bb(-5, 5, -5, 5);
		auto tmp_path = std::filesystem::path{"dummy_plot"};
		Plotter plotter(bb, tmp_path, 400, 400);

		state.ResumeTiming();
		BarnesHutSimulation::simulate_epochs(plotter, uni, number_epochs, false, 1);
		//benchmark::DoNotOptimize(autStat.print());
	}	
}

static void benchmark_barnes_hut_with_collisions(benchmark::State& state) {
	const auto number_bodies = state.range(0);

	for (auto _ : state) {
		state.PauseTiming();
		// initialize universe
		Universe uni;
		InputGenerator::create_random_universe(number_bodies, uni);
		// create dummy plotter
		BoundingBox bb(-5, 5, -5, 5);
		auto tmp_path = std::filesystem::path{"dummy_plot"};
		Plotter plotter(bb, tmp_path, 400, 400);

		state.ResumeTiming();
		BarnesHutSimulationWithCollisions::simulate_epochs(plotter, uni, 1, false, 1);
		//benchmark::DoNotOptimize(autStat.print());
	}	
}

static void benchmark_construct_quadtree(benchmark::State& state) {
	const auto number_bodies = state.range(0);
	const std::int8_t construct_mode = state.range(1);

	for (auto _ : state) {
		state.PauseTiming();
		// initialize universe
		Universe uni;
		InputGenerator::create_random_universe(number_bodies, uni);
		BoundingBox bb = uni.get_bounding_box();

		state.ResumeTiming();
		Quadtree(uni, bb, construct_mode);
	}	
}

static void benchmark_calculate_cumulative_masses(benchmark::State& state) {
	const auto number_bodies = state.range(0);

	for (auto _ : state) {
		state.PauseTiming();
		// initialize universe
		Universe uni;
		InputGenerator::create_random_universe(number_bodies, uni);
		BoundingBox bb = uni.get_bounding_box();
		Quadtree qt = Quadtree(uni, bb, 2);
		
		state.ResumeTiming();
		qt.calculate_cumulative_masses();
	}	
}

static void benchmark_find_collisions(benchmark::State& state) {
	const auto number_bodies = state.range(0);

	for (auto _ : state) {
		state.PauseTiming();
		// initialize universe
		Universe uni;
		InputGenerator::create_random_universe(number_bodies, uni);
		BoundingBox bb = uni.get_bounding_box();
		
		state.ResumeTiming();
		BarnesHutSimulationWithCollisions::find_collisions(uni);
	}	
}

static void benchmark_find_collisions_parallel(benchmark::State& state) {
	const auto number_bodies = state.range(0);

	for (auto _ : state) {
		state.PauseTiming();
		// initialize universe
		Universe uni;
		InputGenerator::create_random_universe(number_bodies, uni);
		BoundingBox bb = uni.get_bounding_box();
		
		state.ResumeTiming();
		BarnesHutSimulationWithCollisions::find_collisions_parallel(uni);
	}	
}


int main(int argc, char** argv) {
	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv))
		return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
	return 0;
}

BENCHMARK(benchmark_get_bounding_box_sequential)->Unit(benchmark::kMillisecond)->Args({100000});
BENCHMARK(benchmark_get_bounding_box_sequential)->Unit(benchmark::kMillisecond)->Args({10000000});
BENCHMARK(benchmark_get_bounding_box_sequential)->Unit(benchmark::kMillisecond)->Args({100000000});

BENCHMARK(benchmark_get_bounding_box_parallel)->Unit(benchmark::kMillisecond)->Args({100000});
BENCHMARK(benchmark_get_bounding_box_parallel)->Unit(benchmark::kMillisecond)->Args({10000000});
BENCHMARK(benchmark_get_bounding_box_parallel)->Unit(benchmark::kMillisecond)->Args({100000000});
/*
BENCHMARK(benchmark_construct_quadtree)->Unit(benchmark::kMillisecond)->Args({10000, 0});
BENCHMARK(benchmark_construct_quadtree)->Unit(benchmark::kMillisecond)->Args({20000, 0});
BENCHMARK(benchmark_construct_quadtree)->Unit(benchmark::kMillisecond)->Args({200000, 0});
BENCHMARK(benchmark_construct_quadtree)->Unit(benchmark::kMillisecond)->Args({10000000, 0});
BENCHMARK(benchmark_construct_quadtree)->Unit(benchmark::kMillisecond)->Args({10000, 1});
BENCHMARK(benchmark_construct_quadtree)->Unit(benchmark::kMillisecond)->Args({20000, 1});
BENCHMARK(benchmark_construct_quadtree)->Unit(benchmark::kMillisecond)->Args({10000, 2});
BENCHMARK(benchmark_construct_quadtree)->Unit(benchmark::kMillisecond)->Args({20000, 2});
BENCHMARK(benchmark_construct_quadtree)->Unit(benchmark::kMillisecond)->Args({200000, 2});
BENCHMARK(benchmark_construct_quadtree)->Unit(benchmark::kMillisecond)->Args({10000000, 2});

BENCHMARK(benchmark_calculate_cumulative_masses)->Unit(benchmark::kMillisecond)->Args({10000});
BENCHMARK(benchmark_calculate_cumulative_masses)->Unit(benchmark::kMillisecond)->Args({1000000});
BENCHMARK(benchmark_calculate_cumulative_masses)->Unit(benchmark::kMillisecond)->Args({10000000});

BENCHMARK(benchmark_naive_sequential)->Unit(benchmark::kMillisecond)->Args({100, 1});
BENCHMARK(benchmark_naive_sequential)->Unit(benchmark::kMillisecond)->Args({1000, 1});

BENCHMARK(benchmark_naive_sequential)->Unit(benchmark::kMillisecond)->Args({10000, 1});

BENCHMARK(benchmark_naive_parallel)->Unit(benchmark::kMillisecond)->Args({100, 1});
BENCHMARK(benchmark_naive_parallel)->Unit(benchmark::kMillisecond)->Args({1000, 1});
BENCHMARK(benchmark_naive_parallel)->Unit(benchmark::kMillisecond)->Args({10000, 1});
BENCHMARK(benchmark_naive_parallel)->Unit(benchmark::kMillisecond)->Args({100000, 1});


BENCHMARK(benchmark_barnes_hut)->Unit(benchmark::kMillisecond)->Args({100, 1});
BENCHMARK(benchmark_barnes_hut)->Unit(benchmark::kMillisecond)->Args({1000, 1});
BENCHMARK(benchmark_barnes_hut)->Unit(benchmark::kMillisecond)->Args({10000, 1});
BENCHMARK(benchmark_barnes_hut)->Unit(benchmark::kMillisecond)->Args({100000, 1});
BENCHMARK(benchmark_barnes_hut)->Unit(benchmark::kMillisecond)->Args({1000000, 1});

BENCHMARK(benchmark_find_collisions)->Unit(benchmark::kMillisecond)->Args({1000});
BENCHMARK(benchmark_find_collisions)->Unit(benchmark::kMillisecond)->Args({10000});
BENCHMARK(benchmark_find_collisions)->Unit(benchmark::kMillisecond)->Args({100000});

BENCHMARK(benchmark_find_collisions_parallel)->Unit(benchmark::kMillisecond)->Args({1000});
BENCHMARK(benchmark_find_collisions_parallel)->Unit(benchmark::kMillisecond)->Args({10000});
BENCHMARK(benchmark_find_collisions_parallel)->Unit(benchmark::kMillisecond)->Args({100000});

BENCHMARK(benchmark_barnes_hut_with_collisions)->Unit(benchmark::kMillisecond)->Args({100, 1});
BENCHMARK(benchmark_barnes_hut_with_collisions)->Unit(benchmark::kMillisecond)->Args({1000, 1});
BENCHMARK(benchmark_barnes_hut_with_collisions)->Unit(benchmark::kMillisecond)->Args({10000, 1});
BENCHMARK(benchmark_barnes_hut_with_collisions)->Unit(benchmark::kMillisecond)->Args({100000, 1});
*/