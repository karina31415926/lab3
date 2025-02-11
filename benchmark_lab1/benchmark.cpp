#include "benchmark.h"


#include <cstdint>
#include <vector>
#include <iostream>

#include "simulation/naive_sequential_simulation.h"
#include "input_generator/input_generator.h"


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


int main(int argc, char** argv) {
	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv))
		return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
	return 0;
}

BENCHMARK(benchmark_naive_sequential)->Unit(benchmark::kMillisecond)->Args({100, 1});
BENCHMARK(benchmark_naive_sequential)->Unit(benchmark::kMillisecond)->Args({1000, 1});
BENCHMARK(benchmark_naive_sequential)->Unit(benchmark::kMillisecond)->Args({10000, 1});
BENCHMARK(benchmark_naive_sequential)->Unit(benchmark::kMillisecond)->Args({100000, 1});

