add_library(lab_lib STATIC)
target_sources(
  lab_lib
  PRIVATE 		  
      io/image_parser.cpp
      image/bitmap_image.cpp
      structures/universe.cpp
      structures/vector2d.cpp
      structures/bounding_box.cpp
      
      input_generator/random_universe.cpp
      input_generator/earth_orbit.cpp
      input_generator/random_universe_with_supermassive_blackhole.cpp
      input_generator/two_body_collision.cpp

      simulation/naive_sequential_simulation.cpp
      simulation/naive_parallel_simulation.cpp
      simulation/barnes_hut_simulation.cpp
      simulation/barnes_hut_simulation_with_collisions.cpp
      simulation/naive_cuda_simulation.cu

      plotting/plotter.cpp
      plotting/universe.cpp
      plotting/quadtree.cpp
      plotting/bounding_box.cpp

      quadtree/quadtree.cpp
      quadtree/quadtreeNode.cpp
	
		  # for visual studio
		  ${lab_lib_additional_files})

if(MSVC)
target_compile_options(lab_lib PRIVATE -openmp:llvm)
endif()

target_link_libraries(lab_lib PRIVATE OpenMP::OpenMP_CXX)

target_include_directories(lab_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(lab_lib PROPERTIES ENABLE_EXPORTS 1)
target_link_libraries(lab_lib PUBLIC project_options project_libraries)

FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG 815553211b644b2a64b5dc671c602233c49c065f)
FetchContent_MakeAvailable(cli11)

# lab executable
add_executable(lab)
target_sources(lab PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(lab PRIVATE lab_lib)

target_link_libraries(lab PRIVATE CLI11::CLI11)
set_target_properties(lab PROPERTIES ENABLE_EXPORTS 1)

target_link_libraries(lab_lib PUBLIC project_options project_libraries)

get_target_property(lab_lib_sources lab_lib SOURCES)
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${lab_lib_sources})
