cmake_minimum_required(VERSION 3.20)
project(turing_ring_experiment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TURING_ENABLE_OPENMP "Enable OpenMP parallel batch execution." ON)

find_package(Threads REQUIRED)
find_path(EIGEN3_INCLUDE_DIR Eigen/Eigenvalues PATHS /usr/include/eigen3)
if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Eigen headers not found. Install eigen3 or adjust EIGEN3_INCLUDE_DIR.")
endif()

if(TURING_ENABLE_OPENMP)
    find_package(OpenMP)
endif()

add_library(
    turing_ring_core
    backend/src/boost_json.cpp
    backend/src/types.cpp
    backend/src/model.cpp
    backend/src/analysis.cpp
    backend/src/simulator.cpp
    backend/src/json.cpp
)

target_include_directories(
    turing_ring_core
    PUBLIC
        backend/include
        ${EIGEN3_INCLUDE_DIR}
)

target_compile_options(
    turing_ring_core
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -DNDEBUG -Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/O2 /W4>
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(turing_ring_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(turing_ring_core PUBLIC TURING_ENABLE_OPENMP=1)
endif()

add_executable(turing_ring_backend backend/src/main.cpp)
target_link_libraries(turing_ring_backend PRIVATE turing_ring_core Threads::Threads)
target_compile_definitions(
    turing_ring_backend
    PRIVATE
        BOOST_ASIO_DISABLE_CONCEPTS
)

add_executable(turing_ring_validate backend/src/validate.cpp)
target_link_libraries(turing_ring_validate PRIVATE turing_ring_core)

add_executable(turing_ring_calibrate_capture backend/src/calibrate_capture.cpp)
target_link_libraries(turing_ring_calibrate_capture PRIVATE turing_ring_core)

enable_testing()

add_executable(turing_ring_regression_tests backend/tests/regression.cpp)
target_link_libraries(turing_ring_regression_tests PRIVATE turing_ring_core)
add_test(NAME turing_ring_regression_tests COMMAND turing_ring_regression_tests)
