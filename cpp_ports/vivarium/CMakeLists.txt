cmake_minimum_required(VERSION 3.14)
project(vivarium_cpp VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only library target
add_library(vivarium INTERFACE)
target_include_directories(vivarium INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Main test executable
add_executable(vivarium_test main.cpp)
target_link_libraries(vivarium_test PRIVATE vivarium pthread)
