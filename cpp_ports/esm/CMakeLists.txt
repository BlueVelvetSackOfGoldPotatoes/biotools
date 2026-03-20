cmake_minimum_required(VERSION 3.14)
project(esm_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for release builds
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")

add_executable(esm_demo main.cpp)
target_include_directories(esm_demo PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(test_extended test_extended.cpp)
target_include_directories(test_extended PRIVATE ${CMAKE_SOURCE_DIR})
