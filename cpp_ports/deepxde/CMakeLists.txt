cmake_minimum_required(VERSION 3.14)
project(deepxde_cpp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for release
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(deepxde_demo main.cpp)
target_include_directories(deepxde_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(deepxde_test test_all.cpp)
target_include_directories(deepxde_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
