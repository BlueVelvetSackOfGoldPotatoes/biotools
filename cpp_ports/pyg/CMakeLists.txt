cmake_minimum_required(VERSION 3.14)
project(pyg_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimisation flags for release builds
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

add_executable(pyg_demo main.cpp)
target_include_directories(pyg_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
