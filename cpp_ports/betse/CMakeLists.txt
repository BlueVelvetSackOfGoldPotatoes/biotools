cmake_minimum_required(VERSION 3.14)
project(betse_cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(betse_sim main.cpp)
target_compile_options(betse_sim PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
