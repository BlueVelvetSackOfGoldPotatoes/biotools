cmake_minimum_required(VERSION 3.14)
project(neuropil_tools
    VERSION 1.0.0
    DESCRIPTION "C++17 port of the neuropil_tools Python package for MCell neuropil analysis"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Main demo executable (header-only library + demo)
add_executable(neuropil_demo main.cpp)
target_include_directories(neuropil_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Platform-specific link requirements for std::filesystem
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    target_link_libraries(neuropil_demo PRIVATE stdc++fs)
endif()

# Compiler warnings
if(MSVC)
    target_compile_options(neuropil_demo PRIVATE /W4)
else()
    target_compile_options(neuropil_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()
