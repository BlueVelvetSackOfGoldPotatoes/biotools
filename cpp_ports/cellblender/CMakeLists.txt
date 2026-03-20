cmake_minimum_required(VERSION 3.14)
project(cellblender_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Main executable
add_executable(cellblender_demo main.cpp)

# The entire library is header-only (cellblender.h)
target_include_directories(cellblender_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link filesystem library if needed (some compilers require it)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    target_link_libraries(cellblender_demo stdc++fs)
endif()

# Warnings
if(MSVC)
    target_compile_options(cellblender_demo PRIVATE /W4)
else()
    target_compile_options(cellblender_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()
