cmake_minimum_required(VERSION 3.14)
project(modulus_cpp VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Release build by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address")

# Main demo (original)
add_executable(modulus_demo main.cpp)
target_include_directories(modulus_demo PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(modulus_demo PRIVATE _USE_MATH_DEFINES)

# Extended demo (all new modules)
add_executable(modulus_extended test_extended.cpp)
target_include_directories(modulus_extended PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(modulus_extended PRIVATE _USE_MATH_DEFINES)

# Link math library on Unix
if(UNIX)
    target_link_libraries(modulus_demo PRIVATE m)
    target_link_libraries(modulus_extended PRIVATE m)
endif()
