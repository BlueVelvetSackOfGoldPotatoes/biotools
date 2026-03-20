# ---------------------------------------------------------------------------
# CMakeLists.txt - BETSEE C++17/Qt5 Port
# BioElectric Tissue Simulation Engine Environment
# Original Python/PySide2 by Alexis Pietak & Cecil Curry.
#
# Build instructions:
#
#   With Qt5 (full GUI):
#     mkdir build && cd build
#     cmake .. -DCMAKE_BUILD_TYPE=Release
#     cmake --build .
#
#   Without Qt5 (CLI-only mode, YAML parser + parameter management):
#     mkdir build && cd build
#     cmake .. -DCMAKE_BUILD_TYPE=Release -DFORCE_NO_QT=ON
#     cmake --build .
#
# ---------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

project(betsee
    VERSION 1.1.2.0
    DESCRIPTION "BETSEE - BioElectric Tissue Simulation Engine Environment"
    HOMEPAGE_URL "https://gitlab.com/betse/betsee"
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# C++17 standard requirement
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(FORCE_NO_QT "Build without Qt5 even if available" OFF)

# ---------------------------------------------------------------------------
# Qt5 detection (optional)
# ---------------------------------------------------------------------------
set(BETSEE_HAS_QT5 OFF)

if(NOT FORCE_NO_QT)
    # Try to find Qt5
    find_package(Qt5 QUIET COMPONENTS
        Core
        Gui
        Widgets
    )

    if(Qt5_FOUND)
        set(BETSEE_HAS_QT5 ON)
        message(STATUS "Qt5 found: version ${Qt5Core_VERSION}")
        message(STATUS "  Qt5Core:    ${Qt5Core_DIR}")
        message(STATUS "  Qt5Gui:     ${Qt5Gui_DIR}")
        message(STATUS "  Qt5Widgets: ${Qt5Widgets_DIR}")

        # Enable Qt MOC (Meta-Object Compiler) for signal/slot support
        set(CMAKE_AUTOMOC ON)
        # Enable Qt RCC (Resource Compiler) for resource files
        set(CMAKE_AUTORCC ON)
        # Enable Qt UIC (User Interface Compiler) for .ui files
        set(CMAKE_AUTOUIC ON)
    else()
        message(STATUS "Qt5 not found - building in CLI-only mode")
        message(STATUS "  Install Qt5 development libraries for full GUI support:")
        message(STATUS "    Ubuntu/Debian: sudo apt install qtbase5-dev")
        message(STATUS "    Fedora:        sudo dnf install qt5-qtbase-devel")
        message(STATUS "    macOS:         brew install qt@5")
        message(STATUS "    Arch:          sudo pacman -S qt5-base")
    endif()
else()
    message(STATUS "Qt5 disabled by FORCE_NO_QT option - building in CLI-only mode")
endif()

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
set(BETSEE_SOURCES
    main.cpp
)

set(BETSEE_HEADERS
    betsee.h
)

# ---------------------------------------------------------------------------
# Create executable target
# ---------------------------------------------------------------------------
add_executable(betsee ${BETSEE_SOURCES} ${BETSEE_HEADERS})

# ---------------------------------------------------------------------------
# Compiler settings
# ---------------------------------------------------------------------------

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(betsee PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
    )
elseif(MSVC)
    target_compile_options(betsee PRIVATE
        /W4
        /wd4100  # unreferenced formal parameter
    )
endif()

# ---------------------------------------------------------------------------
# Qt5 linkage (conditional)
# ---------------------------------------------------------------------------
if(BETSEE_HAS_QT5)
    # Define HAS_QT5 preprocessor macro to enable Qt-dependent code
    target_compile_definitions(betsee PRIVATE HAS_QT5)

    # Link against Qt5 libraries
    target_link_libraries(betsee PRIVATE
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
    )

    message(STATUS "Building BETSEE with full Qt5 GUI support")
else()
    message(STATUS "Building BETSEE in CLI-only mode (no Qt5)")
endif()

# ---------------------------------------------------------------------------
# Filesystem library linkage (needed for std::filesystem on some compilers)
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # GCC < 9 needs explicit linkage for std::filesystem
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
        target_link_libraries(betsee PRIVATE stdc++fs)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Clang < 9 needs explicit linkage for std::filesystem
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
        target_link_libraries(betsee PRIVATE c++fs)
    endif()
endif()

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS betsee
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== BETSEE Build Configuration ===")
message(STATUS "  Version:      ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler:     ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build Type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  Qt5 GUI:      ${BETSEE_HAS_QT5}")
message(STATUS "  Install:      ${CMAKE_INSTALL_PREFIX}")
message(STATUS "=================================")
message(STATUS "")
