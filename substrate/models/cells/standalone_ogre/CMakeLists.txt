cmake_minimum_required(VERSION 3.20)
project(cells_standalone_ogre LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CELL_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(CELL_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")

find_package(OGRE CONFIG REQUIRED COMPONENTS Bites RTShaderSystem)

function(cell_link_ogre_target target)
    foreach(candidate IN LISTS ARGN)
        if(TARGET ${candidate})
            target_link_libraries(${target} PUBLIC ${candidate})
            return()
        endif()
    endforeach()
    message(FATAL_ERROR "Missing expected OGRE target for ${target}: ${ARGN}")
endfunction()

add_library(cell_sim STATIC
    ${CELL_SRC_DIR}/cell_models.cpp
)
target_include_directories(cell_sim PUBLIC ${CELL_REPO_ROOT})

add_library(cell_render_ogre STATIC
    render/OgreBridge.cpp
    render/MembraneRenderable.cpp
    render/ParticlesRenderable.cpp
)
target_include_directories(cell_render_ogre PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CELL_REPO_ROOT})
target_link_libraries(cell_render_ogre PUBLIC cell_sim)
cell_link_ogre_target(cell_render_ogre OGRE::OgreMain OgreMain)
cell_link_ogre_target(cell_render_ogre OGRE::Bites OgreBites)
cell_link_ogre_target(cell_render_ogre OGRE::RTShaderSystem OgreRTShaderSystem)

add_executable(cell_app
    app/main.cpp
    app/CellApp.cpp
)
target_include_directories(cell_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CELL_REPO_ROOT})
target_link_libraries(cell_app PRIVATE cell_sim cell_render_ogre)
cell_link_ogre_target(cell_app OGRE::OgreMain OgreMain)
cell_link_ogre_target(cell_app OGRE::Bites OgreBites)
cell_link_ogre_target(cell_app OGRE::RTShaderSystem OgreRTShaderSystem)

configure_file(resources.cfg ${CMAKE_CURRENT_BINARY_DIR}/resources.cfg COPYONLY)
file(COPY media DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
