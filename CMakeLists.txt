project (singleeyefitter)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_DEBUG_POSTFIX "d")

include(ExternalProject)

if (NOT MSVC)
    if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
        if (GCC_VERSION VERSION_LESS 4.7)
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
        else ()
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
        endif ()
    else ()
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    endif ()
endif ()

ExternalProject_Add(spii
  GIT_REPOSITORY https://github.com/PetterS/spii.git
  GIT_TAG 3.0.0
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/spii
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>)
ExternalProject_Add_Step(spii runcmake
    COMMAND ${CMAKE_COMMAND} "${CMAKE_BINARY_DIR}"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Re-running cmake on ${CMAKE_PROJECT_NAME} project"
    DEPENDEES install)

add_subdirectory (lib)
add_subdirectory (cmd)
