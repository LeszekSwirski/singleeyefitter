find_package (OpenCV REQUIRED)
find_package (Boost REQUIRED COMPONENTS serialization filesystem system)
find_package (TBB REQUIRED)
find_package (Eigen 3.0.0 REQUIRED)
find_package (Ceres REQUIRED)

set(PUPILTRACKER_ROOT_DIR "" CACHE PATH "Pupil tracker installation directory")

if(PUPILTRACKER_ROOT_DIR)
    # Do nothing
else()
    message(WARNING "Pupil tracker not found, downloading from github")

    # Forward cache variables, if they are Boost, TBB, OpenCV or specified on the command line
    get_cmake_property(CACHE_VARS CACHE_VARIABLES)
    foreach(CACHE_VAR ${CACHE_VARS})
      get_property(CACHE_VAR_HELPSTRING CACHE ${CACHE_VAR} PROPERTY HELPSTRING)
      if(CACHE_VAR_HELPSTRING STREQUAL "No help, variable specified on the command line."
              OR CACHE_VAR MATCHES "^BOOST_.*"
              OR CACHE_VAR MATCHES "^Boost_.*"
              OR CACHE_VAR MATCHES "^TBB_.*"
              OR CACHE_VAR MATCHES "^OpenCV_.*"
        )
        set(CACHE_VAR_VALUE "${${CACHE_VAR}}")

        get_property(CACHE_VAR_TYPE CACHE ${CACHE_VAR} PROPERTY TYPE)
        if(WIN32 AND (CACHE_VAR_TYPE STREQUAL "PATH" OR CACHE_VAR_TYPE STREQUAL "FILEPATH"))
            STRING(REPLACE "\\" "/" CACHE_VAR_VALUE "${CACHE_VAR_VALUE}")
        endif()

        if(CACHE_VAR_TYPE STREQUAL "UNINITIALIZED")
            set(CACHE_VAR_TYPE "STRING")
        endif()

        list(APPEND CMAKE_CACHE_ARGS -D${CACHE_VAR}:${CACHE_VAR_TYPE}=${CACHE_VAR_VALUE})
      endif()
    endforeach()

    ExternalProject_Add(pupiltracker
        GIT_REPOSITORY https://github.com/LeszekSwirski/pupiltracker.git
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/pupiltracker
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        CMAKE_CACHE_ARGS ${CMAKE_CACHE_ARGS})

    ExternalProject_Add_Step(pupiltracker runcmake
        COMMAND ${CMAKE_COMMAND} "${CMAKE_BINARY_DIR}"
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Re-running cmake on ${CMAKE_PROJECT_NAME} project"
        DEPENDEES install)

    ExternalProject_Get_Property(pupiltracker INSTALL_DIR)
    set(PUPILTRACKER_ROOT_DIR ${INSTALL_DIR})
endif()

set(pupiltracker_INCLUDE_DIRS ${PUPILTRACKER_ROOT_DIR}/include)

find_library (pupiltracker_LIBRARY_RELEASE
              pupiltracker
              PATHS ${PUPILTRACKER_ROOT_DIR}/lib/)
find_library (pupiltracker_LIBRARY_DEBUG
              pupiltrackerd
              PATHS ${PUPILTRACKER_ROOT_DIR}/lib/)

if (pupiltracker_LIBRARY_RELEASE AND pupiltracker_LIBRARY_DEBUG)
    set(pupiltracker_LIBRARIES optimized ${pupiltracker_LIBRARY_RELEASE} debug ${pupiltracker_LIBRARY_DEBUG})
elseif (pupiltracker_LIBRARY_RELEASE)
    set(pupiltracker_LIBRARIES ${pupiltracker_LIBRARY_RELEASE})
elseif (pupiltracker_LIBRARY_DEBUG)
    set(pupiltracker_LIBRARIES ${pupiltracker_LIBRARY_DEBUG})
endif()
message(STATUS "pupiltracker_LIBRARIES:\"${pupiltracker_LIBRARIES}\"")

add_executable (
    singleeyefittercmd

    SingleEyeFitterCmd.cpp
    clipper.cpp
)

if (MSVC)
    add_definitions(-DNOMINMAX)
endif()

include_directories(
    SYSTEM
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${pupiltracker_INCLUDE_DIRS}
)
include_directories(
    ${singleeyefitter_SOURCE_DIR}/lib
)
link_directories(
    ${Boost_LIBRARY_DIRS}
    ${OpenCV_LIB_DIR}
)

target_link_libraries(
    singleeyefittercmd

    singleeyefitter
    ${pupiltracker_LIBRARIES}
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
    ${TBB_LIBRARIES}
    ${CERES_LIBRARIES}
)
