find_package (OpenCV REQUIRED)
find_package (Boost REQUIRED COMPONENTS serialization filesystem system)
find_package (TBB REQUIRED)
find_package (Eigen 3.0.0 REQUIRED)
find_package (Ceres REQUIRED)

set(PUPILTRACKER_INSTALL_DIR "" CACHE PATH "Pupil tracker installation directory")
set(PUPILTRACKER_SOURCE_DIR "" CACHE PATH "Pupil tracker source directory")

if(PUPILTRACKER_INSTALL_DIR)
    # Do nothing
elseif(PUPILTRACKER_SOURCE_DIR)
    message(WARNING "Pupil tracker source found, compiling locally")

    ExternalProject_Add(pupiltracker
        SOURCE_DIR ${PUPILTRACKER_SOURCE_DIR}
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/pupiltracker
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>)

    ExternalProject_Get_Property(pupiltracker INSTALL_DIR)
    set(PUPILTRACKER_INSTALL_DIR ${INSTALL_DIR})
else()
    message(WARNING "Pupil tracker not found, downloading from github")

    ExternalProject_Add(pupiltracker
        GIT_REPOSITORY https://github.com/LeszekSwirski/pupiltracker.git
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/pupiltracker
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>)

    ExternalProject_Get_Property(pupiltracker INSTALL_DIR)
    set(PUPILTRACKER_INSTALL_DIR ${INSTALL_DIR})
endif()

set(pupiltracker_INCLUDE_DIRS ${PUPILTRACKER_INSTALL_DIR}/include)

find_library (pupiltracker_LIBRARY_RELEASE
              pupiltracker
              PATHS ${PUPILTRACKER_INSTALL_DIR}/lib/)
find_library (pupiltracker_LIBRARY_DEBUG
              pupiltrackerd
              PATHS ${PUPILTRACKER_INSTALL_DIR}/lib/)

if (pupiltracker_LIBRARY_RELEASE AND pupiltracker_LIBRARY_DEBUG)
    set(pupiltracker_LIBRARIES optimized ${pupiltracker_LIBRARY_RELEASE} debug ${pupiltracker_LIBRARY_DEBUG})
elseif (pupiltracker_LIBRARY_RELEASE)
    set(pupiltracker_LIBRARIES ${pupiltracker_LIBRARY_RELEASE})
elseif (pupiltracker_LIBRARY_DEBUG)
    set(pupiltracker_LIBRARIES ${pupiltracker_LIBRARY_DEBUG})
endif()

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
