find_package (OpenCV REQUIRED)
find_package (Boost REQUIRED)
find_package (Eigen 3.0.0 REQUIRED)
find_package (Ceres REQUIRED)

add_library (
    singleeyefitter

    SingleEyeFitter.cpp
    cvx.cpp
    utils.cpp
)

if (MSVC)
    add_definitions(-DNOMINMAX)
endif()

ExternalProject_Get_Property(spii INSTALL_DIR)
set(spii_INCLUDE_DIRS ${INSTALL_DIR}/include)
find_library (spii_LIBRARY_RELEASE
              spii
              PATHS ${INSTALL_DIR}/lib/)
find_library (meschach_LIBRARY_RELEASE
              meschach
              PATHS ${INSTALL_DIR}/lib/)
if (spii_LIBRARY_RELEASE AND meschach_LIBRARY_RELEASE)
    set(spii_LIBRARIES ${spii_LIBRARY_RELEASE} ${meschach_LIBRARY_RELEASE})
else()
    set(spii_LIBRARIES "")
endif()
message(STATUS "spii_LIBRARIES:\"${spii_LIBRARIES}\"")

include_directories(
    SYSTEM
	${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${spii_INCLUDE_DIRS}
)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}    
)

target_link_libraries(
    singleeyefitter

    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${spii_LIBRARIES}
)
