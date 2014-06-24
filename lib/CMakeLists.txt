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
set(spii_LIBRARIES ${INSTALL_DIR}/lib/spii.lib ${INSTALL_DIR}/lib/meschach.lib)

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
