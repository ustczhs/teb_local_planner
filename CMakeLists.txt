cmake_minimum_required(VERSION 3.17)
project(teb)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)

# ROS node compilation options
option(ENABLE_ROS1_NODE "Enable ROS1 node compilation" OFF)
option(ENABLE_ROS2_NODE "Enable ROS2 node compilation" ON)

find_package(Boost REQUIRED COMPONENTS system thread graph)
find_package(SUITESPARSE REQUIRED)
find_package(G2O REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)

# ROS2 dependencies (only when ROS2 node is enabled)
if(ENABLE_ROS2_NODE)
    find_package(ament_cmake REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(geometry_msgs REQUIRED)
    find_package(nav_msgs REQUIRED)
    find_package(tf2 REQUIRED)
    find_package(tf2_geometry_msgs REQUIRED)
    find_package(visualization_msgs REQUIRED)
endif()

include_directories(
        inc
        /usr/include/eigen3
        ${SUITESPARSE_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)

add_library(lib_teb
        src/teb_config.cpp
        src/obstacles.cpp
        src/visualization.cpp
        src/optimal_planner.cpp
        src/timed_elastic_band.cpp
        src/frenet_reference.cpp
        src/config_loader.cpp
        src/sensor_proc.cpp
)

add_executable(teb main.cpp)

add_executable(planner_test planner_test.cpp)

target_link_libraries(
        teb
        lib_teb
        ${SUITESPARSE_LIBRARIES}
        ${G2O_LIBRARIES}
        -lboost_system -lboost_thread -lboost_graph
        ${OpenCV_LIBS}
        fmt::fmt
)

target_link_libraries(
        planner_test
        lib_teb
        ${SUITESPARSE_LIBRARIES}
        ${G2O_LIBRARIES}
        -lboost_system -lboost_thread -lboost_graph
        ${OpenCV_LIBS}
        fmt::fmt
        yaml-cpp
)

# ROS2 node compilation (only when enabled)
if(ENABLE_ROS2_NODE)
    add_executable(teb_ros2_node node/ros2_node.cpp)

    ament_target_dependencies(teb_ros2_node
        rclcpp
        sensor_msgs
        geometry_msgs
        nav_msgs
        tf2
        tf2_geometry_msgs
        visualization_msgs
    )

    target_link_libraries(teb_ros2_node
        lib_teb
        ${SUITESPARSE_LIBRARIES}
        ${G2O_LIBRARIES}
        -lboost_system -lboost_thread -lboost_graph
        ${OpenCV_LIBS}
        fmt::fmt
        yaml-cpp
    )

    # Install executable
    install(TARGETS teb_ros2_node
        DESTINATION lib/${PROJECT_NAME}
    )
endif()

# ROS1 node compilation (placeholder for future implementation)
if(ENABLE_ROS1_NODE)
    # TODO: Add ROS1 node compilation when needed
    message(STATUS "ROS1 node compilation enabled but not implemented yet")
endif()
