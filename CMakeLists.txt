cmake_minimum_required(VERSION 3.16)
project(teb)

# 强制开启C++17，解决std::clamp编译报错 + 完整C++17特性支持
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)

# ROS node compilation options - ROS1 Noetic 环境 正确配置
option(ENABLE_ROS1_NODE "Enable ROS1 node compilation" OFF)
option(ENABLE_ROS2_NODE "Enable ROS2 node compilation" ON)

# 基础必装依赖 (所有环境通用)
find_package(Boost REQUIRED COMPONENTS system thread graph)
find_package(SUITESPARSE REQUIRED)
find_package(G2O REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)



# 头文件目录 (包含ROS1头文件)
include_directories(
        inc
        /usr/include/eigen3
        ${SUITESPARSE_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR}
)



# TEB核心库编译 (保留你所有的源文件，无删减)
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

# 链接核心库依赖
target_link_libraries(lib_teb
        ${SUITESPARSE_LIBRARIES}
        ${G2O_LIBRARIES}
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        fmt::fmt
        yaml-cpp
)

# 编译纯C++主程序 teb (非ROS节点，保留)
add_executable(teb main.cpp)
target_link_libraries(
        teb
        lib_teb
        ${SUITESPARSE_LIBRARIES}
        ${G2O_LIBRARIES}
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        fmt::fmt
        yaml-cpp
)

# 编译测试程序 planner_test (保留)
add_executable(planner_test planner_test.cpp)
target_link_libraries(
        planner_test
        lib_teb
        ${SUITESPARSE_LIBRARIES}
        ${G2O_LIBRARIES}
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        fmt::fmt
        yaml-cpp
)

# ========== ROS1 Noetic 节点编译 (补全实现，非占位符) ==========
if(ENABLE_ROS1_NODE)
    find_package(catkin REQUIRED COMPONENTS
        roscpp
        sensor_msgs
        geometry_msgs
        nav_msgs
        tf
        tf2_ros
        visualization_msgs
        std_msgs
        dynamic_reconfigure
    )

    include_directories(
        ${catkin_INCLUDE_DIRS}
    )

    # 编译ROS1节点文件 (如果你的ROS1节点代码在node/ros1_node.cpp)
    add_executable(teb_ros1_node node/ros1_node.cpp)
    target_link_libraries(teb_ros1_node
        lib_teb
        ${catkin_LIBRARIES}
        ${SUITESPARSE_LIBRARIES}
        ${G2O_LIBRARIES}
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        fmt::fmt
        yaml-cpp
    )
    
    # ROS1 安装规则 (必须配置，ROS1规范)
    install(TARGETS teb_ros1_node teb planner_test
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
    
    install(TARGETS lib_teb
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    )
endif()

# ROS2 分支完全屏蔽 (无改动，保持OFF)
if(ENABLE_ROS2_NODE)
    find_package(ament_cmake REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(geometry_msgs REQUIRED)
    find_package(nav_msgs REQUIRED)
    find_package(tf2 REQUIRED)
    find_package(tf2_geometry_msgs REQUIRED)
    find_package(visualization_msgs REQUIRED)

    # 查找PCL库（用于传感器融合）
    find_package(PCL REQUIRED)
    find_package(pcl_conversions REQUIRED)
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})

    # TEB ROS2节点
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
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        fmt::fmt
        yaml-cpp
    )

    # 传感器融合节点
    add_executable(sensor_fusion_node interface/ros2_interface.cpp)
    ament_target_dependencies(sensor_fusion_node
        rclcpp
        sensor_msgs
        geometry_msgs
        tf2
        tf2_geometry_msgs
        pcl_conversions
    )
    target_link_libraries(sensor_fusion_node
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        yaml-cpp
    )

    # 安装目标
    install(TARGETS teb_ros2_node sensor_fusion_node
        DESTINATION lib/${PROJECT_NAME}
    )

    # ROS2包配置暂时注释
    # ament_package()
endif()
