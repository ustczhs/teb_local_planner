/**
 * @file ros2_node.cpp
 * @brief TEB Local Planner ROS2节点
 * 
 * 此节点是ROS2接口层，负责：
 * - ROS话题的订阅和发布
 * - 传感器数据的接收和转换
 * - 可视化显示
 * 
 * 核心算法逻辑由 TebPlannerCore 类封装
 */

#include <chrono>
#include <functional>
#include <memory>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <vector>
#include <shared_mutex>
#include <thread>
#include <queue>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/utils.h>

#include "teb_planner_core.h"

using namespace std::chrono_literals;

// ============================================================================
// 工具函数
// ============================================================================

/**
 * @brief 将TEB路径转换为ROS Path消息
 */
nav_msgs::msg::Path trajectoryToPath(const std::vector<teb_local_planner::Pose2D>& trajectory) {
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = rclcpp::Clock().now();
    path_msg.header.frame_id = "map";
    
    // 预分配容量
    path_msg.poses.reserve(trajectory.size());
    
    for (const auto& pose : trajectory) {
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header = path_msg.header;
        pose_msg.pose.position.x = pose.x;
        pose_msg.pose.position.y = pose.y;
        pose_msg.pose.position.z = 0.0;
        
        tf2::Quaternion q;
        q.setRPY(0, 0, pose.theta);
        pose_msg.pose.orientation.x = q.x();
        pose_msg.pose.orientation.y = q.y();
        pose_msg.pose.orientation.z = q.z();
        pose_msg.pose.orientation.w = q.w();
        
        path_msg.poses.emplace_back(std::move(pose_msg));
    }
    return path_msg;
}

/**
 * @brief 将TEB Pose2D转换为ROS PoseStamped消息
 */
geometry_msgs::msg::PoseStamped poseToPoseStamped(const teb_local_planner::Pose2D& pose) {
    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp = rclcpp::Clock().now();
    msg.header.frame_id = "map";
    msg.pose.position.x = pose.x;
    msg.pose.position.y = pose.y;
    msg.pose.position.z = 0.0;
    
    tf2::Quaternion q;
    q.setRPY(0, 0, pose.theta);
    msg.pose.orientation.x = q.x();
    msg.pose.orientation.y = q.y();
    msg.pose.orientation.z = q.z();
    msg.pose.orientation.w = q.w();
    
    return msg;
}

/**
 * @brief 将ROS Path转换为TEB Path2D
 */
teb_local_planner::Path2D rosPathToPath2D(const nav_msgs::msg::Path& path) {
    teb_local_planner::Path2D path2d;
    // 预分配容量
    path2d.poses.reserve(path.poses.size());
    
    // 将path的路径点的角度更新为利用前后两个点计算得出的航向角，从0号点开始计算，需要考虑最后一个点的情况
    for (size_t i = 0; i < path.poses.size() - 1; ++i) {
        double yaw = std::atan2(path.poses[i+1].pose.position.y - path.poses[i].pose.position.y, path.poses[i+1].pose.position.x - path.poses[i].pose.position.x);
        path2d.poses.emplace_back(path.poses[i].pose.position.x,
                                  path.poses[i].pose.position.y,
                                  yaw);
    }
    // 最后一个点
    double yaw = std::atan2(path.poses[path.poses.size() - 1].pose.position.y - path.poses[path.poses.size() - 2].pose.position.y, path.poses[path.poses.size() - 1].pose.position.x - path.poses[path.poses.size() - 2].pose.position.x);
    path2d.poses.emplace_back(path.poses[path.poses.size() - 1].pose.position.x,
                              path.poses[path.poses.size() - 1].pose.position.y,
                              yaw);
    return path2d;
}

// ============================================================================
// 主节点类
// ============================================================================

class TebRos2Node : public rclcpp::Node {
public:
    TebRos2Node()
        : Node("teb_local_planner"),
          goal_received_(false),
          goal_reached_(false) {
        
        // 声明参数
        declare_parameter<std::string>("config_file", "../config/config.yaml");
        std::string config_file;
        get_parameter("config_file", config_file);
        
        // 初始化规划器核心
        initializePlanner(config_file);
        
        // 设置ROS接口
        setupROSInterfaces();
        
        // 设置定时器
        setupTimers();
        
        RCLCPP_INFO(get_logger(), "TEB ROS2 Node initialized successfully");
    }
    
    // 析构函数：停止障碍物处理线程
    ~TebRos2Node() {
        stopObstacleProcessing();
    }

private:
    // 规划器核心
    std::unique_ptr<teb_local_planner::TebPlannerCore> planner_;
    std::unique_ptr<teb_local_planner::SensorProcessor> sensor_processor_;
    
    // ROS2接口 - 订阅者
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr current_pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr amcl_pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    
    // ROS2接口 - 发布者
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr aim_pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr obstacle_pub_;
    
    rclcpp::TimerBase::SharedPtr planning_timer_;
    
    // 状态
    teb_local_planner::Pose2D current_pose_;
    teb_local_planner::PoseData current_pose_data_;

    bool goal_received_;
    bool goal_reached_;
    geometry_msgs::msg::Twist current_velocity_;
    // 传感器数据队列和同步
    std::mutex sensor_data_mutex_;
    std::queue<teb_local_planner::LaserScanData> scan_queue_;
    bool has_latest_pose_ = false;
    
    // 障碍物处理线程
    std::thread obstacle_processing_thread_;
    std::atomic<bool> running_{true};
    // 障碍物数据共享
    std::mutex obstacles_mutex_;
    teb_local_planner::ObstContainer all_obstacles_;
    std::vector<Eigen::Vector2d> laser_obstacle_points_;
    std::vector<Eigen::Vector2d> left_boundary_points_;
    std::vector<Eigen::Vector2d> right_boundary_points_;
    
    // =========================================================================
    // 初始化方法
    // =========================================================================
    
    void initializePlanner(const std::string& config_file) {
        planner_ = std::make_unique<teb_local_planner::TebPlannerCore>(config_file);
        
        // 初始化传感器处理器
        sensor_processor_ = std::make_unique<teb_local_planner::SensorProcessor>(planner_->getTebConfig());
        
        // 设置日志回调
        planner_->setLogCallback([this](const std::string& level, const std::string& message) {
            if (level == "INFO") {
                RCLCPP_INFO(get_logger(), "%s", message.c_str());
            } else if (level == "WARN") {
                RCLCPP_WARN(get_logger(), "%s", message.c_str());
            } else if (level == "ERROR") {
                RCLCPP_ERROR(get_logger(), "%s", message.c_str());
            }
        });
        
        RCLCPP_INFO(get_logger(), "Planner core initialized with config: %s", config_file.c_str());
    }
    
    void setupROSInterfaces() {
        // 订阅者
        laser_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", rclcpp::SensorDataQoS(),
            std::bind(&TebRos2Node::laserCallback, this, std::placeholders::_1));
        
        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/odometry/filtered", 10,
            std::bind(&TebRos2Node::odomCallback, this, std::placeholders::_1));
        
        current_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            "/current_pose", 10,
            std::bind(&TebRos2Node::poseCallback, this, std::placeholders::_1));
        
        amcl_pose_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/amcl_pose", 10,
            std::bind(&TebRos2Node::amclPoseCallback, this, std::placeholders::_1));
        
        path_sub_ = create_subscription<nav_msgs::msg::Path>(
            "/plan", 10,
            std::bind(&TebRos2Node::pathCallback, this, std::placeholders::_1));
        
        // 发布者
        aim_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/aim_pose", 10);
        cmd_vel_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>("/cmd_vel1", 10);
        path_pub_ = create_publisher<nav_msgs::msg::Path>("/local_plan_teb", 10);
        obstacle_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/obstacles_marker", 10);
        
        RCLCPP_INFO(get_logger(), "ROS interfaces setup complete");
    }
    
    void setupTimers() {
        // 启动障碍物处理线程
        obstacle_processing_thread_ = std::thread(&TebRos2Node::obstacleProcessingLoop, this);
        
        auto planning_period = std::chrono::duration<double>(1.0 / 20.0);
        planning_timer_ = create_wall_timer(
            planning_period, std::bind(&TebRos2Node::planningTimerCallback, this));
        RCLCPP_INFO(get_logger(), "Planning timer initialized with frequency: 20.0 Hz");
    }
    
    // 停止障碍物处理线程
    void stopObstacleProcessing() {
        running_ = false;
        if (obstacle_processing_thread_.joinable()) {
            obstacle_processing_thread_.join();
        }
    }
    
    // 障碍物处理主循环（独立线程）
    void obstacleProcessingLoop() {
        RCLCPP_INFO(get_logger(), "Obstacle processing thread started");
        
        while (rclcpp::ok() && running_) {
            teb_local_planner::LaserScanData scan_data;
            
            // 从队列获取最新的激光数据
            {
                std::lock_guard<std::mutex> lock(sensor_data_mutex_);                
                if (scan_queue_.empty() || !has_latest_pose_) {
                    // 没有数据时短暂休眠
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
                
                scan_data = std::move(scan_queue_.front());
                scan_queue_.pop();
            }
            
            // 处理激光数据获取障碍物点
            try {
                //加锁
                std::lock_guard<std::mutex> lock(obstacles_mutex_);
                sensor_processor_->laserCallback(scan_data, current_pose_data_);

                // 获取感知层的障碍物点（map坐标系）
                all_obstacles_ = sensor_processor_->getMapObstacles(current_pose_data_);
                // 添加边界障碍物
                planner_->getBoundaryObstacles(all_obstacles_);
            } catch (const std::exception& e) {
                RCLCPP_WARN(get_logger(), "Error processing laser data: %s", e.what());
            }
        }
        
        RCLCPP_INFO(get_logger(), "Obstacle processing thread stopped");
    }
    
    // =========================================================================
    // 回调函数
    // =========================================================================
    
    void laserCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        // 将数据放入队列，由障碍物处理线程读取
        teb_local_planner::LaserScanData scan_data;
        scan_data.angle_min = msg->angle_min;
        scan_data.angle_max = msg->angle_max;
        scan_data.angle_increment = msg->angle_increment;
        scan_data.range_min = msg->range_min;
        scan_data.range_max = msg->range_max;
        scan_data.ranges = std::move(msg->ranges);  // 使用move避免拷贝
        scan_data.timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

        {
            std::lock_guard<std::mutex> lock(sensor_data_mutex_);
            
            // 保持队列中只有最新的数据
            while (scan_queue_.size() >= 1) {
                scan_queue_.pop();
            }
            scan_queue_.push(std::move(scan_data));  // 使用move避免拷贝
        }
    }
    
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        current_pose_.x = msg->pose.pose.position.x;
        current_pose_.y = msg->pose.pose.position.y;
        try {
            current_pose_.theta = tf2::getYaw(msg->pose.pose.orientation);
        } catch (const tf2::TransformException& e) {
            RCLCPP_WARN(get_logger(), "Failed to get yaw from odometry: %s", e.what());
        }
    }
    
    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        current_pose_.x = msg->pose.position.x;
        current_pose_.y = msg->pose.position.y;
        try {
            current_pose_.theta = tf2::getYaw(msg->pose.orientation);
            //current_pose_转换为current_pose
            current_pose_data_.position.x = current_pose_.x;
            current_pose_data_.position.y = current_pose_.y;
            current_pose_data_.position.z = 0.0;
            tf2::Quaternion q;
            q.setRPY(0, 0, current_pose_.theta);
            current_pose_data_.orientation.x = q.x();
            current_pose_data_.orientation.y = q.y();
            current_pose_data_.orientation.z = q.z();
            current_pose_data_.orientation.w = q.w();
            has_latest_pose_ = true;

        } catch (const tf2::TransformException& e) {
            RCLCPP_WARN(get_logger(), "Failed to get yaw from pose: %s", e.what());
        }
    }
    
    void amclPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
        if (!msg) return;
        current_pose_.x = msg->pose.pose.position.x;
        current_pose_.y = msg->pose.pose.position.y;
        try {
            current_pose_.theta = tf2::getYaw(msg->pose.pose.orientation);
            //current_pose_转换为current_pose
            current_pose_data_.position.x = current_pose_.x;
            current_pose_data_.position.y = current_pose_.y;
            current_pose_data_.position.z = 0.0;
            tf2::Quaternion q;
            q.setRPY(0, 0, current_pose_.theta);
            current_pose_data_.orientation.x = q.x();
            current_pose_data_.orientation.y = q.y();
            current_pose_data_.orientation.z = q.z();
            current_pose_data_.orientation.w = q.w();
            has_latest_pose_ = true;
        } catch (const tf2::TransformException& e) {
            RCLCPP_WARN(get_logger(), "Failed to get yaw from AMCL: %s", e.what());
        }
    }
    
    void pathCallback(const nav_msgs::msg::Path::SharedPtr msg) {
        if (!msg || msg->poses.empty()) {
            RCLCPP_WARN(get_logger(), "Received empty or null global path");
            goal_received_ = false;
            return;
        }
        
        // 转换路径
        auto path2d = rosPathToPath2D(*msg);
        
        // 设置到规划器
        planner_->setGlobalPath(path2d);
        
        goal_received_ = true;
        goal_reached_ = false;
        
        RCLCPP_INFO(get_logger(), "Received global path: %zu poses", msg->poses.size());
    }
    
    // =========================================================================
    // 规划循环
    // =========================================================================
    
    void planningTimerCallback() {
        if (!goal_received_ || !planner_->hasValidGlobalPath()) {
            publishZeroVelocity();
            return;
        }
         // 更新机器人位姿
         planner_->setCurrentPose(current_pose_);
         //更新机器人速度，使用上一周期的速度
         planner_->setCurrentVelocity(current_velocity_.linear.x, current_velocity_.linear.y, current_velocity_.angular.z);
        //设置障碍物
        {
            //加锁
            std::lock_guard<std::mutex> lock(obstacles_mutex_);
            planner_->setObstacles(all_obstacles_);
        }
        // 执行规划
        auto result = planner_->plan();
        
        if (result.success) {
            // 发布速度命令
            publishVelocity(result.velocity_x, result.velocity_theta);
            
            // 发布可视化数据
            publishVisualization(result.trajectory, result.aim_pose);
            
            RCLCPP_DEBUG(get_logger(), "Planning successful: vx=%.3f, omega=%.3f",
                        result.velocity_x, result.velocity_theta);
        } else {
            RCLCPP_WARN(get_logger(), "Planning failed: %s", result.message.c_str());
            publishZeroVelocity();
        }
    }
    
    // 发布可视化数据
    void publishVisualization(const std::vector<teb_local_planner::Pose2D>& trajectory,
                              const teb_local_planner::Pose2D& aim_pose) {
        // 发布规划轨迹
        if (!trajectory.empty()) {
            auto path_msg = trajectoryToPath(trajectory);
            path_pub_->publish(path_msg);
        }
        
        // 发布目标点
        auto aim_msg = poseToPoseStamped(aim_pose);
        aim_pose_pub_->publish(aim_msg);
        
        // 发布障碍物和边界
        publishAllObstacles();
    }
    
    // 发布障碍物和边界
    void publishAllObstacles() {
        visualization_msgs::msg::MarkerArray marker_array;
        int marker_id = 0;
            
        {
            std::lock_guard<std::mutex> lock(obstacles_mutex_);
            // 预分配容量避免频繁重新分配
            marker_array.markers.reserve(all_obstacles_.size());
            
            for (const auto& obstacle : all_obstacles_) {
                const Eigen::Vector2d centroid_map = obstacle->getCentroid();
                visualization_msgs::msg::Marker marker;
                marker.header.stamp = now();
                marker.header.frame_id = "map";
                marker.id = marker_id++;
                marker.action = visualization_msgs::msg::Marker::ADD;
                marker.type = visualization_msgs::msg::Marker::SPHERE;
                marker.pose.position.x = centroid_map.x();
                marker.pose.position.y = centroid_map.y();
                marker.pose.position.z = 0.0;
                marker.pose.orientation.w = 1.0;
                marker.color.r = 1.0;
                marker.color.g = 0.0;
                marker.color.b = 0.0;
                marker.color.a = 0.9;
                marker.scale.x = marker.scale.y = marker.scale.z = 0.06;
                marker_array.markers.push_back(std::move(marker));
            }
        }
        obstacle_pub_->publish(marker_array);
    }
    
    void publishVelocity(double vx, double omega) {
        geometry_msgs::msg::TwistStamped cmd_vel;
        cmd_vel.header.stamp = now();
        cmd_vel.header.frame_id = "base_link";
        cmd_vel.twist.linear.x = vx;
        cmd_vel.twist.angular.z = omega;
        cmd_vel_pub_->publish(cmd_vel);
        current_velocity_.linear.x = vx;
        current_velocity_.linear.y = 0.0;
        current_velocity_.angular.z = omega;
        // RCLCPP_INFO(get_logger(), "Published velocity: vx=%.3f, omega=%.3f", vx, omega);
    }
    
    void publishZeroVelocity() {
        geometry_msgs::msg::TwistStamped cmd_vel;
        cmd_vel.header.stamp = now();
        cmd_vel.header.frame_id = "base_link";
        cmd_vel.twist.linear.x = 0.0;
        cmd_vel.twist.angular.z = 0.0;
        cmd_vel_pub_->publish(cmd_vel);
        current_velocity_.linear.x = 0.0;
        current_velocity_.linear.y = 0.0;
        current_velocity_.angular.z = 0.0;
        // RCLCPP_INFO(get_logger(), "Published zero velocity");
    }
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TebRos2Node>());
    rclcpp::shutdown();
    return 0;
}
