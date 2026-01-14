#include <chrono>
#include <functional>
#include <memory>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"

#include "config_loader.h"
#include "optimal_planner.h"
#include "pose_se2.h"
#include "sensor_proc.h"
#include "visualization.h"

using namespace std::chrono_literals;
using namespace teb_local_planner;

class TebRos2Node : public rclcpp::Node {
public:
  TebRos2Node()
      : Node("teb_local_planner"),
        planning_frequency_(20.0), // Default 20Hz, configurable
        goal_received_(false) {

    // Declare parameters
    this->declare_parameter<std::string>("config_file",
                                         "../config/config.yaml");
    this->declare_parameter<double>("planning_frequency", 20.0);

    // Get parameters
    std::string config_file;
    this->get_parameter("config_file", config_file);
    this->get_parameter("planning_frequency", planning_frequency_);

    // Load configuration
    config_ = loadConfig(config_file);
    RCLCPP_INFO(this->get_logger(), "Loaded configuration from %s",
                config_file.c_str());

    // Create components
    sensor_processor_ = std::make_unique<SensorProcessor>(config_.teb_config);
    visual_ = std::make_unique<TebVisualization>(config_.teb_config);
    auto robot_model = boost::make_shared<CircularRobotFootprint>(
        config_.robot.footprint.radius);
    auto visual_ptr = boost::shared_ptr<TebVisualization>(visual_.get());
    planner_ = std::make_unique<TebOptimalPlanner>(
        config_.teb_config, nullptr, robot_model, visual_ptr, nullptr);

    // Create subscriptions
    laser_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", 10,
        std::bind(&TebRos2Node::laserCallback, this, std::placeholders::_1));

    pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/current_pose", 10,
        std::bind(&TebRos2Node::poseCallback, this, std::placeholders::_1));

    path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
        "/global_path", 10,
        std::bind(&TebRos2Node::pathCallback, this, std::placeholders::_1));

    // Create publisher
    cmd_vel_pub_ =
        this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

    // Create planning timer
    auto planning_period =
        std::chrono::duration<double>(1.0 / planning_frequency_);
    planning_timer_ = this->create_wall_timer(
        planning_period, std::bind(&TebRos2Node::planningTimerCallback, this));

    RCLCPP_INFO(this->get_logger(),
                "TEB ROS2 Node initialized with %.1f Hz planning frequency",
                planning_frequency_);
  }

private:
  // Configuration and components
  PlannerConfig config_;
  std::unique_ptr<SensorProcessor> sensor_processor_;
  std::unique_ptr<TebOptimalPlanner> planner_;
  std::unique_ptr<TebVisualization> visual_;

  // ROS2 interfaces
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::TimerBase::SharedPtr planning_timer_;

  // State variables
  double planning_frequency_;
  PoseSE2 current_pose_;
  nav_msgs::msg::Path global_path_;
  bool goal_received_;
  size_t current_goal_index_;

  // Callback functions
  void laserCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
    // Convert ROS message to internal format
    LaserScanData scan_data;
    scan_data.angle_min = msg->angle_min;
    scan_data.angle_max = msg->angle_max;
    scan_data.angle_increment = msg->angle_increment;
    scan_data.range_min = msg->range_min;
    scan_data.range_max = msg->range_max;
    scan_data.ranges = msg->ranges;
    scan_data.timestamp =
        msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

    // Convert pose to internal format
    PoseData current_pose_data;
    current_pose_data.position.x = current_pose_.x();
    current_pose_data.position.y = current_pose_.y();
    current_pose_data.position.z = 0.0;

    // Convert yaw to quaternion
    tf2::Quaternion q;
    q.setRPY(0, 0, current_pose_.theta());
    current_pose_data.orientation.x = q.x();
    current_pose_data.orientation.y = q.y();
    current_pose_data.orientation.z = q.z();
    current_pose_data.orientation.w = q.w();

    // Process laser data
    sensor_processor_->laserCallback(scan_data, current_pose_data);

    RCLCPP_DEBUG(this->get_logger(), "Processed laser scan with %zu ranges",
                 msg->ranges.size());
  }

  void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    // Extract pose from message
    current_pose_.x() = msg->pose.position.x;
    current_pose_.y() = msg->pose.position.y;

    // Convert quaternion to yaw
    tf2::Quaternion q(msg->pose.orientation.x, msg->pose.orientation.y,
                      msg->pose.orientation.z, msg->pose.orientation.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    current_pose_.theta() = yaw;

    RCLCPP_DEBUG(this->get_logger(), "Updated current pose: (%.2f, %.2f, %.2f)",
                 current_pose_.x(), current_pose_.y(), current_pose_.theta());
  }

  void pathCallback(const nav_msgs::msg::Path::SharedPtr msg) {
    global_path_ = *msg;
    current_goal_index_ = 0;
    goal_received_ = !global_path_.poses.empty();

    if (goal_received_) {
      RCLCPP_INFO(this->get_logger(), "Received global path with %zu poses",
                  global_path_.poses.size());
    } else {
      RCLCPP_WARN(this->get_logger(), "Received empty global path");
    }
  }

  void planningTimerCallback() {
    if (!goal_received_ || global_path_.poses.empty()) {
      // No path to follow, stop
      geometry_msgs::msg::Twist cmd_vel;
      cmd_vel.linear.x = 0.0;
      cmd_vel.angular.z = 0.0;
      cmd_vel_pub_->publish(cmd_vel);
      return;
    }

    // Find the next goal point along the path
    PoseSE2 goal_pose = findNextGoal();

    // Get obstacles from sensor processor
    PoseData current_pose_data;
    current_pose_data.position.x = current_pose_.x();
    current_pose_data.position.y = current_pose_.y();
    current_pose_data.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0, 0, current_pose_.theta());
    current_pose_data.orientation.x = q.x();
    current_pose_data.orientation.y = q.y();
    current_pose_data.orientation.z = q.z();
    current_pose_data.orientation.w = q.w();

    ObstContainer obstacles =
        sensor_processor_->getObstaclesForPlanning(current_pose_data);

    // Update planner obstacles
    planner_->setObstVector(&obstacles);

    try {
      // Plan trajectory
      planner_->plan(current_pose_, goal_pose);

      // Get planned trajectory
      std::vector<Eigen::Vector3f> trajectory;
      planner_->getFullTrajectory(trajectory);

      if (!trajectory.empty()) {
        // Extract velocity from trajectory start
        const auto &current_point = trajectory[0];
        geometry_msgs::msg::Twist cmd_vel;

        // Linear velocity (forward/backward)
        cmd_vel.linear.x = current_point[0]; // vx

        // Angular velocity (rotation)
        cmd_vel.angular.z = current_point[1]; // omega (yaw rate)

        // Ensure velocities are within limits
        cmd_vel.linear.x =
            std::clamp(cmd_vel.linear.x, -config_.robot.max_vel_x_backwards,
                       config_.robot.max_vel_x);
        cmd_vel.angular.z =
            std::clamp(cmd_vel.angular.z, -config_.robot.max_vel_theta,
                       config_.robot.max_vel_theta);

        cmd_vel_pub_->publish(cmd_vel);

        RCLCPP_DEBUG(this->get_logger(),
                     "Published cmd_vel: linear=%.2f, angular=%.2f",
                     cmd_vel.linear.x, cmd_vel.angular.z);
      } else {
        RCLCPP_WARN(this->get_logger(), "Empty trajectory from planner");
        // Stop if no valid trajectory
        geometry_msgs::msg::Twist cmd_vel;
        cmd_vel.linear.x = 0.0;
        cmd_vel.angular.z = 0.0;
        cmd_vel_pub_->publish(cmd_vel);
      }

    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Planning failed: %s", e.what());
      // Stop on planning failure
      geometry_msgs::msg::Twist cmd_vel;
      cmd_vel.linear.x = 0.0;
      cmd_vel.angular.z = 0.0;
      cmd_vel_pub_->publish(cmd_vel);
    }
  }

  // Helper function to find the next goal point along the path
  PoseSE2 findNextGoal() {
    // Look ahead distance for local goal
    const double LOOKAHEAD_DISTANCE = 3.0; // meters

    double accumulated_distance = 0.0;
    size_t goal_index = current_goal_index_;

    // Start from current goal index and find point within lookahead distance
    for (size_t i = current_goal_index_; i < global_path_.poses.size() - 1;
         ++i) {
      const auto &current_point = global_path_.poses[i].pose.position;
      const auto &next_point = global_path_.poses[i + 1].pose.position;

      double dx = next_point.x - current_point.x;
      double dy = next_point.y - current_point.y;
      double segment_length = std::sqrt(dx * dx + dy * dy);

      if (accumulated_distance + segment_length >= LOOKAHEAD_DISTANCE) {
        // Interpolate point at lookahead distance
        double remaining_distance = LOOKAHEAD_DISTANCE - accumulated_distance;
        double ratio = remaining_distance / segment_length;

        PoseSE2 goal;
        goal.x() = current_point.x + dx * ratio;
        goal.y() = current_point.y + dy * ratio;

        // Use orientation from the next path point
        tf2::Quaternion q(global_path_.poses[i + 1].pose.orientation.x,
                          global_path_.poses[i + 1].pose.orientation.y,
                          global_path_.poses[i + 1].pose.orientation.z,
                          global_path_.poses[i + 1].pose.orientation.w);
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        goal.theta() = yaw;

        return goal;
      }

      accumulated_distance += segment_length;
      goal_index = i + 1;
    }

    // If we reach the end of the path, use the final point
    const auto &final_pose = global_path_.poses.back().pose;
    PoseSE2 goal;
    goal.x() = final_pose.position.x;
    goal.y() = final_pose.position.y;

    tf2::Quaternion q(final_pose.orientation.x, final_pose.orientation.y,
                      final_pose.orientation.z, final_pose.orientation.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    goal.theta() = yaw;

    // Update current goal index for next iteration
    current_goal_index_ = goal_index;

    return goal;
  }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TebRos2Node>());
  rclcpp::shutdown();
  return 0;
}
