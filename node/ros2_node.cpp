#include <chrono>
#include <functional>
#include <memory>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "visualization_msgs/msg/marker_array.hpp"

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
        "/plan", 10,
        std::bind(&TebRos2Node::pathCallback, this, std::placeholders::_1));

    // Create publishers
    cmd_vel_pub_ =
        this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel_teb", 10);
    path_pub_ =
        this->create_publisher<nav_msgs::msg::Path>("/local_plan_teb", 10);
    obstacle_pub_ =
        this->create_publisher<visualization_msgs::msg::MarkerArray>("/obstacles_marker", 10);

    // Create debug point cloud publishers
    debug_raw_points_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("/debug/raw_points", 10);
    debug_downsampled_points_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("/debug/downsampled_points", 10);
    debug_local_filtered_points_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("/debug/local_filtered_points", 10);
    debug_fov_points_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("/debug/fov_points", 10);
    debug_non_fov_points_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("/debug/non_fov_points", 10);
    debug_fov_clusters_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("/debug/fov_clusters", 10);
    debug_non_fov_clusters_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("/debug/non_fov_clusters", 10);
    debug_memory_obstacles_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("/debug/memory_obstacles", 10);
    obstacles_points_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("/obstacles_points", 10);
    obstacles_clusters_marker_pub_ =
        this->create_publisher<visualization_msgs::msg::MarkerArray>("/obstacles_clusters_marker", 10);

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
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr obstacle_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_raw_points_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_downsampled_points_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_local_filtered_points_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_fov_points_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_non_fov_points_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_fov_clusters_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_non_fov_clusters_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_memory_obstacles_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obstacles_points_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr obstacles_clusters_marker_pub_;
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
    // RCLCPP_INFO(this->get_logger(), "Obstacles for planning: %zu",
    //             obstacles.size());

    // Helper function to transform point from base_link to map
    auto transformPointToMap = [&](const Eigen::Vector2d& point_base) -> Eigen::Vector2d {
      double robot_x = current_pose_.x();
      double robot_y = current_pose_.y();
      double robot_theta = current_pose_.theta();

      double cos_theta = std::cos(robot_theta);
      double sin_theta = std::sin(robot_theta);

      double map_x = robot_x + cos_theta * point_base.x() - sin_theta * point_base.y();
      double map_y = robot_y + sin_theta * point_base.x() + cos_theta * point_base.y();

      return Eigen::Vector2d(map_x, map_y);
    };

    // Publish obstacles for visualization
    visualization_msgs::msg::MarkerArray marker_array;
    int marker_id = 0;
    for (const auto& obst : obstacles) {
      visualization_msgs::msg::Marker marker;
      marker.header.stamp = this->now();
      marker.header.frame_id = "map";
      marker.id = marker_id++;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.color.r = 1.0;
      marker.color.g = 0.0;
      marker.color.b = 0.0;
      marker.color.a = 1.0;

      const Eigen::Vector2d centroid_base = obst->getCentroid();
      const Eigen::Vector2d centroid_map = transformPointToMap(centroid_base);

      if (dynamic_cast<PointObstacle*>(obst.get())) {
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.pose.position.x = centroid_map[0];
        marker.pose.position.y = centroid_map[1];
        marker.pose.position.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.scale.z = 0.1;
      } else if (auto* circ_obst = dynamic_cast<CircularObstacle*>(obst.get())) {
        marker.type = visualization_msgs::msg::Marker::CYLINDER;
        marker.pose.position.x = centroid_map[0];
        marker.pose.position.y = centroid_map[1];
        marker.pose.position.z = 0.0;
        marker.pose.orientation.w = 1.0;
        double radius = circ_obst->radius();
        marker.scale.x = radius * 2;
        marker.scale.y = radius * 2;
        marker.scale.z = 0.1;
      } else if (auto* line_obst = dynamic_cast<LineObstacle*>(obst.get())) {
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.05;
        const Eigen::Vector2d start_map = transformPointToMap(line_obst->start());
        const Eigen::Vector2d end_map = transformPointToMap(line_obst->end());
        geometry_msgs::msg::Point p1, p2;
        p1.x = start_map.x();
        p1.y = start_map.y();
        p1.z = 0.0;
        p2.x = end_map.x();
        p2.y = end_map.y();
        p2.z = 0.0;
        marker.points.push_back(p1);
        marker.points.push_back(p2);
      } else if (auto* poly_obst = dynamic_cast<PolygonObstacle*>(obst.get())) {
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.05;
        for (const auto& vertex : poly_obst->vertices()) {
          const Eigen::Vector2d vertex_map = transformPointToMap(vertex);
          geometry_msgs::msg::Point p;
          p.x = vertex_map.x();
          p.y = vertex_map.y();
          p.z = 0.0;
          marker.points.push_back(p);
        }
        // Close the polygon
        if (!poly_obst->vertices().empty()) {
          const Eigen::Vector2d vertex_map = transformPointToMap(poly_obst->vertices().front());
          geometry_msgs::msg::Point p;
          p.x = vertex_map.x();
          p.y = vertex_map.y();
          p.z = 0.0;
          marker.points.push_back(p);
        }
      }

      marker_array.markers.push_back(marker);
    }
    obstacle_pub_->publish(marker_array);

    // Publish debug point clouds
    publishDebugPointCloud(sensor_processor_->getDebugRawPoints(), debug_raw_points_pub_);
    publishDebugPointCloud(sensor_processor_->getDebugDownsampledPoints(), debug_downsampled_points_pub_);
    publishDebugPointCloud(sensor_processor_->getDebugLocalFilteredPoints(), debug_local_filtered_points_pub_);
    publishDebugPointCloud(sensor_processor_->getDebugFovPoints(), debug_fov_points_pub_);
    publishDebugPointCloud(sensor_processor_->getDebugNonFovPoints(), debug_non_fov_points_pub_);
    publishDebugPointCloud(sensor_processor_->getDebugFovClusters(), debug_fov_clusters_pub_);
    publishDebugPointCloud(sensor_processor_->getDebugNonFovClusters(), debug_non_fov_clusters_pub_);
    publishDebugPointCloud(sensor_processor_->getDebugMemoryObstacles(), debug_memory_obstacles_pub_);

    // Publish obstacles points and clusters
    auto obstacle_points = sensor_processor_->getObstaclesPointsForPlanning(current_pose_data);
    publishDebugPointCloud(obstacle_points, obstacles_points_pub_);

    // Cluster obstacles and publish markers
    auto clusters = clusterObstaclePoints(obstacle_points);
    publishObstacleClusters(clusters);

    // Update planner obstacles
    planner_->setObstVector(&obstacles);

    try {
      // Plan trajectory
      // RCLCPP_INFO(this->get_logger(), "Planning from (%.2f, %.2f, %.2f) to "
      //                                 "(%.2f, %.2f, %.2f)",
      //             current_pose_.x(), current_pose_.y(), current_pose_.theta(),
      //             goal_pose.x(), goal_pose.y(), goal_pose.theta());
      planner_->plan(current_pose_, goal_pose);

      // Get planned trajectory
      std::vector<Eigen::Vector3f> trajectory;
      planner_->getFullTrajectory(trajectory);

      // Publish the planned path
      nav_msgs::msg::Path path_msg;
      path_msg.header.stamp = this->now();
      path_msg.header.frame_id = "map"; // Assuming odom frame for world coordinates
      for (const auto &point : trajectory) {
        geometry_msgs::msg::PoseStamped pose;
        pose.header = path_msg.header;
        pose.pose.position.x = point[0];
        pose.pose.position.y = point[1];
        pose.pose.position.z = 0.0;
        tf2::Quaternion q;
        q.setRPY(0, 0, point[2]);
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();
        path_msg.poses.push_back(pose);
      }
      path_pub_->publish(path_msg);
      RCLCPP_DEBUG(this->get_logger(), "Published local plan with %zu poses",
                   path_msg.poses.size());

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

  // Helper function to publish debug point cloud
  void publishDebugPointCloud(const std::vector<Eigen::Vector2d>& points,
                              rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher) {
    sensor_msgs::msg::PointCloud2 cloud_msg;
    cloud_msg.header.stamp = this->now();
    cloud_msg.header.frame_id = "map";

    cloud_msg.height = 1;
    cloud_msg.width = points.size();
    cloud_msg.is_dense = true;
    cloud_msg.is_bigendian = false;

    // Define fields
    sensor_msgs::msg::PointField x_field;
    x_field.name = "x";
    x_field.offset = 0;
    x_field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    x_field.count = 1;

    sensor_msgs::msg::PointField y_field;
    y_field.name = "y";
    y_field.offset = 4;
    y_field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    y_field.count = 1;

    sensor_msgs::msg::PointField z_field;
    z_field.name = "z";
    z_field.offset = 8;
    z_field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    z_field.count = 1;

    cloud_msg.fields = {x_field, y_field, z_field};
    cloud_msg.point_step = 12;  // 3 * 4 bytes
    cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;

    // Fill data
    cloud_msg.data.resize(cloud_msg.row_step * cloud_msg.height);
    uint8_t* data_ptr = cloud_msg.data.data();

    for (size_t i = 0; i < points.size(); ++i) {
      float* point_ptr = reinterpret_cast<float*>(data_ptr + i * cloud_msg.point_step);
      point_ptr[0] = static_cast<float>(points[i].x());  // x
      point_ptr[1] = static_cast<float>(points[i].y());  // y
      point_ptr[2] = 0.0f;  // z
    }

    publisher->publish(cloud_msg);
  }

  // Helper function to cluster obstacle points
  std::vector<std::vector<Eigen::Vector2d>> clusterObstaclePoints(const std::vector<Eigen::Vector2d>& points) {
    // Use simple Euclidean clustering with distance threshold 0.3m and min points 2
    const double CLUSTER_DISTANCE = 0.3;
    const int MIN_POINTS = 2;

    std::vector<std::vector<Eigen::Vector2d>> clusters;
    std::vector<bool> visited(points.size(), false);

    for (size_t i = 0; i < points.size(); ++i) {
      if (visited[i]) continue;

      std::vector<Eigen::Vector2d> cluster;
      std::vector<size_t> seeds = {i};
      visited[i] = true;

      while (!seeds.empty()) {
        size_t seed_idx = seeds.back();
        seeds.pop_back();
        cluster.push_back(points[seed_idx]);

        for (size_t j = 0; j < points.size(); ++j) {
          if (visited[j]) continue;

          double distance = (points[j] - points[seed_idx]).norm();
          if (distance <= CLUSTER_DISTANCE) {
            visited[j] = true;
            seeds.push_back(j);
          }
        }
      }

      if (static_cast<int>(cluster.size()) >= MIN_POINTS) {
        clusters.push_back(cluster);
      }
    }

    return clusters;
  }

  // Helper function to publish obstacle clusters as markers
  void publishObstacleClusters(const std::vector<std::vector<Eigen::Vector2d>>& clusters) {
    visualization_msgs::msg::MarkerArray marker_array;
    int marker_id = 0;

    for (const auto& cluster : clusters) {
      if (cluster.empty()) continue;

      // Calculate cluster centroid
      Eigen::Vector2d centroid(0, 0);
      for (const auto& point : cluster) {
        centroid += point;
      }
      centroid /= cluster.size();

      // Calculate cluster radius
      double max_radius = 0.0;
      for (const auto& point : cluster) {
        double dist = (point - centroid).norm();
        max_radius = std::max(max_radius, dist);
      }

      // Create cylinder marker for cluster
      visualization_msgs::msg::Marker marker;
      marker.header.stamp = this->now();
      marker.header.frame_id = "map";
      marker.id = marker_id++;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.type = visualization_msgs::msg::Marker::CYLINDER;
      marker.pose.position.x = centroid.x();
      marker.pose.position.y = centroid.y();
      marker.pose.position.z = 0.0;
      marker.pose.orientation.w = 1.0;
      marker.scale.x = max_radius * 2.0;
      marker.scale.y = max_radius * 2.0;
      marker.scale.z = 0.05;
      marker.color.r = 0.0;
      marker.color.g = 1.0;
      marker.color.b = 0.0;
      marker.color.a = 0.7;

      marker_array.markers.push_back(marker);
    }

    obstacles_clusters_marker_pub_->publish(marker_array);
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
