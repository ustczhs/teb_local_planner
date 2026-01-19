#include <ros/ros.h>
#include <ros/console.h>
#include <ros/time.h>
#include <chrono>
#include <functional>
#include <memory>
#include <cmath>

// ROS1 消息头文件
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Twist.h"
#include "nav_msgs/Path.h"
#include "sensor_msgs/LaserScan.h"
#include "sensor_msgs/PointCloud2.h"
#include "tf/LinearMath/Matrix3x3.h"
#include "tf/LinearMath/Quaternion.h"
#include "visualization_msgs/MarkerArray.h"
#include "sensor_msgs/point_cloud2_iterator.h"

// 自定义头文件（保持不变）
#include "config_loader.h"
#include "optimal_planner.h"
#include "pose_se2.h"
#include "sensor_proc.h"
#include "visualization.h"

using namespace teb_local_planner;

class TebRos1Node {
public:
  TebRos1Node(ros::NodeHandle& nh)
      : nh_(nh),
        planning_frequency_(20.0), // 默认20Hz
        goal_received_(false) {

    // -------------------------- 1. 参数读取（ROS1 API） --------------------------
    std::string config_file;
    nh_.param<std::string>("config_file", config_file, "../config/config.yaml");
    nh_.param<double>("planning_frequency", planning_frequency_, 20.0);

    // 加载配置
    config_ = loadConfig(config_file);
    ROS_INFO("Loaded configuration from %s", config_file.c_str());

    // -------------------------- 2. 创建核心组件 --------------------------
    sensor_processor_ = std::make_unique<SensorProcessor>(config_.teb_config);
    visual_ = std::make_unique<TebVisualization>(config_.teb_config);
    auto robot_model = boost::make_shared<CircularRobotFootprint>(
        config_.robot.footprint.radius);
    auto visual_ptr = boost::shared_ptr<TebVisualization>(visual_.get());
    planner_ = std::make_unique<TebOptimalPlanner>(
        config_.teb_config, nullptr, robot_model, visual_ptr, nullptr);

    // -------------------------- 3. 订阅器（ROS1 API） --------------------------
    laser_sub_ = nh_.subscribe("/scan", 10, &TebRos1Node::laserCallback, this);
    pose_sub_ = nh_.subscribe("/current_pose", 10, &TebRos1Node::poseCallback, this);
    path_sub_ = nh_.subscribe("/plan", 10, &TebRos1Node::pathCallback, this);

    // -------------------------- 4. 发布器（ROS1 API） --------------------------
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel_teb", 10);
    path_pub_ = nh_.advertise<nav_msgs::Path>("/local_plan_teb", 10);
    obstacle_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/obstacles_marker", 10);

    // 调试点云发布器
    debug_raw_points_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/debug/raw_points", 10);
    debug_downsampled_points_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/debug/downsampled_points", 10);
    debug_local_filtered_points_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/debug/local_filtered_points", 10);
    debug_fov_points_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/debug/fov_points", 10);
    debug_non_fov_points_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/debug/non_fov_points", 10);
    debug_fov_clusters_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/debug/fov_clusters", 10);
    debug_non_fov_clusters_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/debug/non_fov_clusters", 10);
    debug_memory_obstacles_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/debug/memory_obstacles", 10);
    obstacles_points_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/obstacles_points", 10);
    obstacles_clusters_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/obstacles_clusters_marker", 10);

    // -------------------------- 5. 规划定时器（ROS1 API） --------------------------
    double planning_period = 1.0 / planning_frequency_;
    planning_timer_ = nh_.createTimer(ros::Duration(planning_period), &TebRos1Node::planningTimerCallback, this);

    ROS_INFO("TEB ROS1 Node initialized with %.1f Hz planning frequency", planning_frequency_);
  }

private:
  // ROS1 核心对象
  ros::NodeHandle nh_;
  ros::Subscriber laser_sub_;
  ros::Subscriber pose_sub_;
  ros::Subscriber path_sub_;
  ros::Publisher cmd_vel_pub_;
  ros::Publisher path_pub_;
  ros::Publisher obstacle_pub_;
  ros::Publisher debug_raw_points_pub_;
  ros::Publisher debug_downsampled_points_pub_;
  ros::Publisher debug_local_filtered_points_pub_;
  ros::Publisher debug_fov_points_pub_;
  ros::Publisher debug_non_fov_points_pub_;
  ros::Publisher debug_fov_clusters_pub_;
  ros::Publisher debug_non_fov_clusters_pub_;
  ros::Publisher debug_memory_obstacles_pub_;
  ros::Publisher obstacles_points_pub_;
  ros::Publisher obstacles_clusters_marker_pub_;
  ros::Timer planning_timer_;

  // 配置和核心组件
  PlannerConfig config_;
  std::unique_ptr<SensorProcessor> sensor_processor_;
  std::unique_ptr<TebOptimalPlanner> planner_;
  std::unique_ptr<TebVisualization> visual_;

  // 状态变量
  double planning_frequency_;
  PoseSE2 current_pose_;
  nav_msgs::Path global_path_;
  bool goal_received_;
  size_t current_goal_index_;

  // -------------------------- 回调函数（适配ROS1消息类型） --------------------------
  void laserCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
    // 转换ROS1消息到内部格式
    LaserScanData scan_data;
    scan_data.angle_min = msg->angle_min;
    scan_data.angle_max = msg->angle_max;
    scan_data.angle_increment = msg->angle_increment;
    scan_data.range_min = msg->range_min;
    scan_data.range_max = msg->range_max;
    scan_data.ranges = msg->ranges;
    scan_data.timestamp = msg->header.stamp.toSec();

    // 构造当前位姿数据
    PoseData current_pose_data;
    current_pose_data.position.x = current_pose_.x();
    current_pose_data.position.y = current_pose_.y();
    current_pose_data.position.z = 0.0;

    // 转换yaw到四元数
    tf::Quaternion q;
    q.setRPY(0, 0, current_pose_.theta());
    current_pose_data.orientation.x = q.x();
    current_pose_data.orientation.y = q.y();
    current_pose_data.orientation.z = q.z();
    current_pose_data.orientation.w = q.w();

    // 处理激光数据
    sensor_processor_->laserCallback(scan_data, current_pose_data);

    ROS_DEBUG("Processed laser scan with %zu ranges", msg->ranges.size());
  }

  void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    // 提取位姿
    current_pose_.x() = msg->pose.position.x;
    current_pose_.y() = msg->pose.position.y;

    // 四元数转yaw（ROS1 tf接口）
    tf::Quaternion q(msg->pose.orientation.x, msg->pose.orientation.y,
                     msg->pose.orientation.z, msg->pose.orientation.w);
    tf::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    current_pose_.theta() = yaw;

    ROS_DEBUG("Updated current pose: (%.2f, %.2f, %.2f)",
              current_pose_.x(), current_pose_.y(), current_pose_.theta());
  }

  void pathCallback(const nav_msgs::Path::ConstPtr& msg) {
    global_path_ = *msg;
    current_goal_index_ = 0;
    goal_received_ = !global_path_.poses.empty();

    if (goal_received_) {
      ROS_INFO("Received global path with %zu poses", global_path_.poses.size());
    } else {
      ROS_WARN("Received empty global path");
    }
  }

  void planningTimerCallback(const ros::TimerEvent& e) {
    if (!goal_received_ || global_path_.poses.empty()) {
      // 无路径时停止机器人
      geometry_msgs::Twist cmd_vel;
      cmd_vel.linear.x = 0.0;
      cmd_vel.angular.z = 0.0;
      cmd_vel_pub_.publish(cmd_vel);
      return;
    }

    // 查找下一个目标点
    PoseSE2 goal_pose = findNextGoal();

    // 构造当前位姿数据
    PoseData current_pose_data;
    current_pose_data.position.x = current_pose_.x();
    current_pose_data.position.y = current_pose_.y();
    current_pose_data.position.z = 0.0;

    tf::Quaternion q;
    q.setRPY(0, 0, current_pose_.theta());
    current_pose_data.orientation.x = q.x();
    current_pose_data.orientation.y = q.y();
    current_pose_data.orientation.z = q.z();
    current_pose_data.orientation.w = q.w();

    // 获取规划用障碍物
    ObstContainer obstacles = sensor_processor_->getObstaclesForPlanning(current_pose_data);

    // 基座坐标系转地图坐标系的辅助函数
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

    // 发布障碍物可视化Marker
    visualization_msgs::MarkerArray marker_array;
    int marker_id = 0;
    for (const auto& obst : obstacles) {
      visualization_msgs::Marker marker;
      marker.header.stamp = ros::Time::now();
      marker.header.frame_id = "map";
      marker.id = marker_id++;
      marker.action = visualization_msgs::Marker::ADD;
      marker.color.r = 1.0;
      marker.color.g = 0.0;
      marker.color.b = 0.0;
      marker.color.a = 1.0;

      const Eigen::Vector2d centroid_base = obst->getCentroid();
      const Eigen::Vector2d centroid_map = transformPointToMap(centroid_base);

      if (dynamic_cast<PointObstacle*>(obst.get())) {
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.pose.position.x = centroid_map[0];
        marker.pose.position.y = centroid_map[1];
        marker.pose.position.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.scale.z = 0.1;
      } else if (auto* circ_obst = dynamic_cast<CircularObstacle*>(obst.get())) {
        marker.type = visualization_msgs::Marker::CYLINDER;
        marker.pose.position.x = centroid_map[0];
        marker.pose.position.y = centroid_map[1];
        marker.pose.position.z = 0.0;
        marker.pose.orientation.w = 1.0;
        double radius = circ_obst->radius();
        marker.scale.x = radius * 2;
        marker.scale.y = radius * 2;
        marker.scale.z = 0.1;
      } else if (auto* line_obst = dynamic_cast<LineObstacle*>(obst.get())) {
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.05;
        const Eigen::Vector2d start_map = transformPointToMap(line_obst->start());
        const Eigen::Vector2d end_map = transformPointToMap(line_obst->end());
        geometry_msgs::Point p1, p2;
        p1.x = start_map.x();
        p1.y = start_map.y();
        p1.z = 0.0;
        p2.x = end_map.x();
        p2.y = end_map.y();
        p2.z = 0.0;
        marker.points.push_back(p1);
        marker.points.push_back(p2);
      } else if (auto* poly_obst = dynamic_cast<PolygonObstacle*>(obst.get())) {
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.05;
        for (const auto& vertex : poly_obst->vertices()) {
          const Eigen::Vector2d vertex_map = transformPointToMap(vertex);
          geometry_msgs::Point p;
          p.x = vertex_map.x();
          p.y = vertex_map.y();
          p.z = 0.0;
          marker.points.push_back(p);
        }
        // 闭合多边形
        if (!poly_obst->vertices().empty()) {
          const Eigen::Vector2d vertex_map = transformPointToMap(poly_obst->vertices().front());
          geometry_msgs::Point p;
          p.x = vertex_map.x();
          p.y = vertex_map.y();
          p.z = 0.0;
          marker.points.push_back(p);
        }
      }

      marker_array.markers.push_back(marker);
    }
    obstacle_pub_.publish(marker_array);

    // 发布调试点云
    publishDebugPointCloud(sensor_processor_->getDebugRawPoints(), debug_raw_points_pub_);
    publishDebugPointCloud(sensor_processor_->getDebugDownsampledPoints(), debug_downsampled_points_pub_);
    publishDebugPointCloud(sensor_processor_->getDebugLocalFilteredPoints(), debug_local_filtered_points_pub_);
    publishDebugPointCloud(sensor_processor_->getDebugFovPoints(), debug_fov_points_pub_);
    publishDebugPointCloud(sensor_processor_->getDebugNonFovPoints(), debug_non_fov_points_pub_);
    publishDebugPointCloud(sensor_processor_->getDebugFovClusters(), debug_fov_clusters_pub_);
    publishDebugPointCloud(sensor_processor_->getDebugNonFovClusters(), debug_non_fov_clusters_pub_);
    publishDebugPointCloud(sensor_processor_->getDebugMemoryObstacles(), debug_memory_obstacles_pub_);

    // 发布障碍物点云和聚类
    auto obstacle_points = sensor_processor_->getObstaclesPointsForPlanning(current_pose_data);
    publishDebugPointCloud(obstacle_points, obstacles_points_pub_);

    // 聚类并发布Marker（需实现clusterObstaclePoints/publishObstacleClusters）
    auto clusters = clusterObstaclePoints(obstacle_points);
    publishObstacleClusters(clusters);

    // 更新规划器障碍物
    planner_->setObstVector(&obstacles);

    try {
      // 规划轨迹
      planner_->plan(current_pose_, goal_pose);

      // 获取规划轨迹
      std::vector<Eigen::Vector3f> trajectory;
      planner_->getFullTrajectory(trajectory);

      // 发布局部路径
      nav_msgs::Path path_msg;
      path_msg.header.stamp = ros::Time::now();
      path_msg.header.frame_id = "map";
      for (const auto &point : trajectory) {
        geometry_msgs::PoseStamped pose;
        pose.header = path_msg.header;
        pose.pose.position.x = point[0];
        pose.pose.position.y = point[1];
        pose.pose.position.z = 0.0;
        tf::Quaternion q;
        q.setRPY(0, 0, point[2]);
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();
        path_msg.poses.push_back(pose);
      }
      path_pub_.publish(path_msg);
      ROS_DEBUG("Published local plan with %zu poses", path_msg.poses.size());

      if (!trajectory.empty()) {
        // 提取速度指令
        const auto &current_point = trajectory[0];
        geometry_msgs::Twist cmd_vel;
        cmd_vel.linear.x = current_point[0]; // 线速度
        cmd_vel.angular.z = current_point[1]; // 角速度

        // 速度限幅
        cmd_vel.linear.x = std::clamp(cmd_vel.linear.x, -config_.robot.max_vel_x_backwards, config_.robot.max_vel_x);
        cmd_vel.angular.z = std::clamp(cmd_vel.angular.z, -config_.robot.max_vel_theta, config_.robot.max_vel_theta);

        cmd_vel_pub_.publish(cmd_vel);
        ROS_DEBUG("Published cmd_vel: linear=%.2f, angular=%.2f", cmd_vel.linear.x, cmd_vel.angular.z);
      } else {
        ROS_WARN("Empty trajectory from planner");
        geometry_msgs::Twist cmd_vel;
        cmd_vel.linear.x = 0.0;
        cmd_vel.angular.z = 0.0;
        cmd_vel_pub_.publish(cmd_vel);
      }

    } catch (const std::exception &e) {
      ROS_ERROR("Planning failed: %s", e.what());
      geometry_msgs::Twist cmd_vel;
      cmd_vel.linear.x = 0.0;
      cmd_vel.angular.z = 0.0;
      cmd_vel_pub_.publish(cmd_vel);
    }
  }

  // -------------------------- 辅助函数 --------------------------
  // 查找下一个目标点（需根据你的逻辑实现，此处占位）
  PoseSE2 findNextGoal() {
    // 示例：取全局路径的最后一个点作为目标
    if (global_path_.poses.empty()) {
      return PoseSE2(0, 0, 0);
    }
    const auto& goal_pose_msg = global_path_.poses.back();
    tf::Quaternion q(goal_pose_msg.pose.orientation.x,
                     goal_pose_msg.pose.orientation.y,
                     goal_pose_msg.pose.orientation.z,
                     goal_pose_msg.pose.orientation.w);
    tf::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    return PoseSE2(goal_pose_msg.pose.position.x, goal_pose_msg.pose.position.y, yaw);
  }

  // 发布调试点云（ROS1版本）
  void publishDebugPointCloud(const std::vector<Eigen::Vector2d>& points,
                              ros::Publisher& publisher) {
    sensor_msgs::PointCloud2 cloud_msg;
    cloud_msg.header.stamp = ros::Time::now();
    cloud_msg.header.frame_id = "map";
    cloud_msg.height = 1;
    cloud_msg.width = points.size();
    cloud_msg.is_dense = true;
    cloud_msg.is_bigendian = false;

    // 定义点云字段（x/y/z）
    sensor_msgs::PointField x_field;
    x_field.name = "x";
    x_field.offset = 0;
    x_field.datatype = sensor_msgs::PointField::FLOAT32;
    x_field.count = 1;

    sensor_msgs::PointField y_field;
    y_field.name = "y";
    y_field.offset = 4;
    y_field.datatype = sensor_msgs::PointField::FLOAT32;
    y_field.count = 1;

    sensor_msgs::PointField z_field;
    z_field.name = "z";
    z_field.offset = 8;
    z_field.datatype = sensor_msgs::PointField::FLOAT32;
    z_field.count = 1;

    cloud_msg.fields.push_back(x_field);
    cloud_msg.fields.push_back(y_field);
    cloud_msg.fields.push_back(z_field);
    cloud_msg.point_step = 12; // x(4) + y(4) + z(4)
    cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;

    // 填充点云数据
    cloud_msg.data.resize(cloud_msg.row_step * cloud_msg.height);
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");

    for (const auto& point : points) {
      *iter_x = point.x();
      *iter_y = point.y();
      *iter_z = 0.0;
      ++iter_x;
      ++iter_y;
      ++iter_z;
    }

    publisher.publish(cloud_msg);
  }

  // 聚类障碍物点云（需根据你的逻辑实现，此处占位）
  std::vector<std::vector<Eigen::Vector2d>> clusterObstaclePoints(const std::vector<Eigen::Vector2d>& points) {
    std::vector<std::vector<Eigen::Vector2d>> clusters;
    // 替换为你的聚类逻辑
    if (!points.empty()) {
      clusters.push_back(points);
    }
    return clusters;
  }

  // 发布障碍物聚类Marker（需根据你的逻辑实现，此处占位）
  void publishObstacleClusters(const std::vector<std::vector<Eigen::Vector2d>>& clusters) {
    visualization_msgs::MarkerArray marker_array;
    // 替换为你的Marker发布逻辑
    obstacles_clusters_marker_pub_.publish(marker_array);
  }
};

// -------------------------- 主函数（ROS1入口） --------------------------
int main(int argc, char** argv) {
  ros::init(argc, argv, "teb_local_planner");
  ros::NodeHandle nh("~"); // 私有命名空间

  TebRos1Node teb_node(nh);

  ros::spin(); // ROS1 自旋

  return 0;
}
