#include "sensor_proc.h"
#include "obstacles.h"
#include "teb_config.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

namespace teb_local_planner {

SensorProcessor::SensorProcessor(const TebConfig &config)
    : config_(config), current_time_(0.0) {}

SensorProcessor::~SensorProcessor() {}

// 降采样函数
std::vector<Eigen::Vector2d>
SensorProcessor::downsamplePoints(const std::vector<Eigen::Vector2d> &points,
                                  int factor) {
  if (factor <= 1)
    return points;

  std::vector<Eigen::Vector2d> downsampled;
  downsampled.reserve(points.size() / factor);

  for (size_t i = 0; i < points.size(); i += factor) {
    downsampled.push_back(points[i]);
  }
  return downsampled;
}

// 四元数到欧拉角转换（返回yaw角度）
double quaternionToYaw(double qx, double qy, double qz, double qw) {
  // 计算yaw角度（绕Z轴旋转）
  double siny_cosp = 2.0 * (qw * qz + qx * qy);
  double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
  return std::atan2(siny_cosp, cosy_cosp);
}

// 将激光雷达数据转换为点云（基座坐标系）
std::vector<Eigen::Vector2d>
SensorProcessor::convertLaserToPoints(const LaserScanData &scan_data) {
  std::vector<Eigen::Vector2d> points;

  // 计算角度增量
  double angle_increment = scan_data.angle_increment;
  double angle_min = scan_data.angle_min;

  for (size_t i = 0; i < scan_data.ranges.size(); ++i) {
    float range = scan_data.ranges[i];

    // 过滤无效数据
    if (std::isnan(range) || std::isinf(range) || range < scan_data.range_min ||
        range > scan_data.range_max) {
      continue;
    }

    // 计算角度
    double angle = angle_min + i * angle_increment;

    // 转换为笛卡尔坐标（基座坐标系）
    double x = range * std::cos(angle);
    double y = range * std::sin(angle);

    points.emplace_back(x, y);
  }

  return points;
}

// 坐标变换：基座坐标系 -> 世界坐标系
std::vector<Eigen::Vector2d> SensorProcessor::transformPointsToWorld(
    const std::vector<Eigen::Vector2d> &points_in_base,
    const PoseData &current_pose) {

  std::vector<Eigen::Vector2d> points_in_world;
  points_in_world.reserve(points_in_base.size());

  // 获取机器人位姿
  double robot_x = current_pose.position.x;
  double robot_y = current_pose.position.y;
  double robot_yaw =
      quaternionToYaw(current_pose.orientation.x, current_pose.orientation.y,
                      current_pose.orientation.z, current_pose.orientation.w);

  // 创建变换矩阵
  Eigen::Rotation2Dd rotation(robot_yaw);
  Eigen::Vector2d translation(robot_x, robot_y);

  for (const auto &point : points_in_base) {
    // 旋转变换 + 平移
    Eigen::Vector2d transformed_point = rotation * point + translation;
    points_in_world.push_back(transformed_point);
  }

  return points_in_world;
}

// 坐标变换：世界坐标系 -> 基座坐标系
std::vector<Eigen::Vector2d> SensorProcessor::transformPointsToBase(
    const std::vector<Eigen::Vector2d> &points_in_world,
    const PoseData &current_pose) {

  std::vector<Eigen::Vector2d> points_in_base;
  points_in_base.reserve(points_in_world.size());

  // 获取机器人位姿
  double robot_x = current_pose.position.x;
  double robot_y = current_pose.position.y;
  double robot_yaw =
      quaternionToYaw(current_pose.orientation.x, current_pose.orientation.y,
                      current_pose.orientation.z, current_pose.orientation.w);

  // 创建逆变换矩阵
  Eigen::Rotation2Dd rotation(-robot_yaw);
  Eigen::Vector2d translation(-robot_x, -robot_y);

  for (const auto &point : points_in_world) {
    // 先平移后旋转
    Eigen::Vector2d transformed_point = rotation * (point + translation);
    points_in_base.push_back(transformed_point);
  }

  return points_in_base;
}

// FOV分离：根据角度判断点是否在FOV内
std::pair<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>>
SensorProcessor::separateFOVPoints(const std::vector<Eigen::Vector2d> &points,
                                   double fov_min_angle, double fov_max_angle) {

  std::vector<Eigen::Vector2d> fov_points;
  std::vector<Eigen::Vector2d> non_fov_points;

  for (const auto &point : points) {
    // 计算点的角度（相对于机器人前进方向）
    double angle = std::atan2(point.y(), point.x());

    // 归一化角度到 [-pi, pi]
    while (angle > M_PI)
      angle -= 2 * M_PI;
    while (angle < -M_PI)
      angle += 2 * M_PI;

    // 判断是否在FOV内
    if (angle >= fov_min_angle && angle <= fov_max_angle) {
      fov_points.push_back(point);
    } else {
      non_fov_points.push_back(point);
    }
  }

  return {fov_points, non_fov_points};
}

// 欧几里德距离聚类
std::vector<std::vector<Eigen::Vector2d>>
SensorProcessor::euclideanClustering(const std::vector<Eigen::Vector2d> &points,
                                     double distance_threshold,
                                     int min_points) {

  std::vector<std::vector<Eigen::Vector2d>> clusters;
  std::vector<bool> visited(points.size(), false);

  for (size_t i = 0; i < points.size(); ++i) {
    if (visited[i])
      continue;

    // 开始新簇
    std::vector<Eigen::Vector2d> cluster;
    std::vector<size_t> seeds = {i};

    visited[i] = true;

    while (!seeds.empty()) {
      size_t seed_idx = seeds.back();
      seeds.pop_back();
      cluster.push_back(points[seed_idx]);

      // 查找邻域点
      for (size_t j = 0; j < points.size(); ++j) {
        if (visited[j])
          continue;

        double distance = (points[j] - points[seed_idx]).norm();
        if (distance <= distance_threshold) {
          visited[j] = true;
          seeds.push_back(j);
        }
      }
    }

    // 只保留满足最小点数的簇
    if (static_cast<int>(cluster.size()) >= min_points) {
      clusters.push_back(cluster);
    }
  }

  return clusters;
}

// 将聚类结果转换为障碍物
ObstContainer SensorProcessor::clustersToObstacles(
    const std::vector<std::vector<Eigen::Vector2d>> &clusters) {

  ObstContainer obstacles;

  for (const auto &cluster : clusters) {
    if (cluster.empty())
      continue;

    // 计算质心
    Eigen::Vector2d centroid(0, 0);
    for (const auto &point : cluster) {
      centroid += point;
    }
    centroid /= cluster.size();

    // 计算半径（到质心的最大距离）
    double max_distance = 0.0;
    for (const auto &point : cluster) {
      double dist = (point - centroid).norm();
      max_distance = std::max(max_distance, dist);
    }

    // 创建圆形障碍物
    ObstaclePtr obstacle(
        new CircularObstacle(centroid.x(), centroid.y(), max_distance));

    obstacles.push_back(obstacle);
  }

  return obstacles;
}

// 激光雷达回调函数
void SensorProcessor::laserCallback(const LaserScanData &scan_data,
                                    const PoseData &current_pose) {

  // 获取当前时间（秒）
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  current_time_ = std::chrono::duration<double>(duration).count();

  // 1. 转换为点云（基座坐标系）
  auto raw_points = convertLaserToPoints(scan_data);

  // 2. 降采样
  auto points = downsamplePoints(raw_points, config_.sensor.downsample_factor);

  // 3. 转换到世界坐标系
  auto world_points = transformPointsToWorld(points, current_pose);

  // 4. FOV分离
  auto [fov_points_world, non_fov_points_world] = separateFOVPoints(
      world_points, config_.sensor.fov_min_angle, config_.sensor.fov_max_angle);

  // 5. FOV内聚类并转换回基座坐标系
  auto fov_clusters_world = euclideanClustering(
      fov_points_world, config_.sensor.cluster_distance_threshold,
      config_.sensor.cluster_min_points);
  auto fov_obstacles_world = clustersToObstacles(fov_clusters_world);
  auto fov_obstacles_base =
      transformObstaclesToBase(fov_obstacles_world, current_pose);

  // 6. FOV外聚类并添加到记忆
  auto non_fov_clusters_world = euclideanClustering(
      non_fov_points_world, config_.sensor.cluster_distance_threshold,
      config_.sensor.cluster_min_points);

  for (const auto &cluster : non_fov_clusters_world) {
    if (cluster.empty())
      continue;

    // 计算簇的质心角度用于快速查找
    Eigen::Vector2d cluster_centroid(0, 0);
    for (const auto &point : cluster) {
      cluster_centroid += point;
    }
    cluster_centroid /= cluster.size();

    double cluster_angle =
        std::atan2(cluster_centroid.y(), cluster_centroid.x());

    // 创建障碍物并添加到记忆
    auto obstacle = clustersToObstacles({cluster}).front();
    memory_.addMemoryObstacle(obstacle, cluster_angle, current_time_);
  }

  // 7. 更新FOV障碍物（直接替换）
  memory_.updateFOVObstacles(fov_obstacles_base);

  // 8. 清理过期记忆
  memory_.cleanExpiredMemory(current_time_,
                             config_.sensor.obstacle_memory_time);
}

// 获取当前所有障碍物（用于规划）
ObstContainer
SensorProcessor::getObstaclesForPlanning(const PoseData &current_pose) {

  ObstContainer all_obstacles;

  // FOV内障碍物（已在基座坐标系）
  auto fov_obs = memory_.getFOVObstacles();
  all_obstacles.insert(all_obstacles.end(), fov_obs.begin(), fov_obs.end());

  // FOV外记忆障碍物转换到当前基座坐标系
  auto memory_obs_world = memory_.getMemoryObstacles();
  auto memory_obs_base =
      transformObstaclesToBase(memory_obs_world, current_pose);
  all_obstacles.insert(all_obstacles.end(), memory_obs_base.begin(),
                       memory_obs_base.end());

  return all_obstacles;
}

// 障碍物坐标变换：世界坐标系 -> 基座坐标系
ObstContainer
SensorProcessor::transformObstaclesToBase(const ObstContainer &obstacles_world,
                                          const PoseData &current_pose) {

  ObstContainer obstacles_base;

  double robot_x = current_pose.position.x;
  double robot_y = current_pose.position.y;
  double robot_yaw =
      quaternionToYaw(current_pose.orientation.x, current_pose.orientation.y,
                      current_pose.orientation.z, current_pose.orientation.w);

  Eigen::Rotation2Dd rotation(-robot_yaw);
  Eigen::Vector2d translation(-robot_x, -robot_y);

  for (const auto &obs : obstacles_world) {
    if (auto circular_obs =
            boost::dynamic_pointer_cast<CircularObstacle>(obs)) {
      // 变换圆形障碍物
      Eigen::Vector2d centroid_world(circular_obs->position().x(),
                                     circular_obs->position().y());
      Eigen::Vector2d centroid_base = rotation * (centroid_world + translation);

      ObstaclePtr new_obs(new CircularObstacle(
          centroid_base.x(), centroid_base.y(), circular_obs->radius()));
      obstacles_base.push_back(new_obs);
    } else if (auto point_obs =
                   boost::dynamic_pointer_cast<PointObstacle>(obs)) {
      // 变换点障碍物
      Eigen::Vector2d pos_world(point_obs->position().x(),
                                point_obs->position().y());
      Eigen::Vector2d pos_base = rotation * (pos_world + translation);

      ObstaclePtr new_obs(new PointObstacle(pos_base.x(), pos_base.y()));
      obstacles_base.push_back(new_obs);
    }
    // 其他类型的障碍物可以类似处理
  }

  return obstacles_base;
}

// 障碍物记忆管理类实现
void ObstacleMemory::updateFOVObstacles(const ObstContainer &new_obstacles) {
  fov_obstacles_ = new_obstacles; // 直接替换，提高效率
}

void ObstacleMemory::addMemoryObstacle(const ObstaclePtr &obs, double angle,
                                       double timestamp) {

  MemoryObstacle mem_obs{obs, timestamp, angle};
  memory_list_.push_back(mem_obs);
}

void ObstacleMemory::cleanExpiredMemory(double current_time,
                                        double memory_duration) {
  auto it = std::remove_if(
      memory_list_.begin(), memory_list_.end(),
      [current_time, memory_duration](const MemoryObstacle &mem_obs) {
        return (current_time - mem_obs.timestamp) > memory_duration;
      });
  memory_list_.erase(it, memory_list_.end());
}

ObstContainer ObstacleMemory::getFOVObstacles() const { return fov_obstacles_; }

ObstContainer ObstacleMemory::getMemoryObstacles() const {
  ObstContainer obstacles;
  obstacles.reserve(memory_list_.size());

  for (const auto &mem_obs : memory_list_) {
    obstacles.push_back(mem_obs.obstacle);
  }

  return obstacles;
}

} // namespace teb_local_planner
