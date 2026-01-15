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

// 局部地图大小过滤函数（在基座坐标系下）
std::vector<Eigen::Vector2d> SensorProcessor::filterByLocalMapSize(
    const std::vector<Eigen::Vector2d> &points, double local_map_size) {
  std::vector<Eigen::Vector2d> filtered_points;

  for (const auto &point : points) {
    double distance = point.norm();
    if (distance <= local_map_size) {
      filtered_points.push_back(point);
    }
  }

  return filtered_points;
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

  // turtlebot4激光雷达相对于base_link顺时针旋转90度
  const double LASER_INSTALL_ANGLE = M_PI / 2.0; // 90度

  for (size_t i = 0; i < scan_data.ranges.size(); ++i) {
    float range = scan_data.ranges[i];

    // 过滤无效数据
    if (std::isnan(range) || std::isinf(range) || range < scan_data.range_min ||
        range > scan_data.range_max) {
      continue;
    }

    // 计算角度，加上激光雷达安装角度偏移
    double angle = angle_min + i * angle_increment + LASER_INSTALL_ANGLE;

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

  // 0. 处理上一帧 FOV 内的点：哪些现在滑出去了，就直接加到 memory（作为 PointObstacle）
  for (const auto& prev_p_world : previous_fov_points_world_) {
    // 转到当前 base_link 坐标系
    auto prev_p_base_vec = transformPointsToBase({prev_p_world}, current_pose);
    Eigen::Vector2d prev_p_base = prev_p_base_vec.front();

    double distance = prev_p_base.norm();
    if (distance > config_.sensor.local_map_size) {
      continue;  // 超出局部地图大小，不记忆
    }

    double angle = std::atan2(prev_p_base.y(), prev_p_base.x());
    // 稳定归一化角度到 [-π, π]
    angle = std::fmod(angle + 3 * M_PI, 2 * M_PI) - M_PI;

    if (angle < config_.sensor.fov_min_angle || angle > config_.sensor.fov_max_angle) {
      // 这个点滑出去了，直接创建 PointObstacle 加到 memory
      ObstaclePtr point_obs(new PointObstacle(prev_p_world.x(), prev_p_world.y()));

      // 存的角度用当前帧的角度（cleanInvalidMemory 会用到）
      double stored_angle = angle;

      memory_.addMemoryObstacle(point_obs, stored_angle, current_time_);

      // 调试日志（强烈推荐加）
      // std::cout << "[SLIDE OUT] Added point to memory: world ("
      //           << prev_p_world.x() << ", " << prev_p_world.y()
      //           << "), current angle " << (angle * 180.0 / M_PI) << "°" << std::endl;
    }
  }

  // 1. 转换为点云（基座坐标系）
  auto raw_points = convertLaserToPoints(scan_data);
  debug_raw_points_ = transformPointsToWorld(raw_points, current_pose);

  // 2. 降采样
  auto points = downsamplePoints(raw_points, config_.sensor.downsample_factor);
  debug_downsampled_points_ = transformPointsToWorld(points, current_pose);

  // 3. 局部地图大小过滤（基座坐标系）
  auto local_points = filterByLocalMapSize(points, config_.sensor.local_map_size);
  debug_local_filtered_points_ = transformPointsToWorld(local_points, current_pose);

  // 4. FOV分离（基座坐标系）
  auto [fov_points_base, non_fov_points_base] = separateFOVPoints(
      local_points, config_.sensor.fov_min_angle, config_.sensor.fov_max_angle);
  debug_fov_points_ = transformPointsToWorld(fov_points_base, current_pose);
  debug_non_fov_points_ = transformPointsToWorld(non_fov_points_base, current_pose);

  // 5. FOV内聚类（在base_link坐标系下进行）
  auto fov_clusters_base = euclideanClustering(
      fov_points_base, config_.sensor.cluster_distance_threshold,
      config_.sensor.cluster_min_points);
  auto fov_obstacles_base = clustersToObstacles(fov_clusters_base);

  // 收集FOV内聚类点云（转换为世界坐标系用于调试）
  debug_fov_clusters_.clear();
  for (const auto& cluster : fov_clusters_base) {
    auto cluster_world = transformPointsToWorld(cluster, current_pose);
    debug_fov_clusters_.insert(debug_fov_clusters_.end(),
                               cluster_world.begin(), cluster_world.end());
  }

  // 6. 删除当前FOV内的记忆障碍物（避免重复）
  memory_.removeMemoryObstaclesInCurrentFOV(config_.sensor.fov_min_angle,
                                            config_.sensor.fov_max_angle,
                                            current_pose);

  // 7. 清理无效记忆障碍物（超时、超距、进入FOV）
  memory_.cleanInvalidMemory(current_time_, config_.sensor.obstacle_memory_time,
                             config_.sensor.local_map_size,
                             config_.sensor.fov_min_angle,
                             config_.sensor.fov_max_angle, current_pose);

  // 8. 更新FOV障碍物（直接替换）
  memory_.updateFOVObstacles(fov_obstacles_base);

  // 9. 保存当前FOV内点用于下一帧（转换为世界坐标系）
  // 清空旧的
  previous_fov_points_world_.clear();

  // 把当前 FOV 内点转到世界坐标保存
  for (const auto& p_base : fov_points_base) {  // fov_points_base 是 base_link 坐标系下的点
    auto p_world_vec = transformPointsToWorld({p_base}, current_pose);
    previous_fov_points_world_.push_back(p_world_vec.front());
  }

  // 调试日志（强烈推荐加）
  // std::cout << "[DEBUG] Saved " << previous_fov_points_world_.size()
  //           << " previous FOV points (world) for next frame" << std::endl;

  // 10. 更新记忆障碍物调试数据
  debug_memory_obstacles_.clear();
  auto memory_obstacles = memory_.getMemoryObstacles();
  for (const auto& obs : memory_obstacles) {
    if (auto circular_obs = boost::dynamic_pointer_cast<CircularObstacle>(obs)) {
      Eigen::Vector2d pos(circular_obs->position().x(), circular_obs->position().y());
      debug_memory_obstacles_.push_back(pos);
    } else if (auto point_obs = boost::dynamic_pointer_cast<PointObstacle>(obs)) {
      Eigen::Vector2d pos(point_obs->position().x(), point_obs->position().y());
      debug_memory_obstacles_.push_back(pos);
    }
  }
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

// 获取当前所有障碍物的点云（map坐标系，用于可视化）
std::vector<Eigen::Vector2d>
SensorProcessor::getObstaclesPointsForPlanning(const PoseData &current_pose) {

  std::vector<Eigen::Vector2d> obstacle_points;

  // FOV内原始点云（与_debug_fov_points一致）
  obstacle_points.insert(obstacle_points.end(),
                         debug_fov_points_.begin(),
                         debug_fov_points_.end());

  // FOV外记忆障碍物点云
  obstacle_points.insert(obstacle_points.end(),
                         debug_memory_obstacles_.begin(),
                         debug_memory_obstacles_.end());

  return obstacle_points;
}

// 障碍物坐标变换：基座坐标系 -> 世界坐标系
ObstContainer
SensorProcessor::transformObstaclesToWorld(const ObstContainer &obstacles_base,
                                           const PoseData &current_pose) {

  ObstContainer obstacles_world;

  double robot_x = current_pose.position.x;
  double robot_y = current_pose.position.y;
  double robot_yaw =
      quaternionToYaw(current_pose.orientation.x, current_pose.orientation.y,
                      current_pose.orientation.z, current_pose.orientation.w);

  Eigen::Rotation2Dd rotation(robot_yaw);
  Eigen::Vector2d translation(robot_x, robot_y);

  for (const auto &obs : obstacles_base) {
    if (auto circular_obs =
            boost::dynamic_pointer_cast<CircularObstacle>(obs)) {
      // 变换圆形障碍物
      Eigen::Vector2d centroid_base(circular_obs->position().x(),
                                    circular_obs->position().y());
      Eigen::Vector2d centroid_world = rotation * centroid_base + translation;

      ObstaclePtr new_obs(new CircularObstacle(
          centroid_world.x(), centroid_world.y(), circular_obs->radius()));
      obstacles_world.push_back(new_obs);
    } else if (auto point_obs =
                   boost::dynamic_pointer_cast<PointObstacle>(obs)) {
      // 变换点障碍物
      Eigen::Vector2d pos_base(point_obs->position().x(),
                               point_obs->position().y());
      Eigen::Vector2d pos_world = rotation * pos_base + translation;

      ObstaclePtr new_obs(new PointObstacle(pos_world.x(), pos_world.y()));
      obstacles_world.push_back(new_obs);
    }
    // 其他类型的障碍物可以类似处理
  }

  return obstacles_world;
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

void ObstacleMemory::cleanInvalidMemory(double current_time,
                                        double memory_duration,
                                        double local_map_size,
                                        double fov_min_angle,
                                        double fov_max_angle,
                                        const PoseData &current_pose) {
  // 四元数到欧拉角转换
  auto quaternionToYaw = [](double qx, double qy, double qz, double qw) {
    double siny_cosp = 2.0 * (qw * qz + qx * qy);
    double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    return std::atan2(siny_cosp, cosy_cosp);
  };

  double robot_x = current_pose.position.x;
  double robot_y = current_pose.position.y;
  double robot_yaw = quaternionToYaw(current_pose.orientation.x,
                                     current_pose.orientation.y,
                                     current_pose.orientation.z,
                                     current_pose.orientation.w);

  Eigen::Rotation2Dd rotation(-robot_yaw);
  Eigen::Vector2d translation(-robot_x, -robot_y);

  // std::cout << "[DEBUG] cleanInvalidMemory: checking " << memory_list_.size()
  //           << " memory obstacles, current_time=" << current_time
  //           << ", memory_duration=" << memory_duration
  //           << ", local_map_size=" << local_map_size
  //           << ", fov=[" << (fov_min_angle * 180.0 / M_PI) << ", "
  //           << (fov_max_angle * 180.0 / M_PI) << "] degrees" << std::endl;

  auto it = std::remove_if(
      memory_list_.begin(), memory_list_.end(),
      [&](const MemoryObstacle &mem_obs) {
        double age = current_time - mem_obs.timestamp;
        // std::cout << "[DEBUG] Checking memory obstacle: timestamp=" << mem_obs.timestamp
        //           << ", age=" << age << "s, stored_angle=" << (mem_obs.angle * 180.0 / M_PI) << "°" << std::endl;

        // 1. 检查是否超时
        if (age > memory_duration) {
          // std::cout << "[DEBUG] -> REMOVING: timeout (age=" << age << " > " << memory_duration << ")" << std::endl;
          return true;
        }

        // 2. 将障碍物转到当前base_link坐标系，检查距离和FOV
        Eigen::Vector2d pos_base;
        bool valid_obstacle = false;
        std::string obstacle_type;

        if (auto circular_obs = boost::dynamic_pointer_cast<CircularObstacle>(mem_obs.obstacle)) {
          Eigen::Vector2d centroid_world(circular_obs->position().x(),
                                         circular_obs->position().y());
          pos_base = rotation * (centroid_world + translation);
          valid_obstacle = true;
          obstacle_type = "Circular";
          // std::cout << "[DEBUG] -> Circular obstacle at world (" << centroid_world.x() << ", " << centroid_world.y()
          //           << ") -> base (" << pos_base.x() << ", " << pos_base.y() << ")" << std::endl;
        } else if (auto point_obs = boost::dynamic_pointer_cast<PointObstacle>(mem_obs.obstacle)) {
          Eigen::Vector2d pos_world(point_obs->position().x(),
                                    point_obs->position().y());
          pos_base = rotation * (pos_world + translation);
          valid_obstacle = true;
          obstacle_type = "Point";
          // std::cout << "[DEBUG] -> Point obstacle at world (" << pos_world.x() << ", " << pos_world.y()
          //           << ") -> base (" << pos_base.x() << ", " << pos_base.y() << ")" << std::endl;
        } else {
          // std::cout << "[DEBUG] -> REMOVING: unknown obstacle type" << std::endl;
          return true; // 未知类型，删除
        }

        if (valid_obstacle) {
          double distance = pos_base.norm();
          double current_angle = std::atan2(pos_base.y(), pos_base.x());

          // 归一化角度到 [-pi, pi]
          while (current_angle > M_PI) current_angle -= 2 * M_PI;
          while (current_angle < -M_PI) current_angle += 2 * M_PI;

          // std::cout << "[DEBUG] -> " << obstacle_type << " obstacle: distance=" << distance
          //           << "m, current_angle=" << (current_angle * 180.0 / M_PI) << "°" << std::endl;

          // 如果超出local_map_size范围，删除
          if (distance > local_map_size) {
            // std::cout << "[DEBUG] -> REMOVING: out of range (" << distance << " > " << local_map_size << ")" << std::endl;
            return true;
          }

          // 如果当前角度在FOV内，删除（用最新扫描数据覆盖）
          if (current_angle >= fov_min_angle && current_angle <= fov_max_angle) {
            // std::cout << "[DEBUG] -> REMOVING: in current FOV (" << (current_angle * 180.0 / M_PI)
            //           << "° in [" << (fov_min_angle * 180.0 / M_PI) << ", " << (fov_max_angle * 180.0 / M_PI) << "])" << std::endl;
            return true;
          }

          // std::cout << "[DEBUG] -> KEEPING: valid memory obstacle" << std::endl;
        }

        return false; // 保留
      });

  int removed_count = std::distance(it, memory_list_.end());
  memory_list_.erase(it, memory_list_.end());

  // std::cout << "[DEBUG] cleanInvalidMemory: removed " << removed_count
  //           << " obstacles, " << memory_list_.size() << " remaining" << std::endl;
}

void ObstacleMemory::removeMemoryObstaclesInCurrentFOV(double fov_min_angle, double fov_max_angle,
                                                       const PoseData &current_pose) {
  // 四元数到欧拉角转换
  auto quaternionToYaw = [](double qx, double qy, double qz, double qw) {
    double siny_cosp = 2.0 * (qw * qz + qx * qy);
    double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    return std::atan2(siny_cosp, cosy_cosp);
  };

  double robot_x = current_pose.position.x;
  double robot_y = current_pose.position.y;
  double robot_yaw = quaternionToYaw(current_pose.orientation.x,
                                     current_pose.orientation.y,
                                     current_pose.orientation.z,
                                     current_pose.orientation.w);

  Eigen::Rotation2Dd rotation(-robot_yaw);
  Eigen::Vector2d translation(-robot_x, -robot_y);

  auto it = std::remove_if(
      memory_list_.begin(), memory_list_.end(),
      [&](const MemoryObstacle &mem_obs) {
        Eigen::Vector2d pos_base;
        if (auto circular_obs = boost::dynamic_pointer_cast<CircularObstacle>(mem_obs.obstacle)) {
          Eigen::Vector2d centroid_world(circular_obs->position().x(),
                                         circular_obs->position().y());
          pos_base = rotation * (centroid_world + translation);
        } else if (auto point_obs = boost::dynamic_pointer_cast<PointObstacle>(mem_obs.obstacle)) {
          Eigen::Vector2d pos_world(point_obs->position().x(),
                                    point_obs->position().y());
          pos_base = rotation * (pos_world + translation);
        } else {
          return false; // 未知类型，保留
        }

        double current_angle = std::atan2(pos_base.y(), pos_base.x());
        while (current_angle > M_PI) current_angle -= 2 * M_PI;
        while (current_angle < -M_PI) current_angle += 2 * M_PI;

        // 如果当前角度在FOV内，删除
        return (current_angle >= fov_min_angle && current_angle <= fov_max_angle);
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

// 调试接口实现
std::vector<Eigen::Vector2d> SensorProcessor::getDebugRawPoints() const {
  return debug_raw_points_;
}

std::vector<Eigen::Vector2d> SensorProcessor::getDebugDownsampledPoints() const {
  return debug_downsampled_points_;
}

std::vector<Eigen::Vector2d> SensorProcessor::getDebugLocalFilteredPoints() const {
  return debug_local_filtered_points_;
}

std::vector<Eigen::Vector2d> SensorProcessor::getDebugFovPoints() const {
  return debug_fov_points_;
}

std::vector<Eigen::Vector2d> SensorProcessor::getDebugNonFovPoints() const {
  return debug_non_fov_points_;
}

std::vector<Eigen::Vector2d> SensorProcessor::getDebugFovClusters() const {
  return debug_fov_clusters_;
}

std::vector<Eigen::Vector2d> SensorProcessor::getDebugNonFovClusters() const {
  return debug_non_fov_clusters_;
}

std::vector<Eigen::Vector2d> SensorProcessor::getDebugMemoryObstacles() const {
  return debug_memory_obstacles_;
}

} // namespace teb_local_planner
