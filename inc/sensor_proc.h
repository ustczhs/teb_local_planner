#ifndef TEB_SENSOR_PROC_H
#define TEB_SENSOR_PROC_H

#include "obstacles.h"
#include "teb_config.h"
#include <Eigen/Core>
#include <memory>
#include <vector>

namespace teb_local_planner {

// 简化的激光雷达扫描消息结构（非ROS环境）
struct LaserScanData {
  double angle_min;
  double angle_max;
  double angle_increment;
  double range_min;
  double range_max;
  std::vector<float> ranges;
  double timestamp;
};

// 简化的位姿消息结构（非ROS环境）
struct PoseData {
  struct {
    double x, y, z;
  } position;
  struct {
    double x, y, z, w;
  } orientation;
};

// 障碍物记忆结构
struct MemoryObstacle {
  ObstaclePtr obstacle;
  double timestamp;
  double angle; // 障碍物角度，用于快速FOV判断
};

// 障碍物记忆管理类
class ObstacleMemory {
public:
  ObstacleMemory() = default;
  ~ObstacleMemory() = default;

  // FOV内障碍物更新（直接替换）
  void updateFOVObstacles(const ObstContainer &new_obstacles);

  // 添加记忆障碍物
  void addMemoryObstacle(const ObstaclePtr &obs, double angle,
                         double timestamp);

  // 清理过期记忆
  void cleanExpiredMemory(double current_time, double memory_duration);

  // 清理无效记忆障碍物（FOV内、超出距离、超时）
  void cleanInvalidMemory(double current_time, double memory_duration,
                          double local_map_size, double fov_min_angle,
                          double fov_max_angle, const PoseData &current_pose);

  // 删除当前FOV内的记忆障碍物（避免重复）
  void removeMemoryObstaclesInCurrentFOV(double fov_min_angle, double fov_max_angle,
                                         const PoseData &current_pose);

  // 获取障碍物
  ObstContainer getFOVObstacles() const;
  ObstContainer getMemoryObstacles() const;

private:
  ObstContainer fov_obstacles_; // FOV内当前障碍物（基座坐标系）
  std::vector<MemoryObstacle> memory_list_; // FOV外记忆障碍物（世界坐标系）
};

// 传感器处理器主类
class SensorProcessor {
public:
  explicit SensorProcessor(const TebConfig &config);
  ~SensorProcessor();

  // 激光雷达回调函数
  void laserCallback(const LaserScanData &scan_data,
                     const PoseData &current_pose);

  // 获取当前所有障碍物（用于规划）
  ObstContainer getObstaclesForPlanning(const PoseData &current_pose);

private:
  // 降采样
  std::vector<Eigen::Vector2d>
  downsamplePoints(const std::vector<Eigen::Vector2d> &points, int factor);

  // 局部地图大小过滤
  std::vector<Eigen::Vector2d> filterByLocalMapSize(
      const std::vector<Eigen::Vector2d> &points, double local_map_size);

  // 激光雷达数据转换
  std::vector<Eigen::Vector2d>
  convertLaserToPoints(const LaserScanData &scan_data);

  // 坐标变换
  std::vector<Eigen::Vector2d>
  transformPointsToWorld(const std::vector<Eigen::Vector2d> &points_in_base,
                         const PoseData &current_pose);

  std::vector<Eigen::Vector2d>
  transformPointsToBase(const std::vector<Eigen::Vector2d> &points_in_world,
                        const PoseData &current_pose);

  // FOV分离
  std::pair<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>>
  separateFOVPoints(const std::vector<Eigen::Vector2d> &points,
                    double fov_min_angle, double fov_max_angle);

  // 聚类算法
  std::vector<std::vector<Eigen::Vector2d>>
  euclideanClustering(const std::vector<Eigen::Vector2d> &points,
                      double distance_threshold, int min_points);

  // 聚类转障碍物
  ObstContainer clustersToObstacles(
      const std::vector<std::vector<Eigen::Vector2d>> &clusters);

  // 障碍物坐标变换
  ObstContainer transformObstaclesToBase(const ObstContainer &obstacles_world,
                                         const PoseData &current_pose);

private:
  const TebConfig &config_;
  ObstacleMemory memory_;
  double current_time_;
};

} // namespace teb_local_planner

#endif // TEB_SENSOR_PROC_H
