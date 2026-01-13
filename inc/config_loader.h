#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include "teb_config.h"
#include <Eigen/Core>
#include <string>
#include <vector>

namespace teb_local_planner {

struct RobotConfig {
  std::string type;
  double max_vel_x, max_vel_x_backwards, max_vel_y, max_vel_theta;
  double acc_lim_x, acc_lim_y, acc_lim_theta;
  double min_turning_radius, wheelbase;
  bool cmd_angle_instead_rotvel;

  struct Footprint {
    std::string type;
    double radius;
    std::vector<Eigen::Vector2d> vertices;
  } footprint;
};

struct SimulationConfig {
  double dt;
  double robot_speed;
  double lookahead_distance;

  struct Obstacles {
    int count;
    double amplitude, frequency, phase_offset;
  } obstacles;
};

struct VisualizationConfig {
  bool show_corridor_boundaries;
  bool show_obstacles;
  bool show_trajectory;
  double scale_factor;
};

struct PlannerConfig {
  RobotConfig robot;
  SimulationConfig simulation;
  VisualizationConfig visualization;
  TebConfig teb_config; // TEB配置
};

/**
 * @brief Load configuration from YAML file
 * @param config_file Path to YAML config file
 * @return PlannerConfig structure
 */
PlannerConfig loadConfig(const std::string &config_file);

} // namespace teb_local_planner

#endif // CONFIG_LOADER_H
