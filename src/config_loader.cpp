#include "config_loader.h"
#include <fstream>
#include <iostream>
#include <yaml-cpp/yaml.h>

namespace teb_local_planner {

PlannerConfig loadConfig(const std::string &config_file) {
  PlannerConfig config;

  try {
    YAML::Node yaml = YAML::LoadFile(config_file);

    // Load robot configuration
    if (yaml["robot"]) {
      auto robot_node = yaml["robot"];

      config.robot.type =
          robot_node["type"].as<std::string>("differential_drive");
      config.robot.max_vel_x = robot_node["max_vel_x"].as<double>(0.8);
      config.robot.max_vel_x_backwards =
          robot_node["max_vel_x_backwards"].as<double>(0.3);
      config.robot.max_vel_y = robot_node["max_vel_y"].as<double>(0.0);
      config.robot.max_vel_theta = robot_node["max_vel_theta"].as<double>(0.5);
      config.robot.acc_lim_x = robot_node["acc_lim_x"].as<double>(0.5);
      config.robot.acc_lim_y = robot_node["acc_lim_y"].as<double>(0.5);
      config.robot.acc_lim_theta = robot_node["acc_lim_theta"].as<double>(0.5);
      config.robot.min_turning_radius =
          robot_node["min_turning_radius"].as<double>(0.0);
      config.robot.wheelbase = robot_node["wheelbase"].as<double>(0.5);
      config.robot.cmd_angle_instead_rotvel =
          robot_node["cmd_angle_instead_rotvel"].as<bool>(false);

      // Footprint configuration
      if (robot_node["footprint"]) {
        auto footprint_node = robot_node["footprint"];
        config.robot.footprint.type =
            footprint_node["type"].as<std::string>("circle");
        config.robot.footprint.radius =
            footprint_node["radius"].as<double>(0.2);

        if (footprint_node["vertices"]) {
          auto vertices_node = footprint_node["vertices"];
          for (const auto &vertex : vertices_node) {
            double x = vertex[0].as<double>();
            double y = vertex[1].as<double>();
            config.robot.footprint.vertices.emplace_back(x, y);
          }
        }
      }
    }

    // Load trajectory configuration
    if (yaml["trajectory"]) {
      auto traj_node = yaml["trajectory"];
      config.teb_config.trajectory.dt_ref = traj_node["dt_ref"].as<double>(0.3);
      config.teb_config.trajectory.dt_hysteresis =
          traj_node["dt_hysteresis"].as<double>(0.1);
      config.teb_config.trajectory.min_samples =
          traj_node["min_samples"].as<int>(3);
      config.teb_config.trajectory.max_samples =
          traj_node["max_samples"].as<int>(200);
      config.teb_config.trajectory.force_reinit_new_goal_dist =
          traj_node["force_reinit_new_goal_dist"].as<double>(1.0);
      config.teb_config.trajectory.force_reinit_new_goal_angular =
          traj_node["force_reinit_new_goal_angular"].as<double>(1.57);
    }

    // Load optimization configuration
    if (yaml["optimization"]) {
      auto opt_node = yaml["optimization"];
      config.teb_config.optim.no_inner_iterations =
          opt_node["no_inner_iterations"].as<int>(5);
      config.teb_config.optim.no_outer_iterations =
          opt_node["no_outer_iterations"].as<int>(4);
      config.teb_config.optim.weight_max_vel_x =
          opt_node["weight_max_vel_x"].as<double>(2);
      config.teb_config.optim.weight_max_vel_theta =
          opt_node["weight_max_vel_theta"].as<double>(1);
      config.teb_config.optim.weight_acc_lim_x =
          opt_node["weight_acc_lim_x"].as<double>(1);
      config.teb_config.optim.weight_acc_lim_theta =
          opt_node["weight_acc_lim_theta"].as<double>(1);
      config.teb_config.optim.weight_kinematics_nh =
          opt_node["weight_kinematics_nh"].as<double>(1000);
      config.teb_config.optim.weight_kinematics_forward_drive =
          opt_node["weight_kinematics_forward_drive"].as<double>(1000); // 防止后退
      config.teb_config.optim.weight_obstacle =
          opt_node["weight_obstacle"].as<double>(50);
      config.teb_config.optim.weight_viapoint =
          opt_node["weight_viapoint"].as<double>(500);
    }

    // Load obstacles configuration
    if (yaml["obstacles"]) {
      auto obs_node = yaml["obstacles"];
      config.teb_config.obstacles.min_obstacle_dist =
          obs_node["min_obstacle_dist"].as<double>(0.5);
      config.teb_config.obstacles.inflation_dist =
          obs_node["inflation_dist"].as<double>(0.6);
      config.teb_config.obstacles.obstacle_poses_affected =
          obs_node["obstacle_poses_affected"].as<int>(15);
           config.teb_config.obstacles.obstacle_square_size =
          obs_node["obstacle_square_size"].as<double>(5.0);
            
    }

    // Load sensor configuration
    if (yaml["sensor"]) {
      auto sensor_node = yaml["sensor"];
      config.teb_config.sensor.laser_topic =
          sensor_node["laser_topic"].as<std::string>("/scan");
      config.teb_config.sensor.fov_min_angle =
          sensor_node["fov_min_angle"].as<double>(-1.745);
      config.teb_config.sensor.fov_max_angle =
          sensor_node["fov_max_angle"].as<double>(1.745);
      config.teb_config.sensor.obstacle_memory_time =
          sensor_node["obstacle_memory_time"].as<double>(3.0);
      config.teb_config.sensor.cluster_distance_threshold =
          sensor_node["cluster_distance_threshold"].as<double>(0.1);
      config.teb_config.sensor.cluster_min_points =
          sensor_node["cluster_min_points"].as<int>(3);
      config.teb_config.sensor.downsample_factor =
          sensor_node["downsample_factor"].as<int>(2);
    }

    // Load simulation configuration
    if (yaml["simulation"]) {
      auto sim_node = yaml["simulation"];
      config.simulation.dt = sim_node["dt"].as<double>(0.1);
      config.simulation.robot_speed = sim_node["robot_speed"].as<double>(0.5);
      config.simulation.lookahead_distance =
          sim_node["lookahead_distance"].as<double>(4.0);

      if (sim_node["obstacles"]) {
        auto obs_node = sim_node["obstacles"];
        config.simulation.obstacles.count = obs_node["count"].as<int>(2);
        config.simulation.obstacles.amplitude =
            obs_node["amplitude"].as<double>(0.6);
        config.simulation.obstacles.frequency =
            obs_node["frequency"].as<double>(0.5);
        config.simulation.obstacles.phase_offset =
            obs_node["phase_offset"].as<double>(1.57);
      }
    }

    // Load visualization configuration
    if (yaml["visualization"]) {
      auto vis_node = yaml["visualization"];
      config.visualization.show_corridor_boundaries =
          vis_node["show_corridor_boundaries"].as<bool>(true);
      config.visualization.show_obstacles =
          vis_node["show_obstacles"].as<bool>(true);
      config.visualization.show_trajectory =
          vis_node["show_trajectory"].as<bool>(true);
      config.visualization.scale_factor =
          vis_node["scale_factor"].as<double>(50.0);
    }

    // Set robot parameters in TEB config
    config.teb_config.robot.max_vel_x = config.robot.max_vel_x;
    config.teb_config.robot.max_vel_x_backwards =
        config.robot.max_vel_x_backwards;
    config.teb_config.robot.max_vel_y = config.robot.max_vel_y;
    config.teb_config.robot.max_vel_theta = config.robot.max_vel_theta;
    config.teb_config.robot.acc_lim_x = config.robot.acc_lim_x;
    config.teb_config.robot.acc_lim_y = config.robot.acc_lim_y;
    config.teb_config.robot.acc_lim_theta = config.robot.acc_lim_theta;
    config.teb_config.robot.min_turning_radius =
        config.robot.min_turning_radius;
    config.teb_config.robot.wheelbase = config.robot.wheelbase;
    config.teb_config.robot.cmd_angle_instead_rotvel =
        config.robot.cmd_angle_instead_rotvel;

    std::cout << "Configuration loaded successfully from " << config_file
              << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error loading configuration from " << config_file << ": "
              << e.what() << std::endl;
    std::cerr << "Using default configuration." << std::endl;
  }

  return config;
}

} // namespace teb_local_planner
