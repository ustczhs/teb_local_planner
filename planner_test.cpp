#include "inc/config_loader.h"
#include "inc/frenet_reference.h"
#include "inc/obstacles.h"
#include "inc/optimal_planner.h"
#include "inc/pose_se2.h"
#include "inc/robot_footprint_model.h"
#include "inc/teb_config.h"
#include "inc/teb_types.h"
#include <boost/smart_ptr.hpp>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <thread>

using namespace teb_local_planner;

struct DynamicObstacle {
  Eigen::Vector2d position;
  Eigen::Vector2d velocity;
  double phase; // for sinusoidal motion
  double amplitude;
  double frequency;
  size_t base_idx; // reference path index
};

void updateObstaclePosition(DynamicObstacle &obs,
                            const Eigen::MatrixXd &path_points, double dt,
                            double time) {
  // Sinusoidal lateral motion within corridor
  double lateral_offset =
      obs.amplitude * std::sin(obs.frequency * time + obs.phase);

  // Base position on reference path
  if (obs.base_idx < path_points.rows()) {
    Eigen::Vector2d base_pos(path_points(obs.base_idx, 0),
                             path_points(obs.base_idx, 1));

    // Get path direction
    Eigen::Vector2d direction;
    if (obs.base_idx < path_points.rows() - 1) {
      direction = Eigen::Vector2d(path_points(obs.base_idx + 1, 0),
                                  path_points(obs.base_idx + 1, 1)) -
                  base_pos;
    } else if (obs.base_idx > 0) {
      direction = base_pos - Eigen::Vector2d(path_points(obs.base_idx - 1, 0),
                                             path_points(obs.base_idx - 1, 1));
    } else {
      direction = Eigen::Vector2d(1.0, 0.0);
    }
    direction.normalize();

    // Perpendicular direction for lateral motion
    Eigen::Vector2d normal(-direction.y(), direction.x());

    obs.position = base_pos + lateral_offset * normal;
  }

  // Update base index to move forward
  obs.base_idx =
      std::min(obs.base_idx + 1, static_cast<size_t>(path_points.rows() - 1));
}

int main() {
  // Load configuration
  PlannerConfig config = loadConfig("../config/config.yaml");

  // Load reference path from data.txt
  Eigen::MatrixXd path_points;
  std::vector<std::vector<double>> data;
  std::ifstream file("../test/data.txt");
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::vector<double> row;
    std::istringstream iss(line);
    std::string token;
    while (std::getline(iss, token, ',')) {
      token.erase(token.begin(), std::find_if(token.begin(), token.end(),
                                              [](unsigned char ch) {
                                                return !std::isspace(ch);
                                              }));
      token.erase(
          std::find_if(token.rbegin(), token.rend(),
                       [](unsigned char ch) { return !std::isspace(ch); })
              .base(),
          token.end());
      if (!token.empty()) {
        try {
          row.push_back(std::stod(token));
        } catch (const std::exception &) {
          // Skip invalid numbers
        }
      }
    }
    if (!row.empty()) {
      data.push_back(row);
    }
  }
  if (data.empty()) {
    std::cerr << "Error: No data loaded from test/data.txt" << std::endl;
    return 1;
  }

  path_points.resize(data.size(), data[0].size());
  for (size_t i = 0; i < data.size(); ++i) {
    for (size_t j = 0; j < data[i].size(); ++j) {
      path_points(i, j) = data[i][j];
    }
  }
  std::cout << "Loaded " << data.size() << " path points" << std::endl;

  // Create global Frenet reference
  auto frenet_ref = boost::make_shared<FrenetReference>(path_points);

  // Use loaded TEB configuration
  std::vector<ObstaclePtr> obst_vector;

  // Robot footprint model based on configuration
  RobotFootprintModelPtr robot_model;
  if (config.robot.footprint.type == "circle") {
    robot_model = boost::make_shared<CircularRobotFootprint>(
        config.robot.footprint.radius);
  } else {
    // Default to circle if polygon not supported yet
    robot_model = boost::make_shared<CircularRobotFootprint>(
        config.robot.footprint.radius);
  }

  auto visual = TebVisualizationPtr(new TebVisualization(config.teb_config));
  auto planner = new TebOptimalPlanner(config.teb_config, &obst_vector,
                                       robot_model, visual, nullptr);
  planner->setFrenetReference(frenet_ref);

  // Simulation parameters from config
  const double DT = config.simulation.dt; // simulation time step
  const double LOOKAHEAD_DISTANCE =
      config.simulation.lookahead_distance;                 // planning horizon
  const double ROBOT_SPEED = config.simulation.robot_speed; // robot speed

  // Robot state
  PoseSE2 robot_pose(path_points(0, 0), path_points(0, 1), path_points(0, 2));
  double robot_velocity = ROBOT_SPEED;
  size_t robot_path_idx = 0;

  // Initialize dynamic obstacles from config
  std::vector<DynamicObstacle> obstacles;
  obstacles.resize(config.simulation.obstacles.count);

  for (int i = 0; i < config.simulation.obstacles.count; ++i) {
    obstacles[i].base_idx = 20 + i * 10; // Spread obstacles along path
    obstacles[i].phase = i * config.simulation.obstacles.phase_offset;
    obstacles[i].amplitude = config.simulation.obstacles.amplitude;
    obstacles[i].frequency = config.simulation.obstacles.frequency;
  }

  // Visualization setup
  cv::Mat show_map(600, 1000, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::namedWindow("TEB Planner Test", cv::WINDOW_AUTOSIZE);

  // Draw reference path
  for (size_t i = 0; i < path_points.rows() - 1; ++i) {
    int x = static_cast<int>(path_points(i, 0) * 50.f + 400);
    int y = static_cast<int>(path_points(i, 1) * 50.f + 300);
    int next_x = static_cast<int>(path_points(i + 1, 0) * 50.f + 400);
    int next_y = static_cast<int>(path_points(i + 1, 1) * 50.f + 300);

    cv::line(show_map, cv::Point(x, y), cv::Point(next_x, next_y),
             cv::Scalar(0, 0, 0), 2); // black reference path
  }

  // Draw corridor boundaries
  for (size_t i = 0; i < path_points.rows(); ++i) {
    Eigen::Vector2d pos(path_points(i, 0), path_points(i, 1));
    double corridor_width = frenet_ref->getCorridorDistanceAtIndex(i);
    double half_width = corridor_width / 2.0;

    // Get path direction
    Eigen::Vector2d direction;
    if (i < path_points.rows() - 1) {
      direction =
          Eigen::Vector2d(path_points(i + 1, 0), path_points(i + 1, 1)) - pos;
    } else if (i > 0) {
      direction =
          pos - Eigen::Vector2d(path_points(i - 1, 0), path_points(i - 1, 1));
    } else {
      direction = Eigen::Vector2d(1.0, 0.0);
    }
    direction.normalize();
    Eigen::Vector2d normal(-direction.y(), direction.x());

    // Left and right boundaries
    Eigen::Vector2d left_pos = pos - half_width * normal;
    Eigen::Vector2d right_pos = pos + half_width * normal;

    int x = static_cast<int>(pos.x() * 50.f + 400);
    int y = static_cast<int>(pos.y() * 50.f + 300);
    int lx = static_cast<int>(left_pos.x() * 50.f + 400);
    int ly = static_cast<int>(left_pos.y() * 50.f + 300);
    int rx = static_cast<int>(right_pos.x() * 50.f + 400);
    int ry = static_cast<int>(right_pos.y() * 50.f + 300);

    cv::circle(show_map, cv::Point(lx, ly), 1, cv::Scalar(0, 255, 0),
               -1); // green left boundary
    cv::circle(show_map, cv::Point(rx, ry), 1, cv::Scalar(0, 255, 0),
               -1); // green right boundary
  }

  // Simulation loop
  double simulation_time = 0.0;
  bool initialized = false;

  while (true) {
    // Create fresh frame
    cv::Mat frame = show_map.clone();

    // Update robot position continuously
    if (robot_path_idx < path_points.rows() - 1) {
      // Move robot forward
      double distance_to_move = robot_velocity * DT;
      double remaining_distance = 0.0;

      while (distance_to_move > 0 && robot_path_idx < path_points.rows() - 1) {
        Eigen::Vector2d current_pos(path_points(robot_path_idx, 0),
                                    path_points(robot_path_idx, 1));
        Eigen::Vector2d next_pos(path_points(robot_path_idx + 1, 0),
                                 path_points(robot_path_idx + 1, 1));
        Eigen::Vector2d segment = next_pos - current_pos;
        double segment_length = segment.norm();

        if (remaining_distance + segment_length <= distance_to_move) {
          remaining_distance += segment_length;
          robot_path_idx++;
        } else {
          // Interpolate position along segment
          double ratio =
              (distance_to_move - remaining_distance) / segment_length;
          Eigen::Vector2d new_pos = current_pos + ratio * segment;
          robot_pose.x() = new_pos.x();
          robot_pose.y() = new_pos.y();
          break;
        }
      }
    }

    // Update obstacle positions
    for (auto &obs : obstacles) {
      updateObstaclePosition(obs, path_points, DT, simulation_time);
    }

    // Update obstacles for planner
    obst_vector.clear();
    for (const auto &obs : obstacles) {
      obst_vector.push_back(boost::make_shared<PointObstacle>(
          obs.position.x(), obs.position.y()));

      // Draw obstacles
      int obs_x = static_cast<int>(obs.position.x() * 50.f + 400);
      int obs_y = static_cast<int>(obs.position.y() * 50.f + 300);
      cv::circle(frame, cv::Point(obs_x, obs_y), 4, cv::Scalar(0, 0, 255),
                 -1); // red obstacles
    }

    planner->setObstVector(&obst_vector);

    // Plan trajectory to lookahead distance
    size_t target_idx = robot_path_idx;
    double accumulated_distance = 0.0;

    for (size_t i = robot_path_idx; i < path_points.rows() - 1; ++i) {
      double dx = path_points(i + 1, 0) - path_points(i, 0);
      double dy = path_points(i + 1, 1) - path_points(i, 1);
      double segment_length = std::sqrt(dx * dx + dy * dy);
      accumulated_distance += segment_length;

      if (accumulated_distance >= LOOKAHEAD_DISTANCE) {
        target_idx = i + 1;
        break;
      }
    }

    if (target_idx >= path_points.rows()) {
      target_idx = path_points.rows() - 1;
    }

    PoseSE2 goal(path_points(target_idx, 0), path_points(target_idx, 1),
                 path_points(target_idx, 2));

    // Create current velocity
    Twist current_velocity;
    current_velocity.linear =
        Eigen::Vector3f(robot_velocity, 0.0, 0.0); // forward velocity
    current_velocity.angular = Eigen::Vector3f(0.0, 0.0, 0.0); // no rotation

    // Plan with warm start and current velocity
    planner->plan(robot_pose, goal, &current_velocity);

    // Get and draw planned trajectory
    std::vector<Eigen::Vector3f> planned_path;
    planner->getFullTrajectory(planned_path);

    for (size_t i = 0; i < planned_path.size() - 1; ++i) {
      int x = static_cast<int>(planned_path[i][0] * 50.f + 400);
      int y = static_cast<int>(planned_path[i][1] * 50.f + 300);
      int next_x = static_cast<int>(planned_path[i + 1][0] * 50.f + 400);
      int next_y = static_cast<int>(planned_path[i + 1][1] * 50.f + 300);

      cv::line(frame, cv::Point(x, y), cv::Point(next_x, next_y),
               cv::Scalar(255, 0, 0), 2); // blue planned path
    }

    // Draw robot
    int robot_x = static_cast<int>(robot_pose.x() * 50.f + 400);
    int robot_y = static_cast<int>(robot_pose.y() * 50.f + 300);
    cv::circle(frame, cv::Point(robot_x, robot_y), 6, cv::Scalar(0, 255, 0),
               -1); // green robot

    // Display info
    std::string info = "Time: " + std::to_string(simulation_time).substr(0, 4) +
                       " Pos: " + std::to_string(robot_path_idx) +
                       " Target: " + std::to_string(target_idx) +
                       " Vel: " + std::to_string(robot_velocity).substr(0, 4);
    cv::putText(frame, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 0, 0), 2);

    cv::imshow("TEB Planner Test", frame);

    // Check for exit
    int key = cv::waitKey(10); // 50ms delay
    if (key == 27) {           // ESC
      break;
    }

    // Update simulation time
    simulation_time += DT;

    // Check if reached end
    if (robot_path_idx >= path_points.rows() - 5) {
      std::cout << "Simulation completed!" << std::endl;
      break;
    }
  }

  delete planner;
  return 0;
}
