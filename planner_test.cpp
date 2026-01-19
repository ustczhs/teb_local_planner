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
#include <vector>
#include <limits>
using namespace teb_local_planner;

struct DynamicObstacle {
  Eigen::Vector2d position;
  Eigen::Vector2d velocity;
  double phase; // for sinusoidal motion
  double amplitude;
  double frequency;
  size_t base_idx; // reference path index
};

// 机器人状态结构体
struct RobotState {
  PoseSE2 pose;
  double v;      // 线速度 (m/s)
  double omega;  // 角速度 (rad/s)
  
  RobotState() : v(0.0), omega(0.0) {}
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

// 绘制正方形视野函数 - 稳定版本
void drawSquareVision(cv::Mat& frame, const PoseSE2& robot_pose, double square_size, 
                      const Eigen::MatrixXd& path_points) {
    if (frame.empty()) {
        return;
    }
    
    // 计算正方形的四个角点（世界坐标系）
    double half_size = square_size / 2.0;
    
    Eigen::Vector2d robot_pos(robot_pose.x(), robot_pose.y());
    double robot_theta = robot_pose.theta();
    
    // 正方形的四个角点（相对于机器人中心，在机器人坐标系中）
    std::vector<Eigen::Vector2d> square_corners_local = {
        Eigen::Vector2d(half_size, half_size),    // 右前角
        Eigen::Vector2d(half_size, -half_size),   // 左前角
        Eigen::Vector2d(-half_size, -half_size),  // 左后角
        Eigen::Vector2d(-half_size, half_size)    // 右后角
    };
    
    // 将正方形角点从机器人坐标系转换到世界坐标系
    std::vector<cv::Point> square_points;
    for (const auto& corner_local : square_corners_local) {
        double x_local = corner_local.x();
        double y_local = corner_local.y();
        
        // 将局部坐标旋转机器人朝向角度
        double x_world = robot_pos.x() + x_local * std::cos(robot_theta) - y_local * std::sin(robot_theta);
        double y_world = robot_pos.y() + x_local * std::sin(robot_theta) + y_local * std::cos(robot_theta);
        
        // 转换到图像坐标系
        int x_img = static_cast<int>(x_world * 50.0 + 400);
        int y_img = static_cast<int>(y_world * 50.0 + 300);
        
        // 确保坐标在图像范围内
        x_img = std::max(0, std::min(x_img, frame.cols - 1));
        y_img = std::max(0, std::min(y_img, frame.rows - 1));
        
        square_points.push_back(cv::Point(x_img, y_img));
    }
    
    // 创建掩码用于半透明填充
    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> polys = {square_points};
    cv::fillPoly(mask, polys, cv::Scalar(255));
    
    // 创建黄色覆盖层
    cv::Mat yellow_overlay(frame.size(), frame.type(), cv::Scalar(0, 255, 255));
    
    // 将黄色区域混合到原图（30%透明度）
    for (int i = 0; i < frame.rows; ++i) {
        for (int j = 0; j < frame.cols; ++j) {
            if (mask.at<uchar>(i, j) > 0) {
                frame.at<cv::Vec3b>(i, j)[0] = 
                    cv::saturate_cast<uchar>(frame.at<cv::Vec3b>(i, j)[0] * 0.7);
                frame.at<cv::Vec3b>(i, j)[1] = 
                    cv::saturate_cast<uchar>(frame.at<cv::Vec3b>(i, j)[1] * 0.7 + 255 * 0.3);
                frame.at<cv::Vec3b>(i, j)[2] = 
                    cv::saturate_cast<uchar>(frame.at<cv::Vec3b>(i, j)[2] * 0.7 + 255 * 0.3);
            }
        }
    }
    
    // 绘制正方形边界（更粗的线）
    for (size_t i = 0; i < square_points.size(); ++i) {
        size_t next_i = (i + 1) % square_points.size();
        cv::line(frame, square_points[i], square_points[next_i], 
                cv::Scalar(0, 200, 200), 3, cv::LINE_AA);
    }
    
    // 绘制机器人位置（正方形中心）
    int robot_x = static_cast<int>(robot_pos.x() * 50.0 + 400);
    int robot_y = static_cast<int>(robot_pos.y() * 50.0 + 300);
    cv::circle(frame, cv::Point(robot_x, robot_y), 10, cv::Scalar(0, 0, 255), 2);
    
    // 绘制机器人朝向箭头
    double arrow_length = 20.0;
    int arrow_x = static_cast<int>(robot_x + arrow_length * std::cos(robot_theta));
    int arrow_y = static_cast<int>(robot_y + arrow_length * std::sin(robot_theta));
    cv::arrowedLine(frame, cv::Point(robot_x, robot_y), cv::Point(arrow_x, arrow_y),
                    cv::Scalar(255, 0, 0), 2, cv::LINE_AA, 0, 0.3);
}

// 简化版绘制函数（备用）
void drawSquareVisionSimple(cv::Mat& frame, const PoseSE2& robot_pose, double square_size) {
    if (frame.empty()) return;
    
    double half_size = square_size / 2.0;
    Eigen::Vector2d robot_pos(robot_pose.x(), robot_pose.y());
    double robot_theta = robot_pose.theta();
    
    // 正方形的四个角点
    std::vector<Eigen::Vector2d> corners = {
        Eigen::Vector2d(half_size, half_size),
        Eigen::Vector2d(half_size, -half_size),
        Eigen::Vector2d(-half_size, -half_size),
        Eigen::Vector2d(-half_size, half_size)
    };
    
    std::vector<cv::Point> points;
    for (const auto& corner : corners) {
        double x = robot_pos.x() + corner.x() * std::cos(robot_theta) - corner.y() * std::sin(robot_theta);
        double y = robot_pos.y() + corner.x() * std::sin(robot_theta) + corner.y() * std::cos(robot_theta);
        
        int x_img = static_cast<int>(x * 50.0 + 400);
        int y_img = static_cast<int>(y * 50.0 + 300);
        
        x_img = std::max(0, std::min(x_img, frame.cols - 1));
        y_img = std::max(0, std::min(y_img, frame.rows - 1));
        
        points.push_back(cv::Point(x_img, y_img));
    }
    
    // 只绘制边界
    for (size_t i = 0; i < points.size(); ++i) {
        size_t next_i = (i + 1) % points.size();
        cv::line(frame, points[i], points[next_i], cv::Scalar(0, 200, 200), 2, cv::LINE_AA);
    }
}

// 从TEB规划器获取速度指令（核心辅助函数）
bool getTebVelocityCommand(const TebOptimalPlanner* planner, float& v_out, float& omega_out) {
  float v_y;
  if(planner->getVelocityCommand(v_out, v_y,omega_out,2))
  {
    std::cout<<"TEB Velocity Command - v: "<<v_out<<", omega: "<<omega_out<<std::endl;
    return true;
  }
  else
  {
    std::cout<<"Failed to get velocity command from TEB!"<<std::endl;
    v_out=0;
    omega_out = 0;
    return false;
  }
  //     // 获取规划的轨迹速度数据
  //     std::vector<double> vel_x, vel_theta;
  //     // 2. 修正调用逻辑
  // std::vector<Twist> velocity_profile;  // 匹配函数参数类型
  // planner->getVelocityProfile(velocity_profile);
  // if (velocity_profile.empty())
  // {
  //   // 处理函数调用失败的逻辑
  //   std::cout<<"Failed to get velocity profile!"<<std::endl;
  //   return false; // 或其他错误处理
  // }

  // // 3. 从Twist数组中提取vel_x和vel_theta（按需）
  // for (const auto& twist : velocity_profile) {
  //     vel_x.push_back(twist.linear.x());       // 提取线速度x
  //     vel_theta.push_back(twist.angular.z());  // 提取角速度z（对应theta）
  //     std::cout<<"Velocity - v_x: "<<twist.linear.x()<<", omega: "<<twist.angular.z()<<std::endl;
  // }

  //     // 取轨迹第一个时间步的速度（当前时刻的速度指令）
  //     if (vel_x.empty() || vel_theta.empty()) {
  //         return false;
  //     }

  //     v_out = vel_x[0];          // 线速度（x方向）
  //     omega_out = vel_theta[0];  // 角速度（theta方向）
  //     std::cout<<"TEB Velocity Command - v: "<<v_out<<", omega: "<<omega_out<<std::endl;
  //     return true;
}

// 差分驱动机器人运动学更新（核心：用TEB输出的速度更新位姿）
void updateRobotState(RobotState& robot, double v, double omega, double dt) {
    // 1. 更新朝向（角速度积分）
    double delta_theta = omega * dt;
    robot.pose.theta() += delta_theta;
    // 规范化角度到[-π, π]
    robot.pose.theta() = atan2(sin(robot.pose.theta()), cos(robot.pose.theta()));
    
    // 2. 更新位置
    if (fabs(omega) < 1e-6) {
        // 直线运动
        robot.pose.x() += v * dt * cos(robot.pose.theta());
        robot.pose.y() += v * dt * sin(robot.pose.theta());
    } else {
        // 圆弧运动（差分驱动运动学）
        double radius = v / omega; // 转弯半径
        double icc_x = robot.pose.x() - radius * sin(robot.pose.theta());
        double icc_y = robot.pose.y() + radius * cos(robot.pose.theta());
        
        // 旋转矩阵更新位置
        double cos_theta = cos(delta_theta);
        double sin_theta = sin(delta_theta);
        
        double dx = robot.pose.x() - icc_x;
        double dy = robot.pose.y() - icc_y;
        
        robot.pose.x() = icc_x + dx * cos_theta - dy * sin_theta;
        robot.pose.y() = icc_y + dx * sin_theta + dy * cos_theta;
    }
    
    // 3. 更新机器人状态的速度
    robot.v = v;
    robot.omega = omega;
}

int main() {
  // Load configuration
  PlannerConfig config = loadConfig("../config/config.yaml");

  // Load reference path from data.txt
  Eigen::MatrixXd path_points;
  std::vector<std::vector<double>> data;
  std::ifstream file("../test/data_expanded.txt");
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
  
  // 设置正方形视野大小（从配置读取或使用默认值）
  double square_vision_size = 5.0; // 默认5米
  if (config.teb_config.obstacles.obstacle_square_size > 0) {
    square_vision_size = config.teb_config.obstacles.obstacle_square_size;
    std::cout << "Using square vision size from config: " << square_vision_size << " meters" << std::endl;
  } else {
    std::cout << "Using default square vision size: " << square_vision_size << " meters" << std::endl;
  }
  
  // 设置到规划器中
  planner->setObstacleSquareSize(square_vision_size);

  // Simulation parameters from config
  const double DT = config.simulation.dt; // simulation time step
  const double LOOKAHEAD_DISTANCE =
      config.simulation.lookahead_distance;                 // planning horizon
  const double ROBOT_SPEED = config.simulation.robot_speed; // 最大参考速度

  // 初始化机器人状态
  RobotState robot_state;
  robot_state.pose = PoseSE2(path_points(0, 0), path_points(0, 1), path_points(0, 2));
  robot_state.v = 0.0; // 初始速度为0（由TEB规划器初始化）
  robot_state.omega = 0.0;     // 初始角速度为0
  
  size_t robot_path_idx = 0;

  // Initialize dynamic obstacles from config
  std::vector<DynamicObstacle> obstacles;
  obstacles.resize(config.simulation.obstacles.count);

  for (int i = 0; i < config.simulation.obstacles.count; ++i) {
    obstacles[i].base_idx = 1 + i * 10; // Spread obstacles along path
    obstacles[i].phase = i * config.simulation.obstacles.phase_offset;
    obstacles[i].amplitude = config.simulation.obstacles.amplitude;
    obstacles[i].frequency = config.simulation.obstacles.frequency;
  }

  // Visualization setup
  cv::Mat show_map(600, 1000, CV_8UC3, cv::Scalar(255, 255, 255));
  if (show_map.empty()) {
      std::cerr << "Failed to create show_map!" << std::endl;
      return -1;
  }
  
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
    
    // 只在某些点显示走廊宽度信息，避免输出太多
    if (i % 50 == 0) {
        std::cout << "Corridor width at index " << i << ": " << corridor_width << std::endl;
    }
    
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
  int frame_count = 0;

  while (true) {
    // Create fresh frame
    cv::Mat frame = show_map.clone();
    if (frame.empty()) {
        std::cerr << "Frame is empty!" << std::endl;
        break;
    }

    // 绘制机器人正方形视野
    drawSquareVision(frame, robot_state.pose, square_vision_size, path_points);

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
                 
      // 在障碍物旁边显示其编号
      std::string obs_label = std::to_string(&obs - &obstacles[0]);
      cv::putText(frame, obs_label, cv::Point(obs_x + 10, obs_y - 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }

    planner->setObstVector(&obst_vector);

    // Plan trajectory to lookahead distance
    // 更新robot_path_idx为最近路径点（用于目标点选择）
    robot_path_idx = 0;
    double min_dist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < path_points.rows(); ++i) {
        double dx = path_points(i, 0) - robot_state.pose.x();
        double dy = path_points(i, 1) - robot_state.pose.y();
        double dist = dx*dx + dy*dy;
        if (dist < min_dist) {
            min_dist = dist;
            robot_path_idx = i;
        }
    }
    
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

    // Create current velocity（传入当前机器人速度，用于TEB暖启动）
    Twist current_velocity;
    current_velocity.linear = Eigen::Vector3f(robot_state.v, 0.0, 0.0);
    current_velocity.angular = Eigen::Vector3f(0.0, 0.0, robot_state.omega);

    // Plan with warm start and current velocity
    bool plan_success = planner->plan(robot_state.pose, goal, &current_velocity);

    // Get and draw planned trajectory
    std::vector<Eigen::Vector3f> planned_path;
    planner->getFullTrajectory(planned_path);
    std::cout<<"Planned path size: "<<planned_path.size()<<std::endl;
    // ==================== 核心修改：使用TEB输出的速度更新机器人状态 ====================
    float teb_v = 0.0, teb_omega = 0.0;
    if (plan_success) {
        // 从TEB规划器获取速度指令
        
        if (getTebVelocityCommand(planner, teb_v, teb_omega)) {
            // 限制最大速度（可选，防止TEB输出速度超限）
            teb_v = std::clamp((double)teb_v, -ROBOT_SPEED, ROBOT_SPEED);
            teb_omega = std::clamp(teb_omega, -1.0f, 1.0f); // 限制最大角速度
            std::cout<<"TEB Velocity Command - v: "<<teb_v<<", omega: "<<teb_omega<<std::endl;
            // 使用TEB速度更新机器人位姿和状态
            updateRobotState(robot_state, teb_v, teb_omega, DT);
        } else {
            // 获取速度失败，减速停止
            robot_state.v *= 0.5;
            robot_state.omega = 0.0;
            updateRobotState(robot_state, robot_state.v, robot_state.omega, DT);
        }
    } else {
        // 规划失败，紧急减速
        robot_state.v *= 0.3;
        robot_state.omega = 0.0;
        updateRobotState(robot_state, robot_state.v, robot_state.omega, DT);
    }
    
    // ==================== 绘制规划轨迹 ====================
    for (size_t i = 0; i < planned_path.size() - 1; ++i) {
      int x = static_cast<int>(planned_path[i][0] * 50.f + 400);
      int y = static_cast<int>(planned_path[i][1] * 50.f + 300);
      int next_x = static_cast<int>(planned_path[i + 1][0] * 50.f + 400);
      int next_y = static_cast<int>(planned_path[i + 1][1] * 50.f + 300);

      // 根据规划是否成功选择颜色
      cv::Scalar path_color = plan_success ? cv::Scalar(255, 0, 0) : cv::Scalar(128, 0, 128); // 蓝色或紫色
      cv::line(frame, cv::Point(x, y), cv::Point(next_x, next_y),
               path_color, 2); // blue planned path
    }

    // Draw robot
    int robot_x = static_cast<int>(robot_state.pose.x() * 50.f + 400);
    int robot_y = static_cast<int>(robot_state.pose.y() * 50.f + 300);
    cv::circle(frame, cv::Point(robot_x, robot_y), 8, cv::Scalar(0, 255, 0),
               2); // green robot outline
    cv::circle(frame, cv::Point(robot_x, robot_y), 4, cv::Scalar(0, 200, 0),
               -1); // green robot fill
               
    // 绘制机器人朝向箭头
    double arrow_length = 15.0;
    int arrow_x = static_cast<int>(robot_x + arrow_length * std::cos(robot_state.pose.theta()));
    int arrow_y = static_cast<int>(robot_y + arrow_length * std::sin(robot_state.pose.theta()));
    cv::arrowedLine(frame, cv::Point(robot_x, robot_y), cv::Point(arrow_x, arrow_y),
                    cv::Scalar(0, 100, 0), 2, cv::LINE_AA, 0, 0.3);

    // 绘制目标点
    int goal_x = static_cast<int>(goal.x() * 50.f + 400);
    int goal_y = static_cast<int>(goal.y() * 50.f + 300);
    cv::circle(frame, cv::Point(goal_x, goal_y), 6, cv::Scalar(255, 0, 255),
               2); // magenta goal

    // Display info
    std::string info = "Time: " + std::to_string(simulation_time).substr(0, 4) +
                       " PosIdx: " + std::to_string(robot_path_idx) +
                       " Target: " + std::to_string(target_idx) +
                       " TEB_Vel: " + std::to_string(robot_state.v).substr(0, 4);
    cv::putText(frame, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 0, 0), 2);
                
    // 显示正方形视野大小
    std::string vision_info = "Vision: " + std::to_string(square_vision_size).substr(0, 4) + "m";
    cv::putText(frame, vision_info, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 0, 0), 2);
                
    // 显示障碍物数量
    std::string obs_info = "Obstacles: " + std::to_string(obstacles.size());
    cv::putText(frame, obs_info, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 0, 0), 2);
                
    // 显示规划状态
    std::string plan_status = plan_success ? "Plan: SUCCESS" : "Plan: FAILED";
    cv::Scalar status_color = plan_success ? cv::Scalar(0, 150, 0) : cv::Scalar(0, 0, 150);
    cv::putText(frame, plan_status, cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                status_color, 2);
                
    // 显示TEB输出的速度信息
    std::string vel_info = "TEB_V: " + std::to_string(robot_state.v).substr(0, 4) + 
                          "m/s, TEB_ω: " + std::to_string(robot_state.omega).substr(0, 4) + "rad/s";
    cv::putText(frame, vel_info, cv::Point(10, 150), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 0, 0), 2);

    // 显示帧率（近似）
    frame_count++;
    if (frame_count % 10 == 0) {
        double fps = 1.0 / DT;
        std::string fps_info = "FPS: " + std::to_string(static_cast<int>(fps));
        cv::putText(frame, fps_info, cv::Point(10, 180), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 0, 0), 2);
    }

    cv::imshow("TEB Planner Test", frame);

    // Check for exit
    int key = cv::waitKey(1); // 10ms delay for ~100 FPS
    if (key == 27) {           // ESC
      std::cout << "Simulation stopped by user" << std::endl;
      break;
    } else if (key == ' ') {   // 空格键暂停/继续
      std::cout << "Simulation paused. Press any key to continue..." << std::endl;
      cv::waitKey(0);
    } else if (key == '+') {   // '+' 键增加视野大小
      square_vision_size += 0.5;
      planner->setObstacleSquareSize(square_vision_size);
      std::cout << "Increased vision size to: " << square_vision_size << " meters" << std::endl;
    } else if (key == '-') {   // '-' 键减小视野大小
      square_vision_size = std::max(1.0, square_vision_size - 0.5);
      planner->setObstacleSquareSize(square_vision_size);
      std::cout << "Decreased vision size to: " << square_vision_size << " meters" << std::endl;
    }

    // Update simulation time
    simulation_time += DT;

    // Check if reached end
    if (robot_path_idx >= path_points.rows() - 5) {
      std::cout << "Simulation completed!" << std::endl;
      break;
    }
    usleep(1000);
  }

  delete planner;
  std::cout << "Simulation ended. Total time: " << simulation_time << " seconds" << std::endl;
  return 0;
}
