#include "inc/frenet_reference.h"
#include "inc/obstacles.h"
#include "inc/optimal_planner.h"
#include "inc/pose_se2.h"
#include "inc/robot_footprint_model.h"
#include "inc/teb_config.h"
#include <boost/smart_ptr.hpp>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>

using namespace teb_local_planner;

// 全局变量保存 trackbar 当前值（初始 0，对应 theta=0.0）
static int g_start_theta = 0; // 范围 0~100 → theta -5.0 ~ +5.0
static int g_end_theta = 0;

// 回调函数（必须定义，即使里面空着也要有，否则 warning 可能还出）
void on_start_theta_changed(int pos, void *userdata = nullptr) {
  g_start_theta = pos;
  // 可以在这里加 printf 或其他，但主循环会自动读取，不必重规划
}

void on_end_theta_changed(int pos, void *userdata = nullptr) {
  g_end_theta = pos;
}

int main() {
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
      // Trim whitespace
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
    std::cerr << "Error: No data loaded from ../test/data.txt" << std::endl;
    return 1;
  }
  std::cout << "Loaded " << data.size() << " rows, " << data[0].size()
            << " columns" << std::endl;
  path_points.resize(data.size(), data[0].size());
  for (size_t i = 0; i < data.size(); ++i) {
    for (size_t j = 0; j < data[i].size(); ++j) {
      path_points(i, j) = data[i][j];
    }
  }
  std::cout << "Path points size: " << path_points.rows() << " x "
            << path_points.cols() << std::endl;

  // Create global Frenet reference (for visualization only)
  auto global_frenet_ref = boost::make_shared<FrenetReference>(path_points);

  // 参数配置
  TebConfig config;
  std::vector<ObstaclePtr> obst_vector;
  obst_vector.emplace_back(boost::make_shared<PointObstacle>(0, 0));
  ViaPointContainer via_points;

  // Setup robot shape model
  RobotFootprintModelPtr robot_model =
      boost::make_shared<CircularRobotFootprint>(0.2);

  auto visual = TebVisualizationPtr(new TebVisualization(config));
  auto planner = new TebOptimalPlanner(config, &obst_vector, robot_model,
                                       visual, &via_points);
  boost::shared_ptr<FrenetReference> local_frenet_ref;

  // 创建白色背景
  cv::Mat show_map(600, 1000, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::namedWindow("TEB Planning Demo", cv::WINDOW_AUTOSIZE);

  // 预先绘制完整的参考路径（黑色）
  for (size_t i = 0; i < path_points.rows() - 1; ++i) {
    int x = static_cast<int>(path_points(i, 0) * 50.f + 400);
    int y = static_cast<int>(path_points(i, 1) * 50.f + 300);
    int next_x = static_cast<int>(path_points(i + 1, 0) * 50.f + 400);
    int next_y = static_cast<int>(path_points(i + 1, 1) * 50.f + 300);

    cv::line(show_map, cv::Point(x, y), cv::Point(next_x, next_y),
             cv::Scalar(0, 0, 0), 2); // 黑色参考路径
  }

  // 当前位置索引
  size_t current_idx = 0;
  const double LOOKAHEAD_DISTANCE = 4.0; // 前方3m

  // 动态障碍物状态
  struct DynamicObstacle {
    Eigen::Vector2d position;
    Eigen::Vector2d velocity;
    double last_update_time;
  };

  std::vector<DynamicObstacle> dynamic_obstacles;
  const int NUM_OBSTACLES = 2;
  const double OBSTACLE_SPEED = 0.5; // m/s
  const double DT = 0.1;             // 模拟时间步长

  // 初始化动态障碍物
  for (int i = 0; i < NUM_OBSTACLES; ++i) {
    DynamicObstacle obs;
    // 随机初始位置（前方1-2m）
    size_t rand_idx = 10 + (std::rand() % 20);
    if (rand_idx < path_points.rows()) {
      obs.position =
          Eigen::Vector2d(path_points(rand_idx, 0), path_points(rand_idx, 1));
    } else {
      obs.position = Eigen::Vector2d(path_points(10, 0), path_points(10, 1));
    }
    // 随机速度方向（垂直于路径或随机）
    obs.velocity = Eigen::Vector2d((std::rand() % 100 - 50) / 100.0,
                                   (std::rand() % 100 - 50) / 100.0)
                       .normalized() *
                   OBSTACLE_SPEED;
    obs.last_update_time = 0.0;
    dynamic_obstacles.push_back(obs);
  }

  while (true) {
    // 创建当前帧的图像（复制基础图像）
    cv::Mat frame = show_map.clone();

    // 计算当前点
    PoseSE2 start(path_points(current_idx, 0), path_points(current_idx, 1),
                  path_points(current_idx, 2));

    // 更新动态障碍物位置
    for (auto &obs : dynamic_obstacles) {
      // 更新位置：pos += vel * dt
      obs.position += obs.velocity * DT;
      obs.last_update_time += DT;

      // 边界检查：如果超出当前点前方3m范围，重新初始化
      double dist_to_robot =
          std::sqrt(std::pow(obs.position.x() - start.x(), 2) +
                    std::pow(obs.position.y() - start.y(), 2));

      if (dist_to_robot > 3.0 || dist_to_robot < 0.5) {
        // 重新初始化位置（当前点前方1-2m）
        size_t rand_idx = current_idx + 5 + (std::rand() % 15);
        if (rand_idx < path_points.rows()) {
          obs.position = Eigen::Vector2d(path_points(rand_idx, 0),
                                         path_points(rand_idx, 1));
        } else {
          obs.position = Eigen::Vector2d(path_points(current_idx + 5, 0),
                                         path_points(current_idx + 5, 1));
        }
        // 随机偏移 ±0.3m
        obs.position.x() += (std::rand() % 60 - 30) / 100.0;
        obs.position.y() += (std::rand() % 60 - 30) / 100.0;

        // 重新随机速度
        obs.velocity = Eigen::Vector2d((std::rand() % 100 - 50) / 100.0,
                                       (std::rand() % 100 - 50) / 100.0)
                           .normalized() *
                       OBSTACLE_SPEED;
        obs.last_update_time = 0.0;
      }
    }

    // 创建障碍物
    std::vector<ObstaclePtr> obst_vector;
    for (const auto &obs : dynamic_obstacles) {
      obst_vector.push_back(boost::make_shared<PointObstacle>(
          obs.position.x(), obs.position.y()));

      // 在图像上绘制障碍物（蓝色圆点）
      int obs_x = static_cast<int>(obs.position.x() * 50.f + 400);
      int obs_y = static_cast<int>(obs.position.y() * 50.f + 300);
      cv::circle(frame, cv::Point(obs_x, obs_y), 4, cv::Scalar(255, 0, 0), -1);
    }

    // 更新planner的障碍物
    // planner->setObstVector(&obst_vector);

    // 找到前方3m的目标点
    size_t target_idx = current_idx;
    double lookahead_distance = 0.0;
    for (size_t i = current_idx; i < path_points.rows() - 1; ++i) {
      double dx = path_points(i + 1, 0) - path_points(i, 0);
      double dy = path_points(i + 1, 1) - path_points(i, 1);
      double segment_length = std::sqrt(dx * dx + dy * dy);
      lookahead_distance += segment_length;

      if (lookahead_distance >= LOOKAHEAD_DISTANCE) {
        target_idx = i + 1;
        break;
      }
    }

    if (target_idx >= path_points.rows()) {
      target_idx = path_points.rows() - 1;
    }

    // 计算目标点
    PoseSE2 end(path_points(target_idx, 0), path_points(target_idx, 1),
                path_points(target_idx, 2));

    // 创建局部 Frenet 参考线（从当前点到目标点）
    int local_rows = target_idx - current_idx + 1;
    Eigen::MatrixXd local_path(local_rows, path_points.cols());
    for (int i = 0; i < local_rows; ++i) {
      local_path.row(i) = path_points.row(current_idx + i);
    }
    try {
      // 规划从当前点到目标点的路径
      planner->plan(start, end);

      // 获取规划轨迹并绘制（红色）
      std::vector<Eigen::Vector3f> planned_path;
      planner->getFullTrajectory(planned_path);
      obst_vector.clear();
      planner->setObstVector(&obst_vector);
      for (size_t i = 0; i < planned_path.size() - 1; ++i) {
        int x = static_cast<int>(planned_path[i][0] * 50.f + 400);
        int y = static_cast<int>(planned_path[i][1] * 50.f + 300);
        int next_x = static_cast<int>(planned_path[i + 1][0] * 50.f + 400);
        int next_y = static_cast<int>(planned_path[i + 1][1] * 50.f + 300);

        cv::line(frame, cv::Point(x, y), cv::Point(next_x, next_y),
                 cv::Scalar(0, 0, 255), 2); // 红色实时路径
      }

      // 绘制当前机器人位置（绿色圆点）
      int robot_x = static_cast<int>(start.x() * 50.f + 400);
      int robot_y = static_cast<int>(start.y() * 50.f + 300);
      cv::circle(frame, cv::Point(robot_x, robot_y), 6, cv::Scalar(0, 255, 0),
                 -1);

      // 显示信息
      std::string info = "Current: " + std::to_string(current_idx) +
                         " Target: " + std::to_string(target_idx);
      cv::putText(frame, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                  cv::Scalar(0, 0, 0), 2);

      cv::imshow("TEB Planning Demo", frame);

    } catch (const std::exception &e) {
      std::cerr << "Planning failed at index " << current_idx << ": "
                << e.what() << std::endl;
    }

    // 递增当前位置
    current_idx += 1;
    if (current_idx >= path_points.rows() - 10) { // 留一些余量
      current_idx = 0;                            // 循环演示
    }

    // 按 ESC 退出，空格键暂停
    int key = cv::waitKey(20); // 100ms 延迟
    if (key == 27) {           // ESC
      break;
    } else if (key == 32) { // 空格键暂停
      while (cv::waitKey(0) != 32)
        ; // 再次按空格继续
    }
  }

  delete planner; // 别忘了释放
  return 0;
}
