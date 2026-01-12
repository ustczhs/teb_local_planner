#include "inc/obstacles.h"
#include "inc/optimal_planner.h"
#include "inc/pose_se2.h"
#include "inc/robot_footprint_model.h"
#include "inc/teb_config.h"
#include <boost/smart_ptr.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

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
  // 参数配置
  TebConfig config;
  PoseSE2 start(-2, 0, 0);
  PoseSE2 end(2, 0, 0);
  std::vector<ObstaclePtr> obst_vector;
  obst_vector.emplace_back(boost::make_shared<PointObstacle>(0, 0));
  ViaPointContainer via_points;

  // Setup robot shape model
  RobotFootprintModelPtr robot_model =
      boost::make_shared<CircularRobotFootprint>(0.4);

  auto visual = TebVisualizationPtr(new TebVisualization(config));
  auto planner = new TebOptimalPlanner(config, &obst_vector, robot_model,
                                       visual, &via_points);

  cv::Mat show_map = cv::Mat::zeros(cv::Size(500, 500), CV_8UC3);

  // 创建窗口（提前创建，避免 trackbar 绑定失败）
  cv::namedWindow("path", cv::WINDOW_AUTOSIZE);

  // 创建 trackbar，value 传 nullptr，使用回调
  cv::createTrackbar("start theta", "path", nullptr, 100,
                     on_start_theta_changed);
  cv::createTrackbar("end theta", "path", nullptr, 100, on_end_theta_changed);

  // 可选：设置初始位置（OpenCV 4.5+ 后，trackbar 初始位置默认为 0，无需 value
  // 指针也能设） 如果想强制初始值不为 0，可在循环外调用 cv::setTrackbarPos
  cv::setTrackbarPos("start theta", "path", g_start_theta);
  cv::setTrackbarPos("end theta", "path", g_end_theta);

  while (true) {
    memset(show_map.data, 0, 500 * 500 * 3);

    try {
      // 从全局变量读取当前 theta 值（单位：弧度，0.1 倍率对应 -5~+5 弧度）
      start.theta() = g_start_theta * 0.1;
      end.theta() = g_end_theta * 0.1;

      // 规划
      planner->plan(start, end);

      // 获取完整轨迹并绘制
      std::vector<Eigen::Vector3f> path;
      planner->getFullTrajectory(path);

      for (size_t i = 0; i < path.size() - 1; ++i) {
        int x = static_cast<int>(path[i][0] * 100.f + 250);
        int y = static_cast<int>(path[i][1] * 100.f + 250);
        int next_x = static_cast<int>(path[i + 1][0] * 100.f + 250);
        int next_y = static_cast<int>(path[i + 1][1] * 100.f + 250);

        cv::line(show_map, cv::Point(x, y), cv::Point(next_x, next_y),
                 cv::Scalar(255, 255, 255));
      }

      // 显示障碍物（可选，美观一点）
      cv::circle(show_map, cv::Point(250, 250), 5, cv::Scalar(0, 0, 255),
                 -1); // 障碍点 (0,0)

      cv::imshow("path", show_map);
    } catch (...) {
      std::cerr << "Planning failed, continuing..." << std::endl;
      // 可以 break; 如果想异常就退出
    }

    if (cv::waitKey(10) == 27) { // 按 ESC 退出
      break;
    }
  }

  delete planner; // 别忘了释放
  return 0;
}