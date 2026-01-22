#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>

#include <memory>
#include <vector>
#include <array>
#include <cmath>

namespace sensor_fusion {

/**
 * @brief 高效的传感器融合节点
 * 将3D激光雷达和深度相机数据融合为2D激光扫描
 */
class SensorFusionNode : public rclcpp::Node {
public:
  SensorFusionNode() : Node("sensor_fusion") {
    // 初始化参数
    initParameters();

    // 初始化变换监听器
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // 初始化发布器
    scan_publisher_ = create_publisher<sensor_msgs::msg::LaserScan>(
        "/scan_fused", rclcpp::QoS(10));

    // 初始化订阅器
    initSubscribers();
    // 初始化数据结构
    initDataStructures();
    // 初始化角度查找表
    initAngleTable();
    sleep(1.0); // 确保订阅器准备就绪
    // 初始化融合定时器
    initFusionTimer();

    RCLCPP_INFO(get_logger(), "Sensor fusion node initialized");
    RCLCPP_INFO(get_logger(), "Local map size: %.2f m", local_map_size_);
    RCLCPP_INFO(get_logger(), "Ground threshold: %.3f m", ground_threshold_);
  }

private:
  // ==================== 参数配置 ====================
  double local_map_size_;   // 局部地图大小 (m)
  double ground_threshold_; // 地面高度阈值 (m)
  double angle_min_;        // 扫描最小角度 (rad)
  double angle_max_;        // 扫描最大角度 (rad)
  double angle_increment_;  // 角度分辨率 (rad)
  double range_min_;        // 最小测量距离 (m)
  double range_max_;        // 最大测量距离 (m)
  int scan_size_;           // 扫描点数量
  std::string base_frame_;  // 基座坐标系

  // 传感器配置
  std::string lidar_3d_topic_;
  std::string depth_topic_;
  std::string camera_info_topic_;

  // 传感器坐标系
  std::string lidar_3d_frame_;     // 3D激光雷达坐标系
  std::string depth_camera_frame_; // 深度相机坐标系

  // ==================== ROS2接口 ====================
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_2d_sub_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // ==================== 数据结构 ====================
  std::vector<float> fused_ranges_;      // 融合后的距离数据
  std::vector<float> fused_intensities_; // 融合后的强度数据
  std::vector<size_t> angle_to_index_;   // 角度到索引的映射表

  // 点云缓冲区复用
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_buffer_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud_;

  // 传感器数据缓冲区（用于融合）
  pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_processed_cloud_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr camera_processed_cloud_;
  sensor_msgs::msg::LaserScan::SharedPtr scan_2d_buffer_; // 2D激光雷达scan缓冲区
  rclcpp::Time lidar_last_update_;
  rclcpp::Time camera_last_update_;
  rclcpp::Time scan_2d_last_update_;
  double fusion_timeout_; // 融合超时时间

  // 融合定时器
  rclcpp::TimerBase::SharedPtr fusion_timer_;

  // ==================== 传感器安装参数 ====================
  // 3D激光雷达安装参数 (相对于base_link)
  double lidar3d_x_offset_, lidar3d_y_offset_, lidar3d_z_offset_; // 位置偏移 (m)
  double lidar3d_height_;                                         // 到地面高度 (m)
  double lidar3d_pitch_, lidar3d_roll_, lidar3d_yaw_;             // 安装角度 (rad)
  double lidar3d_ground_tolerance_;                               // 地面过滤容差 (m)

  // 深度相机安装参数 (相对于base_link)
  double camera_x_offset_, camera_y_offset_, camera_z_offset_; // 位置偏移 (m)
  double camera_height_;                                       // 到地面高度 (m)
  double camera_pitch_, camera_roll_, camera_yaw_;             // 安装角度 (rad)
  double camera_ground_tolerance_;                             // 地面过滤容差 (m)

  double scan_2d_rotation_angle_ = M_PI / 2.0; // 2D激光雷达旋转角度 (弧度)
  double scan_2d_x_offset_ = -0.040;           // 2D激光雷达x方向偏移 (米)
  double scan_2d_y_offset_ = 0.000;            // 2D激光雷达y方向偏移 (米)

  // ==================== 核心方法 ====================

  // 从YAML文件读取传感器配置参数
  bool loadSensorConfigFromYaml(const std::string& config_file) {
    try {
      YAML::Node config = YAML::LoadFile(config_file);

      // 读取sensor配置
      if(config["sensor"]) {
        auto sensor = config["sensor"];

        if(sensor["2d_laser_install_angle"]) {
          scan_2d_rotation_angle_ = sensor["2d_laser_install_angle"].as<double>();
          std::cout << "Loaded 2d_laser_install_angle from YAML: " << scan_2d_rotation_angle_ << std::endl;
        }
        if(sensor["2d_laser_x_offset"]) {
          scan_2d_x_offset_ = sensor["2d_laser_x_offset"].as<double>();
          std::cout << "Loaded 2d_laser_x_offset from YAML: " << scan_2d_x_offset_ << std::endl;
        }
        if(sensor["2d_laser_y_offset"]) {
          scan_2d_y_offset_ = sensor["2d_laser_y_offset"].as<double>();
          std::cout << "Loaded 2d_laser_y_offset from YAML: " << scan_2d_y_offset_ << std::endl;
        }
      }

      return true;
    } catch(const std::exception& e) {
      std::cout << "Error loading sensor config from " << config_file << ": " << e.what() << std::endl;
      return false;
    }
  }

  void initParameters() {
    // 首先尝试从config.yaml读取传感器参数
    std::string config_file = "../config/config.yaml";
    if(!loadSensorConfigFromYaml(config_file)) {
      std::cout << "Failed to load sensor config from YAML, using defaults" << std::endl;
    }

    // 声明ROS2参数（作为备用）
    declare_parameter("local_map_size", 4.0);
    declare_parameter("ground_threshold", 0.05);
    declare_parameter("angle_min", -M_PI);
    declare_parameter("angle_max", M_PI);
    declare_parameter("angle_increment", M_PI / 180.0); // 1度分辨率
    declare_parameter("range_min", 0.1);
    declare_parameter("range_max", 10.0);
    declare_parameter("base_frame", "base_link");

    declare_parameter("lidar_3d_topic", "/lidar_3d/points");
    declare_parameter("depth_topic", "/oakd/rgb/preview/depth/points");
    declare_parameter("camera_info_topic", "/depth_camera/depth/camera_info");

    declare_parameter("lidar_3d_frame", "lidar_3d_link");
    declare_parameter("depth_camera_frame", "turtlebot4/oakd_rgb_camera_frame/rgbd_camera");
    declare_parameter("fusion_timeout", 0.5); // 融合超时时间 (秒)

    declare_parameter("sensor.2d_laser_install_angle", scan_2d_rotation_angle_); // 使用YAML中读取的值作为默认值
    declare_parameter("sensor.2d_laser_x_offset", scan_2d_x_offset_);
    declare_parameter("sensor.2d_laser_y_offset", scan_2d_y_offset_);

    // 3D激光雷达安装参数
    declare_parameter("3d_lidar.x_offset", 0.0);
    declare_parameter("3d_lidar.y_offset", 0.0);
    declare_parameter("3d_lidar.z_offset", 0.5);
    declare_parameter("3d_lidar.height", 0.5);
    declare_parameter("3d_lidar.pitch", 0.0);
    declare_parameter("3d_lidar.roll", 0.0);
    declare_parameter("3d_lidar.yaw", 0.0);
    declare_parameter("3d_lidar.ground_tolerance", 0.05);

    // 深度相机安装参数
    declare_parameter("camera.x_offset", -0.06);
    declare_parameter("camera.y_offset", 0.0);
    declare_parameter("camera.z_offset", 0.244);
    declare_parameter("camera.height", 0.324);
    declare_parameter("camera.pitch", 0.0);
    declare_parameter("camera.roll", 0.0);
    declare_parameter("camera.yaw", 0.0);
    declare_parameter("camera.ground_tolerance", 0.1);

    // 获取参数值
    local_map_size_ = get_parameter("local_map_size").as_double();
    ground_threshold_ = get_parameter("ground_threshold").as_double();
    angle_min_ = get_parameter("angle_min").as_double();
    angle_max_ = get_parameter("angle_max").as_double();
    angle_increment_ = get_parameter("angle_increment").as_double();
    range_min_ = get_parameter("range_min").as_double();
    range_max_ = get_parameter("range_max").as_double();
    base_frame_ = get_parameter("base_frame").as_string();

    lidar_3d_topic_ = get_parameter("lidar_3d_topic").as_string();
    depth_topic_ = get_parameter("depth_topic").as_string();
    camera_info_topic_ = get_parameter("camera_info_topic").as_string();

    lidar_3d_frame_ = get_parameter("lidar_3d_frame").as_string();
    depth_camera_frame_ = get_parameter("depth_camera_frame").as_string();

    // 获取传感器安装参数（如果YAML读取失败，会使用ROS2参数）
    scan_2d_rotation_angle_ = get_parameter("sensor.2d_laser_install_angle").as_double();
    scan_2d_x_offset_ = get_parameter("sensor.2d_laser_x_offset").as_double();
    scan_2d_y_offset_ = get_parameter("sensor.2d_laser_y_offset").as_double();

    // 获取传感器安装参数
    lidar3d_x_offset_ = get_parameter("3d_lidar.x_offset").as_double();
    lidar3d_y_offset_ = get_parameter("3d_lidar.y_offset").as_double();
    lidar3d_z_offset_ = get_parameter("3d_lidar.z_offset").as_double();
    lidar3d_height_ = get_parameter("3d_lidar.height").as_double();
    lidar3d_pitch_ = get_parameter("3d_lidar.pitch").as_double();
    lidar3d_roll_ = get_parameter("3d_lidar.roll").as_double();
    lidar3d_yaw_ = get_parameter("3d_lidar.yaw").as_double();
    lidar3d_ground_tolerance_ = get_parameter("3d_lidar.ground_tolerance").as_double();

    camera_x_offset_ = get_parameter("camera.x_offset").as_double();
    camera_y_offset_ = get_parameter("camera.y_offset").as_double();
    camera_z_offset_ = get_parameter("camera.z_offset").as_double();
    camera_height_ = get_parameter("camera.height").as_double();
    camera_pitch_ = get_parameter("camera.pitch").as_double();
    camera_roll_ = get_parameter("camera.roll").as_double();
    camera_yaw_ = get_parameter("camera.yaw").as_double();
    camera_ground_tolerance_ = get_parameter("camera.ground_tolerance").as_double();

    // 计算扫描尺寸
    scan_size_ = static_cast<int>((angle_max_ - angle_min_) / angle_increment_) + 1;
  }

  void initSubscribers() {
    // 3D激光雷达订阅器
    lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_3d_topic_, rclcpp::QoS(10),
        std::bind(&SensorFusionNode::lidarCallback, this, std::placeholders::_1));

    // 深度相机订阅器
    depth_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        depth_topic_, rclcpp::QoS(10),
        std::bind(&SensorFusionNode::depthCallback, this, std::placeholders::_1));

    // 2D激光雷达scan订阅器
    scan_2d_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", rclcpp::QoS(10),
        std::bind(&SensorFusionNode::scan2DCallback, this, std::placeholders::_1));
  }

  void initDataStructures() {
    // 初始化距离和强度数组
    fused_ranges_.resize(scan_size_, std::numeric_limits<float>::infinity());
    fused_intensities_.resize(scan_size_, 0.0f);

    // 初始化点云缓冲区
    cloud_buffer_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    transformed_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());

    // 初始化传感器数据缓冲区
    lidar_processed_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    camera_processed_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
  }

  void initFusionTimer() {
    // 创建10Hz融合定时器
    fusion_timer_ = create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&SensorFusionNode::fusionTimerCallback, this));
  }

  void initAngleTable() {
    // 预计算角度到索引的映射表
    angle_to_index_.resize(360 * 10); // 0.1度分辨率

    for(size_t i = 0; i < angle_to_index_.size(); ++i) {
      double angle = i * 0.1 * M_PI / 180.0 - M_PI; // -π 到 π
      if(angle >= angle_min_ && angle <= angle_max_) {
        size_t index = static_cast<size_t>((angle - angle_min_) / angle_increment_);
        if(index < static_cast<size_t>(scan_size_)) {
          angle_to_index_[i] = index;
        } else {
          angle_to_index_[i] = scan_size_; // 无效索引
        }
      } else {
        angle_to_index_[i] = scan_size_; // 无效索引
      }
    }
  }

  // ==================== 回调函数 ====================

  void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    try {
      // 转换ROS消息到PCL点云
      pcl::fromROSMsg(*msg, *cloud_buffer_);

      // 1. 在传感器坐标系下预滤波（减少计算量）
      filterByDistance(cloud_buffer_, local_map_size_);

      // 2. 坐标变换到base_link
      if(transformPointCloud(cloud_buffer_, transformed_cloud_, lidar_3d_frame_, msg->header.stamp)) {
        lidar_processed_cloud_ = transformed_cloud_->makeShared();
        lidar_last_update_ = msg->header.stamp;
      }

    } catch(const std::exception& e) {
      RCLCPP_WARN(get_logger(), "Error processing 3D lidar data: %s", e.what());
    }
  }

  void depthCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    try {
      // 转换ROS消息到PCL点云
      pcl::fromROSMsg(*msg, *cloud_buffer_);

      // 1. 在传感器坐标系下预滤波（减少计算量）
      filterByDistance(cloud_buffer_, local_map_size_);

      // 2. 坐标变换到base_link
      if(transformPointCloud(cloud_buffer_, transformed_cloud_, depth_camera_frame_, msg->header.stamp)) {
        camera_processed_cloud_ = transformed_cloud_->makeShared();
        camera_last_update_ = msg->header.stamp;
      }

    } catch(const std::exception& e) {
      RCLCPP_WARN(get_logger(), "Error processing depth camera data: %s", e.what());
    }
  }

  void scan2DCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
    try {
      // 1. 将激光雷达scan转换为点云（在激光雷达坐标系中）
      auto points_lidar = convertScanToPoints(*msg);

      // 2. 坐标变换：激光雷达坐标系 → base_link坐标系
      auto points_base = transformScanPointsToBase(points_lidar);

      // 3. 将点云转换回scan（在base_link坐标系中）
      auto processed_msg = convertPointsToScan(points_base);

      // 4. 设置消息头
      processed_msg->header = msg->header;
      processed_msg->header.frame_id = base_frame_;

      // 5. 缓冲处理后的数据
      scan_2d_buffer_ = std::move(processed_msg);
      scan_2d_last_update_ = msg->header.stamp;

    } catch(const std::exception& e) {
      RCLCPP_WARN(get_logger(), "Error processing 2D scan data: %s", e.what());
    }
  }

  // ==================== 数据处理 ====================
  // （已移除独立的processLidar3D和processDepthCamera函数，现在在回调中直接处理）

  // ==================== 工具函数 ====================

  bool transformPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud,
                           const std::string& source_frame,
                           const rclcpp::Time& timestamp) {
    try {
      // 获取变换
      geometry_msgs::msg::TransformStamped transform_stamped =
          tf_buffer_->lookupTransform(base_frame_, source_frame, timestamp,
                                      rclcpp::Duration::from_seconds(0.1));

      // 手动构造Eigen变换矩阵
      Eigen::Affine3d transform = Eigen::Affine3d::Identity();

      // 设置旋转部分
      const auto& rotation = transform_stamped.transform.rotation;
      Eigen::Quaterniond quat(rotation.w, rotation.x, rotation.y, rotation.z);
      transform.rotate(quat);

      // 设置平移部分
      const auto& translation = transform_stamped.transform.translation;
      transform.translation() = Eigen::Vector3d(translation.x, translation.y, translation.z);

      // 应用变换
      pcl::transformPointCloud(*input_cloud, *output_cloud, transform);

      return true;

    } catch(const tf2::TransformException& ex) {
      RCLCPP_WARN(get_logger(), "Transform failed: %s", ex.what());
      return false;
    }
  }

  void filterByDistance(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double max_distance) {
    // 使用PCL的直通滤波器
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-max_distance, max_distance);
    pass.filter(*cloud);

    pass.setInputCloud(cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-max_distance, max_distance);
    pass.filter(*cloud);

    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-max_distance, max_distance);
    pass.filter(*cloud);
  }

  void pointCloudToScan(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    // 重置融合数据
    std::fill(fused_ranges_.begin(), fused_ranges_.end(),
              std::numeric_limits<float>::infinity());
    std::fill(fused_intensities_.begin(), fused_intensities_.end(), 0.0f);

    // 将点云转换为激光扫描
    for(const auto& point : cloud->points) {
      // 计算距离和角度
      double distance = std::sqrt(point.x * point.x + point.y * point.y);
      double angle = std::atan2(point.y, point.x);

      // 范围检查
      if(distance < range_min_ || distance > range_max_ ||
         angle < angle_min_ || angle > angle_max_) {
        continue;
      }

      // 计算角度索引
      size_t angle_index = static_cast<size_t>((angle - angle_min_) / angle_increment_);
      if(angle_index >= static_cast<size_t>(scan_size_)) continue;

      // 距离融合（取最小值）
      if(distance < fused_ranges_[angle_index]) {
        fused_ranges_[angle_index] = distance;
        fused_intensities_[angle_index] = 1.0f; // 3D数据源
      }
    }
  }

  void fusionTimerCallback() {
    // 检查是否有有效数据
    bool has_lidar = !lidar_processed_cloud_->empty();
    bool has_camera = !camera_processed_cloud_->empty();
    bool has_scan_2d = !scan_2d_buffer_->ranges.empty();

    // 如果没有任何有效数据，不发布
    if(!has_lidar && !has_camera && !has_scan_2d) return;

    // 优先级处理：有3D数据优先，无3D数据时直接发布2D scan
    if(has_lidar || has_camera) {
      // 有3D数据，执行完整的融合流程
      pcl::PointCloud<pcl::PointXYZ> fused_cloud;
      if(has_lidar) {
        fused_cloud += *lidar_processed_cloud_;
      }
      if(has_camera) {
        fused_cloud += *camera_processed_cloud_;
      }

      // 地面分割
      removeGroundPointsByGeometry(fused_cloud);

      // 转换为激光扫描
      auto fused_cloud_ptr = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(fused_cloud);
      pointCloudToScan(fused_cloud_ptr);

      // 与2D scan融合
      if(has_scan_2d) {
        fuseWith2DScan();
      }

    } else if(has_scan_2d) {
      // 只有2D数据，直接发布
      publish2DScanDirectly();
    }
  }

  // 直接发布2D激光雷达scan（当只有2D数据时使用）
  void publish2DScanDirectly() {
    if(!scan_2d_buffer_) return;

    // 数据已经在scan2DCallback中处理过了，直接发布
    auto scan_msg = std::make_unique<sensor_msgs::msg::LaserScan>(*scan_2d_buffer_);
    scan_publisher_->publish(std::move(scan_msg));
  }

  // 与2D激光雷达scan融合
  void fuseWith2DScan() {
    if(!scan_2d_buffer_) return;

    // 重新发布已更新的scan数据
    auto scan_msg = std::make_unique<sensor_msgs::msg::LaserScan>();

    // 设置消息头
    scan_msg->header.stamp = scan_2d_buffer_->header.stamp;
    scan_msg->header.frame_id = base_frame_;

    // 设置扫描参数（使用我们自己的参数）
    scan_msg->angle_min = angle_min_;
    scan_msg->angle_max = angle_max_;
    scan_msg->angle_increment = angle_increment_;
    scan_msg->range_min = range_min_;
    scan_msg->range_max = range_max_;

    // 初始化为3D融合结果
    scan_msg->ranges = fused_ranges_;
    scan_msg->intensities = fused_intensities_;

    // 与2D scan融合：对每个角度，取距离更小的值
    for(size_t i = 0; i < fused_ranges_.size(); ++i) {
      // 计算对应的2D scan角度索引
      double fused_angle = angle_min_ + i * angle_increment_;
      double scan_2d_angle = fused_angle - scan_2d_buffer_->angle_min;
      size_t scan_2d_index = static_cast<size_t>(scan_2d_angle / scan_2d_buffer_->angle_increment);

      // 检查索引是否有效
      if(scan_2d_index < scan_2d_buffer_->ranges.size()) {
        float range_3d = fused_ranges_[i];
        float range_2d = scan_2d_buffer_->ranges[scan_2d_index];

        // 取距离更小的值（更近的障碍物）
        if(!std::isinf(range_2d) && (std::isinf(range_3d) || range_2d < range_3d)) {
          scan_msg->ranges[i] = range_2d;
          scan_msg->intensities[i] = 2.0f; // 标记为2D数据源
        }
      }
    }

    // 发布融合后的消息
    scan_publisher_->publish(std::move(scan_msg));
  }

  // 将LaserScan转换为点云（在传感器坐标系中）
  std::vector<Eigen::Vector2d> convertScanToPoints(const sensor_msgs::msg::LaserScan& scan) {

    std::vector<Eigen::Vector2d> points;

    for(size_t i = 0; i < scan.ranges.size(); ++i) {
      if(i % 200 == 0) { // 每200个点打印一次
      }

      float range = scan.ranges[i];

      // 过滤无效数据
      if(std::isnan(range) || std::isinf(range) || range < scan.range_min || range > scan.range_max) {
        continue;
      }

      // 计算角度
      double angle = scan.angle_min + i * scan.angle_increment;

      // 转换为笛卡尔坐标
      double x = range * std::cos(angle);
      double y = range * std::sin(angle);

      points.emplace_back(x, y);
    }

    return points;
  }

  // 将点云从激光雷达坐标系转换到base_link坐标系
  std::vector<Eigen::Vector2d> transformScanPointsToBase(const std::vector<Eigen::Vector2d>& points_lidar) {

    std::vector<Eigen::Vector2d> points_base;
    points_base.reserve(points_lidar.size());

    Eigen::Rotation2Dd rotation(scan_2d_rotation_angle_);

    for(size_t i = 0; i < points_lidar.size(); ++i) {

      const auto& point = points_lidar[i];

      Eigen::Vector2d transformed_point;
      transformed_point.x() = -point.y() + scan_2d_x_offset_; // x_base = -y_lidar + x_offset
      transformed_point.y() = point.x();                      // y_base = x_lidar

      points_base.push_back(transformed_point);
    }
    return points_base;
  }

  // 将点云转换回LaserScan（在base_link坐标系中）
  std::unique_ptr<sensor_msgs::msg::LaserScan> convertPointsToScan(const std::vector<Eigen::Vector2d>& points) {
    auto scan_msg = std::make_unique<sensor_msgs::msg::LaserScan>();

    // 设置扫描参数（使用我们自己的参数）
    scan_msg->angle_min = angle_min_;
    scan_msg->angle_max = angle_max_;
    scan_msg->angle_increment = angle_increment_;
    scan_msg->range_min = range_min_;
    scan_msg->range_max = range_max_;

    // 初始化距离和强度数组
    scan_msg->ranges.assign(scan_size_, std::numeric_limits<float>::infinity());
    scan_msg->intensities.assign(scan_size_, 0.0f);

    // 将点云转换为scan
    for(const auto& point : points) {
      // 计算距离和角度
      double distance = point.norm();
      double angle = std::atan2(point.y(), point.x());

      // 范围检查
      if(distance < range_min_ || distance > range_max_ ||
         angle < angle_min_ || angle > angle_max_) {
        continue;
      }

      // 计算角度索引
      size_t angle_index = static_cast<size_t>((angle - angle_min_) / angle_increment_);
      if(angle_index >= static_cast<size_t>(scan_size_)) continue;

      // 距离融合（取最小值）
      if(distance < scan_msg->ranges[angle_index]) {
        scan_msg->ranges[angle_index] = distance;
        scan_msg->intensities[angle_index] = 2.0f; // 2D数据源
      }
    }

    return scan_msg;
  }

  // 统一的地面过滤函数
  void removeGroundPointsByGeometry(pcl::PointCloud<pcl::PointXYZ>& cloud) {
    if(cloud.empty()) return;

    // 简化的地面模型：假设水平地面，z = 0
    // 可以根据需要扩展为更复杂的地面模型
    const double ground_z_threshold = 0.05; // 5cm地面容差

    pcl::PointCloud<pcl::PointXYZ> filtered_cloud;
    filtered_cloud.reserve(cloud.size());

    for(const auto& point : cloud) {
      // 简化的地面过滤：保留z值大于阈值的点
      if(point.z > ground_z_threshold) {
        filtered_cloud.push_back(point);
      }
    }

    cloud.swap(filtered_cloud);
  }
};

} // namespace sensor_fusion

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<sensor_fusion::SensorFusionNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
