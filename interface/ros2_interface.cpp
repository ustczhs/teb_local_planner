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
  rclcpp::Time lidar_last_update_;
  rclcpp::Time camera_last_update_;
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

  // ==================== 核心方法 ====================

  void initParameters() {
    // 声明参数
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
        fused_intensities_[angle_index] = 1.0f; // 简化强度
      }
    }
    // std::cout << "pointCloudToScan done, fused_ranges_ size: " << fused_ranges_.size() << std::endl;
    auto scan_msg = std::make_unique<sensor_msgs::msg::LaserScan>();

    // 设置消息头
    scan_msg->header.stamp = rclcpp::Time(cloud->header.stamp);
    scan_msg->header.frame_id = base_frame_;
    // scan_msg->header.frame_id = "turtlebot4/rplidar_link/rplidar";
    // 设置扫描参数
    scan_msg->angle_min = angle_min_;
    scan_msg->angle_max = angle_max_;
    scan_msg->angle_increment = angle_increment_;
    scan_msg->range_min = range_min_;
    scan_msg->range_max = range_max_;

    // 设置距离和强度数据
    scan_msg->ranges = fused_ranges_;
    scan_msg->intensities = fused_intensities_;

    // 发布消息
    scan_publisher_->publish(std::move(scan_msg));
  }

  void fusionTimerCallback() {
    // 检查是否有有效数据
    bool has_lidar = !lidar_processed_cloud_->empty();
    bool has_camera = !camera_processed_cloud_->empty();

    // 如果没有任何有效数据，不发布
    if(!has_lidar && !has_camera) return;

    // 3. 融合点云数据
    pcl::PointCloud<pcl::PointXYZ> fused_cloud;
    if(has_lidar) {
      fused_cloud += *lidar_processed_cloud_;
    }
    if(has_camera) {
      fused_cloud += *camera_processed_cloud_;
    }

    // 4. 统一地面分割（在base_link坐标系下）
    removeGroundPointsByGeometry(fused_cloud);

    // 5. 转换为激光扫描并发布
    if(!fused_cloud.empty()) {
      auto fused_cloud_ptr = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(fused_cloud);
      pointCloudToScan(fused_cloud_ptr);
    }
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
