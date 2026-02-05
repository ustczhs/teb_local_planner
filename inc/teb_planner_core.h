/**
 * @file teb_planner_core.h
 * @brief TEB规划器核心类 - 纯C++算法库，无ROS依赖
 * 
 * 此头文件定义了TEB局部规划器的核心功能，包括：
 * - 配置管理
 * - 障碍物处理
 * - 路径边界生成
 * - 目标点选择
 * - 轨迹规划
 * 
 * 设计目标：可独立于ROS使用，作为第三方库调用
 */

#ifndef TEB_PLANNER_CORE_H
#define TEB_PLANNER_CORE_H

#include <memory>
#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <functional>
#include <cmath>
#include <limits>
#include <Eigen/Core>

#include "teb_config.h"
#include "optimal_planner.h"
#include "obstacles.h"
#include "pose_se2.h"
#include "sensor_proc.h"
#include "visualization.h"
#include "robot_footprint_model.h"
#include "teb_types.h"

namespace teb_local_planner {

// 前向声明
class SensorProcessor;
class TebVisualization;

/**
 * @brief 简化的位姿结构（非ROS环境）
 */
struct Pose2D {
    double x = 0.0;
    double y = 0.0;
    double theta = 0.0;
    
    Pose2D() = default;
    Pose2D(double x, double y, double theta) : x(x), y(y), theta(theta) {}
    
    double distanceTo(const Pose2D& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        return std::sqrt(dx * dx + dy * dy);
    }
};

/**
 * @brief 2D点结构
 */
struct Point2D {
    double x = 0.0;
    double y = 0.0;
    
    Point2D() = default;
    Point2D(double x, double y) : x(x), y(y) {}
    
    Eigen::Vector2d toEigen() const { return Eigen::Vector2d(x, y); }
};

/**
 * @brief 简化的路径结构（非ROS环境）
 */
struct Path2D {
    std::vector<Pose2D> poses;
    
    bool empty() const { return poses.empty(); }
    size_t size() const { return poses.size(); }
    
    double getLength() const {
        double length = 0.0;
        for (size_t i = 1; i < poses.size(); ++i) {
            length += poses[i-1].distanceTo(poses[i]);
        }
        return length;
    }
    
    Pose2D operator[](size_t index) const { return poses[index]; }
    Pose2D& operator[](size_t index) { return poses[index]; }
};

/**
 * @brief 规划结果结构
 */
struct PlanningResult {
    bool success = false;
    std::vector<Pose2D> trajectory;  // 规划轨迹
    double velocity_x = 0.0;           // 线速度
    double velocity_theta = 0.0;        // 角速度
    Pose2D aim_pose;                   // 当前目标点
    bool goal_reached = false;         // 是否到达终点
    std::string message;               // 状态消息
};

/**
 * @brief 障碍物数据（用于可视化）
 */
struct ObstacleData {
    struct Obstacle {
        double x, y;
        double radius = 0.0;
        bool is_boundary_left = false;
        bool is_boundary_right = false;
    };
    std::vector<Obstacle> obstacles;
};

/**
 * @brief 边界数据（用于可视化）
 */
struct BoundaryData {
    std::vector<Eigen::Vector2d> left_boundary;
    std::vector<Eigen::Vector2d> right_boundary;
};

/**
 * @brief 回调函数类型定义
 */
using LogCallback = std::function<void(const std::string& level, const std::string& message)>;

/**
 * @class TebPlannerCore
 * @brief TEB规划器核心类
 * 
 * 此类封装了TEB局部规划器的所有核心算法，提供简洁的API供上层调用。
 * 设计为线程安全，可在多线程环境中使用。
 */
class TebPlannerCore {
public:
    /**
     * @brief 默认构造函数
     */
    TebPlannerCore();
    
    /**
     * @brief 构造函数（带配置文件路径）
     * @param config_file YAML配置文件路径
     */
    explicit TebPlannerCore(const std::string& config_file);
    
    /**
     * @brief 析构函数
     */
    ~TebPlannerCore();

    // =========================================================================
    // 初始化与配置
    // =========================================================================
    
    /**
     * @brief 从YAML文件加载配置
     * @param config_file 配置文件路径
     * @return true-成功, false-失败
     */
    bool loadConfig(const std::string& config_file);

    /**
     * @brief 设置日志回调函数
     * @param callback 日志回调函数
     */
    void setLogCallback(LogCallback callback);
    
    // =========================================================================
    // 机器人状态设置
    // =========================================================================
    
    /**
     * @brief 设置机器人当前位姿
     * @param pose 当前位姿
     */
    void setCurrentPose(const Pose2D& pose);
    
    /**
     * @brief 设置机器人当前速度
     * @param vx 线速度 (m/s)
     * @param vy 侧向速度 (m/s)
     * @param omega 角速度 (rad/s)
     */
    void setCurrentVelocity(double vx, double vy, double omega);
    
    /**
     * @brief 获取当前位姿
     * @return 当前位姿
     */
    Pose2D getCurrentPose() const;
    
    // =========================================================================
    // 全局路径设置
    // =========================================================================
    
    /**
     * @brief 设置全局路径
     * @param path 全局路径
     */
    void setGlobalPath(const Path2D& path);
    
    /**
     * @brief 清空全局路径
     */
    void clearGlobalPath();
    
    /**
     * @brief 检查是否有有效的全局路径
     * @return true-有效, false-无效
     */
    bool hasValidGlobalPath() const;

    // =========================================================================
    // 障碍物管理
    // =========================================================================

    /**
     * @brief 设置障碍物
     * @param obstacles 障碍物
     */
    void setObstacles(const ObstContainer& obstacles);

    // =========================================================================
    // 边界管理
    // =========================================================================
    
    /**
     * @brief 生成路径边界
     */
    void generatePathBoundaries();

    /**
     * @brief 生成边界障碍物
     */
    void generateBoundaryObstacles();

    /**
     * @brief 获取边界障碍物
     * @param obstacles 障碍物容器
     */
    void getBoundaryObstacles(ObstContainer& obstacles) const;
    /**
     * @brief 设置边界使能状态
     * @param enabled 是否启用
     */
    void setBoundaryEnabled(bool enabled);
    
    /**
     * @brief 检查边界是否启用
     * @return true-启用, false-禁用
     */
    bool isBoundaryEnabled() const;
    
    /**
     * @brief 获取边界数据
     * @return 边界数据
     */
    BoundaryData getBoundaryData() const;

    // =========================================================================
    // 核心规划API
    // =========================================================================
    
    /**
     * @brief 执行一次规划
     * @return 规划结果
     */
    PlanningResult plan();

    // =========================================================================
    // 状态查询
    // =========================================================================
    
    /**
     * @brief 检查是否到达终点
     * @return true-到达, false-未到达
     */
    bool isGoalReached() const;
    
    /**
     * @brief 获取距离终点的距离
     * @return 距离 (米)
     */
    double distanceToGoal() const;
    
    /**
     * @brief 获取规划器状态描述
     * @return 状态字符串
     */
    std::string getStatus() const;
    
    /**
     * @brief 获取配置参数描述
     * @return 配置字符串
     */
    std::string getConfigString() const;

    // =========================================================================
    // 工具方法
    // =========================================================================
    
    /**
     * @brief 在路径上查找最近的位姿索引
     * @param pose 参考位姿
     * @return 路径上的索引
     */
    int findNearestPoseIndex(const Pose2D& pose) const;
    
    /**
     * @brief 计算路径上的平均段长
     * @param start_index 起始索引
     * @return 平均段长
     */
    double calculateAverageSegmentLength(size_t start_index) const;
    
    /**
     * @brief 检查点是否安全（无碰撞）
     * @param x x坐标
     * @param y y坐标
     * @param theta 航向角
     * @return true-安全, false-不安全
     */
    bool isPointSafe(double x, double y, double theta);
    
    /**
     * @brief 获取路径点航向角
     * @param index 索引
     * @return 航向角
     */
    double getPathPoseYaw(size_t index) const;

    // =========================================================================
    // 高级配置
    // =========================================================================
    
    /**
     * @brief 设置前视距离
     * @param distance 距离 (米)
     */
    void setLookaheadDistance(double distance);
    
    /**
     * @brief 设置最小障碍物距离
     * @param distance 距离 (米)
     */
    void setMinObstacleDistance(double distance);
    
    /**
     * @brief 设置障碍物权重
     * @param weight 权重值
     */
    void setObstacleWeight(double weight);
    
    /**
     * @brief 设置路径跟随权重
     * @param weight 权重值
     */
    void setViapointWeight(double weight);
    
    /**
     * @brief 获取TEB配置（用于初始化其他组件）
     * @return TEB配置引用
     */
    const TebConfig& getTebConfig() const;

private:
    // =========================================================================
    // 内部方法
    // =========================================================================
    

    /**
     * @brief 构建TEB配置
     */
    void buildTebConfig();
    
    /**
     * @brief 初始化规划器
     */
    void initializePlanner();
    
    /**
     * @brief 内部规划逻辑
     * @param update_obstacles 是否更新障碍物
     * @param update_goal 是否更新目标点
     * @return 规划结果
     */
    PlanningResult planInternal();
    
    /**
     * @brief 选择下一个目标点
     * @return 目标位姿
     */
    Pose2D selectNextGoal();
    
    /**
     * @brief 检查是否接近终点
     * @return 是否接近终点
     */
    bool ifGoalApproach() const;
    
    /**
     * @brief 获取最终接近速度
     * @return 速度命令
     */
    std::tuple<double, double> calculateFinalApproachVelocity();
    
    /**
     * @brief 日志输出
     */
    void logInfo(const std::string& message);
    void logWarn(const std::string& message);
    void logError(const std::string& message);
    void logDebug(const std::string& message);
    
    /**
     * @brief 内部状态检查
     * @return 状态描述
     */
    std::string checkStatus() const;

    // =========================================================================
    // 成员变量
    // =========================================================================
    
    // 配置
    TebConfig teb_config_;
    bool config_loaded_ = false;
    
    // 核心组件
    std::unique_ptr<TebOptimalPlanner> planner_;
    std::unique_ptr<SensorProcessor> sensor_processor_;
    std::unique_ptr<TebVisualization> visual_;
    RobotFootprintModelPtr robot_model_;
    
    // 机器人状态
    Pose2D current_pose_;
    Twist current_velocity_;
    bool velocity_set_ = false;
    
    // 全局路径
    Path2D global_path_;
    size_t current_goal_index_ = 0;
    Pose2D final_goal_pose_;
    bool goal_reached_ = false;
    
    // 障碍物
    ObstContainer obstacles_;
    std::mutex obstacle_mutex_;
    
    // 边界
    std::vector<CircularObstaclePtr> left_boundary_obstacles_;
    std::vector<CircularObstaclePtr> right_boundary_obstacles_;
    bool boundary_obstacles_enabled_ = true;
    double boundary_obstacles_point_interval_ = 0.1;
    double boundary_obstacles_inflation_ = 0.05;
    double boundary_obstacles_max_obstacle_dist_ = 6.0;
    double path_boundary_width_ = 1.0;
    double path_boundary_point_interval_ = 0.05;
    double path_boundary_inflation_ = 0.05;
    bool path_boundary_enabled_ = true;
    BoundaryData cached_boundary_;
    
    // 回调函数
    LogCallback log_callback_;
    
    // 状态标志
    std::atomic<bool> initialized_{false};
    std::atomic<bool> global_path_received_{false};
    
    // 配置参数缓存（用于快速访问）
    struct FastConfig {
        double lookahead_distance = 2.0;
        double min_obstacle_dist = 0.5;
        double goal_reached_threshold = 0.05;
        double approach_goal_distance = 1.0;
        double max_vel_x = 0.6;
        double max_vel_theta = 0.8;
        double footprint_radius = 0.17;
        int max_longitudinal_expand = 20;
        int longitudinal_step_points = 100;
        int max_lateral_search = 5;
        int lateral_search_step = 1;
        double lateral_safety_margin = 0.02;
        int max_expand_index_limit = 800;
        int avg_segment_calc_points = 8;
        double min_obstacle_dist_goal = 0.1;
        double angle_gain = 2.0;
        double final_angle_gain = 3.0;
        double speed_distance_factor = 0.5;
        bool enable_direct_approach = true;
        double max_angle_diff_for_forward = 0.5;
        double final_approach_speed = 0.15; 
        std::string type = "circle";
        double max_vel_x_backwards = 0.0;
        double max_vel_y = 0.0;
        double acc_lim_x = 1.0;
        double acc_lim_y = 0.0;
        double acc_lim_theta = 1.5;
        double min_turning_radius = 0.1;
        double wheelbase = 0.28;
        bool cmd_angle_instead_rotvel = false;
        std::vector<Eigen::Vector2d> footprint_vertices;
        
        // 障碍物配置
        double inflation_dist = 0.3;
        int obstacle_poses_affected = 15;
        double obstacle_square_size = 0.0;
        
        // 同伦类规划
        bool enable_homotopy_class_planning = false;
    } fast_config_;
};

} // namespace teb_local_planner

#endif // TEB_PLANNER_CORE_H
