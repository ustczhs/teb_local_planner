/**
 * @file TebPlanner.h
 * @brief TEB局部规划器核心算法层 - ROS无关的纯算法实现
 * 
 * 该模块包含所有规划计算功能，可独立于ROS使用：
 * - 配置管理
 * - 障碍物处理
 * - 路径边界生成
 * - 目标点选择
 * - TEB轨迹规划
 */

#ifndef TEB_PLANNER_H
#define TEB_PLANNER_H

#include <memory>
#include <vector>
#include <string>
#include <atomic>
#include <shared_mutex>
#include <cmath>
#include <limits>
#include <functional>

#include <Eigen/Core>
#include <boost/smart_ptr.hpp>

#include "teb_config.h"
#include "optimal_planner.h"
#include "pose_se2.h"
#include "obstacles.h"
#include "sensor_proc.h"
#include "visualization.h"

namespace teb_local_planner {

// =============================================================================
// 障碍物数据结构
// =============================================================================

/**
 * @brief 边界障碍物结构
 */
struct BoundaryObstacle {
    Eigen::Vector2d position;
    double inflation_radius;
    bool is_left;
};

/**
 * @brief 路径边界结构
 */
struct PathBoundaries {
    std::vector<Eigen::Vector2d> left_points;
    std::vector<Eigen::Vector2d> right_points;
    std::vector<ObstaclePtr> left_obstacles;
    std::vector<ObstaclePtr> right_obstacles;
    double boundary_width;
    bool is_valid;
};

// =============================================================================
// 规划结果结构
// =============================================================================

/**
 * @brief 速度命令结构
 */
struct VelocityCommand {
    double linear_x;
    double linear_y;
    double angular_z;
    bool is_valid;
    std::string message;
};

/**
 * @brief 规划结果结构
 */
struct PlanningResult {
    std::vector<Eigen::Vector3f> trajectory;  // [x, y, theta]
    std::vector<Twist> velocity_profile;
    VelocityCommand cmd_vel;
    Eigen::Vector2d aim_point;
    bool success;
    std::string message;
};

// =============================================================================
// 机器人状态结构
// =============================================================================

/**
 * @brief 机器人状态
 */
struct RobotState {
    Eigen::Vector2d position;
    double theta;
    Twist velocity;
    bool has_velocity;
};

// =============================================================================
// 路径数据结构
// =============================================================================

/**
 * @brief 2D路径点
 */
struct PathPoint {
    Eigen::Vector2d position;
    double theta;
};

// =============================================================================
// 规划器配置类 (简化版)
// =============================================================================

/**
 * @brief 规划器配置 - 封装所有可调参数
 */
class PlannerConfig {
public:
    // 机器人参数
    double max_vel_x;
    double max_vel_x_backwards;
    double max_vel_y;
    double max_vel_theta;
    double acc_lim_x;
    double acc_lim_y;
    double acc_lim_theta;
    double min_turning_radius;
    double wheelbase;
    bool cmd_angle_instead_rotvel;
    double footprint_radius;
    std::string footprint_type;
    
    // 轨迹参数
    double dt_ref;
    double dt_hysteresis;
    int min_samples;
    int max_samples;
    double force_reinit_new_goal_dist;
    double force_reinit_new_goal_angular;
    
    // 优化参数
    int no_inner_iterations;
    int no_outer_iterations;
    double weight_max_vel_x;
    double weight_max_vel_theta;
    double weight_acc_lim_x;
    double weight_acc_lim_theta;
    double weight_kinematics_nh;
    double weight_kinematics_forward_drive;
    double weight_obstacle;
    double weight_viapoint;
    
    // 障碍物参数
    double min_obstacle_dist;
    double inflation_dist;
    int obstacle_poses_affected;
    double obstacle_square_size;
    
    // 目标跟踪参数
    double goal_reached_threshold;
    double approach_goal_distance;
    double final_approach_speed;
    double max_angle_diff_for_forward;
    double angle_gain;
    double final_angle_gain;
    double speed_distance_factor;
    bool enable_direct_approach;
    
    // 目标查找参数
    double lookahead_distance;
    double min_obstacle_dist_goal;
    int max_longitudinal_expand;
    int longitudinal_step_points;
    int max_lateral_search;
    int lateral_search_step;
    double lateral_safety_margin;
    int max_expand_index_limit;
    int avg_segment_calc_points;
    
    // 路径边界参数
    double boundary_width;
    double boundary_point_interval;
    double boundary_inflation;
    double boundary_max_dist;
    bool boundary_enabled;
    
    // 传感器参数
    double fov_min_angle;
    double fov_max_angle;
    double obstacle_memory_time;
    double cluster_distance_threshold;
    int cluster_min_points;
    int downsample_factor;
    double local_map_size;
    
    /**
     * @brief 默认构造函数
     */
    PlannerConfig() {
        // 机器人参数
        max_vel_x = 0.6;
        max_vel_x_backwards = 0.0;
        max_vel_y = 0.0;
        max_vel_theta = 0.8;
        acc_lim_x = 1.0;
        acc_lim_y = 0.0;
        acc_lim_theta = 1.5;
        min_turning_radius = 0.1;
        wheelbase = 0.28;
        cmd_angle_instead_rotvel = false;
        footprint_radius = 0.17;
        footprint_type = "circle";
        
        // 轨迹参数
        dt_ref = 0.15;
        dt_hysteresis = 0.05;
        min_samples = 5;
        max_samples = 80;
        force_reinit_new_goal_dist = 0.5;
        force_reinit_new_goal_angular = 0.785;
        
        // 优化参数
        no_inner_iterations = 5;
        no_outer_iterations = 3;
        weight_max_vel_x = 2.0;
        weight_max_vel_theta = 1.0;
        weight_acc_lim_x = 1.0;
        weight_acc_lim_theta = 1.0;
        weight_kinematics_nh = 1000.0;
        weight_kinematics_forward_drive = 1000.0;
        weight_obstacle = 50.0;
        weight_viapoint = 0.0;
        
        // 障碍物参数
        min_obstacle_dist = 0.05;
        inflation_dist = 0.06;
        obstacle_poses_affected = 15;
        obstacle_square_size = 3.0;
        
        // 目标跟踪参数
        goal_reached_threshold = 0.05;
        approach_goal_distance = 1.0;
        final_approach_speed = 0.15;
        max_angle_diff_for_forward = 0.5;
        angle_gain = 2.0;
        final_angle_gain = 3.0;
        speed_distance_factor = 0.5;
        enable_direct_approach = true;
        
        // 目标查找参数
        lookahead_distance = 1.5;
        min_obstacle_dist_goal = 0.1;
        max_longitudinal_expand = 20;
        longitudinal_step_points = 100;
        max_lateral_search = 5;
        lateral_search_step = 1;
        lateral_safety_margin = 0.02;
        max_expand_index_limit = 800;
        avg_segment_calc_points = 8;
        
        // 路径边界参数
        boundary_width = 1.0;
        boundary_point_interval = 0.1;
        boundary_inflation = 0.05;
        boundary_max_dist = 6.0;
        boundary_enabled = true;
        
        // 传感器参数
        fov_min_angle = -1.57;
        fov_max_angle = 1.57;
        obstacle_memory_time = 2.0;
        cluster_distance_threshold = 0.2;
        cluster_min_points = 3;
        downsample_factor = 2;
        local_map_size = 3.0;
    }
    
    /**
     * @brief 从YAML节点加载配置
     */
    void loadFromYaml(const YAML::Node& node);
    
    /**
     * @brief 验证配置有效性
     */
    bool validate() const;
};

// =============================================================================
// TEB规划器核心类
// =============================================================================

/**
 * @class TebPlanner
 * @brief TEB局部规划器 - 纯算法实现，不依赖ROS
 * 
 * 该类封装了所有规划计算功能：
 * - 障碍物管理
 * - 路径边界生成
 * - 目标点选择
 * - TEB轨迹优化
 */
class TebPlanner {
public:
    /**
     * @brief 默认构造函数
     */
    TebPlanner();
    
    /**
     * @brief 带配置构造函数
     * @param config 规划器配置
     */
    explicit TebPlanner(const PlannerConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~TebPlanner();
    
    // =========================================================================
    // 初始化和配置
    // =========================================================================
    
    /**
     * @brief 初始化规划器
     * @param config 规划器配置
     * @return true-成功, false-失败
     */
    bool initialize(const PlannerConfig& config);
    
    /**
     * @brief 重置规划器状态
     */
    void reset();
    
    /**
     * @brief 更新配置
     * @param config 新配置
     */
    void updateConfig(const PlannerConfig& config);
    
    /**
     * @brief 获取当前配置
     * @return 当前配置
     */
    const PlannerConfig& getConfig() const;
    
    // =========================================================================
    // 障碍物管理
    // =========================================================================
    
    /**
     * @brief 设置障碍物
     * @param obstacles 障碍物容器
     */
    void setObstacles(const ObstContainer& obstacles);
    
    /**
     * @brief 添加障碍物
     * @param obstacle 障碍物指针
     */
    void addObstacle(const ObstaclePtr& obstacle);
    
    /**
     * @brief 清空障碍物
     */
    void clearObstacles();
    
    /**
     * @brief 获取当前障碍物
     * @return 障碍物容器引用
     */
    const ObstContainer& getObstacles() const;
    
    // =========================================================================
    // 路径边界管理
    // =========================================================================
    
    /**
     * @brief 设置全局路径
     * @param path 路径点序列
     */
    void setGlobalPath(const std::vector<PathPoint>& path);
    
    /**
     * @brief 生成路径边界
     * @return 生成的边界
     */
    const PathBoundaries& generatePathBoundaries();
    
    /**
     * @brief 获取路径边界
     * @return 路径边界引用
     */
    const PathBoundaries& getPathBoundaries() const;
    
    /**
     * @brief 更新边界障碍物
     */
    void updateBoundaryObstacles();
    
    // =========================================================================
    // 核心规划功能
    // =========================================================================
    
    /**
     * @brief 执行一次规划
     * @param current_state 当前机器人状态
     * @param goal_point 目标点
     * @return 规划结果
     */
    PlanningResult plan(const RobotState& current_state, const Eigen::Vector2d& goal_point);
    
    /**
     * @brief 执行完整规划（使用全局路径）
     * @param current_state 当前机器人状态
     * @return 规划结果
     */
    PlanningResult planWithGlobalPath(const RobotState& current_state);
    
    /**
     * @brief 设置是否使用直接接近模式
     * @param enable 是否启用
     */
    void setDirectApproach(bool enable);
    
    /**
     * @brief 检查是否到达终点
     * @param current_state 当前状态
     * @return true-到达
     */
    bool isGoalReached(const RobotState& current_state) const;
    
    /**
     * @brief 计算到终点的距离
     * @return 距离
     */
    double distanceToGoal() const;
    
    // =========================================================================
    // 辅助功能
    // =========================================================================
    
    /**
     * @brief 计算点到障碍物的最小距离
     * @param point 点坐标
     * @return 最小距离
     */
    double getMinObstacleDistance(const Eigen::Vector2d& point) const;
    
    /**
     * @brief 检查点是否安全（无碰撞）
     * @param point 点坐标
     * @param min_dist 最小安全距离
     * @return true-安全
     */
    bool isPointSafe(const Eigen::Vector2d& point, double min_dist) const;
    
    /**
     * @brief 获取最近路径点索引
     * @param position 位置
     * @return 索引
     */
    int findNearestPathIndex(const Eigen::Vector2d& position) const;
    
    /**
     * @brief 计算路径段平均长度
     * @param start_index 起始索引
     * @return 平均长度
     */
    double calculateAverageSegmentLength(size_t start_index) const;
    
    /**
     * @brief 设置终点位置
     * @param goal 终点位置
     */
    void setGoal(const Eigen::Vector2d& goal);
    
    /**
     * @brief 获取终点位置
     * @return 终点位置
     */
    const Eigen::Vector2d& getGoal() const;
    
    /**
     * @brief 获取最终目标点
     * @return 目标点
     */
    const Eigen::Vector2d& getFinalGoal() const;
    
    /**
     * @brief 设置最终目标点
     * @param goal 目标点
     */
    void setFinalGoal(const Eigen::Vector2d& goal);
    
    /**
     * @brief 获取规划器内部状态（用于调试）
     */
    struct DebugInfo {
        Eigen::Vector2d current_goal_point;
        int current_goal_index;
        bool is_using_direct_approach;
        int obstacles_count;
        int boundary_obstacles_count;
        double last_planning_time_ms;
        bool last_success;
        std::string last_message;
    };
    
    DebugInfo getDebugInfo() const;

private:
    // =========================================================================
    // 私有成员变量
    // =========================================================================
    
    PlannerConfig config_;              // 规划配置
    TebConfig teb_config_;            // TEB配置
    ObstContainer obstacles_;          // 障碍物容器
    
    std::vector<PathPoint> global_path_;    // 全局路径
    Eigen::Vector2d goal_point_;          // 当前目标点
    Eigen::Vector2d final_goal_;          // 最终目标点
    bool use_direct_approach_;            // 是否使用直接接近
    
    PathBoundaries boundaries_;        // 路径边界
    ObstContainer boundary_obstacles_; // 边界障碍物
    
    TebOptimalPlannerPtr planner_;      // TEB规划器
    RobotFootprintModelPtr robot_model_; // 机器人模型
    
    std::unique_ptr<SensorProcessor> sensor_processor_; // 传感器处理器
    
    // 状态变量
    int current_goal_index_;          // 当前目标索引
    bool is_initialized_;             // 初始化标志
    bool goal_received_;              // 目标接收标志
    bool goal_reached_;               // 目标到达标志
    
    // 调试信息
    mutable DebugInfo debug_info_;
    double last_planning_time_ms_;
    
    // =========================================================================
    // 私有成员函数
    // =========================================================================
    
    /**
     * @brief 构建TEB配置
     */
    void buildTebConfig();
    
    /**
     * @brief 初始化TEB规划器
     */
    bool initPlanner();
    
    /**
     * @brief 选择下一个目标点
     * @param current_state 当前状态
     * @return 选中的目标点
     */
    Eigen::Vector2d selectNextGoal(const RobotState& current_state);
    
    /**
     * @brief 检查是否应该直接接近终点
     * @param current_state 当前状态
     * @return true-应该直接接近
     */
    bool shouldUseDirectApproach(const RobotState& current_state) const;
    
    /**
     * @brief 计算直接接近速度
     * @param current_state 当前状态
     * @return 速度命令
     */
    VelocityCommand calculateDirectApproachVelocity(const RobotState& current_state);
    
    /**
     * @brief 创建边界障碍物
     */
    void createBoundaryObstacles();
    
    /**
     * @brief 将全局路径转换为via-points
     */
    void extractViaPoints();
    
    /**
     * @brief 获取via-points容器
     */
    ViaPointContainer getViaPoints() const;
    
    /**
     * @brief 更新调试信息
     * @param result 规划结果
     */
    void updateDebugInfo(const PlanningResult& result);
};

// 共享指针类型定义
typedef boost::shared_ptr<TebPlanner> TebPlannerPtr;
typedef boost::shared_ptr<const TebPlanner> TebPlannerConstPtr;

} // namespace teb_local_planner

#endif // TEB_PLANNER_H
