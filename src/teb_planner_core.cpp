/**
 * @file teb_planner_core.cpp
 * @brief TEB规划器核心类实现 - 纯C++算法库，无ROS依赖
 */

#include "teb_planner_core.h"
#include <fstream>
#include <sstream>
#include <yaml-cpp/yaml.h>

namespace teb_local_planner {

// ============================================================================
// 构造函数与析构函数
// ============================================================================

TebPlannerCore::TebPlannerCore() {
}

TebPlannerCore::TebPlannerCore(const std::string& config_file) {
    loadConfig(config_file);
}

TebPlannerCore::~TebPlannerCore() = default;



// 让参数与config.yaml关联的实现
bool TebPlannerCore::loadConfig(const std::string& config_file) {
    try {
        YAML::Node config = YAML::LoadFile(config_file);

        if (!config) {
            logError("Failed to load config file: " + config_file);
            return false;
        }

        // 解析参数节点，依照config.yaml的结构优先级查找
        YAML::Node params = config["teb_local_planner"]["parameters"];
        if (!params) params = config["parameters"];
        if (!params) params = config;

        if (!params) {
            logError("Cannot find configuration parameters in config file");
            return false;
        }

        // 关联robot参数
        if (params["robot"]) {
            const auto& robot = params["robot"];
            if (robot["type"]) fast_config_.type = robot["type"].as<std::string>();
            if (robot["max_vel_x"]) fast_config_.max_vel_x = robot["max_vel_x"].as<double>();
            if (robot["max_vel_theta"]) fast_config_.max_vel_theta = robot["max_vel_theta"].as<double>();
            if (robot["wheelbase"]) fast_config_.wheelbase = robot["wheelbase"].as<double>();
            if (robot["cmd_angle_instead_rotvel"]) fast_config_.cmd_angle_instead_rotvel = robot["cmd_angle_instead_rotvel"].as<bool>();
            
            // 处理footprint类型
            if (robot["footprint"]) {
                const auto& footprint = robot["footprint"];
                if (footprint["type"]) {
                    std::string footprint_type = footprint["type"].as<std::string>();
                    if (footprint_type == "circle") {
                        if (footprint["radius"]) {
                            fast_config_.footprint_radius = footprint["radius"].as<double>();
                            robot_model_ = boost::make_shared<CircularRobotFootprint>(fast_config_.footprint_radius);
                        }
                    } else if (footprint_type == "polygon") {
                        // 手动解析 polygon vertices
                        if (footprint["vertices"]) {
                            Point2dContainer vertices;
                            for (const auto& v : footprint["vertices"]) {
                                vertices.emplace_back(v[0].as<double>(), v[1].as<double>());
                            }
                            robot_model_ = boost::make_shared<PolygonRobotFootprint>(vertices);
                        }
                    }
                }
            }
            
            if (robot["min_turning_radius"]) fast_config_.min_turning_radius = robot["min_turning_radius"].as<double>();
            if (robot["acc_lim_x"]) fast_config_.acc_lim_x = robot["acc_lim_x"].as<double>();
            if (robot["acc_lim_y"]) fast_config_.acc_lim_y = robot["acc_lim_y"].as<double>();
            if (robot["acc_lim_theta"]) fast_config_.acc_lim_theta = robot["acc_lim_theta"].as<double>();
            if (robot["max_vel_x_backwards"]) fast_config_.max_vel_x_backwards = robot["max_vel_x_backwards"].as<double>();
            if (robot["max_vel_y"]) fast_config_.max_vel_y = robot["max_vel_y"].as<double>();
        }

        // 轨迹配置 trajectory
        if (params["trajectory"]) {
            const auto& traj = params["trajectory"];
            if (traj["dt_ref"]) teb_config_.trajectory.dt_ref = traj["dt_ref"].as<double>();
            if (traj["dt_hysteresis"]) teb_config_.trajectory.dt_hysteresis = traj["dt_hysteresis"].as<double>();
            if (traj["min_samples"]) teb_config_.trajectory.min_samples = traj["min_samples"].as<int>();
            if (traj["max_samples"]) teb_config_.trajectory.max_samples = traj["max_samples"].as<int>();
            if (traj["force_reinit_new_goal_dist"]) teb_config_.trajectory.force_reinit_new_goal_dist = traj["force_reinit_new_goal_dist"].as<double>();
            if (traj["force_reinit_new_goal_angular"]) teb_config_.trajectory.force_reinit_new_goal_angular = traj["force_reinit_new_goal_angular"].as<double>();
        }

        // 优化配置 optimization
        if (params["optimization"]) {
            const auto& optim = params["optimization"];
            if (optim["no_inner_iterations"]) teb_config_.optim.no_inner_iterations = optim["no_inner_iterations"].as<int>();
            if (optim["no_outer_iterations"]) teb_config_.optim.no_outer_iterations = optim["no_outer_iterations"].as<int>();
            if (optim["weight_obstacle"]) teb_config_.optim.weight_obstacle = optim["weight_obstacle"].as<double>();
            if (optim["weight_viapoint"]) teb_config_.optim.weight_viapoint = optim["weight_viapoint"].as<double>();
            if (optim["weight_kinematics_nh"]) teb_config_.optim.weight_kinematics_nh = optim["weight_kinematics_nh"].as<double>();
            if (optim["weight_kinematics_forward_drive"]) teb_config_.optim.weight_kinematics_forward_drive = optim["weight_kinematics_forward_drive"].as<double>();
            if (optim["weight_max_vel_x"]) teb_config_.optim.weight_max_vel_x = optim["weight_max_vel_x"].as<double>();
            if (optim["weight_max_vel_theta"]) teb_config_.optim.weight_max_vel_theta = optim["weight_max_vel_theta"].as<double>();
            if (optim["weight_acc_lim_x"]) teb_config_.optim.weight_acc_lim_x = optim["weight_acc_lim_x"].as<double>();
            if (optim["weight_acc_lim_theta"]) teb_config_.optim.weight_acc_lim_theta = optim["weight_acc_lim_theta"].as<double>(); 
            if (optim["optimization_activate"]) teb_config_.optim.optimization_activate = optim["optimization_activate"].as<bool>();
            if (optim["enable_homotopy_class_planning"]) teb_config_.hcp.enable_homotopy_class_planning = optim["enable_homotopy_class_planning"].as<bool>();
        }

        // 障碍物 obstacles
        if (params["obstacles"]) {
            const auto& obs = params["obstacles"];
            if (obs["min_obstacle_dist"]) {
                teb_config_.obstacles.min_obstacle_dist = obs["min_obstacle_dist"].as<double>();
            }
            if (obs["inflation_dist"]) teb_config_.obstacles.inflation_dist = obs["inflation_dist"].as<double>();
            if (obs["obstacle_poses_affected"]) teb_config_.obstacles.obstacle_poses_affected = obs["obstacle_poses_affected"].as<int>();
            if (obs["obstacle_square_size"]) teb_config_.obstacles.obstacle_square_size = obs["obstacle_square_size"].as<double>();
        }

        // 目标追踪 goal_tracking
        if (params["goal_tracking"]) {
            const auto& goal = params["goal_tracking"];
            if (goal["goal_reached_threshold"]) fast_config_.goal_reached_threshold = goal["goal_reached_threshold"].as<double>();
            if (goal["approach_goal_distance"]) fast_config_.approach_goal_distance = goal["approach_goal_distance"].as<double>();
            if (goal["final_approach_speed"]) fast_config_.final_approach_speed = goal["final_approach_speed"].as<double>();
            if (goal["max_angle_diff_for_forward"]) fast_config_.max_angle_diff_for_forward = goal["max_angle_diff_for_forward"].as<double>();
            if (goal["angle_gain"]) fast_config_.angle_gain = goal["angle_gain"].as<double>();
            if (goal["final_angle_gain"]) fast_config_.final_angle_gain = goal["final_angle_gain"].as<double>();
            if (goal["speed_distance_factor"]) fast_config_.speed_distance_factor = goal["speed_distance_factor"].as<double>();
            if (goal["enable_direct_approach"]) fast_config_.enable_direct_approach = goal["enable_direct_approach"].as<bool>();
        }
        // 查找下一个目标点 find_next_goal
        if (params["find_next_goal"]) {
            const auto& find = params["find_next_goal"];
            if (find["lookahead_distance"]) fast_config_.lookahead_distance = find["lookahead_distance"].as<double>();
            if (find["min_obstacle_dist"]) fast_config_.min_obstacle_dist = find["min_obstacle_dist"].as<double>();
            if (find["max_longitudinal_expand"]) fast_config_.max_longitudinal_expand = find["max_longitudinal_expand"].as<int>();
            if (find["longitudinal_step_points"]) fast_config_.longitudinal_step_points = find["longitudinal_step_points"].as<int>();
            if (find["max_lateral_search"]) fast_config_.max_lateral_search = find["max_lateral_search"].as<int>();
            if (find["lateral_search_step"]) fast_config_.lateral_search_step = find["lateral_search_step"].as<int>();
            if (find["lateral_safety_margin"]) fast_config_.lateral_safety_margin = find["lateral_safety_margin"].as<double>();
            if (find["max_expand_index_limit"]) fast_config_.max_expand_index_limit = find["max_expand_index_limit"].as<int>();
            if (find["avg_segment_calc_points"]) fast_config_.avg_segment_calc_points = find["avg_segment_calc_points"].as<int>();
        }

        // 边界障碍物 boundary_obstacles
        if (params["boundary_obstacles"]) {
            const auto& boundary = params["boundary_obstacles"];
            if (boundary["enable"]) boundary_obstacles_enabled_ = boundary["enable"].as<bool>();
            if (boundary["inflation"]) boundary_obstacles_inflation_ = boundary["inflation"].as<double>();
            if (boundary["point_interval"]) boundary_obstacles_point_interval_ = boundary["point_interval"].as<double>();
            if (boundary["max_obstacle_dist"]) boundary_obstacles_max_obstacle_dist_ = boundary["max_obstacle_dist"].as<double>();
        }
        // 路径边界 path_boundary
        if (params["path_boundary"]) {
            const auto& pb = params["path_boundary"];
            if (pb["enabled"]) path_boundary_enabled_ = pb["enabled"].as<bool>();
            if (pb["boundary_width"]) path_boundary_width_ = pb["boundary_width"].as<double>();
            if (pb["point_interval"]) path_boundary_point_interval_ = pb["point_interval"].as<double>();
        }

        // 生成TEB配置并初始化
        buildTebConfig();
        initializePlanner();

        config_loaded_ = true;
        logInfo("Configuration loaded successfully from: " + config_file);

        return true;

    } catch (const YAML::Exception& e) {
        logError("Failed to parse config file: " + std::string(e.what()));
        return false;
    } catch (const std::exception& e) {
        logError("Error loading config: " + std::string(e.what()));
        return false;
    }
}


void TebPlannerCore::setLogCallback(LogCallback callback) {
    log_callback_ = callback;
}

// ============================================================================
// 机器人状态设置
// ============================================================================

void TebPlannerCore::setCurrentPose(const Pose2D& pose) {
    current_pose_ = pose;
}

void TebPlannerCore::setCurrentVelocity(double vx, double vy, double omega) {
    current_velocity_.linear.x() = vx;
    current_velocity_.linear.y() = vy;
    current_velocity_.angular.z() = omega;
}

Pose2D TebPlannerCore::getCurrentPose() const {
    return current_pose_;
}

// ============================================================================
// 全局路径设置
// ============================================================================

void TebPlannerCore::setGlobalPath(const Path2D& path) {
    global_path_ = path;
    
    if (!global_path_.empty()) {
        final_goal_pose_ = global_path_.poses.back();
        current_goal_index_ = 0;
        goal_reached_ = false;
        global_path_received_ = true;
        
        // 生成路径边界
        if (path_boundary_enabled_) {
            generatePathBoundaries();
        }
        
        logInfo("Global path received: " + std::to_string(path.size()) + " poses");
    }
}

void TebPlannerCore::clearGlobalPath() {
    global_path_.poses.clear();
    global_path_received_ = false;
    goal_reached_ = false;
    current_goal_index_ = 0;
}

bool TebPlannerCore::hasValidGlobalPath() const {
    return global_path_received_ && !global_path_.empty();
}

// ============================================================================
// 障碍物管理
// ============================================================================

void TebPlannerCore::setObstacles(const ObstContainer& obstacles)
{
    obstacles_ = obstacles;
    planner_->setObstVector(&obstacles_);   // 设置障碍物容器
}
// ============================================================================
// 边界管理
// ============================================================================

void TebPlannerCore::generatePathBoundaries() {
    if (global_path_.empty() || global_path_.poses.size() < 2) {
        logWarn("Path too short to generate boundaries");
        return;
    }
        
    cached_boundary_.left_boundary.clear();
    cached_boundary_.right_boundary.clear();
    
    const double half_width = path_boundary_width_ / 2.0;
    const size_t n = global_path_.poses.size();
    
    // 预提取路径位置
    std::vector<Eigen::Vector2d> path_positions;
    path_positions.reserve(n);
    for (const auto& pose : global_path_.poses) {
        path_positions.emplace_back(pose.x, pose.y);
    }
    
    // 计算平均段长
    double avg_segment_length = half_width;
    if (n > 1) {
        double total_length = 0.0;
        for (size_t i = 0; i < n - 1; ++i) {
            total_length += (path_positions[i + 1] - path_positions[i]).norm();
        }
        avg_segment_length = total_length / std::max(n - 1, size_t(1));
    }
    
    // 生成边界点
    for (size_t i = 0; i < n; ++i) {
        // 计算路径方向
        Eigen::Vector2d direction;
        if (i < n - 1) {
            direction = path_positions[i + 1] - path_positions[i];
        } else if (i > 0) {
            direction = path_positions[i] - path_positions[i - 1];
        } else {
            direction = Eigen::Vector2d(1.0, 0.0);
        }
        
        if (direction.squaredNorm() > std::numeric_limits<double>::epsilon()) {
            direction.normalize();
        }
        
        Eigen::Vector2d normal(-direction.y(), direction.x());
        const Eigen::Vector2d& pos = path_positions[i];
        
        Eigen::Vector2d left_pos = pos - half_width * normal;
        Eigen::Vector2d right_pos = pos + half_width * normal;
        
        // 缓存边界点用于可视化
        cached_boundary_.left_boundary.push_back({left_pos.x(), left_pos.y()});
        cached_boundary_.right_boundary.push_back({right_pos.x(), right_pos.y()});
    }
   
}
//生成边界障碍物
void TebPlannerCore::generateBoundaryObstacles() {
    // 清理旧边界障碍物
    left_boundary_obstacles_.clear();
    right_boundary_obstacles_.clear();

    // 采样参数
    const double point_interval = path_boundary_point_interval_;
    const double inflation = boundary_obstacles_inflation_;
    const double max_dist = boundary_obstacles_max_obstacle_dist_;

    const auto& left_boundary = cached_boundary_.left_boundary;
    const auto& right_boundary = cached_boundary_.right_boundary;

    // 辅助采样函数，采样并生成障碍物
    auto sampleBoundary = [=](const std::vector<Eigen::Vector2d>& boundary,
                              std::vector<boost::shared_ptr<CircularObstacle>>& obstacles) {
        if (boundary.empty()) return;

        Eigen::Vector2d robot_pos(current_pose_.x, current_pose_.y);

        // 第一采样点
        Eigen::Vector2d prev = boundary[0];
        if ((prev - robot_pos).norm() <= max_dist) {
            obstacles.push_back(boost::make_shared<CircularObstacle>(prev.x(), prev.y(), inflation));
        }

        double acc_dist = 0.0;

        // 对边界点做等距采样
        for (size_t i = 1; i < boundary.size(); ++i) {
            const Eigen::Vector2d& pt = boundary[i];
            double seg_dist = (pt - prev).norm();
            double seg_acc = acc_dist + seg_dist;

            // 间隔采样本段，acc_dist用于累积采样位置
            while (seg_acc >= point_interval) {
                double remain = point_interval - acc_dist;
                double ratio = (remain) / seg_dist;
                Eigen::Vector2d interp = prev + ratio * (pt - prev);
                if ((interp - robot_pos).norm() <= max_dist) {
                    obstacles.push_back(boost::make_shared<CircularObstacle>(interp.x(), interp.y(), inflation));
                }
                prev = interp;
                seg_dist = (pt - prev).norm();
                acc_dist = 0.0;
                seg_acc = seg_dist;
            }
            acc_dist += seg_dist;
            prev = pt;

            // 最后端点采样
            if ((pt - robot_pos).norm() <= max_dist) {
                obstacles.push_back(boost::make_shared<CircularObstacle>(pt.x(), pt.y(), inflation));
            }
        }
    };

    sampleBoundary(left_boundary, left_boundary_obstacles_);
    sampleBoundary(right_boundary, right_boundary_obstacles_);
}
//获取边界障碍物
void TebPlannerCore::getBoundaryObstacles(ObstContainer& obstacles) const {
    obstacles.insert(obstacles.end(), left_boundary_obstacles_.begin(), left_boundary_obstacles_.end());
    obstacles.insert(obstacles.end(), right_boundary_obstacles_.begin(), right_boundary_obstacles_.end());
}

//设置边界使能状态
void TebPlannerCore::setBoundaryEnabled(bool enabled) {
    path_boundary_enabled_ = enabled;
}

bool TebPlannerCore::isBoundaryEnabled() const {
    return path_boundary_enabled_;
}

BoundaryData TebPlannerCore::getBoundaryData() const {
    return cached_boundary_;
}

// ============================================================================
// 核心规划API
// ============================================================================


PlanningResult TebPlannerCore::plan() {
    return planInternal();
}

PlanningResult TebPlannerCore::planInternal() {
    PlanningResult result;
    
    // 状态检查
    if (!hasValidGlobalPath()) {
        result.success = false;
        result.message = "No valid global path";
        return result;
    } 
   
    // 检查是否到达终点
    if (distanceToGoal() < fast_config_.goal_reached_threshold) {
        goal_reached_ = true;
        result.success = true;
        result.velocity_x = 0;
        result.velocity_theta = 0;
        result.goal_reached = true;
        result.message = "Goal reached";
        clearGlobalPath();
        return result;
    }
    if (ifGoalApproach()) {
        //接近终点，计算最终接近速度
        auto [vx, omega] = calculateFinalApproachVelocity();
        result.velocity_x = vx;
        result.velocity_theta = omega;
        result.goal_reached = false;
        result.message = "Goal approach";
        result.success = true;

        return result;
    }
    goal_reached_ = false;
     // 选择目标点
     Pose2D goal_pose;
     goal_pose = selectNextGoal();
     result.aim_pose = goal_pose;
     
    // 执行TEB规划
    try {
        PoseSE2 start(current_pose_.x, current_pose_.y, current_pose_.theta);
        PoseSE2 goal(goal_pose.x, goal_pose.y, goal_pose.theta);
        bool plan_success=false;
        {
            std::lock_guard<std::mutex> lock(obstacle_mutex_);
            plan_success = planner_->plan(start, goal, &current_velocity_);
        }
       
        if (plan_success) {
            // 获取轨迹
            std::vector<Eigen::Vector3f> traj;
            planner_->getFullTrajectory(traj);
            
            result.trajectory.clear();
            for (const auto& p : traj) {
                result.trajectory.push_back({static_cast<double>(p[0]), 
                                           static_cast<double>(p[1]), 
                                           static_cast<double>(p[2])});
            }
            
            // 获取速度
            float vx = 0.0f, omega = 0.0f;
            std::vector<Twist> vel_profile;
            planner_->getVelocityProfile(vel_profile);
            
            if (!vel_profile.empty()) {
                vx = vel_profile[1].linear.x();
                omega = vel_profile[1].angular.z();
            }
            
            // 限制速度
            result.velocity_x = std::clamp(static_cast<double>(vx), 
                                         -fast_config_.max_vel_x_backwards, 
                                         fast_config_.max_vel_x);
            result.velocity_theta = std::clamp(static_cast<double>(omega), 
                                             -fast_config_.max_vel_theta, 
                                             fast_config_.max_vel_theta);
            
            result.success = true;
            result.message = "Planning successful";
        } else {
            result.success = false;
            result.message = "TEB planning failed";
        }
        
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Planning exception: ") + e.what();
    }
    
    return result;
}

// ============================================================================
// 目标点选择
// ============================================================================

Pose2D TebPlannerCore::selectNextGoal() {
    if (global_path_.empty()) {
        return current_pose_;
    }
    
    int current_index = findNearestPoseIndex(current_pose_);
    if (current_index < 0) {
        return current_pose_;
    }
    
    // 记录上一次的目标索引
    int last_goal_index = static_cast<int>(current_goal_index_);
    int start_index = std::max(current_index, last_goal_index);
    
    // 在前视距离内查找安全点
    double accumulated_distance = 0.0;
    const double min_obstacle_dist = fast_config_.min_obstacle_dist;
    
    for (size_t i = start_index; i < global_path_.poses.size() - 1; ++i) {
        const auto& current_pt = global_path_.poses[i];
        const auto& next_pt = global_path_.poses[i + 1];
        
        double dx = next_pt.x - current_pt.x;
        double dy = next_pt.y - current_pt.y;
        double segment_length = std::sqrt(dx * dx + dy * dy);
        
        if (accumulated_distance + segment_length >= fast_config_.lookahead_distance) {
            double remaining = fast_config_.lookahead_distance - accumulated_distance;
            double ratio = (segment_length > 1e-6) ? remaining / segment_length : 0.0;
            
            Pose2D goal;
            goal.x = current_pt.x + dx * ratio;
            goal.y = current_pt.y + dy * ratio;
            goal.theta = std::atan2(dy, dx);
            
            if (isPointSafe(goal.x, goal.y, goal.theta)) {
                current_goal_index_ = i;
                return goal;
            }
            break;
        }
        accumulated_distance += segment_length;
    }
    
    // 如果没找到安全点，保持在start_index
    Pose2D goal_pose;
    goal_pose.x = global_path_.poses[start_index].x;
    goal_pose.y = global_path_.poses[start_index].y;
    goal_pose.theta = getPathPoseYaw(start_index);
    
    // 检查是否安全，不安全则纵向搜索
    if (!isPointSafe(goal_pose.x, goal_pose.y, goal_pose.theta)) {
        int max_expand = fast_config_.max_longitudinal_expand;
        int search_index = start_index;
        
        while (!isPointSafe(goal_pose.x, goal_pose.y, goal_pose.theta) && max_expand > 0) {
            max_expand--;
            double avg_seg_len = calculateAverageSegmentLength(start_index);
            int expand_num = (avg_seg_len > 1e-6) ? 
                static_cast<int>(2.0 * fast_config_.footprint_radius / avg_seg_len) : 10;
            
            search_index += expand_num;
            
            if (search_index >= static_cast<int>(global_path_.poses.size()) - 1) {
                search_index = global_path_.poses.size() - 1;
                goal_pose.x = global_path_.poses.back().x;
                goal_pose.y = global_path_.poses.back().y;
                goal_pose.theta = getPathPoseYaw(global_path_.poses.size() - 1);
                break;
            }
            
            goal_pose.x = global_path_.poses[search_index].x;
            goal_pose.y = global_path_.poses[search_index].y;
            goal_pose.theta = getPathPoseYaw(search_index);
        }
        
        current_goal_index_ = search_index;
    } else {
        current_goal_index_ = start_index;
    }
    
    return goal_pose;
}

std::tuple<double, double> TebPlannerCore::calculateFinalApproachVelocity() {
    double dx = final_goal_pose_.x - current_pose_.x;
    double dy = final_goal_pose_.y - current_pose_.y;
    double distance = std::sqrt(dx * dx + dy * dy);
    
    double target_theta = std::atan2(dy, dx);
    double angle_diff = target_theta - current_pose_.theta;
    while (angle_diff > M_PI) angle_diff -= 2.0 * M_PI;
    while (angle_diff < -M_PI) angle_diff += 2.0 * M_PI;
    
    double vx = 0.0;
    double omega = std::clamp(angle_diff * fast_config_.final_angle_gain, 
                             -fast_config_.max_vel_theta, 
                             fast_config_.max_vel_theta);
        
    if (distance > fast_config_.goal_reached_threshold) {
        vx = std::min(fast_config_.final_approach_speed, distance * fast_config_.speed_distance_factor);
        vx = std::max(0.0, vx);
    }
    
    return {vx, omega};
}

// ============================================================================
// 状态查询
// ============================================================================

bool TebPlannerCore::isGoalReached() const {
    return goal_reached_;
}
bool TebPlannerCore::ifGoalApproach() const {
    return current_pose_.distanceTo(final_goal_pose_) < fast_config_.approach_goal_distance;
}
double TebPlannerCore::distanceToGoal() const {
    return current_pose_.distanceTo(final_goal_pose_);
}

std::string TebPlannerCore::getStatus() const {
    return checkStatus();
}

std::string TebPlannerCore::getConfigString() const {
    std::ostringstream oss;
    oss << "TEB Planner Config:\n";
    oss << "  max_vel_x: " << fast_config_.max_vel_x << "\n";
    oss << "  max_vel_theta: " << fast_config_.max_vel_theta << "\n";
    oss << "  lookahead_distance: " << fast_config_.lookahead_distance << "\n";
    oss << "  min_obstacle_dist: " << fast_config_.min_obstacle_dist << "\n";
    oss << "  goal_reached_threshold: " << fast_config_.goal_reached_threshold << "\n";
    oss << "  boundary_enabled: " << (path_boundary_enabled_ ? "true" : "false") << "\n";
    oss << "  boundary_width: " << path_boundary_width_ << "\n";
    return oss.str();
}

std::string TebPlannerCore::checkStatus() const {
    std::ostringstream oss;
    oss << "Status: ";
    
    if (!config_loaded_) {
        oss << "Config not loaded";
    } else if (!hasValidGlobalPath()) {
        oss << "No global path";
    } else if (goal_reached_) {
        oss << "Goal reached";
    } else {
        oss << "Planning";
        oss << ", goal_index: " << current_goal_index_;
        oss << ", dist_to_goal: " << std::fixed << std::setprecision(2) << distanceToGoal();
    }
    
    return oss.str();
}

// ============================================================================
// 工具方法
// ============================================================================

int TebPlannerCore::findNearestPoseIndex(const Pose2D& pose) const {
    if (global_path_.empty()) return -1;
    
    double min_distance = std::numeric_limits<double>::max();
    int nearest_index = 0;
    
    for (size_t i = 0; i < global_path_.poses.size(); ++i) {
        double dx = global_path_.poses[i].x - pose.x;
        double dy = global_path_.poses[i].y - pose.y;
        double distance = std::sqrt(dx * dx + dy * dy);
        
        if (distance < min_distance) {
            min_distance = distance;
            nearest_index = static_cast<int>(i);
        }
    }
    return nearest_index;
}

double TebPlannerCore::calculateAverageSegmentLength(size_t start_index) const {
    if (global_path_.poses.size() < 2 || 
        start_index >= global_path_.poses.size() - 1) {
        return 0.1;
    }
    
    double total_length = 0.0;
    int count = 0;
    size_t end_index = std::min(start_index + 8, global_path_.poses.size() - 1);
    
    for (size_t i = start_index; i < end_index; ++i) {
        const auto& p1 = global_path_.poses[i];
        const auto& p2 = global_path_.poses[i + 1];
        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        total_length += std::sqrt(dx * dx + dy * dy);
        count++;
    }
    
    return (count > 0) ? (total_length / count) : 0.1;
}

bool TebPlannerCore::isPointSafe(double x, double y, double theta) {
    (void)theta;
    
    std::lock_guard<std::mutex> lock(obstacle_mutex_);
    Eigen::Vector2d point(x, y);
    
    // 检查普通障碍物
    for (const auto& obst : obstacles_) {
        double dist = obst->getMinimumDistance(point);
        if (dist < fast_config_.min_obstacle_dist) {
            return false;
        }
    }
    
    return true;
}

double TebPlannerCore::getPathPoseYaw(size_t index) const {
    if (index >= global_path_.poses.size()) return 0.0;
    
    if (index < global_path_.poses.size() - 1) {
        const auto& current = global_path_.poses[index];
        const auto& next = global_path_.poses[index + 1];
        return std::atan2(next.y - current.y, next.x - current.x);
    } else if (index > 0) {
        const auto& prev = global_path_.poses[index - 1];
        const auto& current = global_path_.poses[index];
        return std::atan2(current.y - prev.y, current.x - prev.x);
    }
    
    return 0.0;
}

// ============================================================================
// 高级配置
// ============================================================================

void TebPlannerCore::setLookaheadDistance(double distance) {
    fast_config_.lookahead_distance = distance;
}

void TebPlannerCore::setMinObstacleDistance(double distance) {
    fast_config_.min_obstacle_dist = distance;
    teb_config_.obstacles.min_obstacle_dist = distance;
}

void TebPlannerCore::setObstacleWeight(double weight) {
    teb_config_.optim.weight_obstacle = weight;
}

void TebPlannerCore::setViapointWeight(double weight) {
    teb_config_.optim.weight_viapoint = weight;
}

const TebConfig& TebPlannerCore::getTebConfig() const {
    return teb_config_;
}

// ============================================================================
// 内部方法
// ============================================================================

void TebPlannerCore::buildTebConfig() {
    // 机器人配置
    teb_config_.robot.max_vel_x = fast_config_.max_vel_x;
    teb_config_.robot.max_vel_x_backwards = fast_config_.max_vel_x_backwards;
    teb_config_.robot.max_vel_y = fast_config_.max_vel_y;
    teb_config_.robot.max_vel_theta = fast_config_.max_vel_theta;
    teb_config_.robot.acc_lim_x = fast_config_.acc_lim_x;
    teb_config_.robot.acc_lim_y = fast_config_.acc_lim_y;
    teb_config_.robot.acc_lim_theta = fast_config_.acc_lim_theta;
    teb_config_.robot.min_turning_radius = fast_config_.min_turning_radius;
    teb_config_.robot.wheelbase = fast_config_.wheelbase;
    teb_config_.robot.cmd_angle_instead_rotvel = fast_config_.cmd_angle_instead_rotvel;
    
    // 障碍物配置
    teb_config_.obstacles.min_obstacle_dist = fast_config_.min_obstacle_dist;
    teb_config_.obstacles.inflation_dist = fast_config_.inflation_dist;
    teb_config_.obstacles.obstacle_poses_affected = fast_config_.obstacle_poses_affected;
    teb_config_.obstacles.obstacle_square_size = fast_config_.obstacle_square_size;
    
    // 同伦类规划
    teb_config_.hcp.enable_homotopy_class_planning = fast_config_.enable_homotopy_class_planning;
}

void TebPlannerCore::initializePlanner() {
    // 创建可视化器
    visual_ = std::make_unique<TebVisualization>(teb_config_);
    
    // 创建规划器
    planner_ = std::make_unique<TebOptimalPlanner>(
        teb_config_, 
        &obstacles_, 
        robot_model_, 
        TebVisualizationPtr(visual_.get()), 
        nullptr
    );
    
    initialized_ = true;
    logInfo("Planner initialized");
}

// ============================================================================
// 日志输出
// ============================================================================

void TebPlannerCore::logInfo(const std::string& message) {
    if (log_callback_) {
        log_callback_("INFO", message);
    }
}

void TebPlannerCore::logWarn(const std::string& message) {
    if (log_callback_) {
        log_callback_("WARN", message);
    }
}

void TebPlannerCore::logError(const std::string& message) {
    if (log_callback_) {
        log_callback_("ERROR", message);
    }
}

void TebPlannerCore::logDebug(const std::string& message) {
    if (log_callback_) {
        log_callback_("DEBUG", message);
    }
}

} // namespace teb_local_planner
