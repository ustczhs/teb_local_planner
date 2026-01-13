# TEB Local Planner 配置系统

## 概述

TEB (Timed Elastic Band) 轻量级局部规划模块现已支持完整的YAML配置文件系统，允许用户通过修改配置文件来调整算法性能和行为。

## 配置文件结构

### 1. 机器人配置 (`robot`)
```yaml
robot:
  type: "differential_drive"  # 机器人类型: differential_drive, ackermann, omnidirectional, legged
  max_vel_x: 0.8             # 最大前进速度 (m/s)
  max_vel_x_backwards: 0.3   # 最大后退速度 (m/s)
  max_vel_y: 0.0             # 最大侧向速度 (m/s)
  max_vel_theta: 0.5         # 最大角速度 (rad/s)
  acc_lim_x: 0.5             # 最大线加速度 (m/s²)
  acc_lim_theta: 0.5         # 最大角加速度 (rad/s²)
  footprint:
    type: "circle"           # footprint类型: circle 或 polygon
    radius: 0.2              # 圆形半径 (m)
```

### 2. 轨迹规划配置 (`trajectory`)
```yaml
trajectory:
  dt_ref: 0.3                # 时间分辨率 (s)
  dt_hysteresis: 0.1         # 自动调整滞后
  min_samples: 3             # 最小采样点数
  max_samples: 200           # 最大采样点数 (影响运算效率)
  force_reinit_new_goal_dist: 1.0      # 热启动距离阈值 (m)
  force_reinit_new_goal_angular: 1.57  # 热启动角度阈值 (rad)
```

### 3. 优化配置 (`optimization`)
```yaml
optimization:
  no_inner_iterations: 5     # 内层迭代次数
  no_outer_iterations: 4     # 外层迭代次数
  weight_max_vel_x: 2        # 最大线速度权重
  weight_max_vel_theta: 1    # 最大角速度权重
  weight_acc_lim_x: 1        # 线加速度权重
  weight_kinematics_nh: 1000 # 非全向运动学权重
  weight_obstacle: 50        # 障碍物避让权重
  weight_frenet_corridor: 20 # Frenet走廊约束权重
```

### 4. 障碍物配置 (`obstacles`)
```yaml
obstacles:
  min_obstacle_dist: 0.5     # 最小障碍物距离 (m)
  inflation_dist: 0.6        # 障碍物缓冲区距离 (m)
  obstacle_poses_affected: 15 # 受障碍物影响的位姿数
```

### 5. Frenet走廊约束 (`frenet_corridor`)
```yaml
frenet_corridor:
  enabled: true              # 是否启用Frenet走廊约束
  corridor_width: 1.0        # 走廊宽度 (m)
  min_corridor_width: 0.5    # 最小走廊宽度 (m)
```

### 6. 测试场景配置 (`simulation`)
```yaml
simulation:
  dt: 0.1                    # 仿真时间步长 (s)
  robot_speed: 0.5           # 机器人前进速度 (m/s)
  lookahead_distance: 4.0    # 规划前瞻距离 (m)
  obstacles:
    count: 2                 # 动态障碍物数量
    amplitude: 0.6           # 运动振幅 (m)
    frequency: 0.5           # 运动频率 (Hz)
    phase_offset: 1.57       # 相位偏移 (rad)
```

## 关键参数说明

### 运算效率相关参数
- **`trajectory.max_samples`**: 控制轨迹的采样点数，直接影响计算复杂度
- **`optimization.no_inner_iterations`**: G2O优化器的内层迭代次数
- **`optimization.no_outer_iterations`**: G2O优化器的外层迭代次数
- **`obstacles.obstacle_poses_affected`**: 每个障碍物影响的轨迹点数

### 执行效果相关参数
- **`robot.max_vel_x/theta`**: 机器人最大速度，影响规划的激进程度
- **`robot.acc_lim_x/theta`**: 加速度限制，影响轨迹的平滑性
- **`optimization.weight_obstacle`**: 障碍物避让强度
- **`optimization.weight_frenet_corridor`**: 走廊约束强度
- **`trajectory.force_reinit_new_goal_dist`**: 热启动灵敏度

## 使用方法

### 1. 修改配置
编辑 `config/config.yaml` 文件，根据需要调整参数。

### 2. 编译运行
```bash
cd build
make planner_test
./planner_test
```

### 3. 性能调优建议

#### 提高运算效率
- 降低 `trajectory.max_samples`
- 减少 `optimization.no_inner_iterations`
- 减小 `obstacles.obstacle_poses_affected`

#### 改善规划质量
- 增加 `optimization.weight_obstacle` 提高避障能力
- 调整 `optimization.weight_frenet_corridor` 控制走廊约束强度
- 优化 `robot.max_vel_x/theta` 和加速度限制

#### 热启动优化
- 增大 `trajectory.force_reinit_new_goal_dist` 减少重新初始化
- 调整 `trajectory.dt_ref` 平衡精度和效率

## 示例配置

### 高性能配置 (适合实时应用)
```yaml
trajectory:
  max_samples: 150
  dt_ref: 0.4
optimization:
  no_inner_iterations: 3
  no_outer_iterations: 3
obstacles:
  obstacle_poses_affected: 10
```

### 高精度配置 (适合复杂环境)
```yaml
trajectory:
  max_samples: 300
  dt_ref: 0.2
optimization:
  no_inner_iterations: 7
  no_outer_iterations: 5
  weight_obstacle: 80
  weight_frenet_corridor: 30
obstacles:
  obstacle_poses_affected: 20
```

## 运行结果

程序启动时会显示：
```
Configuration loaded successfully from ../config/config.yaml
Loaded XXX path points
```

可视化窗口显示：
- 黑色线条：参考路径
- 绿色点：走廊边界
- 红色圆点：动态障碍物
- 蓝色线条：规划轨迹
- 绿色圆点：机器人当前位置

按 ESC 键退出仿真。
