import math

def generate_stadium_path(file_name="data.txt"):
    # 基础参数配置
    step_size = 0.05       # 路径点步长: 0.05m
    straight_len = 10.0    # 直线段长度: 10m
    arc_len = 5.0          # 弧线段长度: 5m
    
    # 根据弧长计算圆弧半径 R = L / PI
    radius = arc_len / math.pi 
    curvature_val = 1.0 / radius
    
    # 运动与约束参数 (基于来源 [1-4])
    v_ref_straight = 0.5   # 直线参考速度
    v_ref_curve = 0.3      # 曲线参考速度 (考虑曲率约束降低速度)
    corridor_str = 1.5     # 直线段安全走廊宽度
    corridor_curve = 0.8   # 曲线段安全走廊宽度 (模拟窄道)

    points = []
    current_s = 0.0

    # 1. 第一段直线: (0,0) -> (10,0)
    num_steps = int(straight_len / step_size)
    for i in range(num_steps):
        x = i * step_size
        y = 0.0
        yaw = 0.0
        dt = step_size / v_ref_straight
        points.append([x, y, yaw, v_ref_straight, corridor_str, dt, 0.0, current_s])
        current_s += step_size

    # 2. 第一段圆弧 (右半圆)
    num_steps = int(arc_len / step_size)
    for i in range(num_steps):
        angle = (i * step_size) / radius
        x = straight_len + radius * math.sin(angle)
        y = radius - radius * math.cos(angle)
        yaw = angle
        dt = step_size / v_ref_curve
        points.append([x, y, yaw, v_ref_curve, corridor_curve, dt, curvature_val, current_s])
        current_s += step_size

    # 3. 第二段直线 (返回): (10, 2R) -> (0, 2R)
    num_steps = int(straight_len / step_size)
    for i in range(num_steps):
        x = straight_len - (i * step_size)
        y = 2 * radius
        yaw = math.pi
        dt = step_size / v_ref_straight
        points.append([x, y, yaw, v_ref_straight, corridor_str, dt, 0.0, current_s])
        current_s += step_size

    # 4. 第二段圆弧 (左半圆) 回到起点
    num_steps = int(arc_len / step_size)
    for i in range(num_steps):
        angle = (i * step_size) / radius
        x = 0.0 - radius * math.sin(angle)
        y = radius + radius * math.cos(angle)
        yaw = math.pi + angle
        dt = step_size / v_ref_curve
        points.append([x, y, yaw, v_ref_curve, corridor_curve, dt, curvature_val, current_s])
        current_s += step_size

    # 写入文件
    with open(file_name, "w") as f:
        f.write("# x, y, yaw, ref_v, corridor_dis, dt, kappa, s\n")
        for p in points:
            f.write(", ".join([f"{val:.4f}" for val in p]) + "\n")
    
    print(f"成功生成 {len(points)} 个路径点，已保存至 {file_name}")

if __name__ == "__main__":
    generate_stadium_path()
