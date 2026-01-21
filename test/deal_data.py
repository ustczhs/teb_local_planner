# 读取原始数据文件
input_file = "data.txt"
output_file = "data_expanded.txt"

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        if line.startswith('#'):
            # 保留注释行
            f_out.write(line)
            continue
        
        # 去除首尾空白字符
        line = line.strip()
        if not line:
            f_out.write(line + '\n')
            continue
        
        # 分割数据行
        parts = line.split(',')
        if len(parts) >= 5:
            try:
                # 将corridor_dis列（第5列CMakeFiles/lib_teb.dir/src/teb_config.cpp.o，索引4）扩大1倍
                corridor_dis = float(parts[3].strip())
                new_corridor_dis = corridor_dis * 4.0
                parts[3] = f"{new_corridor_dis:.4f}"
                f_out.write(','.join(parts) + '\n')
            except ValueError:
                f_out.write(line + '\n')
        else:
            f_out.write(line + '\n')

print(f"处理完成！已将走廊边界扩大1倍，结果保存在 {output_file}")