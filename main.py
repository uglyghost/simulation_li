import numpy as np
import matplotlib.pyplot as plt

# 设置光电发射器的尺寸
detector_width = 100  # 宽度（毫米）
detector_height = 50  # 高度（毫米）

# 设置探测器的数量
num_sensors = 16

# 计算探测器位置（等间距分布）
sensor_positions_left = np.linspace(0, detector_height, num_sensors)
sensor_positions_right = np.linspace(0, detector_height, num_sensors)

# 随机生成光子的位置
photon_x = np.random.uniform(0, detector_width)
photon_y = np.random.uniform(0, detector_height)

# 假设光速为c，单位：毫米/纳秒
c = 300  # 光速（毫米/纳秒）

# 计算每个探测器接收到光子的时间
time_left = np.sqrt(photon_x**2 + (sensor_positions_left - photon_y)**2) / c
time_right = np.sqrt((detector_width - photon_x)**2 + (sensor_positions_right - photon_y)**2) / c

# 可视化仿真环境
plt.figure(figsize=(12, 6))

# 绘制光电发射器
plt.plot([0, detector_width], [0, 0], 'k-', lw=2)
plt.plot([0, detector_width], [detector_height, detector_height], 'k-', lw=2)
plt.plot([0, 0], [0, detector_height], 'k-', lw=2)
plt.plot([detector_width, detector_width], [0, detector_height], 'k-', lw=2)

# 绘制探测器位置
plt.scatter([0] * num_sensors, sensor_positions_left, c='blue', label='Left Sensors')
plt.scatter([detector_width] * num_sensors, sensor_positions_right, c='red', label='Right Sensors')

# 绘制光子生成位置
plt.scatter(photon_x, photon_y, c='green', label='Photon Origin')

# 绘制光子到达探测器的路径
for pos in sensor_positions_left:
    plt.plot([photon_x, 0], [photon_y, pos], 'b--')
for pos in sensor_positions_right:
    plt.plot([photon_x, detector_width], [photon_y, pos], 'r--')

# 设置图形参数
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.legend()
plt.title('Photodetector and Sensors Simulation')
plt.grid(True)
plt.show()

# 输出结果
print("Photon origin position: ({:.2f}, {:.2f})".format(photon_x, photon_y))
print("Left sensors detection times:", time_left)
print("Right sensors detection times:", time_right)
