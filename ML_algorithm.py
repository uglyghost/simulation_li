import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 设置参数
detector_width = 100  # 宽度（毫米）
detector_height = 50  # 高度（毫米）
num_sensors = 16
num_samples = 10000  # 样本数量
c = 300  # 光速（毫米/纳秒）

# 生成探测器位置（等间距分布）
sensor_positions_left = np.linspace(0, detector_height, num_sensors)
sensor_positions_right = np.linspace(0, detector_height, num_sensors)

# 生成样本数据
data = []
labels = []

for _ in range(num_samples):
    photon_x = np.random.uniform(0, detector_width)
    photon_y = np.random.uniform(0, detector_height)

    time_left = np.sqrt(photon_x ** 2 + (sensor_positions_left - photon_y) ** 2) / c
    time_right = np.sqrt((detector_width - photon_x) ** 2 + (sensor_positions_right - photon_y) ** 2) / c

    data.append(np.concatenate((time_left, time_right)))
    labels.append([photon_x, photon_y])

data = np.array(data)
labels = np.array(labels)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 定义模型
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    # 'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    # 'Support Vector Machine': SVR(),
    # 'K-Nearest Neighbors': KNeighborsRegressor(),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
}

# 创建结果文件夹
if not os.path.exists('results'):
    os.makedirs('results')

# 训练和评估模型
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append((name, mse, r2))
    print(f'{name} - Mean Squared Error: {mse:.4f}, R^2 Score: {r2:.4f}')

    # 保存预测结果到单独的txt文件
    predictions_df = pd.DataFrame({'True_X': y_test[:, 0], 'True_Y': y_test[:, 1],
                                   'Pred_X': y_pred[:, 0], 'Pred_Y': y_pred[:, 1]})
    predictions_df.to_csv(f'results/{name.replace(" ", "_")}_predictions.txt', index=False, sep='\t')

# 汇总结果
results_df = pd.DataFrame(results, columns=['Model', 'Mean Squared Error', 'R^2 Score'])
print(results_df)

# 绘制性能结果的直方图
plt.figure(figsize=(12, 6))
x_labels = results_df['Model']
mse_values = results_df['Mean Squared Error']
r2_values = results_df['R^2 Score']

x = np.arange(len(x_labels))

fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:red'
ax1.set_xlabel('Model')
ax1.set_ylabel('Mean Squared Error', color=color)
ax1.bar(x - 0.2, mse_values, width=0.4, color=color, align='center')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, rotation=45, ha='right')

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('R^2 Score', color=color)
ax2.bar(x + 0.2, r2_values, width=0.4, color=color, align='center')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Model Performance Comparison')
plt.show()

# 保存汇总结果到文本文件
results_df.to_csv('results/summary_results.txt', index=False, sep='\t')
