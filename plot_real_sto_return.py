import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import CubicSpline

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取CSV文件
csv_path = r'/home/qu/f-IRL/train_sac_optimal_with_logs/AntFH-v0/sac_optimal_reward/1/2026_03_15_00_41_50/progress.csv'
df = pd.read_csv(csv_path)

# 三次样条插值，让曲线平滑（仍穿过每个评估点）
x = df['timestep'].values[:300]
y = df['real_sto_return'].values[:300]
x_dense = np.linspace(x.min(), x.max(), 500)
y_smooth = CubicSpline(x, y)(x_dense)

# 创建图表
plt.figure(figsize=(12, 6))
plt.plot(x_dense, y_smooth, linewidth=2, color='blue')
plt.xlabel('Eviroment Steps', fontsize=12)
plt.ylabel('Real Sto Return', fontsize=12)
plt.title('Ant', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# 保存图表
output_path = 'real_sto_return_curve.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'图表已保存到: {output_path}')

# 显示图表
plt.show()

# 打印统计信息
# print(f'\n统计信息:')
# print(f'最小值: {df["Real Sto Return"].min():.2f}')
# print(f'最大值: {df["Real Sto Return"].max():.2f}')
# print(f'平均值: {df["Real Sto Return"].mean():.2f}')
# print(f'最终值: {df["Real Sto Return"].iloc[-1]:.2f}')
