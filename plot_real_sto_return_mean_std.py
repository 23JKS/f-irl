import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import CubicSpline

# 设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ================= 配置区域 =================
CSV_PATHS = [
    '/home/qu/f-IRL/train_sac_optimal_with_logs/HopperFH-v0/sac_optimal_reward/1/2026_03_14_09_14_19/progress.csv',
    '/home/qu/f-IRL/train_sac_optimal_with_logs/HopperFH-v0/sac_optimal_reward/1/2026_03_14_12_53_48/progress.csv',
    '/home/qu/f-IRL/train_sac_optimal_with_logs/HopperFH-v0/sac_optimal_reward/1/2026_03_14_20_00_11/progress.csv',
]

REWARD_COL = 'real_sto_return'  # 奖励列名
TITLE = 'Hopper'
OUTPUT_PATH = 'real_sto_return_mean_std_no_smooth.png'
N_POINTS = 300  # 截取前多少行数据进行对齐
# ===========================================

def load_and_align(csv_paths, reward_col, n_points):
    """
    加载多份 CSV，截取前 n_points 行，
    将每份数据插值到统一的时间轴上，然后计算均值和标准差。
    注意：这里保留了必要的‘时间轴对齐’插值，否则不同步长的数据无法计算均值。
    """
    dfs = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            dfs.append(df)
        except FileNotFoundError:
            print(f"警告：文件未找到 {p}")
    
    if len(dfs) == 0:
        raise ValueError("没有成功加载任何文件")

    # 1. 提取各文件的前 n_points 个数据点
    x_lists = []
    y_lists = []
    
    for df in dfs:
        if len(df) < n_points:
            print(f"警告：文件行数不足 {n_points}，使用全部行数 {len(df)}")
            current_n = len(df)
        else:
            current_n = n_points
            
        x_vals = df['timestep'].values[:current_n]
        y_vals = df[reward_col].values[:current_n]
        
        # 过滤掉可能的 NaN
        mask = ~np.isnan(y_vals) & ~np.isnan(x_vals)
        x_lists.append(x_vals[mask])
        y_lists.append(y_vals[mask])

    # 2. 构建统一的密集时间轴 (Common X) - 必须步骤，用于对齐不同频率的数据
    if not x_lists:
        raise ValueError("没有有效数据用于对齐")
        
    min_max = max([x.min() for x in x_lists])
    max_min = min([x.max() for x in x_lists])
    
    if min_max >= max_min:
        raise ValueError("时间步范围没有重叠，无法对齐")
    
    # 在重叠范围内生成 500 个点作为基准对齐轴
    x_common = np.linspace(min_max, max_min, 500)

    # 3. 对每个文件进行插值，映射到 x_common (这是为了对齐，不是为了平滑曲线外观)
    ys_aligned = []
    for x_orig, y_orig in zip(x_lists, y_lists):
        if len(x_orig) < 2:
            continue
        cs = CubicSpline(x_orig, y_orig)
        y_interpolated = cs(x_common)
        ys_aligned.append(y_interpolated)
    
    ys_aligned = np.array(ys_aligned) # shape: (n_runs, 500)
    
    # 4. 计算均值和标准差
    mean_vals = np.mean(ys_aligned, axis=0)
    std_vals = np.std(ys_aligned, axis=0, ddof=1)
    # 计算标准误差
    # std_vals = np.std(ys_aligned, axis=0, ddof=1)/np.sqrt(len(ys_aligned))
    
    return x_common, mean_vals, std_vals

def main():
    try:
        # 加载并对齐数据
        x, mean, std = load_and_align(CSV_PATHS, REWARD_COL, N_POINTS)
    except Exception as e:
        print(f"处理数据时出错: {e}")
        return

    # 【修改点】不再进行二次平滑插值
    # 直接使用对齐后计算出的 x, mean, std 进行绘图
    x_dense = x       # 直接使用对齐后的轴
    mean_plot = mean  # 直接使用计算出的均值
    std_plot = std    # 直接使用计算出的标准差

    # 计算上下界
    lower = mean_plot - std_plot
    upper = mean_plot + std_plot

    # 6. 绘图
    plt.figure(figsize=(12, 6))
    
    # 绘制阴影区域 (Mean ± Std)
    plt.fill_between(x_dense, lower, upper, color='red', alpha=0.25)
    
    # 绘制均值曲线
    plt.plot(x_dense, mean_plot, linewidth=2, color='red', label='Mean')
    
    plt.xlabel('Environment Steps', fontsize=12)
    plt.ylabel(f'Real Sto Return (mean ± std)', fontsize=12)
    plt.title(TITLE, fontsize=14, fontweight='bold')
    
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # 保存
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f'图表已保存到: {OUTPUT_PATH}')
    
    plt.show()

if __name__ == '__main__':
    main()