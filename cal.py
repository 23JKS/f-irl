import pandas as pd

file_path = '/home/qu/f-IRL/logs/AntFH-v0/exp-16/fkl/1/2026_01_12_00_09_22/progress.csv'

# ✅ 使用 sep=','，并且跳过以 @ 开头的行
df = pd.read_csv(file_path, comment='@', header=None, sep=',')

# 提取第 182~201 行（物理行号） → iloc[181:201]
subset = df.iloc[481:501, 0]  # 第一列

# 转为数值（其实已经是数值，但保险起见）
subset = pd.to_numeric(subset, errors='coerce')

mean_value = subset.mean()
print("均值:", mean_value)