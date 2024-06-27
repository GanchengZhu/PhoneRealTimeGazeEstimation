import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_excel("filtered_data/draw.xlsx")

font = {'family': 'Arial',
        'size': 16,}
plt.rc('font', **font)

# 设置时间轴
time = np.arange(len(data.x_filtered_one)) * 100 / 3

# 创建子图
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# 设置子图标题
titles = ['(A) Ground-truth', '(B) Raw Output', '(C) Heuristic filter', '(D) One Euro filter']

# 设置线的颜色
colors = ["#2878b5", "#c82423"]

# 用循环填充子图数据和标题
for i, ax in enumerate(axs.flat):
    ax.set_xlabel('Time (ms)', fontsize=16)
    ax.set_ylabel('Centimeters', fontsize=16)

    if i == 0:  # Groundtruth
        ax.plot(time, data.x_gt, label='Horizontal signal', color=colors[0])
        ax.plot(time, data.y_gt, label='Vertical signal', color=colors[1])
        ax.legend(frameon=True, fontsize=12)
    elif i == 1:  # Raw Output (示例数据)
        ax.plot(time, data.x_pre, label='Horizontal signal', color=colors[0])
        ax.plot(time, data.y_pre, label='Vertical signal', color=colors[1])
        ax.legend(frameon=True, fontsize=12)
    elif i == 2:  # Heuristic filter (示例数据)
        ax.plot(time, data.x_filtered_heu, label='Horizontal signal', color=colors[0])
        ax.plot(time, data.y_filtered_heu, label='Vertical signal', color=colors[1])
        ax.legend(frameon=True, fontsize=12)
    elif i == 3:  # One Euro filter (示例数据)
        ax.plot(time, data.x_filtered_one, label='Horizontal signal', color=colors[0])
        ax.plot(time, data.y_filtered_one, label='Vertical signal', color=colors[1])
        ax.legend(frameon=True, fontsize=12)

    # 设置子图标题
    ax.set_title(titles[i], fontsize=16, y=-.3)

# 调整子图布局
plt.tight_layout()
# 调整子图间距
plt.subplots_adjust(hspace=0.4)
# 显示图形
plt.show()
