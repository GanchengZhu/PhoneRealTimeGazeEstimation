#!/usr/bin/bash
# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import json

import numpy as np
import pandas
import pandas as pd
from matplotlib import pyplot as plt


def getME(labels, pres):
    euclid_distance = np.sqrt((labels[:, 0] - pres[:, 0]) ** 2
                              + (labels[:, 1] - pres[:, 1]) ** 2)
    return np.average(euclid_distance)


def rms_s2s(df):
    pre_rms_s2s = np.sqrt(np.mean(np.diff(df.x_pre.values) ** 2 + np.diff(df.y_pre.values) ** 2))
    filtered_rms_s2s = np.sqrt(np.mean(np.diff(df.x_filtered.values) ** 2 + np.diff(df.y_filtered.values) ** 2))
    # np.diff(df.x_filtered.values)
    # np.diff(df.y_filtered.values)
    return pre_rms_s2s, filtered_rms_s2s


def compute_sample_to_sample_data_loss(df, task_type):
    # RMS-S2S, the Root Mean Square sample-to-sample deviation
    rms_s2s_list_no_filtered = []
    rms_s2s_list_filtered = []

    diff = (np.diff(df.x_gt.values) == 0.0) & (np.diff(df.y_gt.values) == 0.0)
    last_i = 0
    # print(task_type)
    # print(np.argwhere(diff == False).flatten())
    for i in np.argwhere(diff == False).flatten():
        df_cut = df.iloc[last_i: i + 1]
        no_filtered_rms_s2s, filtered_rms_s2s = rms_s2s(df_cut)

        rms_s2s_list_no_filtered.append(no_filtered_rms_s2s)
        rms_s2s_list_filtered.append(filtered_rms_s2s)

        last_i = i + 1
    df_cut = df.iloc[last_i:]
    no_filtered_rms_s2s, filtered_rms_s2s = rms_s2s(df_cut)

    rms_s2s_list_no_filtered.append(no_filtered_rms_s2s)
    rms_s2s_list_filtered.append(filtered_rms_s2s)
    # print(np.array(rms_s2s_list_no_filtered).shape)
    return rms_s2s_list_no_filtered, rms_s2s_list_filtered
    # print()


def data_plotting(json_file, task_type, filter_type):
    with open(json_file) as f:
        data = json.load(f)
        result_no_filtered = []
        result_filtered = []
        for key in data.keys():
            # for every participant
            participant_data = data[key]
            gt_points = np.array(participant_data['gt'])
            pre_points = np.array(participant_data['pre'])
            filtered_points = np.array(participant_data["filtered"])

            start = gt_points.shape[0] - filtered_points.shape[0]
            # print(filtered_points)
            # filtered_points = np.array(json.loads(filtered_points))

            df = pandas.DataFrame(data={
                "gt_ms": np.round(np.arange(len(gt_points[start + 1:, 0])) * 100 / 3),
                "x_gt": gt_points[start + 1:, 0],
                "y_gt": gt_points[start + 1:, 1],
                "x_pre": pre_points[start + 1:, 0],
                "y_pre": pre_points[start + 1:, 1],
                "x_filtered": filtered_points[1:, 0],
                "y_filtered": filtered_points[1:, 1],
            })
            if task_type == "zigzag":
                rms_s2s_list_no_filtered, rms_s2s_list_filtered = compute_sample_to_sample_data_loss(df, key)
                result_no_filtered.append(rms_s2s_list_no_filtered)
                result_filtered.append(rms_s2s_list_filtered)

            df.plot(x='gt_ms', y=['x_pre', 'y_pre', "x_filtered", "y_filtered"])
            df.to_csv(f"filtered_data/{task_type}_{filter_type}_{key}.csv")
            print(key)
            # df.plot(x='gt_ms', y=['x_gt', 'y_gt'])
            plt.savefig(f'filtered_data/{task_type}_{filter_type}_{key}.png')
            plt.close()
        if result_filtered:
            print(filter_type)
            print(np.mean(result_no_filtered, axis=0))
            print(np.mean(result_filtered, axis=0))

            return np.mean(result_no_filtered, axis=0), np.mean(result_filtered, axis=0)


plot_dict = {}

for task_type in ['zigzag']:
    for filter_type in ['HEURISTIC_FILTER', 'ONE_EURO_FILTER']:
        # for filter_type in ['ONE_EURO_FILTER']:
        no_filter_result, filter_result = \
            data_plotting(f'filtered/{task_type}_{filter_type}.json',
                          task_type=task_type, filter_type=filter_type)
        plot_dict["raw"] = no_filter_result
        if filter_type == "HEURISTIC_FILTER":
            plot_dict["heuristic filter"] = filter_result
        else:
            plot_dict["one Euro filter"] = filter_result

# 设置字体
font = {'family': 'Arial',
        'size': 10,
        'weight': 'bold'}
plt.rc('font', **font)
colors = [
      "#2878b5", "#54B345","#c82423",
]
# 创建图形
fig, ax = plt.subplots(figsize=(10, 5))

# 设置x轴位置
x = np.arange(len(plot_dict["raw"]))

# 绘制柱状图
bar_width = 0.2  # 柱子的宽度
bar_positions_raw = x - bar_width - 0.05
bar_positions_heuristic = x
bar_positions_one_euro = x + bar_width + 0.05

ax.bar(bar_positions_raw, plot_dict["raw"], width=bar_width, label="Raw", color=colors[0])
ax.bar(bar_positions_heuristic, plot_dict["heuristic filter"], width=bar_width, label="Heuristic Filter",  color=colors[1])
ax.bar(bar_positions_one_euro, plot_dict["one Euro filter"], width=bar_width, label="One Euro Filter", color=colors[2])

# 添加标签和标题
ax.set_xlabel('Data Points',fontsize=10, weight='bold')
ax.set_ylabel('RMS',fontsize=10, weight='bold')
# ax.set_title('Data Comparison')
ax.set_xticks(x)
ax.set_xticklabels([f'{i + 1}' for i in range(len(x))])

# 添加图例
ax.legend(frameon=True, fontsize=10)

# 显示图像
plt.tight_layout()
plt.show()

pd.DataFrame(plot_dict).to_csv("filters.csv")

for task_type in ['smooth_pursuit']:
    for filter_type in ['HEURISTIC_FILTER', 'ONE_EURO_FILTER']:
        # for filter_type in ['ONE_EURO_FILTER']:
        data_plotting(f'filtered/{task_type}_{filter_type}.json', task_type=task_type, filter_type=filter_type)
        print(task_type)