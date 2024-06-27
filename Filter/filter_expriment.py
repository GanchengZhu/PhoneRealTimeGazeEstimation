#!/usr/bin/bash
# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import glob
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas

from HeuristicFilter import HeuristicFilterManager
from OneEuroFilterManager import OneEuroFilterManager


def getME(labels, pres):
    euclid_distance = np.sqrt((labels[:, 0] - pres[:, 0]) ** 2
                              + (labels[:, 1] - pres[:, 1]) ** 2)
    return np.average(euclid_distance)

heuristic_filter = HeuristicFilterManager(look_ahead=2)
one_euro_filter = OneEuroFilterManager(count=2, freq=30, minCutOff=0.5, beta=0.001, dCutOff=1.0)

# Smooth pursuit feature data
subj_sp = glob.glob("feature/*_random_sp/Iter_6_feature.npz")
# Zigzag feature data
subj_zigzag = glob.glob("feature/*_sp_zigzag/Iter_6_feature.npz")

point_dict = {}

for subj in subj_zigzag:
    print(subj)
    subject_feature = np.load(subj, allow_pickle=True)
    subject_feature = subject_feature['arr_0']
    if "cyz_sp_zigzag" not in subj:
        subject_feature = subject_feature[np.argsort(subject_feature[:, 1])]
    # break
    # heuristic_filtered_points = []
    # one_euro_filtered_points = []
    gt_points = subject_feature[:, 2:4]
    pre_points = subject_feature[:, 4:6]

    point_dict[subj.split("\\")[1]] = {}
    point_dict[subj.split("\\")[1]]["gt"] = gt_points.tolist()
    point_dict[subj.split("\\")[1]]["pre"] = pre_points.tolist()

json.dump(point_dict, open('zigzag.json', 'w'))

point_dict = {}
for subj in subj_sp:
    print(subj)
    subject_feature = np.load(subj, allow_pickle=True)
    subject_feature = subject_feature['arr_0']
    subject_feature = subject_feature[np.argsort(subject_feature[:, 1])]
    # break
    # heuristic_filtered_points = []
    # one_euro_filtered_points = []
    gt_points = subject_feature[:, 2:4]
    pre_points = subject_feature[:, 4:6]

    point_dict[subj.split("\\")[1]] = {}
    point_dict[subj.split("\\")[1]]["gt"] = gt_points.tolist()
    point_dict[subj.split("\\")[1]]["pre"] = pre_points.tolist()

json.dump(point_dict, open('smooth_pursuit.json', 'w'))
    # for n, point in enumerate(pre_points):
    #     heuristic_filter.filter_values(timestamp=None, x=point[0], y=point[1])
    #     one_euro_filter.filterValues(timestamp=int(n * 100 / 3), x=point[0], y=point[1])
    #     value = heuristic_filter.get_filtered_values()
    #     if value:
    #         heuristic_filtered_points.append(heuristic_filter.get_filtered_values())
    #     else:
    #         heuristic_filtered_points.append([np.nan, np.nan])
    #     one_euro_filtered_points.append(one_euro_filter.getFilteredValues())
    #
    # heuristic_filtered_points = np.array(heuristic_filtered_points)
    # one_euro_filtered_points = np.array(one_euro_filtered_points)
    # print(getME(gt_points[6:], heuristic_filtered_points[6:]))
    # print(getME(gt_points, pre_points))
    #
    # df = pandas.DataFrame(data={
    #     "gt_ms": np.round(subject_feature[6:, 1] * 100 / 3),
    #     "x_gt": gt_points[6:, 0],
    #     "y_gt": gt_points[6:, 1],
    #     "x_svr_pre": pre_points[6:, 0],
    #     "y_svr_pre": pre_points[6:, 1],
    #     "x_filtered": heuristic_filtered_points[6:, 0],
    #     "y_filtered": heuristic_filtered_points[6:, 1],
    # })
    # # df.plot(x='gt_ms', y=['x_svr_pre', 'x_filtered', 'x_gt'])
    # df.plot(x='gt_ms', y=['x_svr_pre', 'y_svr_pre', "x_filtered", "y_filtered"])
    # plt.show()

# for subj in subj_zigzag:
#     feature = np.load(subj, allow_pickle=True)
#     feature = feature['arr_0']




