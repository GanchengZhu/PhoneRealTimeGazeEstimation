# -*- coding: utf-8 -*-


import os
import random

import numpy as np
import pandas as pd
from sklearn.svm import SVR


def select_data_by_id(data, int_id, n_point=13):
    folder = data[:, 0]
    idx = np.where(folder == int_id)
    if type(n_point) == int:
        # Ground Truth points
        data = np.squeeze(data[idx, :], axis=0)
        gt = data[:, 2:4]
        # gt_unique_indices = gt.copy()
        # Map gt to a unique point array.
        unique_point, unique_index = np.unique(
            gt.view(gt.dtype.descr * gt.shape[1]),
            return_index=True
        )

        # filter
        full_choice_list = np.arange(0, len(unique_point))
        np.random.shuffle(full_choice_list)
        choice_list = full_choice_list[:n_point]
        unique_point = unique_point[choice_list]

        keep_idx = np.zeros(shape=(data.shape[0],), dtype=bool)

        # Get gt unique index and keep points for train
        for n, element in enumerate(unique_point):
            keep_idx = (keep_idx | ((gt[:, 0] == element[0]) & (gt[:, 1] == element[1])))

        return data[keep_idx, :]
        # keep_for_train = _keep(unique)
        # data[idx, :]
    else:
        return np.squeeze(data[idx, :], axis=0)


# def dis(p1, p2):
#     return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

n_point = 5
random.seed(0)
np.random.seed(0)
data_source = "calibration_all"

smooth_pursuit_data = np.load(os.path.join("../calibration_data", data_source, "smooth_pursuit.npz"))
smooth_pursuit_data = smooth_pursuit_data["arr_0"]

zigzag_data = np.load(os.path.join("../calibration_data", data_source, "zigzag.npz"))
zigzag_data = zigzag_data["arr_0"]

all_result = []
id_map = pd.read_excel("../calibration_data/id_list.xlsx")

train_feature_dataset_list = []
train_x_dataset_list = []
train_y_dataset_list = []
test_feature_dataset_list = []
test_x_dataset_list = []
test_y_dataset_list = []

for sub_item in np.unique(id_map.subj.values):
    smooth_pursuit_id = id_map[(id_map.subj == sub_item) & (id_map.task == "lissajous")].folder.values
    zigzag_id = id_map[(id_map.subj == sub_item) & (id_map.task == "smooth")].folder.values
    smooth_pursuit_id = random.choice(smooth_pursuit_id)
    zigzag_id = random.choice(zigzag_id)

    test_dataset = select_data_by_id(smooth_pursuit_data, smooth_pursuit_id)
    train_dataset = select_data_by_id(zigzag_data, zigzag_id, n_point=n_point)

    train_feature = train_dataset[:, 6:]
    train_label_x, train_label_y = train_dataset[:, 2], train_dataset[:, 3]

    test_feature = test_dataset[:, 6:]
    test_label_x, test_label_y = test_dataset[:, 2], test_dataset[:, 3]

    train_feature_dataset_list.append(train_feature)
    test_feature_dataset_list.append(test_feature)
    train_x_dataset_list.append(train_label_x)
    train_y_dataset_list.append(train_label_y)
    test_x_dataset_list.append(test_label_x)
    test_y_dataset_list.append(test_label_y)


# 定义适应度函数 - SVR交叉验证误差
def svr_opt_func(params):
    all_mean_euclidean = 0
    C, gamma, epsilon = params
    for n in range(len(train_feature_dataset_list)):
        test_feature = test_feature_dataset_list[n]
        train_feature = train_feature_dataset_list[n]
        train_label_x, train_label_y = \
            train_x_dataset_list[n], train_y_dataset_list[n]
        test_label_x, test_label_y = \
            test_x_dataset_list[n], test_y_dataset_list[n]

        regression_x = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
        regression_y = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
        regression_x.fit(train_feature, train_label_x)
        regression_y.fit(train_feature, train_label_y)
        svr_pre_x = regression_x.predict(test_feature)
        svr_pre_y = regression_y.predict(test_feature)
        mean_euclidean = np.mean(np.sqrt((svr_pre_x - test_label_x) ** 2 +
                                         (svr_pre_y - test_label_y) ** 2))
        all_mean_euclidean += mean_euclidean
    return all_mean_euclidean / len(train_feature_dataset_list)

def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {
        "svr_opt_func": ["svr_opt_func", [0.1, 0.001, 0.01], [1000, 10, 0.1], 3],
    }
    return param.get(a, "nothing")

