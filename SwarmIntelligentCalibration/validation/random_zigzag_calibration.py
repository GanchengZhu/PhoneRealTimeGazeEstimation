import argparse
import os.path
import random

import numpy as np
import pandas as pd
from sklearn import linear_model, neighbors
from sklearn.svm import SVR
from xgboost import XGBRegressor


def dis(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def check_point(data):
    gt = data[:, 2:4].copy()
    gt_unique_indices = gt.copy()
    # Map gt to a unique point array.
    unique, unique_index = np.unique(
        gt.view(gt.dtype.descr * gt.shape[1]),
        return_index=True
    )
    print(len(unique))


def select_data_by_id(data, int_id, n_point="All"):
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


def create_svm(x_c=10.0, x_p=0.1, x_gamma=5e-3, y_c=10.0, y_p=0.1, y_gamma=5e-3):
    svm_x = SVR(kernel='rbf', C=x_c, gamma=x_gamma, epsilon=x_p)
    svm_y = SVR(kernel='rbf', C=y_c, gamma=y_gamma, epsilon=y_p)
    return svm_x, svm_y


def create_xgb():
    xgb_x = XGBRegressor()
    xgb_y = XGBRegressor()
    return xgb_x, xgb_y


def create_ridge(alpha_x=.1, alpha_y=.1):
    ridge_x = linear_model.Ridge(alpha=alpha_x)
    ridge_y = linear_model.Ridge(alpha=alpha_y)
    return ridge_x, ridge_y


def create_k_neighbors():
    knn_x = neighbors.KNeighborsRegressor()
    knn_y = neighbors.KNeighborsRegressor()
    return knn_x, knn_y


def calibration(train_dataset, test_dataset, regression_type="SVR"):
    train_feature = train_dataset[:, 6:]
    train_label_x, train_label_y = train_dataset[:, 2], train_dataset[:, 3]

    test_feature = test_dataset[:, 6:]
    test_label_x, test_label_y = test_dataset[:, 2], test_dataset[:, 3]

    if regression_type == "RIDGE":
        regression_x, regression_y = create_ridge()

    elif regression_type == "XGB":
        regression_x, regression_y = create_xgb()

    elif regression_type == "KNN":
        regression_x, regression_y = create_k_neighbors()

    elif regression_type == "SVR":
        regression_x, regression_y = create_svm()

    elif regression_type == "JAYA-SVR":
        regression_x, regression_y = create_svm(1.1019672e+01, 1.0000000e-01, 1.0000000e-03, 1.1019672e+01,
                                                1.0000000e-01, 1.0000000e-03, )
    elif regression_type == "PSO-SVR":
        regression_x, regression_y = create_svm(4.79385068e+01, 1.00000000e-01, 1.00000000e-03, 4.79385068e+01,
                                                1.00000000e-01, 1.00000000e-03, )

    elif regression_type == "MVO-SVR":
        regression_x, regression_y = create_svm(3.47527619e+00, 6.80388839e-02, 1.00000000e-03, 3.47527619e+00,
                                                6.80388839e-02, 1.00000000e-03, )

    else:
        raise Exception("Not found regression:%s" % regression_type)

    regression_x.fit(train_feature, train_label_x)
    regression_y.fit(train_feature, train_label_y)

    svr_pre_x = regression_x.predict(test_feature)
    svr_pre_y = regression_y.predict(test_feature)

    mean_euclidean = np.mean(np.sqrt((svr_pre_x - test_label_x) ** 2 +
                                     (svr_pre_y - test_label_y) ** 2))
    return mean_euclidean


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    parser = argparse.ArgumentParser(description='Run SVR Calibration Experiment')
    parser.add_argument('--data_source', type=str, help='data source')
    parser.add_argument('--random_as_train', default=False, action='store_true', help='data source')
    parser.add_argument('--regression_type', default="SVR", type=str, help='data source')
    # parser.add_argument('--regression_type', default="SVR", type=str, help='data source')
    # parser.add_argument('--mnn_path', type=str, help='mnn path')
    args = parser.parse_args()
    data_source = args.data_source
    regression_type = args.regression_type

    random_data = np.load(os.path.join("calibration_data", data_source, "random.npz"))
    random_data = random_data["arr_0"]

    smooth_pursuit_data = np.load(os.path.join("calibration_data", data_source, "smooth_pursuit.npz"))
    smooth_pursuit_data = smooth_pursuit_data["arr_0"]

    zigzag_data = np.load(os.path.join("calibration_data", data_source, "zigzag.npz"))
    zigzag_data = zigzag_data["arr_0"]

    all_result = []
    id_map = pd.read_excel("../calibration_data/id_list.xlsx")
    for n_point in range(1, 25):
        result = []

        for sub_item in np.unique(id_map.subj.values):
            smooth_pursuit_id = id_map[(id_map.subj == sub_item) & (id_map.task == "lissajous")].folder.values
            random_id = id_map[(id_map.subj == sub_item) & (id_map.task == "random")].folder.values
            zigzag_id = id_map[(id_map.subj == sub_item) & (id_map.task == "smooth")].folder.values

            smooth_pursuit_id = random.choice(smooth_pursuit_id)
            random_id = random.choice(random_id)
            zigzag_id = random.choice(zigzag_id)

            test_dataset = select_data_by_id(smooth_pursuit_data, smooth_pursuit_id)
            if args.random_as_train:
                train_dataset = select_data_by_id(random_data, random_id, n_point=n_point)
            else:
                train_dataset = select_data_by_id(zigzag_data, zigzag_id, n_point=n_point)
            result.append(calibration(train_dataset, test_dataset, regression_type=regression_type))

        all_result.append(result)

    if args.random_as_train:
        data_type = "results/random"
    else:
        data_type = "results/zigzag"

    if not os.path.exists(data_type):
        os.makedirs(data_type)

    np.save(os.path.join(data_type, data_source) + "_" + regression_type, all_result)
