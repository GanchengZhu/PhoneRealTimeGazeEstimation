#!/usr/bin/bash
# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

# !/usr/bin/bash
# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.mobile_optimizer import optimize_for_mobile

import data_reader
import MGazeNet

import tqdm


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def do_evaluation(net, dataset, task_name, feature_path):
    print(task_name)
    net.eval()
    feature_npy = os.path.join(
        feature_path, task_name
    )


    with torch.no_grad():
        buffer_list = []
        for j, data in tqdm.tqdm(enumerate(dataset)):
            data["faceImg"] = data["faceImg"].to(device)
            data["leftEyeImg"] = data["leftEyeImg"].to(device)
            data['rightEyeImg'] = data['rightEyeImg'].to(device)
            data['rects'] = data['rects'].to(device)
            labels = data["label"].numpy()
            sub_id = np.array(data["folder"]).reshape((-1, 1))
            frame_id = np.array(data["frame"]).reshape((-1, 1))

            gazes = net(data["leftEyeImg"], data["rightEyeImg"], data['faceImg'], data['rects'])

            gazes = gazes.cpu().detach().numpy()
            condi_data = np.concatenate([sub_id, frame_id, labels, gazes], axis=1)
            condi_data = condi_data.astype(np.float32)
            buffer_list.append(condi_data)

        buffer_list = np.concatenate(buffer_list, axis=0)
        buffer_list.astype(np.float32)
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)

        np.savez_compressed(feature_npy, buffer_list)


if __name__ == "__main__":
    # Usage
    # python predict_features.py --model_path [model_path] --npy_save_dir [npy_save_dir]
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="model path")
    parser.add_argument('--npy_save_dir', type=str, help="npy save folder")
    args = parser.parse_args()

    model_path = args.model_path
    npy_save_dir = args.npy_save_dir
    create_dir(npy_save_dir)

    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Test Set: tests")

    print("Read data")
    # dataset = data_reader.txtload(path, "test", 512, num_workers=20, shuffle=False)
    ban_list = []

    our_lissajous_data_path = ["/media/robotics/ca9601d4-7476-4123-b138-cac7340caaa8/gazeCapture-202211/lissajous"]
    our_smooth_data_path = ["/media/robotics/ca9601d4-7476-4123-b138-cac7340caaa8/gazeCapture-202211/smooth"]
    our_random_data_path = ["/media/robotics/ca9601d4-7476-4123-b138-cac7340caaa8/gazeCapture-202211/random"]

    our_lissajous_test_dataset = data_reader.txtload(our_lissajous_data_path, "test", 128, num_workers=14,
                                                     iTracker_format=0)
    our_smooth_test_dataset = data_reader.txtload(our_smooth_data_path, "test", 128, num_workers=14,
                                                  iTracker_format=0)
    our_random_test_dataset = data_reader.txtload(our_random_data_path, "test", 128, num_workers=14,
                                                  iTracker_format=0)

    net = MGazeNet.MGazeNet()
    net.to(device)
    state_dict = torch.load(model_path)
    try:
      net.load_state_dict(state_dict)
      net = nn.DataParallel(net)
    except:
      net = nn.DataParallel(net)
      net.load_state_dict(state_dict)
    net.eval()

    do_evaluation(net, our_lissajous_test_dataset, task_name="smooth_pursuit", feature_path=npy_save_dir)
    do_evaluation(net, our_smooth_test_dataset, task_name="zigzag", feature_path=npy_save_dir)
    do_evaluation(net, our_random_test_dataset, task_name="random", feature_path=npy_save_dir)

    # Using MGazeNet models trained by all dataset and our dataset respectively to do SVM calibration
    # from 1 to all point, draw a curve
    # Using random point as svm calibration train dataset, other as test dataset
    # Using Zigzag point as svm calibration train dataset, other as test dataset