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

import data_reader
import tqdm

import itracker
import AFFNet
import MGazeNet


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def do_evaluation(net, dataset, epoch, itracker_format=1):
    net.eval()

    count = 0
    total = 0

    with torch.no_grad():
        for j, data in tqdm.tqdm(enumerate(dataset)):
            data["faceImg"] = data["faceImg"].to(device)
            data["leftEyeImg"] = data["leftEyeImg"].to(device)
            data['rightEyeImg'] = data['rightEyeImg'].to(device)

            if not itracker_format:
                data['rects'] = data['rects'].to(device)
            else:
                data['grid'] = data['grid'].to(device)

            labels = data["label"].numpy()
            sub_id = np.array(data["folder"]).reshape((-1, 1))
            frame_id = np.array(data["frame"]).reshape((-1, 1))

            if not itracker_format:
                gazes = net(data["leftEyeImg"], data["rightEyeImg"], data['faceImg'], data['rects'])
            else:
                gazes = net(data)
            # print(gazes.shape)
            # names = data["frame"]
            gazes = gazes.cpu().detach().numpy()
            condi_data = np.concatenate([sub_id, frame_id, labels, gazes], axis=1)
            condi_data = condi_data.astype(np.float32)
            # print(condi_data.shape)
            euclid_distance = np.sqrt((condi_data[:, 4] - condi_data[:, 2]) ** 2
                                      + (condi_data[:, 5] - condi_data[:, 3]) ** 2)

            acc = euclid_distance.sum()
            total += acc
            count += 1

        count = count * 64
        loger = f"Total Num: {count}, avg: {total / count} \n"
        print(f"epoch: {epoch}, logger: {loger}")
        return total / count, count


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help="the file of configuration")
    args = parser.parse_args()

    yaml_config = yaml.load(open(args.config_file), Loader=yaml.Loader)
    config = yaml_config["test"]
    data_path = config["data"]["path"]
    model_name = config["load"]["model_name"]
    result_path = os.path.join("results", config["load"]["load_path"])
    create_dir(result_path)

    # feature_path = config["data"]["feature_path"]

    save_path = os.path.join(config["load"]["load_path"], "checkpoint")

    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Test Set: tests")
    # gazecap_data_path = ["/media/robotics/ca9601d4-7476-4123-b138-cac7340caaa8/gaze_estimation_project/gazecapture_raw"]
    gazecap_data_path = [
        "/home/robotics2/data_ssd03/bk_all_datasets/train_datasets/gazeCapture-raw"
    ]
    # our_lissajous_data_path = ["/media/robotics/ca9601d4-7476-4123-b138-cac7340caaa8/gazeCapture-202211/lissajous"]
    # our_smooth_data_path = ["/media/robotics/ca9601d4-7476-4123-b138-cac7340caaa8/gazeCapture-202211/smooth"]
    # our_random_data_path = ["/media/robotics/ca9601d4-7476-4123-b138-cac7340caaa8/gazeCapture-202211/random"]

    print("Read data")
    # dataset = data_reader.gaze_loader(path, "test", 512, num_workers=20, shuffle=False)
    if model_name == "itracker":
        if "zjugaze" in model_name:
            gazecap_test_dataset = data_reader.gaze_loader(gazecap_data_path, "test", 64, num_workers=14, itracker_format=1,
                                                       device="phone", ori_filter=[1])
        else:
            gazecap_test_dataset = data_reader.gaze_loader(gazecap_data_path, "test", 64, num_workers=14, itracker_format=1,
                                                       device="phone")
        # our_lissajous_test_dataset = data_reader.gaze_loader(our_lissajous_data_path, "test", 64, num_workers=8,
        #                                                  itracker_format=1)
        # our_smooth_test_dataset = data_reader.gaze_loader(our_smooth_data_path, "test", 64, num_workers=8,
        #                                               itracker_format=1)
        # our_random_test_dataset = data_reader.gaze_loader(our_random_data_path, "test", 64, num_workers=8,
        #                                               itracker_format=1)
    else:
        if "zjugaze" in model_name:
            gazecap_test_dataset = data_reader.gaze_loader(gazecap_data_path, "test", 64, num_workers=14, itracker_format=1,
                                                       device="phone", ori_filter=[1])
        else:
            gazecap_test_dataset = data_reader.gaze_loader(gazecap_data_path, "test", 64, num_workers=14, itracker_format=1,
                                                       device="phone")
        # our_lissajous_test_dataset = data_reader.gaze_loader(our_lissajous_data_path, "test", 64, num_workers=8,
        #                                                  itracker_format=0)
        # our_smooth_test_dataset = data_reader.gaze_loader(our_smooth_data_path, "test", 64, num_workers=8,
        #                                               itracker_format=0)
        # our_random_test_dataset = data_reader.gaze_loader(our_random_data_path, "test", 64, num_workers=8,
        #                                               itracker_format=0)
    begin = config["load"]["begin_step"]
    end = config["load"]["end_step"]
    step = config["load"]["steps"]

    gazecap_all_result = []
    lissajous_all_result = []
    smooth_all_result = []
    random_all_result = []
    itracker_format = 1
    for save_iter in range(begin, end + step, step):
        print("Model building")
        if model_name == "AFFNet":
            net = AFFNet.model()
            itracker_format = 0
        elif model_name == "itracker":
            net = itracker.ITrackerModel()
            itracker_format = 1
        else:
            net = MGazeNet.MGazeNet()
            itracker_format = 0

        net.to(device)
        state_dict = torch.load(os.path.join(save_path, f"Iter_{save_iter}_{model_name}.pt"), map_location='cuda:0')
        try:
            net.load_state_dict(state_dict)
            net = nn.DataParallel(net)
        except:
            net = nn.DataParallel(net)
            net.load_state_dict(state_dict)

        net.eval()

        print(f"Test {save_iter}")
        gazecap_res, gazecap_count = do_evaluation(net, gazecap_test_dataset, epoch=save_iter,
                                                   itracker_format=itracker_format)
        gazecap_all_result.append(pd.DataFrame(
            {"Iter": save_iter, "ME": gazecap_res, "count": gazecap_count}, index=[0]))

        # lissajous_res, lissajous_count = do_evaluation(net, our_lissajous_test_dataset, epoch=save_iter, itracker_format=itracker_format)
        # lissajous_all_result.append(pd.DataFrame(
        #     {"Iter": save_iter, "ME": lissajous_res, "count": lissajous_count}, index=[0]))

        # smooth_res, smooth_count = do_evaluation(net, our_smooth_test_dataset, epoch=save_iter, itracker_format=itracker_format)
        # smooth_all_result.append(pd.DataFrame(
        #     {"Iter": save_iter, "ME": smooth_res, "count": smooth_count}, index=[0]))

        # random_res, random_count = do_evaluation(net, our_random_test_dataset, epoch=save_iter, itracker_format=itracker_format)
        # random_all_result.append(pd.DataFrame(
        #     {"Iter": save_iter, "ME": random_res, "count":random_count}, index=[0]))

    pd.concat(gazecap_all_result).to_csv(os.path.join(result_path, "evaluation_gazecap.csv"))
    # pd.concat(lissajous_all_result).to_csv(os.path.join(result_path, "evaluation_lissajous.csv"))
    # pd.concat(smooth_all_result).to_csv(os.path.join(result_path, "evaluation_smooth.csv"))
    # pd.concat(random_all_result).to_csv(os.path.join(result_path, "evaluation_random.csv"))

