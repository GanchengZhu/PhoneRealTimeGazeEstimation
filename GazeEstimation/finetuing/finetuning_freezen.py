#!/usr/bin/bash
# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import argparse
import json
import os
# for personalized calibration by fine-tuning tech
import os.path
import random
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import MGazeNet


# randomly move the bounding box around
def aug_line(line, width, height):
    bbox = np.array(line[2:5])
    bias = round(30 * random.uniform(-1, 1))
    bias = max(np.max(-bbox[0, [0, 2]]), bias)
    bias = max(np.max(-2 * bbox[1:, [0, 2]] + 0.5), bias)

    line[2][0] += int(round(bias))
    line[2][1] += int(round(bias))
    line[2][2] += int(round(bias))
    line[2][3] += int(round(bias))

    line[3][0] += int(round(0.5 * bias))
    line[3][1] += int(round(0.5 * bias))
    line[3][2] += int(round(0.5 * bias))
    line[3][3] += int(round(0.5 * bias))

    line[4][0] += int(round(0.5 * bias))
    line[4][1] += int(round(0.5 * bias))
    line[4][2] += int(round(0.5 * bias))
    line[4][3] += int(round(0.5 * bias))

    line[5][2] = line[2][2]
    line[5][3] = line[2][0]

    line[5][6] = line[3][2]
    line[5][7] = line[3][0]

    line[5][10] = line[4][2]
    line[5][11] = line[4][0]
    return line


class loader(Dataset):

    def __init__(self, data_path_list, subject_list, data_type, ban_list=None, device="", iTracker_format=False,
                 ori_filter=[]):
        if ban_list is None:
            ban_list = []
        self.lines = []
        self.labels = {}
        self.data_path_list = data_path_list
        self.data_type = data_type
        self.iTracker_format = iTracker_format

        for n, data_path in enumerate(data_path_list):
            subjects = subject_list[n]

            subjects = ["%05d" % i for i in subjects]
            # subjects.sort()
            # print(subjects)
            for subject in subjects:
                subject_path = os.path.join(data_path, subject)
                # print(subject_path)
                if ((
                        not os.path.isdir(subject_path)) or subject == '01185'
                        or subject == '02585'
                        or subject == '01730' or subject == '02065'):
                    continue

                if ban_list:
                    if subject in ban_list:
                        continue

                info_json = json.load(open(os.path.join(subject_path, "info.json"), "r"))
                current_data_type = info_json["Dataset"]

                device_name = info_json["DeviceName"]
                if device and device not in device_name.lower():
                    # print(device_name)
                    continue

                self.labels[subject] = json.load(open(os.path.join(subject_path, "dotInfo.json"), "r"))

                if not current_data_type == data_type:
                    continue

                face_file = open(os.path.join(subject_path, "appleFace.json"))
                left_file = open(os.path.join(subject_path, "appleLeftEye.json"))
                right_file = open(os.path.join(subject_path, "appleRightEye.json"))
                grid_file = open(os.path.join(subject_path, "faceGrid.json"))

                screen_file = open(os.path.join(subject_path, "screen.json"))

                face_json = json.load(face_file)
                left_json = json.load(left_file)
                right_json = json.load(right_file)
                
                json_avialiable = 1
                try:
                    grid_json = json.load(grid_file)
                    screen_json = json.load(screen_file)
                except:
                    json_avialiable = 0

                # print(os.path.join(subject_path, "frames", "*.jpg"))
                for idx, _ in enumerate(face_json["X"]):
                    # idx = int(n_frame[-9:-4])
                    if json_avialiable:
                        if ((not int(face_json["IsValid"][idx])) or (not int(left_json["IsValid"][idx])) or (
                            not int(right_json["IsValid"][idx])) or not int(grid_json["IsValid"][idx])):
                            continue
                    else:
                        if (not int(face_json["IsValid"][idx])) or (not int(left_json["IsValid"][idx])) or (
                            not int(right_json["IsValid"][idx])):
                            continue
                    # get face
                    tl_x_face = int(face_json["X"][idx])
                    tl_y_face = int(face_json["Y"][idx])
                    w_face = int(face_json["W"][idx])
                    h_face = int(face_json["H"][idx])
                    br_x = tl_x_face + w_face
                    br_y = tl_y_face + h_face
                    # face = img[tl_y_face:br_y, tl_x_face:br_x]

                    # get left eye
                    tl_x = tl_x_face + int(left_json["X"][idx])
                    tl_y = tl_y_face + int(left_json["Y"][idx])
                    w_left = int(left_json["W"][idx])
                    h_left = int(left_json["H"][idx])
                    br_x = tl_x + w_left
                    br_y = tl_y + h_left
                    # left_eye = img[tl_y:br_y, tl_x:br_x]

                    # get right eye
                    tr_x = tl_x_face + int(right_json["X"][idx])
                    tr_y = tl_y_face + int(right_json["Y"][idx])
                    w_right = int(right_json["W"][idx])
                    h_right = int(right_json["H"][idx])
                    br_x = tr_x + w_right
                    br_y = tr_y + h_right
                    if json_avialiable:
                        grid_x = int(grid_json["X"][idx])
                        grid_y = int(grid_json["Y"][idx])
                        grid_w = int(grid_json["W"][idx])
                        grid_h = int(grid_json["H"][idx])
                    else:
                        grid_x, grid_y, grid_w, grid_h = 0,0,0,0

                    if w_face <= 0 or h_face <= 0 \
                            or w_right <= 0 or h_right <= 0 or w_left <= 0 or h_left <= 0 \
                            or tl_x < 0 or tl_y < 0 or tr_y < 0 or tr_x < 0 \
                            or tl_x_face < 0 or tl_y_face < 0:
                        continue

                    if ori_filter and screen_json["Orientation"][idx] not in ori_filter:
                        continue

                    # right_eye = img[tr_y:br_y, tr_x:br_x]
                    self.lines.append(
                        [
                            subject,  # 0
                            idx,  # 1
                            [tl_y_face, tl_y_face + h_face, tl_x_face, tl_x_face + w_face, ],  # 2
                            [tl_y, tl_y + h_left, tl_x, tl_x + w_left, ],  # 3
                            [tr_y, tr_y + h_right, tr_x, tr_x + w_right, ],  # 4
                            [w_face, h_face,
                             tl_x_face, tl_y_face,
                             w_left, h_left,
                             tl_x, tl_y,
                             w_right, h_right,
                             tr_x, tr_y],  # 5
                            data_path,
                            [grid_y, grid_y + grid_h, grid_x, grid_x + grid_w]
                        ]
                    )

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        #   0        1     2     3     4     5     6
        # subject, frame, face, left, right, rect, (8pts), root_path

        line = self.lines[idx]

        subject_path = os.path.join(line[6], line[0])
        img = cv2.imread(os.path.join(subject_path, "frames", "%05d" % line[1] + ".jpg"))
        # img = np.array(img)
        height = img.shape[0]
        width = img.shape[1]

        # origin = copy.deepcopy(line)
        # print(line[2])
        # if not (self.data_type == 'test'):
        #     line = aug_line(copy.deepcopy(line), width, height)

        face_img = img[line[2][0]:line[2][1], line[2][2]:line[2][3]]
        leftEye_img = img[line[3][0]:line[3][1], line[3][2]:line[3][3]]
        rightEye_img = img[line[4][0]:line[4][1], line[4][2]:line[4][3]]

        # try:
        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img / 255.
        face_img = face_img.transpose(2, 0, 1)

        if self.iTracker_format:
            eye_size = (224, 224)
        else:
            eye_size = (112, 112)
        leftEye_img = cv2.resize(leftEye_img, eye_size)
        leftEye_img = cv2.cvtColor(leftEye_img, cv2.COLOR_BGR2RGB)
        leftEye_img = leftEye_img / 255.
        leftEye_img = leftEye_img.transpose(2, 0, 1)

        rightEye_img = cv2.resize(rightEye_img, eye_size)
        rightEye_img = cv2.cvtColor(rightEye_img, cv2.COLOR_BGR2RGB)
        rightEye_img = cv2.flip(rightEye_img, 1)
        rightEye_img = rightEye_img / 255.
        rightEye_img = rightEye_img.transpose(2, 0, 1)

        rects = np.array(line[5]) / np.array([width, height] * 6)
        # ex_label = line[6]
        label = np.array([self.labels[line[0]]["XCam"][line[1]],
                          self.labels[line[0]]["YCam"][line[1]]])
        if self.iTracker_format:
            face_grid = np.zeros(shape=(25, 25))
            face_grid[line[7][0]:line[7][1], line[7][2]:line[7][3]] = 1
            face_grid = np.expand_dims(face_grid, 0)
            return {"faceImg": torch.from_numpy(face_img).type(torch.FloatTensor),
                    "leftEyeImg": torch.from_numpy(leftEye_img).type(torch.FloatTensor),
                    "rightEyeImg": torch.from_numpy(rightEye_img).type(torch.FloatTensor),
                    # "rects": torch.from_numpy(rects).type(torch.FloatTensor),
                    "label": torch.from_numpy(label).type(torch.FloatTensor),
                    "grid": torch.from_numpy(face_grid).type(torch.FloatTensor),
                    "folder": line[0],
                    "frame": line[1]}
        else:
            return {"faceImg": torch.from_numpy(face_img).type(torch.FloatTensor),
                    "leftEyeImg": torch.from_numpy(leftEye_img).type(torch.FloatTensor),
                    "rightEyeImg": torch.from_numpy(rightEye_img).type(torch.FloatTensor),
                    "rects": torch.from_numpy(rects).type(torch.FloatTensor),
                    "label": torch.from_numpy(label).type(torch.FloatTensor),
                    "folder": line[0],
                    "frame": line[1]}


def txt_load(path_list, subject_list, data_type, batch_size, shuffle=False, num_workers=0,
             ban_list=None, device_="", iTracker_format=False, ori_filter=None):
    if ban_list is None:
        ban_list = []
    if ori_filter is None:
        ori_filter = []
    dataset = loader(path_list, subject_list, data_type, ban_list=ban_list, device=device_,
                     iTracker_format=iTracker_format,
                     ori_filter=ori_filter)
    print("[Read Data]: GazeCapture Dataset")
    print("[Read Data]: Total num: {:d}".format(len(dataset)))
    print("[Read Data]: Dataset type: {:s}".format(data_type))
    if type == "train":
        load = DataLoader(dataset, batch_size=batch_size, pin_memory=False, shuffle=shuffle,
                          num_workers=num_workers)
        return None, load
    else:
        load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return load


def dis(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def do_evaluation(net, test_loader, iTracker_format=False):
    net.eval()
    count = 0
    total = 0
    print("Testing")
    with torch.no_grad():
        for j, data in tqdm(enumerate(test_loader)):
            data["faceImg"] = data["faceImg"].to(device)
            data["leftEyeImg"] = data["leftEyeImg"].to(device)
            data['rightEyeImg'] = data['rightEyeImg'].to(device)

            if not iTracker_format:
                data['rects'] = data['rects'].to(device)
            else:
                data['grid'] = data['grid'].to(device)

            labels = data["label"].cpu().detach().numpy()
            sub_id = np.array(data["folder"]).reshape((-1, 1))
            frame_id = np.array(data["frame"]).reshape((-1, 1))

            if not iTracker_format:
                gazes = net(data["leftEyeImg"], data["rightEyeImg"], data['faceImg'], data['rects'])
            else:
                gazes = net(data)

            gazes = gazes.cpu().detach().numpy()
            # condi_data = np.concatenate([sub_id, frame_id, labels, gazes], axis=1)
            # condi_data = condi_data.astype(np.float32)
            # print(condi_data.shape)
            euclid_distance = np.sqrt((labels[:, 0] - gazes[:, 0]) ** 2
                                      + (labels[:, 1] - gazes[:, 1]) ** 2)
            count += euclid_distance.size
            acc = euclid_distance.sum()
            total += acc

        loger = f"Total Num: {count}, avg: {total / count} \n"
        print(loger)
    return total / count, count


def runSession(train_loader, test_loader, unique_id):
    n_epoch = 20
    print("Model building")
    net = MGazeNet.MGazeNet()

    net.train()
    if args.model_path:
      state_dict = torch.load(args.model_path, map_location='cuda:0')
      try:
          net.load_state_dict(state_dict)
          net = nn.DataParallel(net)
      except:
          net = nn.DataParallel(net)
          net.load_state_dict(state_dict)

    net.to(device)

    for k, v in net.named_parameters():
        if k not in ['module.fc.0.weight', 'module.fc.0.bias', 'module.fc.2.weight', 'module.fc.2.bias']:
            v.requires_grad_(False)
        else:
            v.requires_grad_(True)
     
    print("optimizer building")
    loss_op = nn.SmoothL1Loss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), 5e-5,
                                 weight_decay=0.0005)

    length = len(train_loader)

    time_begin = time.time()

    result_ = []
    for epoch in range(1, n_epoch + 1):
        print("Training")
        for i, (data) in enumerate(train_loader):
            data["faceImg"] = data["faceImg"].to(device)
            data["leftEyeImg"] = data["leftEyeImg"].to(device)
            data['rightEyeImg'] = data['rightEyeImg'].to(device)
            data['rects'] = data['rects'].to(device)
            label = data["label"].to(device)

            gaze = net(data["leftEyeImg"], data["rightEyeImg"], data['faceImg'], data['rects'])
            # print(gaze.shape)
            loss = loss_op(gaze[:, :2], label) * 4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            time_remain = (length - i - 1) * (
                    (time.time() - time_begin) / (i + 1)) / 3600  # time estimation for current epoch
            epoch_time = (length - 1) * ((time.time() - time_begin) / (i + 1)) / 3600  # time estimation for 1 epoch
            # person_time = epoch_time * (config["params"]["epoch"])                  #time estimation for 1 subject
            time_remain_total = time_remain + \
                                epoch_time * (n_epoch - epoch)
            # person_time * (len(subjects) - subject_i - 1)
            log = f"[{epoch}/{n_epoch}]: [{i}/{length}] loss:{loss:.5f} time:{time_remain:.2f}h total:{time_remain_total:.2f}h"
            # if i % 20 == 0:
            print(log)
            sys.stdout.flush()

        result_.append(do_evaluation(net, test_loader))
    np.save(os.path.join(result_dir, unique_id), result_)


if __name__ == '__main__':
    our_lissajous_data_path = ["/media/robotics/data_gc/GANCHENG_1TB_BK/gazeCapture-202211/lissajous/"]
    our_smooth_data_path = ["/media/robotics/data_gc/GANCHENG_1TB_BK/gazeCapture-202211/smooth/"]
    our_random_data_path = ["/media/robotics/data_gc/GANCHENG_1TB_BK/gazeCapture-202211/random/"]

    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(description='Run Fine-tuning Experiment')
    parser.add_argument('--model_path', type=str, default="", help='model path')
    parser.add_argument('--data_source', type=str, help='model path')
    args = parser.parse_args()

    if args.model_path:
        result_dir = f"result_finetuning_freezen/{args.data_source}"
    else:
        result_dir = f"result_no_finetuning/{args.data_source}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Read data")

    id_map = pd.read_excel("id_list.xlsx")

    result = []
    for sub_item in np.unique(id_map.subj.values):
        smooth_pursuit_id = id_map[(id_map.subj == sub_item) & (id_map.task == "lissajous")].folder.values
        random_id = id_map[(id_map.subj == sub_item) & (id_map.task == "random")].folder.values
        zigzag_id = id_map[(id_map.subj == sub_item) & (id_map.task == "smooth")].folder.values

        # print(random_id)
        zigzag_dataset = txt_load(our_smooth_data_path, [zigzag_id], "train", 256, num_workers=14, shuffle=True)
        random_dataset = txt_load(our_random_data_path, [random_id], "train", 256, num_workers=14, shuffle=True)
        smooth_pursuit_dataset = txt_load(our_lissajous_data_path, [smooth_pursuit_id], "train", 256, num_workers=14,
                                          shuffle=True)
        runSession(zigzag_dataset, random_dataset, "%s_zigzag_random" % sub_item)
        runSession(zigzag_dataset, smooth_pursuit_dataset, "%s_zigzag_sp" % sub_item)
        runSession(random_dataset, zigzag_dataset, "%s_random_zigzag" % sub_item)
        runSession(random_dataset, smooth_pursuit_dataset, "%s_random_sp" % sub_item)
        runSession(smooth_pursuit_dataset, random_dataset, "%s_sp_random" % sub_item)
        runSession(smooth_pursuit_dataset, zigzag_dataset, "%s_sp_zigzag" % sub_item)
