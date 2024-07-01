import copy
import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


# from albumentations import *
# from torch.utils.data.distributed import DistributedSampler


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


class GazeDataset(Dataset):

    def __init__(self, data_path_list, data_type, device="phone", itracker_format=False, ori_filter=None):
        self.lines = []
        self.labels = {}
        self.data_path_list = data_path_list
        self.data_type = data_type
        self.itracker_format = itracker_format
        # subjects = []

        for data_path in data_path_list:
            # data_path_raw = data_path.replace("gaze_capture_re_preprocessing", "gazecapture_raw")
            subjects = os.listdir(data_path)
            subjects.sort()
            # print(subjects)
            for subject in subjects:
                subject_path = os.path.join(data_path, subject)
                # print(subject_path)
                if ((
                        not os.path.isdir(subject_path)) or subject == '01185'
                        or subject == '02585'
                        or subject == '01730' or subject == '02065'):
                    continue

                info_json = json.load(open(os.path.join(subject_path, "info.json"), "r"))
                current_data_type = info_json["Dataset"]
                device_name = info_json["DeviceName"]

                if device not in device_name.lower():
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
                grid_json = json.load(grid_file)
                try:
                    screen_json = json.load(screen_file)
                except:
                    pass

                # print(os.path.join(subject_path, "frames", "*.jpg"))
                for idx, _ in enumerate(face_json["X"]):
                    # idx = int(n_frame[-9:-4])
                    if (not int(face_json["IsValid"][idx])) or (not int(left_json["IsValid"][idx])) or (
                            not int(right_json["IsValid"][idx])) or not int(grid_json["IsValid"][idx]):
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

                    grid_x = int(grid_json["X"][idx])
                    grid_y = int(grid_json["Y"][idx])
                    grid_w = int(grid_json["W"][idx])
                    grid_h = int(grid_json["H"][idx])

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
                # test on iPhone datas only
                # if data_type=="test" and device_name[:6]!="iPhone":
                #     continue
                # name_file = open(os.path.join(subject_path, "newFaceLdmk.json"), "r")
                # temp = json.load(name_file)

                # self.lines = self.lines + temp
                # if (len(self.lines)>=150000):
                #    break

        # random.shuffle(self.lines)
        # self.lines = self.lines[:len(self.lines) // 10]

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
        if not (self.data_type == 'test'):
            line = aug_line(copy.deepcopy(line), width, height)

        face_img = img[line[2][0]:line[2][1], line[2][2]:line[2][3]]
        leftEye_img = img[line[3][0]:line[3][1], line[3][2]:line[3][3]]
        rightEye_img = img[line[4][0]:line[4][1], line[4][2]:line[4][3]]

        # try:
        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img / 255.
        face_img = face_img.transpose(2, 0, 1)

        if self.itracker_format:
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
        # except:
        #     print("____________")
        #     # print(subject_path)
        #     print(line[0])
        #     print(line[1])
        #     print(line[5])
        rects = np.array(line[5]) / np.array([width, height] * 6)
        # ex_label = line[6]
        label = np.array([self.labels[line[0]]["XCam"][line[1]],
                          self.labels[line[0]]["YCam"][line[1]]])
        if self.itracker_format:
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
                    "frame": line[1]

                    }
        else:
            return {"faceImg": torch.from_numpy(face_img).type(torch.FloatTensor),
                    "leftEyeImg": torch.from_numpy(leftEye_img).type(torch.FloatTensor),
                    "rightEyeImg": torch.from_numpy(rightEye_img).type(torch.FloatTensor),
                    "rects": torch.from_numpy(rects).type(torch.FloatTensor),
                    "label": torch.from_numpy(label).type(torch.FloatTensor),
                    "folder": line[0],
                    "frame": line[1]
                    # "grid": torch.from_numpy(face_grid).type(torch.FloatTensor),
                    }


def gaze_loader(path_list, data_type, batch_size, shuffle=False, num_workers=0, itracker_format=False, device="phone",
                ori_filter=None):
    dataset = GazeDataset(path_list, data_type, itracker_format=itracker_format, device=device,
        ori_filter = ori_filter)
    print("[Read Data]: GazeCapture Dataset")
    print("[Read Data]: Total num: {:d}".format(len(dataset)))
    print("[Read Data]: Dataset type: {:s}".format(data_type))

    if data_type == "train":
        sampler = DistributedSampler(dataset, shuffle=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, sampler=sampler,
                            num_workers=num_workers)
        return sampler, loader
    else:
        load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return load
