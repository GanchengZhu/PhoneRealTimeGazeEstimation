'''
Author: happycodeday yongkai_li@foxmail.com
Date: 2023-03-14 10:17:05
LastEditors: happycodeday yongkai_li@foxmail.com
LastEditTime: 2023-03-26 16:30:13
FilePath: /undefined/home/robotics2/data_ssd03/AFF-Net/train.py
Description:

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved.
'''

import argparse
import datetime
import logging
import multiprocessing
import os
import time

import numpy as np
import torch
import torch.nn as nn
import tqdm
import yaml
from torch import optim

import AFFNet
import MGazeNet
import data_reader
import itracker_model


def reduce_loss(tensor, rank, world_size):
    if rank == -1:
        return
    with torch.no_grad():
        torch.distributed.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def dis(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def configure_logger(log_dir):
    # Create a logger instance
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    # Create a file handler for logging to a file
    file_handler = logging.FileHandler(str(os.path.join(log_dir, "train_log")))
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler for logging to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Define a logging format
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S')
    # Set the formatter for both file and console handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # Add the file and console handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-rank', type=int, help="local gpu id")
    parser.add_argument('--world_size', type=int, help="num of processes")
    parser.add_argument('--config_file', type=str, help="the file of configuration")
    args = parser.parse_args()

    multiprocessing.set_forkserver_preload(["torch"])

    yaml_config = yaml.load(open(args.config_file), Loader=yaml.Loader)
    train_config = yaml_config["train"]
    path = train_config["data"]["path"]
    model_name = train_config["save"]["model_name"]

    exp_path = os.path.join(train_config["save"]["save_path"], "checkpoint")
    dataset_label = train_config["save"]["label"]

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    configure_logger(exp_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://',
                                             timeout=datetime.timedelta(seconds=3600))

    logging.info("Model building")
    if model_name == "AFFNet":
        net = AFFNet.model()
    elif model_name == "iTracker":
        net = itracker_model.ITrackerModel()
    else:
        net = MGazeNet.MGazeNet()

    net.train()
    net.to(device)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank],
                                                  output_device=args.local_rank)

    else:
        net = nn.DataParallel(net)

    logging.info("Read data")
    sampler, dataset = [], []
    if num_gpus > 1:
        itracker_format = model_name == "iTracker"
        sampler, dataset = data_reader.gaze_loader(path, "train", train_config["params"]["batch_size"],
                                                   num_workers=14, itracker_format=itracker_format)
    else:
        logging.warning("not support single gpu")

    logging.info("optimizer building")
    if model_name == "iTracker":
        lossfunc = train_config["params"]["loss"]
        loss_op = getattr(nn, lossfunc)().cuda()
    else:
        loss_op = nn.SmoothL1Loss().cuda()

    # loss_op = nn.L1Loss().cuda()
    base_lr = train_config["params"]["lr"]
    cur_step = 0
    decay_steps = train_config["params"]["decay_step"]
    optimizer = torch.optim.Adam(net.parameters(), base_lr,
                                 weight_decay=0.0005)
    logging.info("Traning")
    length = len(dataset)
    cur_decay_index = 0

    if model_name == "iTracker":
        decay_ratio = train_config["params"]["decay"]

        optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_ratio)

    for epoch in range(1, train_config["params"]["epoch"] + 1):
        net.train()

        if model_name != "iTracker":
            if cur_decay_index < len(decay_steps) and epoch == decay_steps[cur_decay_index]:
                base_lr = base_lr * train_config["params"]["decay"]
                cur_decay_index = cur_decay_index + 1
                for param_group in optimizer.param_groups:
                    param_group["lr"] = base_lr

        if num_gpus > 1:
            sampler.set_epoch(epoch)

        epoch_loss = 0
        time_begin = time.time()

        for i, (data) in tqdm.tqdm(enumerate(dataset)):
            if model_name == "iTracker":
                # Acquire data
                data["faceImg"] = data["faceImg"].to(device)
                data["leftEyeImg"] = data["leftEyeImg"].to(device)
                data['rightEyeImg'] = data['rightEyeImg'].to(device)
                data['grid'] = data['grid'].to(device)
                label = data['label'].to(device)
                # forward
                gaze = net(data)
                # loss calculation
                loss = loss_op(gaze, label)
            else:
                data["faceImg"] = data["faceImg"].to(device)
                data["leftEyeImg"] = data["leftEyeImg"].to(device)
                data['rightEyeImg'] = data['rightEyeImg'].to(device)
                data['rects'] = data['rects'].to(device)
                label = data["label"].to(device)
                gaze = net(data["leftEyeImg"], data["rightEyeImg"], data['faceImg'], data['rects'])

                if model_name == "MGazeNet":
                    loss = loss_op(gaze[:, :2], label)
                else:
                    loss = loss_op(gaze, label) * 4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            reduce_loss(loss, args.local_rank, args.world_size)
            if model_name == "iTracker":
                scheduler.step()

            time_remain = (length - i - 1) * (
                    (time.time() - time_begin) / (i + 1)) / 3600  # time estimation for current epoch
            epoch_time = (length - 1) * ((time.time() - time_begin) / (i + 1)) / 3600  # time estimation for 1 epoch
            # person_time = epoch_time * (config["params"]["epoch"])                  #time estimation for 1 subject
            time_remain_total = time_remain + \
                                epoch_time * (train_config["params"]["epoch"] - epoch)
            # person_time * (len(subjects) - subject_i - 1)
            epoch_loss += loss.detach()
            # logging.info(
            #     f"[{epoch}/{config['params']['epoch']}]: [{i}/{length}] loss:{loss.item():.5f} lr:{base_lr} time:{time_remain:.2f}h total:{time_remain_total:.2f}h")

        logging.info(f"task name: training test, epoch: {epoch}, loss: {epoch_loss / len(dataset)}")

        if num_gpus > 1 and torch.distributed.get_rank() == 0:
            # do_evaluation(net, test_dataset, task_name="test_dataset", epoch=epoch, test_config=test_config)
            if epoch % train_config["save"]["step"] == 0:
                torch.save(net.module.state_dict(),
                           os.path.join(exp_path, f"Iter_{epoch}_{model_name}_{dataset_label}.pt"))
        else:
            if epoch % train_config["save"]["step"] == 0:
                torch.save(net.module.state_dict(),
                           os.path.join(exp_path, f"Iter_{epoch}_{model_name}_{dataset_label}.pt"))
        if num_gpus > 1:
            torch.distributed.barrier()
